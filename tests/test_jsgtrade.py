"""Offline tests for the JS Global order bridge.

The broker is NEVER contacted: every network call is served by a fake
session and the journal is redirected to a temp file. These assert the
payload contract, the double-key gates, and that a trading PIN can never
leak into a response, journal line or pending file.
"""

from __future__ import annotations

import json
import time

import pytest

from kairos import jsgsync, jsgtrade
from kairos.execution import AuditJournal
from kairos.jsgsync import JsGlobalError

CAPS = [{"symbol": "OGDC", "market": "REG", "upperCap": 340.0, "lowerLock": 290.0}]


class FakeResponse:
    def __init__(self, text: str = "12345", status_code: int = 200,
                 payload=None) -> None:
        self.text, self.status_code = text, status_code
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


class FakeSession:
    def __init__(self, caps=None, post_text: str = "12345") -> None:
        self.posts: list[tuple[str, dict]] = []
        self.caps = caps or []
        self.post_text = post_text

    def get(self, url: str, params=None, timeout=None) -> FakeResponse:
        if url.endswith("/Home/GetUpperLowerCap"):
            return FakeResponse("", 200, self.caps)
        return FakeResponse(json.dumps([{"accountNo": "90001",
                                         "accountName": "TEST ACCOUNT"}]))

    def post(self, url: str, data=None, timeout=None) -> FakeResponse:
        self.posts.append((url, dict(data or {})))
        return FakeResponse(self.post_text)


class Gate:
    def __init__(self, engaged: bool = False) -> None:
        self._engaged = engaged

    def engaged(self) -> bool:
        return self._engaged


def _conf(**over):
    base = dict(url="https://portal.invalid", login="tester",
                password="hunter2-test", portal_url="https://portal.invalid",
                allow_orders=False, trading_pin="", order_account="",
                market="REG", max_order_notional=250_000.0)
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def isolated_conf(monkeypatch):
    """Pin the bridge to a synthetic config; no real broker, no lockout."""
    from kairos import jsgsync
    monkeypatch.setattr(jsgsync, "_conf", lambda: _conf())
    monkeypatch.setattr(jsgtrade, "_conf", lambda: _conf())


@pytest.fixture()
def sb(tmp_path, monkeypatch):
    """Temp pending file + journal, no kill switch, order key OFF, no caps."""
    monkeypatch.setattr(jsgtrade, "PENDING_PATH", tmp_path / "pending.json")
    jrn = AuditJournal(path=tmp_path / "audit.jsonl")
    monkeypatch.setattr(jsgtrade, "JOURNAL", jrn)
    monkeypatch.setattr(jsgtrade, "KILL", Gate(False))
    monkeypatch.setattr(jsgtrade, "_caps", None)
    monkeypatch.setattr(jsgtrade, "_caps_retry_at", 0.0)
    return tmp_path, jrn


def arm(monkeypatch, ses: FakeSession, **over):
    conf = _conf(allow_orders=True, order_account="90001")
    conf.update(over)
    monkeypatch.setattr(jsgsync, "_conf", lambda: dict(conf))
    monkeypatch.setattr(jsgtrade, "_conf", lambda: dict(conf))
    monkeypatch.setattr(jsgtrade, "_sess", lambda: ses)
    return conf


def test_stage_refuses_without_allow_orders(sb):
    with pytest.raises(JsGlobalError, match="disabled"):
        jsgtrade.stage("OGDC", "BUY", 10, 300.0)


def test_kill_switch_blocks_staging(sb, monkeypatch):
    arm(monkeypatch, FakeSession(CAPS))
    monkeypatch.setattr(jsgtrade, "KILL", Gate(True))
    with pytest.raises(JsGlobalError, match="kill switch"):
        jsgtrade.stage("OGDC", "BUY", 10, 300.0)


def test_side_and_order_type_use_broker_codes(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses)
    row = jsgtrade.stage("ogdc", "sell", 25, 300.0, "stop-loss")
    out = jsgtrade.ack(row["id"], pin="1234")
    url, payload = ses.posts[0]
    assert url.endswith("/Home/PlaceOrder")
    assert payload["BuySell"] == "SEL"            # broker code, not "SELL"
    assert payload["OrderType"] == "Stop Loss"    # sell side sends the label
    assert payload["Account"] == "90001" and payload["Exchange"] == "KSE"
    assert payload["Market"] == "REG"
    assert payload["Volume"] == 25 and payload["Price"] == 300.0
    assert payload["LimitPrice"] == 0             # only LIMIT carries one
    assert out["ok"] and out["order_no"] == "12345"


def test_buy_limit_spelling_and_limit_price(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses)
    row = jsgtrade.stage("OGDC", "BUY", 10, 310.5, "limit")
    jsgtrade.ack(row["id"], pin="1234")
    _, payload = ses.posts[0]
    assert payload["BuySell"] == "BUY" and payload["OrderType"] == "Limit"
    assert payload["LimitPrice"] == 310.5


def test_account_defaults_to_portal_accountno(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses, order_account="")
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    out = jsgtrade.ack(row["id"], pin="1234")
    assert out["account"] == "90001"


def test_pin_never_leaks(sb, monkeypatch):
    tmp_path, jrn = sb
    ses = FakeSession(CAPS, post_text="Order 778 accepted PIN 1234")
    arm(monkeypatch, ses)
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    out = jsgtrade.ack(row["id"], pin="1234")
    assert "1234" not in json.dumps(out) and "***" in out["response"]
    assert "1234" not in (tmp_path / "pending.json").read_text(encoding="utf-8")
    assert "1234" not in json.dumps(jrn.tail(50))


def test_price_outside_broker_band_rejected(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses)
    with pytest.raises(JsGlobalError, match="band"):
        jsgtrade.stage("OGDC", "BUY", 5, 999.0)
    assert ses.posts == []


def test_a_nan_price_is_refused_before_the_broker(sb, monkeypatch):
    """Every guard on this path is a comparison, and NaN loses all of them.

    `price <= 0` is False, `qty * price > cap` is False, `lo <= price <= hi` is
    False, so a hole in the price feed used to hand the broker an order with an
    unusable price while every check above it reported a clean pass.
    """
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(JsGlobalError, match="finite"):
            jsgtrade.stage("OGDC", "BUY", 5, bad)
    with pytest.raises(JsGlobalError, match="finite"):
        jsgtrade.stage("OGDC", "BUY", float("nan"), 300.0)
    assert ses.posts == []


def test_an_unreadable_band_is_written_down_not_skipped(sb, monkeypatch):
    """The bridge claims a price band; the journal has to say whether it ran.

    A caps fetch that fails is cached for five minutes because every retry
    means another portal login and this account locks out on repetition, so
    staging proceeds without the band and the broker enforces it. That is a
    deliberate trade, not a bug — what was missing is the record. An order that
    went out unchecked and one that was checked read identically afterwards.
    """
    _tmp, jrn = sb
    ses = FakeSession(CAPS)              # the fake caps table covers OGDC only
    arm(monkeypatch, ses)
    jsgtrade.stage("ENGRO", "BUY", 5, 300.0)
    events = [r["event"] for r in jrn.tail(50)]
    assert "JSG_BAND_UNCHECKED" in events
    assert "JSG_STAGED" in events


def test_notional_cap_enforced(sb, monkeypatch):
    arm(monkeypatch, FakeSession(CAPS))
    with pytest.raises(JsGlobalError, match="cap"):
        jsgtrade.stage("OGDC", "BUY", 100_000, 300.0)


def test_unknown_side_or_market_fails_closed(sb, monkeypatch):
    arm(monkeypatch, FakeSession(CAPS))
    with pytest.raises(JsGlobalError):
        jsgtrade.stage("OGDC", "HOLD", 5, 300.0)
    with pytest.raises(JsGlobalError):
        jsgtrade.stage("OGDC", "BUY", 5, 300.0, market="FOREX")
    with pytest.raises(JsGlobalError):
        jsgtrade.stage("OGDC", "BUY", 5, 300.0, order_type="ICEBERG")


def test_pending_expires_after_ttl(sb, monkeypatch):
    arm(monkeypatch, FakeSession(CAPS))
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    rows = jsgtrade._load()
    rows[0]["t"] = time.time() - jsgtrade.TTL_S - 1
    jsgtrade._save(rows)
    assert jsgtrade.pending() == []
    with pytest.raises(JsGlobalError, match="no pending order"):
        jsgtrade.ack(row["id"], pin="1234")


def test_missing_pin_blocks_ack(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses)
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    with pytest.raises(JsGlobalError, match="PIN required"):
        jsgtrade.ack(row["id"])          # nothing stored — human must type
    assert ses.posts == []


def test_scrub_is_whole_token_only():
    # a 4-digit PIN is often a substring of the order number it returns
    assert jsgtrade._scrub("12345", "1234") == "12345"
    assert jsgtrade._scrub("Order 9 PIN 1234 rejected", "1234") == \
        "Order 9 PIN *** rejected"


def test_rejected_broker_answer_is_not_ok(sb, monkeypatch):
    arm(monkeypatch, FakeSession(CAPS, post_text="Invalid PIN"))
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    assert jsgtrade.ack(row["id"], pin="1234")["ok"] is False


def test_transport_loss_drops_the_staged_copy(sb, monkeypatch):
    """A POST that died in transit must not stay re-ackable — that is how a
    blind retry becomes two real orders."""
    import requests

    ses = FakeSession(CAPS)

    def boom(url, data=None, timeout=None):
        raise requests.exceptions.ReadTimeout("read timed out")

    ses.post = boom
    arm(monkeypatch, ses)
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    with pytest.raises(JsGlobalError, match="may or may not"):
        jsgtrade.ack(row["id"], pin="1234")
    assert jsgtrade.pending() == []


def test_failed_band_fetch_does_not_relogin(sb, monkeypatch):
    """Every caps retry is another portal login attempt — the account locks
    out on repetition, so a failure gets a negative cache."""
    calls = {"n": 0}

    class Dead(FakeSession):
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            raise RuntimeError("portal unreachable")

    arm(monkeypatch, Dead(CAPS))
    assert jsgtrade._cap_map() == {}
    jsgtrade.band("OGDC")
    jsgtrade.band("ENGRO")
    assert calls["n"] == 1                       # one attempt, then cooled off


def test_place_now_uses_stored_pin(sb, monkeypatch):
    ses = FakeSession(CAPS)
    arm(monkeypatch, ses, trading_pin="4321")
    assert jsgtrade.enabled()
    out = jsgtrade.place_now("OGDC", "BUY", 5, 300.0)
    assert out["ok"] and ses.posts[0][1]["PIN"] == "4321"
    assert jsgtrade.pending() == []
