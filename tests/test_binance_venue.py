"""Offline tests for Binance — the third venue.

Nothing here touches the network: every request is served by a fake `_send`, and
the journal and kill switch are redirected to a temp directory. What is being
pinned down is the shape the other two venues already have to obey — a refusal
that names which door is shut, an order that is journaled before it is sent, a
transport failure that is reported as unconfirmed rather than as a rejection, an
account that answers `None` instead of a flattering zero, and a signature that
never reaches a log line.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from kairos import binance as B
from kairos import data as D
from kairos.config import CFG
from kairos.execution import Fill, OrderIntent


CONF = {"api_key": "111111", "secret_key": "99999",
        "base_url": "https://binance.invalid", "allow_orders": True,
        "timeout_s": 5}
PERMIT = {"enableReading": True, "enableSpotAndMarginTrading": True,
          "ipRestrict": False}
ROWS = [[1735689600000 + i * 86400000, "100", "105", "99", "104", "1000"]
        for i in range(20)]


def bars(rows: list[list]) -> pd.DataFrame:
    """A kline-shaped frame — what `BinanceClient.klines` really hands back.

    The stubs return this rather than a raw list-of-lists so a test that saves to
    the store exercises the same columns the production client produces.
    """
    return pd.DataFrame(
        [{"date": pd.to_datetime(r[0], unit="ms").tz_localize(None).date(),
          "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
          "close": float(r[4]), "volume": float(r[5])} for r in rows])


class StubBinance:
    """The data layer's view of the venue: bars on request, counted."""

    def __init__(self, rows: list[list] | None = None, alive: bool = True):
        self.rows = rows if rows is not None else ROWS
        self._alive = alive
        self.pairs: list[str] = []

    def klines(self, pair, interval="1d", limit=500):
        self.pairs.append(pair)
        return bars(self.rows)

    def alive(self):
        return self._alive


class StubMT5:
    """The MT5 half of the loop: enough for symbol lookup, no terminal."""

    def resolve_symbol(self, key: str) -> str | None:
        return key



class FakeSend:
    """Stands in for the transport and remembers every call in order."""

    def __init__(self, replies: dict[str, object] | None = None,
                 fail: Exception | None = None) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, path, params=None, signed=False, method="GET"):
        self.calls.append((path, method, signed))
        if self.fail is not None:
            raise self.fail
        for key, reply in self.replies.items():
            if path.startswith(key):
                if callable(reply):
                    return reply(path, params)
                return reply
        return {}

    def paths(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture()
def venue(tmp_path, monkeypatch):
    """A client with a key pair, an armed config and a journal on a temp disk."""
    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    monkeypatch.setattr(B, "_conf", lambda: dict(CONF))
    monkeypatch.setattr(B.JOURNAL, "path", tmp_path / "audit.jsonl")
    monkeypatch.setattr(B.JOURNAL, "last_error", None)
    client = B.BinanceClient()
    client._perms = dict(PERMIT)          # armed without a network read
    client._perms_at = 9e12
    events: list[tuple[str, dict]] = []
    real_log = B.JOURNAL.log

    def spy(event: str, payload: dict) -> bool:
        events.append((event, payload))
        return real_log(event, payload)

    monkeypatch.setattr(B.JOURNAL, "log", spy)
    client.events = events                # type: ignore[attr-defined]
    return client


def _arm(monkeypatch, client, **over):
    """Point the module-level gates at this fake client."""
    conf = dict(CONF)
    conf.update(over)
    monkeypatch.setattr(B, "_conf", lambda: dict(conf))
    monkeypatch.setattr(B, "BINANCE", client)


def _section(monkeypatch, conf: dict):
    """Point the module's *real* config reader at one section.

    Patching `B._conf` itself would test the stub: the coercion of a YAML string
    lives inside `_conf`, and an armed gate that is only correct while the reader
    is unpatched is not a gate worth having.
    """
    for var in ("BINANCE_API_KEY", "BINANCE_SECRET", "BINANCE_ALLOW_ORDERS",
                "BINANCE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(CFG, "section",
                        lambda name: dict(conf) if name == "binance" else {})


def test_enabled_needs_both_keys(monkeypatch):
    _section(monkeypatch, CONF)
    assert B.configured() and B.enabled()
    _section(monkeypatch, {**CONF, "allow_orders": False})
    assert B.configured() and not B.enabled()
    _section(monkeypatch, {**CONF, "secret_key": ""})
    assert not B.configured() and not B.enabled()
    _section(monkeypatch, {**CONF, "api_key": ""})
    assert not B.configured() and not B.enabled()
    # a config file's "false" is a string, and a non-empty string is truthy
    _section(monkeypatch, {**CONF, "allow_orders": "false"})
    assert not B.enabled()
    _section(monkeypatch, {**CONF, "allow_orders": "0"})
    assert not B.enabled()
    _section(monkeypatch, {**CONF, "allow_orders": "true"})
    assert B.enabled()


def test_env_overrides_arm_the_venue(monkeypatch):
    _section(monkeypatch, CONF)
    monkeypatch.setenv("BINANCE_ALLOW_ORDERS", "0")
    assert not B.enabled()          # the env key is a second author, not a rubber
    monkeypatch.setenv("BINANCE_API_KEY", "4321")
    assert B.configured() and B._conf()["api_key"] == "4321"


def test_disabled_sends_nothing(venue, monkeypatch):
    _arm(monkeypatch, venue, allow_orders=False)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5, venue=B.VENUE)
    assert venue.place_order(intent) is None
    assert fake.calls == []
    assert venue.events[0][0] == "BINANCE_REFUSED"
    assert "allow_orders" in venue.events[0][1]["reason"]


def test_kill_switch_blocks_an_armed_venue(venue, monkeypatch):
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    from kairos.execution import KILL
    monkeypatch.setattr(KILL, "engaged", lambda: True)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5)
    assert venue.place_order(intent) is None
    assert fake.calls == []
    assert "kill switch" in venue.events[0][1]["reason"]
    # refused, not merely absent: the caller can tell "we never asked" apart from
    # "we asked and the venue did not answer"
    assert venue.outcome(intent.client_order_id) == "REFUSED"


def test_readonly_key_is_named_not_rejected(venue, monkeypatch):
    venue._perms = {"enableReading": True, "enableSpotAndMarginTrading": False}
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    assert venue.place_order(OrderIntent(symbol="BTC", side="BUY",
                                         quantity=0.5)) is None
    assert fake.calls == []
    reason = venue.events[0][1]["reason"]
    assert "enableSpotAndMarginTrading=false" in reason
    assert "recreate" in reason


def test_unreadable_permission_is_unknown_not_permitted(venue, monkeypatch):
    venue._perms = {}                          # the probe answered nothing
    monkeypatch.setattr(B, "BINANCE", venue)
    fake = FakeSend(fail=B.BinanceError("refused by stub"))
    monkeypatch.setattr(venue, "_send", fake)
    ok, why = venue.tradeable()
    assert ok is False
    assert "unknown" in why and "nothing was sent" in why


def test_a_name_this_venue_has_no_market_for(venue, monkeypatch):
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    assert venue.place_order(OrderIntent(symbol="OGDC", side="BUY",
                                         quantity=100)) is None
    assert fake.calls == []
    assert "no Binance spot pair" in venue.events[0][1]["reason"]


def test_intent_lands_before_the_request(venue, monkeypatch):
    _arm(monkeypatch, venue)
    fake = FakeSend({"/api/v3/order": {"orderId": 4321, "status": "FILLED",
                                       "symbol": "BTCUSDT", "executedQty": "0.5",
                                       "fills": [{"price": "104.0", "qty": "0.5",
                                                  "commission": "0.05"}]}})
    monkeypatch.setattr(venue, "_send", fake)
    fill = venue.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))
    assert isinstance(fill, Fill) and fill.venue == B.VENUE
    assert [e[0] for e in venue.events] == ["BINANCE_INTENT", "BINANCE_PLACED"]
    assert venue.events[0][1]["cid"] == fill.client_order_id
    assert fake.paths()[0] == "/api/v3/order"      # only after the journal row


def test_failing_journal_stops_the_order(venue, monkeypatch):
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    monkeypatch.setattr(venue, "tradeable", lambda: (True, "spot trading permitted"))
    monkeypatch.setattr(B.JOURNAL, "log", lambda *a, **k: False)
    monkeypatch.setattr(B.JOURNAL, "last_error", None)
    assert venue.place_order(OrderIntent(symbol="BTC", side="BUY",
                                         quantity=0.5)) is None
    assert fake.calls == []


def test_an_unanswered_post_is_unconfirmed_not_refused(venue, monkeypatch):
    _arm(monkeypatch, venue)
    err = B.BinanceError("/api/v3/order unreachable (URLError: name not resolved)")
    monkeypatch.setattr(venue, "_send", FakeSend(fail=err))
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5)
    assert venue.place_order(intent) is None
    assert venue.outcome(intent.client_order_id) == "UNCONFIRMED"
    assert venue.events[-1][0] == "BINANCE_UNCONFIRMED"
    assert venue.outcome("not-my-order") == "UNKNOWN"


def test_a_signed_failure_never_carries_the_signature(venue, monkeypatch):
    body = (b'{"msg":"Invalid"}&signature=deadbeef1234&recvWindow=5000')
    exc = urllib.error.HTTPError("https://binance.invalid/api/v3/account"
                                 "?signature=deadbeef1234", 400, "Bad", {}, None)
    exc.read = lambda: body                       # type: ignore[method-assign]

    def fake_urlopen(req, timeout=None):
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(B.BinanceError) as got:
        venue._send("/api/v3/account", {"a": 1}, signed=True)
    text = str(got.value)
    assert "deadbeef" not in text and "signature=[redacted]" in text
    assert venue.error == text


def test_limit_order_carries_price_and_quantity(venue, monkeypatch):
    _arm(monkeypatch, venue)
    seen: dict = {}

    def capture(path, params=None, signed=False, method="GET"):
        seen.update(params or {})
        return {"orderId": 1234, "status": "NEW", "symbol": "BTCUSDT",
                "executedQty": "0", "fills": []}

    monkeypatch.setattr(venue, "_send", capture)
    fill = venue.place_order(OrderIntent(symbol="BTC", side="SELL", quantity=0.25,
                                         order_type="LIMIT", limit_price=9999.0))
    assert fill is not None and fill.status == "NEW"
    assert seen["symbol"] == "BTCUSDT" and seen["timeInForce"] == "GTC"
    assert seen["price"] == "9999" and seen["quantity"] == "0.25"
    assert "quoteOrderQty" not in seen


def test_limit_without_a_price_is_refused(venue, monkeypatch):
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    assert venue.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=1,
                                         order_type="LIMIT")) is None
    assert fake.calls == []


def test_a_number_the_venue_cannot_express_is_refused(venue, monkeypatch):
    """1e-9 is a positive finite number, and as text it is "0".

    An order for zero quantity, or a limit at zero, is not a small order — it is
    a wrong one, and the journal must say so instead of the venue answering.
    """
    _arm(monkeypatch, venue)
    fake = FakeSend()
    monkeypatch.setattr(venue, "_send", fake)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=1e-9)
    assert venue.place_order(intent) is None
    assert fake.calls == []
    assert venue.outcome(intent.client_order_id) == "REFUSED"
    assert "can express" in venue.events[-1][1]["reason"]


def test_account_none_on_failure_never_zero(venue, monkeypatch):
    monkeypatch.setattr(venue, "_send", FakeSend(fail=B.BinanceError("down")))
    assert venue.account() is None
    assert venue.positions() is None


def test_account_prices_holdings_and_names_what_it_could_not(venue, monkeypatch):
    def balances(path, params=None, signed=False, method="GET"):
        if path.startswith("/api/v3/account"):
            return {"canTrade": True, "balances": [
                {"asset": "USDT", "free": "1000.0", "locked": "0"},
                {"asset": "BTC", "free": "0.5", "locked": "0"},
                {"asset": "WEIRD", "free": "3.0", "locked": "0"},
                {"asset": "ETH", "free": "0", "locked": "0"}]}
        if path.startswith("/api/v3/ticker/price"):
            return {"price": {"BTCUSDT": "64000.0", "WEIRDUSDT": ""}[
                (params or {})["symbol"]]}
        raise AssertionError(path)

    monkeypatch.setattr(venue, "_send", balances)
    acc = venue.account()
    assert acc is not None
    assert acc["equity"] == pytest.approx(1000.0 + 0.5 * 64000.0)
    assert acc["unvalued"] == ["WEIRD"]
    assert "WEIRD" in acc["note"]
    # an asset with no price is not valued at zero and is not silently dropped
    assert acc["open_positions"] == 3
    pos = venue.positions()
    assert pos is not None
    syms = {p["symbol"] for p in pos}
    assert syms == {"BTCUSDT", "WEIRDUSDT"}      # USDT is cash, not a position
    btc = next(p for p in pos if p["symbol"] == "BTCUSDT")
    assert btc["qty"] == 0.5 and btc["last"] == 64000.0
    assert btc["entry"] is None and btc["pnl"] is None
    weird = next(p for p in pos if p["symbol"] == "WEIRDUSDT")
    assert weird["last"] is None                 # held, counted, honestly unpriced


def test_klines_land_in_the_store_column_order(venue):
    venue._send = lambda path, params=None, signed=False, method="GET": ROWS
    df = venue.klines("BTCUSDT")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(df) == 20 and float(df["close"].iloc[-1]) == 104.0


def test_state_reads_no_network_and_no_secret(venue, monkeypatch):
    _arm(monkeypatch, venue)
    monkeypatch.setattr(venue, "_send",
                        FakeSend(fail=AssertionError("state() must not call out")))
    st = venue.state()
    assert st["armed"] is True and st["probed"] is False
    assert st["key_can_trade"] is True and st["perms_probed"] is True
    assert json.dumps(st)
    for text in (json.dumps(st), str(st)):
        assert "111111" not in text and "99999" not in text


def test_clock_skew_is_measured_once_before_a_signed_call(venue, monkeypatch):
    calls: list[str] = []

    class Resp:
        def read(self):
            return json.dumps({"serverTime": 1735689600000000}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", ""))
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    venue._send("/api/v3/account", {}, signed=True)
    venue._send("/api/v3/account", {}, signed=True)
    assert len([c for c in calls if "/api/v3/time" in c]) == 1
    assert venue._clock_skew_ms != 0


def test_signed_call_without_a_secret_is_not_attempted(venue, monkeypatch):
    monkeypatch.setattr(B, "_conf", lambda: {**CONF, "secret_key": ""})
    venue._perms, venue._perms_at = None, 0.0     # nothing probed yet

    def no_transport(req, timeout=None):
        raise AssertionError("a signed request went out with no secret to sign it")

    monkeypatch.setattr(urllib.request, "urlopen", no_transport)
    assert venue.account() is None
    # a None with no reason attached is the same shape as an outage
    assert "needs a key pair" in (venue.error or "")
    assert venue.positions() is None
    # without a key pair even the key's own permissions are unknowable
    assert venue.permissions() == {}
    ok, why = venue.tradeable()
    assert ok is False and "unknown" in why


# ------------------------------------------------------------------ registry
def test_crypto_names_carry_a_pair_and_resolve_from_it():
    from kairos.universe import resolve

    btc = resolve("BTC")
    assert btc is not None and btc.binance == "BTCUSDT"
    assert resolve("BTCUSDT") is btc
    assert resolve("OGDC").binance is None
    assert resolve("GOLD").binance is None


# ------------------------------------------------------------------ data layer
def test_binance_answers_when_yahoo_is_down(monkeypatch, tmp_path):
    from kairos.universe import resolve

    monkeypatch.setattr(D.STORE, "dir", tmp_path)
    md = D.MarketData()
    stub = StubBinance()
    monkeypatch.setattr(D, "_binance", lambda: stub)
    monkeypatch.setattr(md.yahoo, "history", lambda sym, days=None: None)
    df = md.history(resolve("BTC"), use_cache=True)
    assert df is not None and len(df) == 20
    assert stub.pairs == ["BTCUSDT"]
    # the failover must not overwrite the accumulated Yahoo record
    assert not list(tmp_path.glob("BTC.csv"))


def test_a_name_yahoo_does_not_carry_is_persisted_from_the_venue(monkeypatch,
                                                                 tmp_path):
    from kairos.universe import Instrument

    monkeypatch.setattr(D.STORE, "dir", tmp_path)
    md = D.MarketData()
    away = Instrument(key="TESTCOIN", name="Test Coin", asset_class="crypto",
                      sector="Crypto", currency="USD", market="GLOBAL",
                      binance="TESTUSDT")
    stub = StubBinance()
    monkeypatch.setattr(D, "_binance", lambda: stub)
    df = md.history(away, use_cache=True)
    assert df is not None and (tmp_path / "TESTCOIN.csv").exists()
    assert stub.pairs == ["TESTUSDT"]
    # second read is served from the 900 s cache, not a second request
    assert md.history(away, use_cache=True) is not None
    assert stub.pairs == ["TESTUSDT"]
    stored = D.STORE.load("TESTCOIN")
    assert stored is not None and len(stored) == 20
    assert float(stored["close"].iloc[-1]) == float(ROWS[-1][4])


# ------------------------------------------------------------------- health
def test_binance_is_listed_but_alone_does_not_read_online(monkeypatch):
    from kairos import health as H
    from kairos.data import SourceHealth

    class MD:
        def health(self, force=False):
            return SourceHealth(binance=True)     # every primary source down

    class Eng:
        md = MD()

    import kairos.engine as E
    monkeypatch.setattr(E, "ENGINE", Eng())
    monkeypatch.setattr(H, "WSL", type("W", (), {"wsl_available": staticmethod(
        lambda: False), "models_ready": staticmethod(lambda: {})})())
    out = H.system_health()
    assert out["market_data"]["binance"] == "ok"
    assert out["data_online"] is False


# ---------------------------------------------------------------- autopilot
def _loop(monkeypatch, tmp_path, bn_armed: bool):
    """An Autopilot with just enough state to run the venue-facing helpers.

    Built with `__new__` so no thread starts and no broker attaches; `mt5` is a
    stub because symbol lookup asks both venues what name it would use.
    """
    from kairos import autopilot as AP

    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    ap = AP.Autopilot.__new__(AP.Autopilot)
    ap.st = ap._defaults()
    ap._bn_armed = bn_armed
    ap._thread = None
    ap._lock = threading.Lock()
    ap.mt5 = StubMT5()
    from kairos.execution import JOURNAL
    monkeypatch.setattr(JOURNAL, "path", tmp_path / "audit.jsonl")
    ap.acted: list[str] = []          # type: ignore[attr-defined]
    monkeypatch.setattr(ap, "_act", lambda *a: ap.acted.append(str(a)))
    return ap


def test_crypto_goes_to_binance_only_when_armed(monkeypatch, tmp_path):
    from kairos.universe import resolve

    ap = _loop(monkeypatch, tmp_path, bn_armed=False)
    btc = resolve("BTC")
    assert ap._venue_symbols(btc.key, False, True) == [btc.key.upper()]
    ap._bn_armed = True
    syms = ap._venue_symbols(btc.key, False, True)
    assert syms[0] == "BTCUSDT"
    assert btc.key.upper() in syms     # the MT5 candidate is still checked
    assert ap._venue_symbols("OGDC", True, True) == ["OGDC"]


def test_reconcile_keeps_a_binance_entry_when_mt5_is_down(monkeypatch, tmp_path):
    ap = _loop(monkeypatch, tmp_path, bn_armed=True)
    ap.st["held"] = {"BTC": {"qty": 0.5}}
    held = {"BTCUSDT": {"symbol": "BTCUSDT", "qty": 0.5}}
    ap._reconcile(held, live=True, mt5_up=False, psx_up=False, pos_known=True,
                  bn_up=True)
    assert ap.st["held"] == {"BTC": {"qty": 0.5}}
    assert ap.acted == []


def test_reconcile_still_drops_a_vanished_binance_entry(monkeypatch, tmp_path):
    ap = _loop(monkeypatch, tmp_path, bn_armed=True)
    ap.st["held"] = {"BTC": {"qty": 0.5}}
    ap._reconcile({}, live=True, mt5_up=False, psx_up=False, pos_known=True,
                  bn_up=True)
    assert ap.st["held"] == {}
    assert ap.acted


def test_reconcile_keeps_crypto_when_no_venue_answered(monkeypatch, tmp_path):
    """A cycle where neither terminal nor exchange replied is not a flat book."""
    ap = _loop(monkeypatch, tmp_path, bn_armed=False)
    ap.st["held"] = {"BTC": {"qty": 0.5}}
    ap._reconcile({}, live=True, mt5_up=False, psx_up=False, pos_known=True,
                  bn_up=False)
    assert ap.st["held"] == {"BTC": {"qty": 0.5}}
    assert ap.acted == []


def test_paper_reconcile_still_runs(monkeypatch, tmp_path):
    """The venue guards must not switch off reconciliation in paper mode."""
    ap = _loop(monkeypatch, tmp_path, bn_armed=False)
    ap.st["held"] = {"BTC": {"qty": 0.5}}
    ap._reconcile({}, live=False, mt5_up=False, psx_up=False, pos_known=True)
    assert ap.st["held"] == {}
    assert ap.acted


def test_a_spot_answer_with_no_fills_array_still_reports_what_it_paid(
        venue, monkeypatch):
    """Spot's order response names no `fills` at all — only the quote value that
    was spent. Reading the missing array as the price made every live Binance
    fill look like it was struck for free."""
    _arm(monkeypatch, venue)
    monkeypatch.setattr(venue, "_send", FakeSend({
        "/api/v3/order": {"orderId": 7, "status": "FILLED", "symbol": "BTCUSDT",
                          "executedQty": "0.5", "cummulativeQuoteQty": "52.0"}}))

    fill = venue.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))

    assert fill is not None
    assert fill.quantity == 0.5
    assert fill.price == 104.0, "a live fill was journalled as free"


def test_an_answer_with_no_executed_quantity_is_not_a_trade(venue, monkeypatch):
    """A Fill carries what left the venue. Falling back to the quantity that was
    asked for turns an empty answer into a booked trade."""
    _arm(monkeypatch, venue)
    monkeypatch.setattr(venue, "_send", FakeSend({
        "/api/v3/order": {"orderId": 8, "status": "EXPIRED", "symbol": "BTCUSDT",
                          "cummulativeQuoteQty": "0"}}))

    fill = venue.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))

    assert fill is not None
    assert fill.quantity == 0.0, "the requested size was reported as executed"
    assert fill.price == 0.0
    assert fill.status == "EXPIRED"
