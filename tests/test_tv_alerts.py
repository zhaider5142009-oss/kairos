"""TradingView alerts land in a journal, and nowhere near an order.

The webhook is the system's only inbound write from a third-party cloud, so the
tests are about what it refuses at least as much as what it accepts: a disabled
route that 404s, a blank secret that refuses everyone (an empty shared secret is
no secret, and "unset means open" is how webhooks become public write
endpoints), an unknown symbol the universe cannot price, a price that is not a
positive finite number, and a body big enough to be an attack rather than an
alert. The master API token is deliberately not required here — TradingView
would have to hold the token that also unlocks the desk's real-book reads — and
one test pins both halves of that: the webhook answers without it, and the desk
does not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from kairos import tvalerts as TV

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _body(**over):
    payload = {"symbol": "WHEAT", "side": "buy", "price": 581.5,
               "timeframe": "1D", "note": "crossed above the 50-day"}
    payload.update(over)
    return payload


# ---------------------------------------------------------------- unit
def test_a_blank_secret_refuses_every_caller():
    assert TV.secret_matches("", "") is False
    assert TV.secret_matches("anything", "") is False
    assert TV.secret_matches("s3cret", "s3cret") is True
    assert TV.secret_matches("s3cret", "other") is False


def test_parse_accepts_the_documented_shape():
    a = TV.parse(_body(), now=NOW)
    assert (a.symbol, a.side, a.price, a.timeframe) == ("WHEAT", 1, 581.5, "1D")
    assert a.received_at == "2026-10-24T12:00:00+00:00"
    assert len(a.digest) == 16


def test_a_prefixed_symbol_is_normalised():
    assert TV.parse(_body(symbol=" psx:wheat "), now=NOW).symbol == "WHEAT"


def test_sell_and_short_are_the_same_vote():
    assert TV.parse(_body(side="sell"), now=NOW).side == -1
    assert TV.parse(_body(side="SHORT"), now=NOW).side == -1


@pytest.mark.parametrize("payload", [
    [],
    {"side": "buy", "price": 1.0},
    _body(symbol="not-a-listed-instrument"),
    _body(side="hold"),
    _body(price=0),
    _body(price=-3.2),
    _body(price=float("nan")),
    _body(price=float("inf")),
    _body(price=True),
    _body(price="581.5"),
])
def test_malformed_payloads_are_refused(payload):
    with pytest.raises(TV.AlertError):
        TV.parse(payload, now=NOW)


def test_the_digest_separates_two_receipts_of_the_same_alert():
    later = datetime(2026, 10, 24, 12, 0, 1, tzinfo=timezone.utc)
    assert TV.parse(_body(), now=NOW).digest != TV.parse(_body(), now=later).digest


def test_record_appends_one_strict_line_after_an_unterminated_tail(tmp_path):
    journal = tmp_path / "tv_alerts.jsonl"
    journal.write_text('{"ts": "2026-10-24T11:59:00", "event": "TRUNCATED"', encoding="utf-8")
    path = TV.record(TV.parse(_body(), now=NOW), journal)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"ts": "2026-10-24T11:59:00", "event": "TRUNCATED"'
    record = json.loads(lines[1])
    assert record["event"] == "TV_ALERT"
    assert record["symbol"] == "WHEAT"
    assert record["side"] == 1


# ---------------------------------------------------------------- route
@pytest.fixture
def client(monkeypatch):
    from kairos.api import server as SV
    from kairos.autopilot import AUTOPILOT

    monkeypatch.setattr(AUTOPILOT, "start",
                        lambda *a, **k: {"ok": False, "error": "test stub"})
    with TestClient(SV.app) as c:
        yield c


@pytest.fixture
def webhook(monkeypatch, tmp_path):
    state = {"enabled": True, "secret": "s3cret", "max_bytes": TV.MAX_BODY,
             "journal": tmp_path / "tv_alerts.jsonl"}
    monkeypatch.setattr(TV, "settings", lambda: dict(state))
    return state


def test_a_disabled_webhook_is_not_there(client, webhook):
    webhook["enabled"] = False
    r = client.post("/api/tv/alert?secret=s3cret", json=_body())
    assert r.status_code == 404
    assert not webhook["journal"].exists()


def test_a_blank_secret_refuses_the_route(client, webhook):
    webhook["secret"] = ""
    r = client.post("/api/tv/alert?secret=", json=_body())
    assert r.status_code == 503
    assert not webhook["journal"].exists()


def test_a_wrong_secret_writes_nothing(client, webhook):
    r = client.post("/api/tv/alert?secret=guess", json=_body())
    assert r.status_code == 403
    assert not webhook["journal"].exists()


def test_a_body_that_is_not_json_is_a_400(client, webhook):
    r = client.post("/api/tv/alert?secret=s3cret", content=b"not json")
    assert r.status_code == 400
    assert not webhook["journal"].exists()


def test_an_unpriceable_symbol_is_a_422(client, webhook):
    r = client.post("/api/tv/alert?secret=s3cret", json=_body(symbol="FAKE"))
    assert r.status_code == 422
    assert not webhook["journal"].exists()


def test_an_oversized_body_is_refused_before_it_is_parsed(client, webhook):
    webhook["max_bytes"] = 64
    r = client.post("/api/tv/alert?secret=s3cret", json=_body(note="x" * 200))
    assert r.status_code == 413
    assert not webhook["journal"].exists()


def test_a_valid_alert_is_recorded(client, webhook):
    r = client.post("/api/tv/alert?secret=s3cret", json=_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["symbol"] == "WHEAT" and body["side"] == 1
    record = json.loads(webhook["journal"].read_text(encoding="utf-8").splitlines()[-1])
    assert record["digest"] == body["digest"]
    assert record["price"] == 581.5


def test_the_webhook_does_not_hand_out_the_master_token(client, webhook, monkeypatch):
    from kairos.api import server as SV

    monkeypatch.setattr(SV, "AUTH_TOKEN", "master-secret")
    monkeypatch.setattr(SV, "AUTH_MODE", "all")
    ok = client.post("/api/tv/alert?secret=s3cret", json=_body())
    assert ok.status_code == 200, ok.text
    desk = client.get("/api/desk")
    assert desk.status_code == 401
