"""The audit trail is the product, so it cannot be allowed to stop silently.

`AuditJournal.log` wrapped every write in `except Exception: pass` and `tail`
answered any problem by returning nothing at all. Both are the shape that made
the pack's central claim — "every prediction, gate decision, arming event and
kill-switch flip appended to on-disk JSONL" — unfalsifiable: a full disk, a
locked file or one torn line turned the trail into an empty one, and an empty
trail and a dead one print the same words.

Nothing here touches a broker: the order doors are tested against a fake
terminal and a stubbed stage.
"""

import json

import pytest

from kairos import execution as E
from kairos import jsgtrade
from kairos.execution import AuditJournal, Mt5Bridge, OrderIntent


@pytest.fixture
def journal(tmp_path):
    return AuditJournal(path=tmp_path / "audit.jsonl")


def _blocked(tmp_path):
    """A path whose parent is a regular file — no write can land there."""
    blocker = tmp_path / "blocker"
    blocker.write_text("something replaced the directory", encoding="utf-8")
    return blocker / "audit.jsonl"


# -------------------------------------------------------------------- writing
def test_a_record_lands_as_its_own_line(journal):
    assert journal.log("HELLO", {"a": 1}) is True
    journal.log("HELLO", {"a": 2})
    assert [r["a"] for r in journal.tail()] == [1, 2]
    assert journal.last_error is None


def test_a_torn_last_line_does_not_swallow_the_next_record(journal):
    """A half-written row is what a crash mid-append, a copied file or a
    restored backup leaves behind. Appending onto it welds two records into one
    line no parser accepts, and the trail loses both of them."""
    journal.path.write_text('{"event": "ONE", "a": 1}\n{"event": "TORN", "a"',
                            encoding="utf-8")
    assert journal.log("TWO", {"a": 2}) is True
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "TWO"
    assert json.loads(lines[0])["a"] == 1
    assert [r["event"] for r in journal.tail(50)] == ["ONE", "TWO"]
    assert journal.skipped == 1


def test_a_failed_write_is_reported_instead_of_swallowed(tmp_path, journal):
    journal.path = _blocked(tmp_path)
    assert journal.log("HELLO", {"a": 1}) is False
    assert journal.last_error and "HELLO" in journal.last_error


def test_the_error_clears_when_the_trail_recovers(tmp_path, journal):
    journal.path = _blocked(tmp_path)
    assert journal.log("OFFLINE", {}) is False
    journal.path = tmp_path / "audit.jsonl"
    assert journal.log("BACK", {"a": 1}) is True
    assert journal.last_error is None
    assert journal.tail()[-1]["event"] == "BACK"


# ---------------------------------------------------------------------- reading
def test_one_corrupt_line_does_not_lose_the_trail(journal):
    journal.log("A", {})
    with journal.path.open("a", encoding="utf-8") as fh:
        fh.write("this line is not json\n")
    journal.log("B", {})
    assert [r["event"] for r in journal.tail(50)] == ["A", "B"]
    assert journal.skipped == 1


def test_a_missing_file_is_an_empty_trail_not_a_broken_one(journal):
    assert journal.tail() == []
    assert journal.skipped == 0
    assert journal.last_error is None


def test_a_trail_that_cannot_be_read_says_so(tmp_path, journal):
    journal.log("A", {})
    unreadable = tmp_path / "not-a-file"
    unreadable.mkdir()
    journal.path = unreadable
    assert journal.tail() == []
    assert journal.skipped == 0
    assert journal.last_error and "unreadable" in journal.last_error


def test_the_desk_endpoint_reports_damage_not_an_empty_ok(tmp_path, monkeypatch):
    """The desk journal panel is built on this response, so an `ok` with no
    rows is exactly how a dead trail gets read as a quiet session."""
    from kairos.api import server
    jrn = AuditJournal(path=tmp_path / "audit.jsonl")
    jrn.log("A", {})
    noise = tmp_path / "noise"
    noise.mkdir()
    jrn.path = noise
    monkeypatch.setattr(E, "JOURNAL", jrn)
    out = server.journal(last=10)
    assert out["entries"] == []
    assert out["ok"] is False and "unreadable" in out["error"]


# ------------------------------------------------------- the money-path promise
class _DeadJournal:
    """A trail that answers, and answers no."""

    last_error = "disk full"
    skipped = 0

    def log(self, event, payload=None):
        return False

    def tail(self, n: int = 20):
        return []


def test_no_psx_order_is_staged_when_the_trail_cannot_record(monkeypatch):
    """An order with no record behind it is unauditable by construction."""
    monkeypatch.setattr(jsgtrade, "JOURNAL", _DeadJournal())
    reached: list = []
    monkeypatch.setattr(jsgtrade, "stage", lambda *a, **k: reached.append(a))
    with pytest.raises(jsgtrade.JsGlobalError):
        jsgtrade.place_now("OGDC", "BUY", 5, 300.0)
    assert reached == []


class _Terminal:
    """A fake MetaTrader module: everything answers, until it is asked to send."""

    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10000
    TRADE_RETCODE_DONE_PARTIAL = 10001

    def __init__(self) -> None:
        self.sent = 0

    def symbol_info(self, symbol):
        return type("I", (), {"visible": True})()

    def symbol_info_tick(self, symbol):
        return type("T", (), {"bid": 99.9, "ask": 100.1})()

    def order_send(self, request):
        self.sent += 1
        raise AssertionError("a live order reached the terminal in an offline test")


def _bridge() -> Mt5Bridge:
    b = Mt5Bridge()
    b.connected = True
    b._mt5 = _Terminal()
    return b


def test_no_mt5_order_is_sent_when_the_trail_cannot_record(monkeypatch):
    monkeypatch.setattr(E, "JOURNAL", _DeadJournal())
    b = _bridge()
    assert b.place_order(OrderIntent(symbol="XAUUSD", side="BUY",
                                     quantity=0.01, venue="MT5")) is None
    assert b._mt5.sent == 0


def test_an_mt5_order_goes_out_while_the_trail_is_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "JOURNAL", AuditJournal(path=tmp_path / "audit.jsonl"))

    class _Send(_Terminal):
        def order_send(self, request):
            self.sent += 1
            return type("R", (), {"retcode": 10000, "order": 4321,
                                  "symbol": "XAUUSD", "price": 100.1,
                                  "volume": 0.01, "comment": ""})()

    b = _bridge()
    b._mt5 = _Send()
    fill = b.place_order(OrderIntent(symbol="XAUUSD", side="BUY",
                                     quantity=0.01, venue="MT5"))
    assert b._mt5.sent == 1
    assert fill is not None and fill.broker_order_id == "4321"
