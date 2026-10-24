"""A live fill is booked, so the exits that protect it exist.

`st["held"]` is where the stop, the target, the entry date and the trailing
ratchet live. It used to be written only for paper fills and PSX tickets, so an
MT5 or Binance position the venue confirmed came back as `LIVE-FILLED` and was
managed by nothing: the stop and target had no row to be read from, the ratchet
had nothing to ratchet, and maturity had no entry date to age from. The only way
such a position could get out was a signal flip.

These run the real cycle against a stubbed terminal — book the live entry, then
drive the tape through its stop and expect the position to be closed at the
venue, not just in the book.
"""

from __future__ import annotations

import kairos.autopilot as AP
from test_autopilot_cycle import _AGREE
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_psx_exit_working import _Journal


def test_a_confirmed_live_entry_is_booked_with_its_risk_row(monkeypatch, tmp_path):
    res = _prediction(monkeypatch, battery=_AGREE)
    term = _Terminal([])
    ap = _autopilot(monkeypatch, tmp_path, term, _Journal(), res, managed=False)

    ap.cycle()

    assert len(term.deals) == 1, "no entry went out"
    hb = ap.st["held"].get("GOLD")
    assert hb is not None, "the venue holds a trade the loop never booked"
    assert hb["side"] == "BUY"
    assert float(hb["qty"]) > 0
    assert hb["entry"] == float(res["price"])
    # the same stop the position was sized on, not a different one
    assert hb["stop"] == (res.get("confidence_band") or {}).get("p10")
    assert hb["started"]


def test_a_booked_entry_exits_when_its_stop_is_breached(monkeypatch, tmp_path):
    res = _prediction(monkeypatch, battery=_AGREE)
    term = _Terminal([])
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)

    ap.cycle()
    assert term.closed == [], "the entry closed itself on the way in"

    # Move the stop above the live price: the next read of the same tape is a
    # breach, and the position has to leave the terminal as well as the book.
    hb = ap.st["held"]["GOLD"]
    hb["stop"] = hb["stop0"] = float(res["price"]) + 1.0

    ap.cycle()

    assert term.closed == [4321], "the stop never reached the terminal"
    assert term.rows == []
    assert "GOLD" not in ap.st["held"]
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 1


def test_an_unverified_live_entry_is_booked_anyway(monkeypatch, tmp_path):
    """An order the venue acknowledged but the position list could not confirm may
    still be live. Forgetting it there leaves a trade with no stop and nobody
    managing it; the row costs nothing, because `_reconcile` drops it the moment
    the venue is read clearly and does not report the position."""
    res = _prediction(monkeypatch, battery=_AGREE)
    term = _Terminal([])
    term.positions = lambda: []
    monkeypatch.setattr(AP.time, "sleep", lambda s: None)
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)

    ap.cycle()

    assert ap.st["signals"][0]["action"] == "LIVE-UNVERIFIED"
    assert "GOLD" in ap.st["held"]
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 0
