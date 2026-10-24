"""Confirming a live fill polls the venue instead of sleeping a fixed second.

An acknowledgement is not a position, so both live paths re-read the venue's own
position list before trusting a fill. The check used to sleep a flat second and
then read once, which meant two things: every clean fill paid a second of dead
latency on the order path, and a venue that simply took longer than that to show
the position got the same "unverified" answer as one that never filled at all.

The budget is unchanged — attempts times poll is the same second — but it
returns on the first read that sees the fill, and a slow venue is now
distinguishable from an absent one. These run the real cycle against a stubbed
terminal and count the sleeps.

The stub reads are gated on the order having gone out: the same list is what the
cycle reads *before* deciding to enter, and a row that is already there would
make this a position to manage rather than an entry to confirm.
"""

from __future__ import annotations

import kairos.autopilot as AP
from test_autopilot_cycle import _AGREE
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_psx_exit_working import _Journal


def _entry(monkeypatch, tmp_path, positions=None):
    """One live MT5 entry; `positions(term)` stands in for the venue's list."""
    res = _prediction(monkeypatch, battery=_AGREE)
    term = _Terminal([])
    if positions is not None:
        term.positions = lambda: positions(term)
    slept: list[float] = []
    monkeypatch.setattr(AP.time, "sleep", lambda s: slept.append(s))
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)
    return ap, term, slept, journal


def test_a_fill_the_venue_already_shows_is_not_kept_waiting(monkeypatch, tmp_path):
    # the stub terminal shows the position the moment the deal is accepted
    ap, term, slept, _journal = _entry(monkeypatch, tmp_path)

    ap.cycle()

    assert slept == [], f"the order path still waits {slept} before it looks"
    assert len(term.deals) == 1, "the entry never went out"
    assert ap.st["signals"][0]["action"] == "LIVE-FILLED"


def test_a_slow_venue_is_confirmed_by_the_read_that_shows_it(monkeypatch, tmp_path):
    reads = {"n": 0}

    def positions(term):
        if not term.deals:
            return []
        reads["n"] += 1
        return list(term.rows) if reads["n"] >= 3 else []

    ap, _term, slept, _journal = _entry(monkeypatch, tmp_path, positions)

    ap.cycle()

    assert reads["n"] == 3, "the venue was not polled until it answered"
    assert slept == [AP.FILL_CONFIRM_POLL_S, AP.FILL_CONFIRM_POLL_S]
    assert ap.st["signals"][0]["action"] == "LIVE-FILLED"


def test_a_venue_that_never_shows_the_fill_is_still_not_trusted(monkeypatch, tmp_path):
    ap, _term, slept, journal = _entry(monkeypatch, tmp_path, lambda term: [])

    ap.cycle()

    assert slept == [AP.FILL_CONFIRM_POLL_S] * (AP.FILL_CONFIRM_ATTEMPTS - 1)
    assert ap.st["signals"][0]["action"] == "LIVE-UNVERIFIED"
    assert journal.count("AUTOPILOT_UNVERIFIED") == 1


def test_an_unreadable_position_list_is_not_a_confirmation(monkeypatch, tmp_path):
    """None is the terminal's "the request failed", not "the account is flat",
    and neither one is evidence that this fill exists."""
    ap, _term, _slept, journal = _entry(
        monkeypatch, tmp_path, lambda term: None if term.deals else [])

    ap.cycle()

    assert ap.st["signals"][0]["action"] == "LIVE-UNVERIFIED"
    assert journal.count("AUTOPILOT_UNVERIFIED") == 1


def test_the_whole_budget_is_no_longer_than_the_second_it_replaced():
    assert (AP.FILL_CONFIRM_ATTEMPTS - 1) * AP.FILL_CONFIRM_POLL_S <= 1.0
