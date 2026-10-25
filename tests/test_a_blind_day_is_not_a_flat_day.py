"""A book that cannot say what it lost today is not a book that lost nothing.

The spot venue answers balances, not a day figure, so its account read carries
`day_pnl: None` every single time. Two guards read that field and both treated
absence as zero. `_merge_account` summed the None as 0, so a crypto loss inside a
dollar-and-rupee desk made the merged day look smaller than any account was
actually carrying; and the per-book daily brake skipped the row, so the one book
where a fast loss happens was the one book with no daily stop at all. Neither
reads as broken — a clean number is exactly what a fail-open guard looks like.

The fix gives the spot book the same treatment as the rupee book: today's move
measured against the first usable equity of the calendar day. A degraded read
stamps nothing (a zero baseline would pin the loss at nothing all session), and
the merge now refuses to total a funded book it cannot measure while still
letting an empty book contribute its honest nothing.
"""

from __future__ import annotations

import kairos.autopilot as AP
from kairos import binance as BN

from test_breakers_read_their_own_book import _limits, _two_books
from test_live_arming import _book
from test_mt5_close_ticket import _Terminal

PSX_EQ = 1_000_000.0
SPOT_EQ = 20_000.0


class _Spot:
    """A spot account in the shape `BinanceClient.account()` returns."""

    def __init__(self, equity: float = SPOT_EQ):
        self.equity = equity

    def account(self):
        return {"venue": "BINANCE", "currency": "USDT",
                "equity": round(self.equity, 2),
                "balance": round(self.equity, 2), "day_pnl": None,
                "open_positions": 0}


def _arm(monkeypatch, spot: _Spot) -> _Spot:
    monkeypatch.setattr(BN, "BINANCE", spot)
    return spot


def _ap() -> AP.Autopilot:
    ap = AP.Autopilot()
    ap.st = ap._defaults()
    return ap


# ------------------------------------------------------------------ baseline
def test_a_spot_book_is_measured_from_its_first_read_of_the_day(monkeypatch):
    """None becomes a number only by having something to measure from."""
    ap = _ap()
    spot = _arm(monkeypatch, _Spot())

    first = ap._venue_accounts(False, None, True)["BINANCE"]
    assert first["day_pnl"] == 0.0                  # nothing lost since we looked
    assert first["day_pnl_from"].startswith("spot equity")
    assert ap.st["venue_day_start"]["BINANCE"]["equity"] == SPOT_EQ

    spot.equity = SPOT_EQ - 800.0
    second = ap._venue_accounts(False, None, True)["BINANCE"]
    assert second["day_pnl"] == -800.0


def test_a_zero_read_never_stamps_the_spot_baseline(monkeypatch):
    """The rupee book's lesson, applied to the spot book.

    A zero latching here is sticky: every later reading measures at most a few
    dollars against zero and the brake stays silent for the session on a book
    that was losing the whole time.
    """
    ap = _ap()
    spot = _arm(monkeypatch, _Spot(equity=0.0))

    blind = ap._venue_accounts(False, None, True)["BINANCE"]
    assert blind["day_pnl"] is None                 # unknown, not a small loss
    assert ap.st["venue_day_start"] == {}           # nothing latched from it

    spot.equity = SPOT_EQ
    assert ap._venue_accounts(False, None, True)["BINANCE"]["day_pnl"] == 0.0
    assert ap.st["venue_day_start"]["BINANCE"]["equity"] == SPOT_EQ


def test_a_new_day_takes_a_new_baseline(monkeypatch):
    ap = _ap()
    spot = _arm(monkeypatch, _Spot())
    ap._venue_accounts(False, None, True)
    ap.st["venue_day_start"]["BINANCE"]["date"] = "2000-01-01"
    spot.equity = SPOT_EQ - 900.0
    assert ap._venue_accounts(False, None, True)["BINANCE"]["day_pnl"] == 0.0


# --------------------------------------------------------------- terminal day
def test_a_loss_the_terminal_already_banked_is_still_todays_loss(monkeypatch):
    """`profit` is floating P&L, not the day.

    A terminal that closes -5,000 USD reports position profit back at zero and a
    balance and equity of 95,000. Reading that `profit` as the day's loss says
    the account is flat on a day it paid a loss out of, and the daily brake stays
    shut for the rest of the session. The change in equity is the day; the
    broker's figure is kept beside it as `floating_pnl`.
    """
    ap = _ap()
    ap.mt5 = _Terminal([])                          # 100,000 USD at the open
    first = ap._venue_accounts(True, None, False)["MT5"]
    assert first["day_pnl"] == 0.0
    assert first["floating_pnl"] == 0.0

    ap.mt5.account = lambda: {                      # the losing trade is closed
        "venue": "MT5", "currency": "USD", "equity": 95_000.0,
        "balance": 95_000.0, "day_pnl": 0.0, "open_positions": 0}
    second = ap._venue_accounts(True, None, False)["MT5"]
    assert second["day_pnl"] == -5_000.0
    assert ap._venue_day_pnl("MT5", 95_000.0) == -5_000.0


def test_a_banked_loss_opens_the_brake_the_cycle_actually_reads(monkeypatch):
    _limits(monkeypatch, brake=0.02)
    ap = _ap()
    ap.mt5 = _Terminal([])
    ap._venue_accounts(True, None, False)           # 100,000 stamped as the open
    ap.mt5.account = lambda: {
        "venue": "MT5", "currency": "USD", "equity": 95_000.0,
        "balance": 95_000.0, "day_pnl": 0.0, "open_positions": 0}
    ap._venue_accts = ap._venue_accounts(True, None, False)
    posture = ap._risk_posture(ap._merge_account(ap._venue_accts), [])

    assert posture["day_pnl"] == -5_000.0
    assert posture["block_entries"] is True
    assert any("daily loss" in b for b in posture["blockers"]), \
        str(posture["blockers"])


# ------------------------------------------------------------------- merging
def _part(venue: str, equity: float, day) -> dict:
    return {"venue": venue, "equity": equity, "balance": equity,
            "day_pnl": day, "open_positions": 0}


def test_a_funded_blind_book_makes_the_merged_day_unknown():
    """The number the desk prints must not be smaller than a book's loss.

    -5,000 USD on the terminal plus an unmeasurable spot book of 20,000 USDT is
    not "-5,000"; it is "nobody knows yet", and the daily brake has to read it
    that way.
    """
    ap = _ap()
    merged = ap._merge_account({
        "MT5": _part("MT5", 100_000.0, -5_000.0),
        "BINANCE": _part("BINANCE", 20_000.0, None)})
    assert merged["day_pnl"] is None
    assert merged["equity"] == 120_000.0            # the balance still totals


def test_an_empty_book_does_not_blind_the_ones_that_answered():
    """Nothing to lose is a measured zero, not an unknown."""
    ap = _ap()
    merged = ap._merge_account({
        "MT5": _part("MT5", 100_000.0, -5_000.0),
        "BINANCE": _part("BINANCE", 0.0, None)})
    assert merged["day_pnl"] == -5_000.0


# --------------------------------------------------------------------- cycle
def test_a_crypto_day_trips_its_own_brake_under_a_small_merged_one(
        monkeypatch):
    """End to end on the money path, in the direction that costs money.

    -1,000 USDT is 5% of the crypto book and 0.09% of the blended total, so the
    merged figure clears a 2% brake while the book that took the loss is past
    the limit the rules allow it in a day.
    """
    _limits(monkeypatch, brake=0.02)
    ap, _acct, _pos = _two_books()
    spot = _arm(monkeypatch, _Spot())
    book = _book(cash=PSX_EQ, holdings=[])

    ap._venue_accts = ap._venue_accounts(True, book, True)   # baseline latched
    ap._risk_posture(ap._merge_account(ap._venue_accts),
                     ap._venue_positions(True, book))

    spot.equity = SPOT_EQ - 1_000.0
    ap._venue_accts = ap._venue_accounts(True, book, True)
    posture = ap._risk_posture(ap._merge_account(ap._venue_accts),
                               ap._venue_positions(True, book))

    assert ap._merge_account(ap._venue_accts)["day_pnl"] == -1_000.0
    assert posture["block_entries"] is True
    assert any("BINANCE" in b and "brake" in b for b in posture["blockers"]), \
        str(posture["blockers"])


def test_the_posture_does_not_name_a_breach_on_a_flat_spot_day(monkeypatch):
    """Control: the same three-book desk, with nothing lost anywhere."""
    _limits(monkeypatch, brake=0.02)
    ap, _acct, _pos = _two_books()
    _arm(monkeypatch, _Spot())
    book = _book(cash=PSX_EQ, holdings=[])

    ap._venue_accts = ap._venue_accounts(True, book, True)
    ap._risk_posture(ap._merge_account(ap._venue_accts),
                     ap._venue_positions(True, book))
    posture = ap._risk_posture(ap._merge_account(ap._venue_accts),
                               ap._venue_positions(True, book))

    assert posture["block_entries"] is False, str(posture["blockers"])
