"""A circuit breaker has to read the book it exists to stop, in that book's money.

The previous pass took the cross-currency sum off the sizing path and left it
where a sum belongs — as an upper bound the desk prints. The breakers went on
dividing by it. With a dollar CFD account and a rupee portal book both live, a
70,000 USD position on a 100,000 USD terminal account is 6% of a 1,100,000
"book", and 6% is under every cap in the rules file. The daily brake has the
same shape: a 5% loss in dollars netted against a good day in rupees is a 0.1%
loss, and the venue taking the hit keeps trading. Every one of these guards is
fail-open — the loop refuses an order only when a number looks too big, so a
number that is systematically small does not look broken, it looks clean.

So each book is measured against its own equity and the cap is checked against
the most invested book. That needs position rows to say which book they are on,
which the MT5 bridge never said; its rows carry no venue field at all, so they
are tagged on the way out of `_venue_positions`. And a row on a venue whose
account was never read gets its own blocker: that book is not flat, it is dark.
"""

from __future__ import annotations

import kairos.autopilot as AP
from test_autopilot_cycle import _AGREE
from test_live_arming import _book
from test_mt5_close_ticket import _Terminal, _prediction
from test_psx_exit_working import _Journal
from test_size_from_the_funding_book import _both_venues

MT5_EQ = 100_000.0          # the terminal's own USD account
PSX_EQ = 1_000_000.0        # the portal's own PKR book

# A name with no local history, so the cluster budget cannot join it to the
# candidate: the only guard left to refuse the order is the one under test.
USD_POSITION = {"symbol": "TESTX", "side": "BUY", "qty": 70.0,
                "entry": 900.0, "last": 1000.0, "pnl": 7_000.0,
                "ticket": 99999}                    # 70,000 USD = 70% of MT5


def _limits(monkeypatch, cap: float = 60.0, brake: float = 0.02):
    """One risk instruction per test, so a figure in config.yaml cannot decide
    whether a guard fired. The drawdown guard is parked far out: these tests ask
    which book a ratio was measured on, not what the book did last week."""
    from kairos import rules as R
    monkeypatch.setattr(R, "rules", lambda: {
        **R.DEFAULT_RULES, "max_total_exposure_pct": cap,
        "max_equity_drawdown_pct": 100.0})
    monkeypatch.setattr(R, "daily_loss_fraction", lambda *a: (brake, "test brake"))


def _two_books(mt5_rows=None, psx_book=None, mt5_eq=MT5_EQ,
               psx_day_pnl=0.0, mt5_day_pnl=0.0):
    """A posture read the way a live cycle reads it: venue accounts collected
    once, positions tagged with their book, one merged snapshot for the desk."""
    from datetime import datetime
    term = _Terminal(list(mt5_rows or []))
    term.account = lambda: {                          # type: ignore[method-assign]
        "venue": "MT5", "login": "90001", "equity": mt5_eq, "balance": mt5_eq,
        "day_pnl": mt5_day_pnl, "open_positions": len(term.rows), "leverage": 1.0}
    ap = AP.Autopilot()
    ap.st = ap._defaults()
    ap.mt5 = term
    book = psx_book if psx_book is not None else _book(cash=PSX_EQ, holdings=[])
    ap.st["psx_day_start"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "equity": round(book["equity"] - psx_day_pnl, 2)}
    ap._venue_accts = ap._venue_accounts(True, book, False)
    return ap, ap._merge_account(ap._venue_accts), ap._venue_positions(True, book)


# ------------------------------------------------------------------ exposure
def test_a_fully_invested_dollar_book_trips_the_cap_next_to_a_rupee_book(monkeypatch):
    """70% of the CFD account, and the old reading called it 6%.

    The blend added 70,000 USD of notionals to a 1,100,000 total of two
    currencies: under a 60% cap, so the brake that exists to stop an over-exposed
    book let the cycle size the next order. Asserting the exact figure rather
    than "is blocked" is the point — the number the desk prints is the number
    the cap was checked against.
    """
    _limits(monkeypatch)
    ap, acct, pos = _two_books(mt5_rows=[USD_POSITION])
    posture = ap._risk_posture(acct, pos)

    assert posture["exposure_by_venue"] == {"JS_GLOBAL": 0.0, "MT5": 70.0}
    assert posture["exposure_pct"] == 70.0
    assert posture["block_entries"] is True
    assert any("cap" in b and "MT5" in b for b in posture["blockers"]), \
        str(posture["blockers"])


def test_the_rupee_book_trips_the_cap_on_its_own_rupees(monkeypatch):
    """The other direction: 700,000 PKR of shares on a million-rupee book is 70%,
    whatever the dollar account next to it holds. Attribution matters here — the
    blocker has to name JS_GLOBAL, because an operator who reads "MT5 at 70%" and
    finds a flat terminal stops trusting the brake."""
    _limits(monkeypatch)
    shares = [{"symbol": "OGDC", "qty": 7_000.0, "avg": 90.0, "mtm": 100.0,
               "value": 700_000.0, "pending_sell": 0.0, "account": "99999"}]
    book = _book(cash=300_000.0, holdings=shares)
    ap, acct, pos = _two_books(psx_book=book)
    posture = ap._risk_posture(acct, pos)

    assert posture["exposure_by_venue"] == {"JS_GLOBAL": 70.0, "MT5": 0.0}
    assert posture["exposure_pct"] == 70.0
    assert any("JS_GLOBAL" in b and "cap" in b for b in posture["blockers"]), \
        str(posture["blockers"])


def test_rupee_holdings_do_not_look_like_exposure_on_a_flat_dollar_book(monkeypatch):
    """Control, and the half of the bug that overstates.

    150,000 PKR against a 100,000 USD account the portal cannot spend reads as
    13.6% of the pair; the rupee book itself is 15% invested. Neither number is
    near the cap, so nothing blocks — but the figure the desk shows has to be the
    book that is actually invested, or the next pass that grows the holding trips
    a cap nobody can point at.
    """
    _limits(monkeypatch)
    shares = [{"symbol": "OGDC", "qty": 1_500.0, "avg": 90.0, "mtm": 100.0,
               "value": 150_000.0, "pending_sell": 0.0, "account": "99999"}]
    book = _book(cash=850_000.0, holdings=shares)
    ap, acct, pos = _two_books(psx_book=book)
    posture = ap._risk_posture(acct, pos)

    assert posture["exposure_pct"] == 15.0
    assert posture["exposure_by_venue"]["MT5"] == 0.0
    assert posture["block_entries"] is False, str(posture["blockers"])


def test_an_unmeasurable_book_can_raise_the_number_but_never_lower_it(monkeypatch):
    """The worst-book rule must not quietly drop notional it cannot divide.

    A per-book ratio exists only where equity is readable, so taking the maximum
    of the measurable books alone would report the rupee book's 0% while 70,000
    USD sat unmeasured next to it — the understatement this pass exists to end.
    The blend still counts that notional, so the headline is the higher of the
    two, and the book itself is named as blind.
    """
    _limits(monkeypatch)
    ap, acct, pos = _two_books(mt5_rows=[USD_POSITION], mt5_eq=0.0)
    posture = ap._risk_posture(acct, pos)

    assert ap._venue_accts["MT5"]["equity"] == 0.0
    assert posture["exposure_by_venue"] == {"JS_GLOBAL": 0.0}
    assert posture["exposure_pct"] == 7.0      # 70,000 / 1,000,000, never 0.0
    assert any(b.startswith("MT5:") for b in posture["blockers"]), \
        str(posture["blockers"])
    assert posture["block_entries"] is True


def test_rows_on_a_venue_that_never_reported_an_account_are_dark_not_flat(
        monkeypatch):
    """A third venue's positions with no account behind them.

    The loop only sizes on a venue whose account was read, but rows reach the
    posture from the broker list, and a book that contributes notionals to no
    ratio is exactly the blind row this file already refuses to sum as zero. It
    has to be named: "gross exposure 6%" and a flat-looking MT5 is a picture of a
    risk book that is not there.
    """
    _limits(monkeypatch)
    ap, acct, pos = _two_books()
    pos = pos + [{"symbol": "BTCUSDT", "side": "BUY", "qty": 0.4, "last": 25_000.0,
                  "venue": "BINANCE"}]
    posture = ap._risk_posture(acct, pos)

    assert "BINANCE" not in ap._venue_accts
    assert any("BINANCE" in b and "never" in b for b in posture["blockers"]), \
        str(posture["blockers"])
    assert posture["block_entries"] is True


def test_one_reported_book_does_not_make_the_other_book_measurable(monkeypatch):
    """Darkness is a fact about a row, not about how many venues answered.

    Only the terminal reported an account, and a Binance holding is sitting on
    someone else's money: 10,000 USD of BTCUSDT divided by a 100,000 USD CFD
    account prints 10% and reads as a small book. The blend cannot be trusted for
    that row and no per-book ratio can hold it, so it gets named — and it gets
    named whether the portal answered beside the terminal or not.
    """
    _limits(monkeypatch)
    ap, _merged, _rows = _two_books()
    ap._venue_accts = {"MT5": ap._venue_accts["MT5"]}
    acct = ap._merge_account(ap._venue_accts)
    pos = ap._venue_positions(True, None) + [
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.4, "last": 25_000.0,
         "venue": "BINANCE"}]
    posture = ap._risk_posture(acct, pos)

    assert len(ap._venue_accts) == 1
    assert posture["exposure_pct"] == 10.0      # the blend, which cannot be right
    assert not any("cap" in b for b in posture["blockers"])
    assert any("BINANCE" in b and "never" in b for b in posture["blockers"]), \
        str(posture["blockers"])
    assert posture["block_entries"] is True


# ---------------------------------------------------------------- daily loss
def test_a_dollar_loss_breaches_its_own_brake_under_a_small_merged_one(
        monkeypatch):
    """-5,000 USD on a 100,000 USD account is a 5% day. Netted against +4,000 PKR
    on a million-rupee book it becomes -1,000 of a 1,100,000 total: 0.09%, under
    a 2% brake, so the terminal kept opening trades all afternoon on a book that
    had already lost what the rules allow in a day."""
    _limits(monkeypatch)
    ap, acct, pos = _two_books(mt5_day_pnl=-5_000.0,
                               psx_day_pnl=4_000.0)
    posture = ap._risk_posture(acct, pos)

    assert acct["day_pnl"] == -1_000.0          # the merged figure that hid it
    assert posture["block_entries"] is True
    assert any("MT5" in b and "brake" in b for b in posture["blockers"]), \
        str(posture["blockers"])


def test_a_good_dollar_day_does_not_borrow_the_rupee_book_for_a_breach(
        monkeypatch):
    """The reverse must not fire either: -3,000 PKR on the portal is 0.3% of its
    own book, and no ratio in this fixture reaches a 2% brake."""
    _limits(monkeypatch)
    ap, acct, pos = _two_books(mt5_day_pnl=0.0, psx_day_pnl=-3_000.0)
    posture = ap._risk_posture(acct, pos)

    assert posture["block_entries"] is False, str(posture["blockers"])


# ------------------------------------------------------------------ one book
def test_a_single_book_is_measured_exactly_as_it_always_was(monkeypatch):
    """Paper, or one live venue: `_venue_accts` is empty, so the snapshot already
    is that book's equity and there is no attribution to make. Nothing tightens
    here — if this test starts needing a venue name, the split is being applied
    where there is only one book to split."""
    _limits(monkeypatch)
    ap = AP.Autopilot()
    ap.st = ap._defaults()
    posture = ap._risk_posture({"venue": "PAPER", "equity": MT5_EQ,
                               "day_pnl": 0.0, "open_positions": 1},
                              [{"symbol": "GOLD", "side": "BUY", "qty": 70.0,
                                "entry": 900.0, "last": 1000.0, "pnl": 0.0}])

    assert ap._venue_accts == {}
    assert posture["exposure_pct"] == 70.0
    assert posture["exposure_by_venue"] == {}
    cap = [b for b in posture["blockers"] if "cap" in b]
    assert cap and " on " not in cap[0], str(posture["blockers"])
    # paper rows carry no venue, because there is only one book and it is not
    # named; an untagged row must not read as a holding on a stranger's account
    assert not any("never" in b for b in posture["blockers"]), \
        str(posture["blockers"])


def test_the_terminal_names_its_rows_because_the_bridge_never_did(monkeypatch):
    """The whole split rests on this tag.

    `MT5Bridge.positions()` builds rows from the MetaTrader structs and never
    writes a venue key — it is the only venue that doesn't. Untagged, its
    notionals match no account, so the most invested book in the fixture reads as
    0% and the cap passes with a number nobody can see.
    """
    term = _Terminal([USD_POSITION])
    ap = AP.Autopilot()
    ap.st = ap._defaults()
    ap.mt5 = term
    rows = ap._venue_positions(True, None)

    assert [r["venue"] for r in rows] == ["MT5"]
    assert ap._venue_positions(True, _book(holdings=[]))[0]["venue"] == "MT5"
    assert all(r["venue"] == "JS_GLOBAL"
               for r in ap._venue_positions(False, _book())[:2])


# --------------------------------------------------------------------- cycle
def test_the_breaker_stops_the_order_it_was_written_to_stop(monkeypatch, tmp_path):
    """End to end, on the money path.

    A signal that earned its size in every previous test, an idle rupee book, and
    a terminal already 70% invested. The loop sends nothing and says which book
    is the reason; on the blended denominator the same cycle placed the trade.
    """
    _limits(monkeypatch)
    res = _prediction(monkeypatch, battery=_AGREE)
    journal = _Journal()
    term, ap = _both_venues(monkeypatch, tmp_path, res, journal)
    term.rows.append(dict(USD_POSITION))           # type: ignore[attr-defined]

    out = ap.cycle()

    assert out["trades"] == 0, str(ap.st["signals"][0])
    assert term.deals == []
    signal = ap.st["signals"][0]
    assert signal["action"] == "BREAKER", str(signal)
    assert "MT5" in signal["detail"], signal["detail"]
    assert ap.status()["risk_posture"]["exposure_pct"] == 70.0
