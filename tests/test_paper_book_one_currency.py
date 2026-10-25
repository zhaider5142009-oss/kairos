"""A paper book that adds dollars to rupees is not a book.

The ledger is one number, and every surface that reads it calls it rupees: the desk
prints `free cash Rs`, the track record prints `Capital Rs` and `realized P&L Rs`,
and the daily-loss brake divides `day_pnl` by an equity it counts in rupees. The
book itself did not know that. `submit` deducted `price * qty` from cash whichever
money the tape printed, so GOLD — which the registry states in dollars — was charged
at its *dollar* figure against a rupee million. The live ledger holds two fills of
1.2 oz at $4,183 each: Rs 5,020 spent, a position worth Rs 1.39 million, and the same
277x understatement in `day_pnl` (the number the brake divides by), in `realized_pnl`
(the number the published track record prints), and in an equity the book grew out of
thin air.

The fix is the same doctrine #96 applied one layer up: the book declares its money,
converts a leg at the published cross *before* it touches the ledger, and refuses —
`None`, never a rate of one, never par — when the cross has no print. An uncountable
holding is blindness, not a smaller number.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import kairos.data as D
import kairos.execution as E
from kairos.config import CFG
from kairos.execution import OrderIntent, PaperBroker
from test_autopilot_cycle import _AGREE, _BULL, _run_cycle

RATE = 300.0
GOLD = 4183.0


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def dirpath(monkeypatch, tmp_path):
    """A data directory of its own, and a kill switch that cannot reach the desk.

    `PaperBroker` resolves its state path from `CFG.data_dir` in its constructor,
    so patching the config before building the book is what keeps this file from
    reading, sizing against, and then overwriting the operator's real ledger.
    """
    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    monkeypatch.setattr(E.KILL, "path", tmp_path / "KILL_SWITCH")
    return tmp_path


@pytest.fixture
def rate(monkeypatch) -> float:
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    return RATE


@pytest.fixture
def no_rate(monkeypatch):
    monkeypatch.setattr(D, "usd_pkr", lambda: None)


@pytest.fixture
def events(monkeypatch) -> list:
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(E.JOURNAL, "log",
                        lambda event, payload=None: seen.append((event, payload or {})))
    return seen


def _config(monkeypatch, **paper):
    """Pin the `paper:` section, so a fixture tests the book and not whatever the
    operator's config.yaml happens to hold."""
    real = CFG.section
    monkeypatch.setattr(CFG, "section",
                        lambda name: paper if name == "paper" else real(name))


def _paper(monkeypatch, dirpath, capital: float = 1_000_000.0, **paper) -> PaperBroker:
    _config(monkeypatch, **paper)
    book = PaperBroker(capital)
    assert book.state_path == dirpath / "paper_account.json"
    return book


def _write_state(dirpath, **extra) -> dict:
    state = {"capital": 1_000_000.0, "cash": 1_000_000.0, "positions": {},
             "fills": [], "realized_pnl": 0.0, "day_pnl": 0.0, "day_stamp": ""}
    state.update(extra)
    (dirpath / "paper_account.json").write_text(json.dumps(state), encoding="utf-8")
    return state


def _trade(book, symbol, qty, price, side="BUY"):
    return book.submit(OrderIntent(symbol=symbol, side=side, quantity=qty,
                                   venue="PAPER"), price)


# ------------------------------------------------------------------ the money
def test_the_book_declares_the_money_it_counts_in(monkeypatch, dirpath):
    book = _paper(monkeypatch, dirpath)
    assert book.currency == "PKR"
    # Every reader of this figure calls it rupees, so it has to be able to ask.
    assert book.account()["currency"] == "PKR"


def test_a_dollar_funded_book_says_so(monkeypatch, dirpath):
    book = _paper(monkeypatch, dirpath, currency="USD")
    assert book.currency == "USD"
    assert book.account()["currency"] == "USD"


def test_the_ledger_on_disk_outvotes_the_config(monkeypatch, dirpath):
    """Relabelling a ledger that already holds trades is the same mixing in a new
    place, so the money in the file wins over the money in the config."""
    _write_state(dirpath, currency="USD", cash=500_000.0)
    book = _paper(monkeypatch, dirpath, currency="PKR")
    assert book.currency == "USD"
    assert book.cash == 500_000.0


def test_the_money_is_written_with_the_ledger(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    _trade(book, "OGDC", 100.0, 150.0)
    saved = json.loads(book.state_path.read_text(encoding="utf-8"))
    assert saved["currency"] == "PKR"


def test_a_reopened_book_keeps_the_money_it_was_opened_with(monkeypatch, dirpath,
                                                            rate):
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    again = _paper(monkeypatch, dirpath)
    assert again.currency == "PKR"
    assert again.positions["GOLD"]["ccy"] == "USD"


# --------------------------------------------------------------------- the cross
def test_the_books_own_money_never_asks_the_tape(monkeypatch, dirpath):
    """The rupee leg of a rupee book is the common case and must cost no read."""
    monkeypatch.setattr(D, "usd_pkr", lambda: pytest.fail("read the tape"))
    book = _paper(monkeypatch, dirpath)
    assert book.rate("PKR") == 1.0
    assert book.rate("pkr") == 1.0
    # A name the registry does not state resolves to the book's own money.
    assert book.rate("") == 1.0


def test_a_dollar_on_a_rupee_book_is_the_published_cross(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    assert book.rate("USD") == RATE
    assert book.rate("usd") == RATE


def test_a_currency_this_desk_does_not_hold_has_no_rate(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    assert book.rate("EUR") is None


def test_no_print_is_no_rate_not_a_rate_of_one(monkeypatch, dirpath, no_rate):
    """The whole refusal design rests on this: a missing cross is a hole in the
    book, and 1.0 would fill it with a 277x-sized trade."""
    book = _paper(monkeypatch, dirpath)
    assert book.rate("USD") is None


def test_a_dollar_book_prices_rupees_by_the_same_cross(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath, currency="USD")
    assert book.rate("PKR") == pytest.approx(1 / RATE)
    assert book.rate("USD") == 1.0


# ------------------------------------------------------------------ the size
def test_a_rupee_ticket_on_a_dollar_name_buys_rupee_worth(monkeypatch, dirpath, rate):
    """Rs 150,000 of gold at $4,183 is 0.12 oz. The bare division says 35.9 oz —
    Rs 45 million of metal on a Rs 1 million book."""
    book = _paper(monkeypatch, dirpath)
    units = book.units_for("GOLD", 150_000.0, GOLD)
    assert units == pytest.approx(150_000.0 / (GOLD * RATE))
    assert 150_000.0 / GOLD / units == pytest.approx(RATE)


def test_a_rupee_ticket_on_a_rupee_name_needs_no_cross(monkeypatch, dirpath):
    monkeypatch.setattr(D, "usd_pkr", lambda: pytest.fail("read the tape"))
    book = _paper(monkeypatch, dirpath)
    assert book.units_for("OGDC", 150_000.0, 150.0) == 1000.0


@pytest.mark.parametrize("notional, price", [(0.0, GOLD), (-1.0, GOLD),
                                             (150_000.0, 0.0), (150_000.0, -5.0)])
def test_a_ticket_that_states_no_size_gets_no_size(monkeypatch, dirpath, rate,
                                                   notional, price):
    assert _paper(monkeypatch, dirpath).units_for("GOLD", notional, price) is None


def test_no_cross_is_no_size(monkeypatch, dirpath, no_rate):
    book = _paper(monkeypatch, dirpath)
    assert book.units_for("GOLD", 150_000.0, GOLD) is None


def test_a_dollar_book_sizes_a_rupee_name_down(monkeypatch, dirpath, rate):
    """$500,000 of OGDC at Rs 150 buys 11,111 shares, not 3,333,333."""
    book = _paper(monkeypatch, dirpath, currency="USD")
    assert book.units_for("OGDC", 500_000.0, 150.0) == pytest.approx(
        500_000.0 / (150.0 / RATE))


# ------------------------------------------------------------------ the fill
def test_a_dollar_fill_is_charged_in_rupees(monkeypatch, dirpath, rate, events):
    book = _paper(monkeypatch, dirpath)
    fill = _trade(book, "GOLD", 0.1, GOLD)
    assert fill is not None
    charged = 1_000_000.0 - book.cash
    # The venue's own print times the cross, not the venue's print.
    assert charged == pytest.approx((fill.price * 0.1 + fill.commission) * RATE)
    # ... which is the number the old code never produced: it charged Rs 418.
    assert charged > 100_000.0
    assert book.positions["GOLD"]["ccy"] == "USD"


def test_the_fill_journals_the_money_and_the_cross(monkeypatch, dirpath, rate, events):
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    line = [p for e, p in events if e == "PAPER_FILL"][0]
    assert line["currency"] == "USD" and line["book"] == "PKR"
    assert line["fx"] == RATE


def test_a_dollar_position_the_book_cannot_fund_is_refused(monkeypatch, dirpath,
                                                           rate, events):
    """The live ledger's own ticket: 1.2 oz at $4,183 is Rs 1.51 million, and on a
    Rs 1 million cash book the answer is no — not a negative balance called leverage."""
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "GOLD", 1.2, GOLD) is None
    assert book.cash == 1_000_000.0
    assert "GOLD" not in book.positions
    assert not book.fills
    line = [p for e, p in events if e == "PAPER_NO_BUYING_POWER"][0]
    assert line["cost"] > 1_000_000.0
    assert line["book"] == "PKR"


def test_a_rupee_ticket_bigger_than_the_book_is_refused(monkeypatch, dirpath, rate,
                                                       events):
    """Same control, one currency: nothing downstream checks order value for a
    MARKET order, and the desk lets a human type any size."""
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "OGDC", 20_000.0, 150.0) is None
    assert book.cash == 1_000_000.0
    assert [e for e, _ in events] == ["PAPER_NO_BUYING_POWER"]


def test_a_buy_that_closes_a_short_is_not_a_purchase(monkeypatch, dirpath, rate,
                                                     events):
    """The refusal is a buying-power test, and closing a leg hands cash back. A
    short that ran away must still be closable: the book takes the loss and shows a
    negative balance, which is a number the brake reads, instead of a position
    nobody is allowed to exit."""
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "OGDC", 1000.0, 150.0, side="SELL") is not None
    assert book.cash > 1_000_000.0
    fill = _trade(book, "OGDC", 1000.0, 1_500.0)
    assert fill is not None
    assert "OGDC" not in book.positions
    assert book.cash < 0.0
    open_, close = book.fills[0]["price"], book.fills[1]["price"]
    loss = (close - open_) * 1000.0
    fees = book.fills[0]["commission"] + book.fills[1]["commission"]
    # rupees, and every rupee of it: no leg of this is a dollar figure
    assert book.realized_pnl == pytest.approx(-loss - fees, rel=1e-9)
    assert [e for e, _ in events] == ["PAPER_FILL", "PAPER_FILL"]


def test_a_ticket_the_book_cannot_count_is_refused(monkeypatch, dirpath, no_rate,
                                                   events):
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "GOLD", 0.1, GOLD) is None
    assert book.cash == 1_000_000.0 and book.positions == {} and book.fills == []
    line = [p for e, p in events if e == "PAPER_NO_FX"][0]
    assert line == {"symbol": "GOLD", "side": "BUY", "currency": "USD", "book": "PKR"}
    assert not book.state_path.exists()


def test_a_rupee_leg_needs_no_cross_to_be_taken(monkeypatch, dirpath, no_rate, events):
    """A book that cannot read dollars must still trade its own scrips — refusing
    these would turn a currency hole into an outage."""
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "OGDC", 100.0, 150.0) is not None
    assert [e for e, _ in events] == ["PAPER_FILL"]


# ---------------------------------------------------------------- the position row
def test_a_position_row_names_its_money(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    _trade(book, "OGDC", 100.0, 150.0)
    rows = {r["symbol"]: r for r in book.open_positions()}
    assert rows["GOLD"]["currency"] == "USD"
    assert rows["OGDC"]["currency"] == "PKR"
    # entry/last/pnl stay the venue's own print — the label says what they mean
    assert rows["GOLD"]["pnl"] == 0.0


def test_a_legacy_row_gets_its_money_from_the_registry(monkeypatch, dirpath, rate):
    """Rows written before `ccy` existed carry no label, and the registry is the
    same fact the fill would have written."""
    _write_state(dirpath, positions={"GOLD": {"qty": 0.1, "entry": 4000.0,
                                               "last": 4200.0, "commission": 1.0}})
    book = _paper(monkeypatch, dirpath)
    assert book.open_positions()[0]["currency"] == "USD"
    assert book.account()["equity"] == pytest.approx(
        1_000_000.0 + 0.1 * 4200.0 * RATE)


def test_the_money_on_the_row_wins_over_the_registry(monkeypatch, dirpath, rate):
    """A name outside the registry is counted at par, so the row has to be able to
    overrule that — and a registry that starts knowing a name cannot move a leg
    that was filled in another money."""
    _write_state(dirpath, positions={"TESTX": {"qty": 1.0, "entry": 100.0,
                                                "last": 110.0, "ccy": "USD"}})
    book = _paper(monkeypatch, dirpath)
    assert book._ccy("TESTX", book.positions["TESTX"]) == "USD"
    assert book.account()["equity"] == pytest.approx(1_000_000.0 + 110.0 * RATE)
    # ... while the same unknown name with no label on it is not foreign
    assert book._ccy("TESTX") == "PKR"


# --------------------------------------------------------------------- equity
def test_equity_counts_a_dollar_holding_in_rupees(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    book.mark("GOLD", 4300.0)
    acct = book.account()
    assert acct["unmarked"] == []
    assert acct["equity"] == pytest.approx(
        book.cash + 0.1 * 4300.0 * RATE, rel=1e-9)
    # and the mark moved equity by the *dollar* move times the cross
    assert acct["equity"] > book.cash + 10_000.0


def test_a_dollar_holding_with_no_cross_blinds_the_book(monkeypatch, dirpath, rate):
    _write_state(dirpath, positions={"GOLD": {"qty": 0.1, "entry": GOLD,
                                               "last": GOLD, "ccy": "USD"},
                                     "OGDC": {"qty": 100.0, "entry": 150.0,
                                              "last": 150.0, "ccy": "PKR"}})
    book = _paper(monkeypatch, dirpath)
    assert book.account()["equity"] is not None
    monkeypatch.setattr(D, "usd_pkr", lambda: None)
    acct = book.account()
    assert acct["equity"] is None
    # the rupee leg is still readable; one dark leg is what makes the number unknown
    assert acct["unmarked"] == ["GOLD"]
    assert acct["balance"] == 1_000_000.0


def test_a_flat_row_is_not_a_dark_one(monkeypatch, dirpath, rate):
    """Nothing held and something unreadable are different facts, and the breaker
    text has to keep them apart."""
    _write_state(dirpath, positions={"GOLD": {"qty": 0.0, "entry": 0.0,
                                               "last": 0.0, "ccy": "USD"}})
    acct = _paper(monkeypatch, dirpath).account()
    assert acct["equity"] == 1_000_000.0 and acct["unmarked"] == []


@pytest.mark.parametrize("qty, last", [(0.1, 0.0), (0.1, -5.0),
                                        ("n/a", 4183.0), (None, 4183.0)])
def test_an_unmeasurable_holding_is_dark(monkeypatch, dirpath, rate, qty, last):
    _write_state(dirpath, positions={"GOLD": {"qty": qty, "entry": GOLD,
                                               "last": last, "ccy": "USD"}})
    acct = _paper(monkeypatch, dirpath).account()
    assert acct["equity"] is None
    assert acct["unmarked"] == ["GOLD"]


def test_a_ledger_holding_a_nan_is_refused_on_read(monkeypatch, dirpath, rate, events):
    """#74 stopped the book from *writing* one; Python's `json` still parses it
    back, and the next `_save()` — which `account()` calls on the desk's read path
    — raised out of a successful read. The reader now uses the writer's rule, and
    a ledger that breaks it follows the damaged-file path: keep the evidence,
    engage the brake, refuse to size against it."""
    _write_state(dirpath, positions={"GOLD": {"qty": 0.1, "entry": GOLD,
                                               "last": float("nan"), "ccy": "USD"}})
    book = _paper(monkeypatch, dirpath)
    assert [e[0] for e in events] == ["PAPER_STATE_DAMAGED", "KILL_SWITCH_ENGAGED"]
    assert book.positions == {}
    assert E.KILL.engaged()
    assert list(dirpath.glob("paper_account.json.*.damaged"))


# ----------------------------------------------------------------- realised money
def test_a_round_trip_realises_in_book_money(monkeypatch, dirpath, rate):
    """The published `realized P&L Rs` line reads this field. A dollar commission
    printed as a rupee one is the same 277x understatement in the track record."""
    book = _paper(monkeypatch, dirpath)
    buy = _trade(book, "GOLD", 0.1, GOLD)
    sell = _trade(book, "GOLD", 0.1, GOLD, side="SELL")
    assert sell is not None
    # the quote is the same on both sides, so the loss is the slip and the fees —
    # all of it multiplied by the cross before it reached the ledger
    slip = (buy.price - sell.price) * 0.1
    dollars = slip + buy.commission + sell.commission
    assert book.realized_pnl == pytest.approx(-dollars * RATE, rel=1e-9)
    assert book.day_pnl == pytest.approx(-dollars * RATE, rel=1e-9)
    assert dollars < 2.0 < abs(book.realized_pnl)


def test_a_marked_up_dollar_leg_realises_the_rupee_gain(monkeypatch, dirpath, rate):
    """0.1 oz up $115 is Rs 34,500 of gain, and the published track record prints
    the rupee one."""
    book = _paper(monkeypatch, dirpath)
    buy = _trade(book, "GOLD", 0.1, GOLD)
    sell = _trade(book, "GOLD", 0.1, 4300.0, side="SELL")
    gain = (sell.price - buy.price) * 0.1
    fees = buy.commission + sell.commission
    assert book.realized_pnl == pytest.approx((gain - fees) * RATE, rel=1e-9)
    assert book.realized_pnl > 2_000.0
    assert "GOLD" not in book.positions


# ------------------------------------------------------- the cross at both ends
def test_a_leg_closed_at_another_cross_books_the_translation(monkeypatch, dirpath):
    """The published cash moved by more than the trade did, and nothing said so.

    A dollar leg is converted twice — in on the entry's cross, out on the exit's —
    and only `cash` carried the difference between the two. The live ledger showed
    it as a Rs 6,262.07 fall in equity beside a Rs 3,501.69 `realized P&L`: two
    numbers a reader cannot reconcile and no third line to explain them. The
    translation is now its own realised figure.
    """
    seen = {"rate": RATE}
    monkeypatch.setattr(D, "usd_pkr", lambda: seen["rate"])
    book = _paper(monkeypatch, dirpath)
    buy = _trade(book, "GOLD", 0.1, GOLD)
    seen["rate"] = RATE + 2.0
    sell = _trade(book, "GOLD", 0.1, GOLD, side="SELL")
    assert sell is not None

    # the price P&L is measured at the exit cross, the entry fee at the entry's
    trading = -buy.commission * RATE + (sell.price - buy.price) * 0.1 * (RATE + 2.0) \
        - sell.commission * (RATE + 2.0)
    assert book.realized_pnl == pytest.approx(trading, rel=1e-9)
    # ... and the notional that crossed twice is its own line
    assert book.fx_pnl == pytest.approx(0.1 * buy.price * 2.0, rel=1e-9)
    # together they are the cash the round trip moved, to the rupee
    assert book.realized_pnl + book.fx_pnl == pytest.approx(
        book.cash - book.capital, rel=1e-9)
    acct = book.account()
    assert acct["realized_total"] == pytest.approx(acct["equity"] - book.capital,
                                                  rel=1e-9)
    assert acct["fx_pnl"] == pytest.approx(book.fx_pnl)


def test_the_translation_is_the_day_s_pnl_too(monkeypatch, dirpath):
    """It is money that moved today, and the daily-loss brake divides today."""
    seen = {"rate": RATE}
    monkeypatch.setattr(D, "usd_pkr", lambda: seen["rate"])
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    seen["rate"] = RATE + 2.0
    _trade(book, "GOLD", 0.1, GOLD, side="SELL")
    assert book.day_pnl == pytest.approx(book.realized_pnl + book.fx_pnl, rel=1e-9)


def test_a_translation_survives_a_restart(monkeypatch, dirpath):
    seen = {"rate": RATE}
    monkeypatch.setattr(D, "usd_pkr", lambda: seen["rate"])
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    seen["rate"] = RATE + 2.0
    _trade(book, "GOLD", 0.1, GOLD, side="SELL")
    again = _paper(monkeypatch, dirpath)
    assert again.fx_pnl == pytest.approx(book.fx_pnl, rel=1e-12)


def test_a_rupee_leg_has_no_translation(monkeypatch, dirpath):
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    book = _paper(monkeypatch, dirpath)
    _trade(book, "OGDC", 100.0, 150.0)
    _trade(book, "OGDC", 100.0, 160.0, side="SELL")
    assert book.fx_pnl == 0.0


def test_a_ticket_that_crosses_zero_opens_at_its_own_price(monkeypatch, dirpath):
    """A buy larger than the short it closes is a new long, not the old one carried.

    The old branch kept the short's entry for the remainder, so every later P&L
    on that position was measured against a price it was never filled at.
    """
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    book = _paper(monkeypatch, dirpath)
    short = _trade(book, "OGDC", 100.0, 150.0, side="SELL")
    flip = _trade(book, "OGDC", 300.0, 160.0)
    assert short is not None and flip is not None
    pos = book.positions["OGDC"]
    assert pos["qty"] == pytest.approx(200.0)
    assert pos["entry"] == pytest.approx(flip.price, rel=1e-12)
    # the short leg realised against the short's own entry, at the flip's price
    fees = short.commission + flip.commission
    assert book.realized_pnl == pytest.approx(
        (flip.price - short.price) * -100.0 - fees, rel=1e-9)


def test_a_legacy_row_closes_without_inventing_a_translation(monkeypatch, dirpath):
    """A row written before the entry cross existed has none to measure against.

    Zero is the only supportable claim for it; guessing today's cross would book a
    translation the book never actually paid.
    """
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    _write_state(dirpath, positions={"GOLD": {"qty": 0.2, "entry": 4000.0,
                                               "last": 4000.0, "commission": 1.0,
                                               "ccy": "USD"}})
    book = _paper(monkeypatch, dirpath)
    assert _trade(book, "GOLD", 0.2, 4000.0, side="SELL") is not None
    assert book.fx_pnl == 0.0
    assert "GOLD" not in book.positions


# ------------------------------------------------------------------- the curve
def test_the_curve_counts_a_dollar_leg_as_a_dollar_leg(monkeypatch, dirpath, rate):
    book = _paper(monkeypatch, dirpath)
    buy = _trade(book, "GOLD", 0.1, GOLD)
    sell = _trade(book, "GOLD", 0.1, 4300.0, side="SELL")
    curve = book.equity_curve()
    assert len(curve) == 2
    gain = (sell.price - buy.price) * 0.1 * RATE
    fees = (buy.commission + sell.commission) * RATE
    assert curve.iloc[0] == pytest.approx(1_000_000.0 - buy.commission * RATE)
    assert curve.iloc[-1] == pytest.approx(1_000_000.0 + gain - fees, rel=1e-9)
    assert curve.iloc[-1] > 1_000_000.0


def test_the_curve_leaves_out_a_leg_it_cannot_count(monkeypatch, dirpath, rate):
    """Not at par, and not silently in: the curve is a published figure, and the
    line above says what it was rebuilt from."""
    book = _paper(monkeypatch, dirpath)
    _trade(book, "GOLD", 0.1, GOLD)
    _trade(book, "OGDC", 100.0, 150.0)
    monkeypatch.setattr(D, "usd_pkr", lambda: None)
    curve = book.equity_curve()
    assert len(curve) == 1
    fee = book.fills[1]["commission"]
    assert curve.iloc[-1] == pytest.approx(1_000_000.0 - fee)
    assert curve.index[0] == pd.Timestamp(book.fills[1]["ts"])


# ----------------------------------------------------------------- the real cycle
def test_a_cycle_buys_what_the_rupee_capital_actually_buys(monkeypatch, tmp_path):
    """One production cycle, one dollar name, one rupee book."""
    ap, _res, book, out = _run_cycle(monkeypatch, tmp_path, _BULL, "fx_size",
                                     battery=_AGREE)
    assert out["trades"] == 1, ap.st["signals"][0]["action"]
    pos = book.positions["GOLD"]
    charged = 1_000_000.0 - book.cash
    # the cash that left the book is the rupee cost of the position, not its
    # dollar print
    assert charged == pytest.approx(pos["qty"] * pos["entry"] * RATE, rel=0.02)
    # and the holding is worth a rupee fraction of a rupee book, not 277 of them
    assert pos["qty"] * pos["last"] * RATE <= 0.16 * 1_000_000.0
    assert book.account()["unmarked"] == []
    # buying at the mark neither creates nor destroys equity
    assert book.account()["equity"] == pytest.approx(1_000_000.0, rel=0.02)


def test_a_cycle_with_no_cross_places_nothing(monkeypatch, tmp_path):
    """The same cycle with the tape unable to answer: no order, no position, and a
    veto the desk can read instead of a 277x trade."""
    ap, _res, book, out = _run_cycle(monkeypatch, tmp_path, _BULL, "fx_none",
                                     battery=_AGREE, rate=None)
    assert out["trades"] == 0
    assert book.positions == {} and book.cash == 1_000_000.0
    row = ap.st["signals"][0]
    assert row["action"] == "NO-FX"
    assert "no USD->PKR rate" in row["detail"]
