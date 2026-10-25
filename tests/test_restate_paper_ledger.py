"""Restating the ledger's money without restating its history.

`data/paper_account.json` is the track record: the realized figure in it is printed
to investors as `realized P&L Rs …`. It was written before the book kept one
currency (#97), so its only round trip — 1.2 oz of GOLD sold and bought back — is
recorded as `realized_pnl: -13.05`, the dollar arithmetic with an Rs label, when the
rupee figure is about -3,500. `tools/restate_paper_ledger.py` rebuilds the money from
the ledger's own fills, leg by leg, through `PaperBroker.submit`.

Three things this file has to be sure of, because the script touches the record:

1. the restated round trip is counted in rupees at each leg's own print;
2. a leg already in the book's money comes back untouched — the script must not
   convert a rupee scrip twice, which would invent a loss of its own;
3. a replay measures the ledger and never writes to it: no state file, no journal
   event, no kill switch, whatever the live file looks like.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_TOOLS = str(ROOT / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import restate_paper_ledger as RS  # noqa: E402  (the restatement is the thing under test)
from kairos import execution as E  # noqa: E402

# The two fills the live ledger actually holds, and the two prints the USDPKR series
# really carried on those days. Kept verbatim so this file grades the same round trip
# the published track record describes.
SELL_TS, BUY_TS = "2026-10-17T14:11:22", "2026-10-18T00:43:23"
SELL_PX, BUY_PX = 4179.009549949999, 4183.19065005
SELL_FEE, BUY_FEE = 4.011849167951999, 4.015863024048
RATE_17, RATE_18 = 268.0496520996094, 268.60009765625


def _bars() -> pd.DataFrame:
    return pd.DataFrame({"date": [dt.date(2026, 10, 16), dt.date(2026, 10, 17),
                                  dt.date(2026, 10, 18), dt.date(2026, 10, 20)],
                         "close": [267.0, RATE_17, RATE_18, 276.75]})


def _fill(ts: str, side: str, symbol: str, qty: float, price: float,
          fee: float) -> dict:
    return {"ts": ts, "side": side, "symbol": symbol, "quantity": qty,
            "price": price, "commission": fee, "venue": "PAPER"}


def _state(fills: list[dict], **over) -> dict:
    state = {"capital": 1_000_000.0, "cash": 1_000_000.0, "positions": {},
             "fills": fills, "realized_pnl": 0.0, "day_pnl": 0.0,
             "day_stamp": ""}
    state.update(over)
    return state


def _gold_trip() -> list[dict]:
    return [_fill(SELL_TS, "SELL", "GOLD", 1.2, SELL_PX, SELL_FEE),
            _fill(BUY_TS, "BUY", "GOLD", 1.2, BUY_PX, BUY_FEE)]


def _replay(tmp_path: Path, state: dict, bars=None):
    return RS._replay(state, _bars() if bars is None else bars, tmp_path)


# ----------------------------------------------------------------------- the cross
def test_a_leg_sees_the_print_of_its_own_day_not_todays():
    bars = _bars()
    assert RS._fx_on(bars, dt.date(2026, 10, 17), "USD", "PKR") == RATE_17
    assert RS._fx_on(bars, dt.date(2026, 10, 19), "USD", "PKR") == RATE_18
    # The weekend reads Friday, because Friday is the print that was live.
    assert RS._fx_on(bars, dt.date(2026, 10, 20), "USD", "PKR") == 276.75


def test_a_print_before_the_first_bar_is_not_guessed():
    assert RS._fx_on(_bars(), dt.date(2026, 10, 15), "USD", "PKR") is None


def test_the_book_money_needs_no_tape_and_another_pair_has_no_print():
    assert RS._fx_on(_bars(), dt.date(2026, 10, 17), "PKR", "PKR") == 1.0
    assert RS._fx_on(_bars(), dt.date(2026, 10, 17), "EUR", "PKR") is None
    assert RS._fx_on(None, dt.date(2026, 10, 17), "USD", "PKR") is None
    # The rupee book prices a dollar in the mirror of the same print, not a new one.
    assert RS._fx_on(_bars(), dt.date(2026, 10, 17), "PKR", "USD") == pytest.approx(
        1.0 / RATE_17)


def test_a_bar_that_is_not_a_number_is_no_print():
    bars = pd.DataFrame({"date": [dt.date(2026, 10, 17)], "close": [float("nan")]})
    assert RS._fx_on(bars, dt.date(2026, 10, 17), "USD", "PKR") is None
    bars = pd.DataFrame({"date": [dt.date(2026, 10, 17)], "close": ["n/a"]})
    assert RS._fx_on(bars, dt.date(2026, 10, 17), "USD", "PKR") is None


# ------------------------------------------------------------------- the round trip
def test_the_dollar_round_trip_comes_back_in_rupees(tmp_path):
    book, stamps, problem = _replay(tmp_path, _state(_gold_trip()))
    assert problem is None
    # The file says -13.05. Both legs converted at their own day's print put the loss
    # at about Rs 3,500 — the difference is the bug, and the restatement has to be on
    # the rupee side of it.
    assert book.realized_pnl == pytest.approx(-3501.7, abs=1.0)
    assert book.realized_pnl < -100.0
    # ... and the principal's own move is the second realised line. It used to be
    # in `cash` and nowhere else, so the file's cash fell Rs 6,262.07 while its
    # realised line said 3,501.69 and no figure named the other 2,760.38.
    assert book.fx_pnl == pytest.approx(-2760.4, abs=1.0)
    assert book.realized_pnl + book.fx_pnl == pytest.approx(
        book.cash - 1_000_000.0, rel=1e-9)
    # Cash is the truth of the two prints: the short is opened at Friday's rate and
    # bought back at Saturday's, so the principal crosses twice and the rate drift
    # between them lands on the ledger, not in `realized`.
    assert book.cash == pytest.approx(
        1_000_000.0 + (SELL_PX * 1.2 - SELL_FEE) * RATE_17
        - (BUY_PX * 1.2 + BUY_FEE) * RATE_18, rel=1e-9)
    assert book.currency == "PKR"
    assert book.positions == {}
    assert [d for d, _x in stamps] == [dt.date(2026, 10, 17), dt.date(2026, 10, 18)]


def test_a_rupee_leg_is_left_in_rupees(tmp_path):
    """The restatement must not convert what was never foreign.

    OGDC is priced in rupees on a rupee book, so the figure in the file is already
    the money the track record publishes. Multiplying it by the cross anyway would
    invent a 268x loss on a scrip that lost nothing of the kind.
    """
    fills = [_fill("2026-10-17T14:11:22", "BUY", "OGDC", 1000.0, 300.0, 6.0),
             _fill("2026-10-18T00:43:23", "SELL", "OGDC", 1000.0, 306.0, 6.12)]
    book, _stamps, problem = _replay(tmp_path, _state(fills))
    assert problem is None
    # (306 - 300) * 1000 - 6 - 6.12, exactly as the file records it, at a rate of 1.
    assert book.realized_pnl == pytest.approx(5987.88, rel=1e-9)
    assert book.cash == pytest.approx(1_000_000.0 + 5987.88, rel=1e-9)


def test_a_leg_with_no_print_before_it_refuses_the_whole_restatement(tmp_path):
    fills = [_fill("2026-10-01T09:00:00", "BUY", "GOLD", 1.0, 4000.0, 3.2)]
    book, _stamps, problem = _replay(tmp_path, _state(fills))
    assert problem is not None and "GOLD" in problem
    assert book.fills == []


def test_a_leg_that_costs_more_than_the_book_cannot_be_rebuilt(tmp_path):
    """Replaying a ticket the restated book cannot fund is not a restatement.

    The old ledger would have taken 1.2 oz of gold for Rs 5,020 because it never
    converted. A rupee-correct book refuses it — and a script that quietly wrote the
    remaining legs would publish a shorter trade history than the one that happened.
    """
    fills = [_fill(SELL_TS, "BUY", "GOLD", 1.2, SELL_PX, SELL_FEE),
             _fill(BUY_TS, "BUY", "GOLD", 1.2, BUY_PX, BUY_FEE)]
    book, _stamps, problem = _replay(tmp_path, _state(fills))
    assert problem is not None and "cannot be rebuilt" in problem


def test_a_leg_without_a_usable_price_stops_the_restatement(tmp_path):
    fills = [_fill(SELL_TS, "SELL", "GOLD", 1.2, 0.0, 0.0)]
    _book, _stamps, problem = _replay(tmp_path, _state(fills))
    assert problem is not None and "no usable price" in problem


def test_the_position_the_ledger_claims_to_hold_is_the_one_the_fills_make(tmp_path):
    fills = [_fill(SELL_TS, "BUY", "OGDC", 1000.0, 300.0, 6.0)]
    state = _state(fills, positions={"OGDC": {"qty": 1000.0, "entry": 300.0,
                                               "last": 311.0, "commission": 6.0,
                                               "ccy": "PKR"}})
    book, _stamps, problem = _replay(tmp_path, state)
    assert problem is None
    assert book.positions["OGDC"]["qty"] == pytest.approx(1000.0)
    assert book.positions["OGDC"]["entry"] == pytest.approx(300.0)
    # `last` is the desk's mark, not a number the fills can regenerate: the replay is
    # compared on size and entry only, so a marked-up holding is not read as a lie.
    assert book.positions["OGDC"]["last"] == pytest.approx(300.0)


def test_a_dollar_name_is_sized_from_the_day_the_leg_struck(tmp_path):
    """Same fills, different prints: the restatement follows the tape, not the clock."""
    bars = pd.DataFrame({"date": [dt.date(2026, 10, 17), dt.date(2026, 10, 18)],
                         "close": [100.0, 100.0]})
    book, _stamps, problem = _replay(tmp_path, _state(_gold_trip()), bars)
    assert problem is None
    assert book.realized_pnl == pytest.approx(
        ((BUY_PX - SELL_PX) * -1.2 - SELL_FEE - BUY_FEE) * 100.0, rel=1e-6)


# ------------------------------------------------------------- it must not touch
def test_a_replay_writes_nothing_anywhere(tmp_path, monkeypatch):
    """Report-only means report-only: the script may be run against a live ledger.

    `PaperBroker` persists on every fill and journals every one, so the replay has to
    be silent about both — otherwise measuring the book doubles its trade history and
    fills the audit trail with events that never happened.
    """
    seen: list = []
    monkeypatch.setattr(E.JOURNAL, "log",
                        lambda event, payload=None: seen.append(event))
    real_dir = E.CFG.data_dir
    live = tmp_path / "live"
    live.mkdir()
    (live / "paper_account.json").write_text(json.dumps(_state(_gold_trip())),
                                             encoding="utf-8")
    monkeypatch.setattr(E.CFG, "data_dir", live)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    state = json.loads((live / "paper_account.json").read_text(encoding="utf-8"))
    before = (live / "paper_account.json").read_bytes()
    book, _stamps, problem = RS._replay(state, _bars(), scratch)
    assert problem is None
    assert E.CFG.data_dir == live, "the replay left the data dir pointed at its own"
    assert seen == [], f"a replay journalled {seen}"
    assert (live / "paper_account.json").read_bytes() == before, \
        "a report-only run wrote into the live ledger"
    assert list(live.iterdir()) == [live / "paper_account.json"], \
        "a report-only run wrote beside the live ledger"
    assert not E.KILL.engaged()
    assert not (scratch / "replay_account.json").exists(), \
        "the replay book persisted a state of its own"


def test_a_damaged_live_ledger_is_still_untouched_by_a_report(tmp_path, monkeypatch):
    """A replay book that cannot read its own state engages the kill switch.

    That is the right answer for the production book and the wrong one for a script
    that was only asked to report, so the replay must never be constructed against the
    live data dir — here is the file that used to make it do exactly that.
    """
    live = tmp_path / "live"
    live.mkdir()
    (live / "paper_account.json").write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(E.CFG, "data_dir", live)
    seen: list = []
    monkeypatch.setattr(E.JOURNAL, "log",
                        lambda event, payload=None: seen.append(event))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    book, _stamps, problem = RS._replay(_state(_gold_trip()), _bars(), scratch)
    assert problem is None
    assert (live / "paper_account.json").exists(), \
        "a report-only run renamed the live ledger it could not read"
    assert seen == []
    assert not E.KILL.engaged()
    assert book.currency == "PKR"


def test_the_kill_switch_is_not_engaged_by_a_good_replay(tmp_path):
    E.KILL.path = tmp_path / "KILL_SWITCH"
    book, _stamps, problem = _replay(tmp_path, _state(_gold_trip()))
    assert problem is None and not E.KILL.engaged()


# ------------------------------------------------------------------ the guard rail
class _StubStore:
    """The tape the script reads, without reading the machine's cache."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def load(self, symbol: str):
        return None if symbol != "USDPKR" else self._bars.copy()


def _run(tmp_path, monkeypatch, fills, bars=None, apply=False):
    """Drive the script's own entry point against a temporary ledger."""
    live = tmp_path / "data"
    live.mkdir(exist_ok=True)
    path = live / "paper_account.json"
    path.write_text(json.dumps(_state(fills)), encoding="utf-8")
    monkeypatch.setattr(RS.CFG, "data_dir", live)
    monkeypatch.setattr(RS, "STORE", _StubStore(_bars() if bars is None else bars))
    monkeypatch.setattr(sys, "argv",
                        ["restate_paper_ledger.py"] + (["--apply"] if apply else []))
    return RS.main(), path


def test_a_truncated_history_is_refused(tmp_path, monkeypatch, capsys):
    """`_save` keeps the last 500 fills, so a full window has already lost some.

    Replaying what survives would publish a ledger whose trades are fewer than the
    ones that happened — a neater lie than the one being fixed.
    """
    fills = [_fill(f"2026-10-17T{h:02d}:00:00", "BUY", "OGDC", 1.0, 300.0, 0.06)
             for h in range(RS.FILLS_KEPT)]
    code, path = _run(tmp_path, monkeypatch, fills)
    assert code == 1
    assert "Refusing" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8"))["fills"] == fills


def test_a_report_only_run_leaves_the_ledger_byte_for_byte(tmp_path, monkeypatch):
    code, path = _run(tmp_path, monkeypatch, _gold_trip())
    before = path.read_bytes()
    assert code == 0
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path], "a report wrote beside the ledger"


def test_no_series_means_no_restatement(tmp_path, monkeypatch, capsys):
    class _Empty(_StubStore):
        def load(self, symbol: str):
            return None

    monkeypatch.setattr(RS, "STORE", _Empty(_bars()))
    live = tmp_path / "data"
    live.mkdir(exist_ok=True)
    path = live / "paper_account.json"
    path.write_text(json.dumps(_state(_gold_trip())), encoding="utf-8")
    monkeypatch.setattr(RS.CFG, "data_dir", live)
    monkeypatch.setattr(sys, "argv", ["restate_paper_ledger.py", "--apply"])
    assert RS.main() == 1
    assert "no USDPKR series" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8"))["realized_pnl"] == 0.0


def test_apply_restates_the_money_and_keeps_the_history(tmp_path, monkeypatch):
    code, path = _run(tmp_path, monkeypatch, _gold_trip(), apply=True)
    assert code == 0
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["realized_pnl"] == pytest.approx(-3501.7, abs=1.0)
    assert out["fx_pnl"] == pytest.approx(-2760.4, abs=1.0)
    assert out["realized_pnl"] + out["fx_pnl"] == pytest.approx(
        out["cash"] - out["capital"], rel=1e-9)
    assert out["currency"] == "PKR"
    assert out["cash"] == pytest.approx(
        1_000_000.0 + (SELL_PX * 1.2 - SELL_FEE) * RATE_17
        - (BUY_PX * 1.2 + BUY_FEE) * RATE_18, rel=1e-9)
    # The trades are the record. A restatement that re-dated them would replace one
    # untrue file with a differently untrue one.
    assert [(f["ts"], f["side"], f["symbol"], f["price"]) for f in out["fills"]] == \
           [(f["ts"], f["side"], f["symbol"], f["price"]) for f in _gold_trip()]
    assert out["positions"] == {}
    assert out["day_pnl"] == 0.0
    kept = [p.name for p in path.parent.iterdir() if "pre-restate" in p.name]
    assert len(kept) == 1, "the ledger it replaced was not kept"
    assert json.loads((path.parent / kept[0]).read_text(
        encoding="utf-8"))["realized_pnl"] == 0.0


def test_restating_a_restated_ledger_changes_nothing(tmp_path, monkeypatch):
    """Re-running the script is a report, not a compounding restatement."""
    code, path = _run(tmp_path, monkeypatch, _gold_trip(), apply=True)
    assert code == 0
    once = json.loads(path.read_text(encoding="utf-8"))
    state = _state(_gold_trip(), cash=once["cash"], currency=once["currency"],
                   realized_pnl=once["realized_pnl"], day_stamp=once["day_stamp"],
                   day_pnl=once["day_pnl"])
    monkeypatch.setattr(RS.CFG, "data_dir", tmp_path)
    book, _stamps, problem = RS._replay(state, _bars(), tmp_path / "scratch")
    assert problem is None
    assert book.realized_pnl == pytest.approx(once["realized_pnl"], rel=1e-9)
    assert book.cash == pytest.approx(once["cash"], rel=1e-9)
