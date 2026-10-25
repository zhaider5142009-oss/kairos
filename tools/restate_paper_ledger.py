"""Restate the paper ledger in the money it has always claimed to be counted in.

The book keeps one money now (#97): a fill in another currency is converted at the
published cross before it touches cash, and `realized P&L Rs` in the published track
record is a rupee figure. The ledger on disk predates that. Its only two trades are
1.2 oz of GOLD sold and bought back — a dollar round trip — and the file records it
as `realized_pnl: -13.05` next to a rupee million, printed to investors as
"realized P&L Rs -13.05" when the trade lost about Rs 3,500.

This script rebuilds the ledger's money from its own `fills`, leg by leg, through
`PaperBroker.submit` itself, so the arithmetic restating the book is the arithmetic
that will keep it. The only thing overridden is *which cross a leg sees*: each leg is
converted at the USDPKR print on or before its own date, the way the live book
converts a leg at the print of its own day.

The fills themselves are not rewritten. The trades happened when the tape said they
happened, at the prices it recorded; only the money figures beside them were wrong,
and a restatement that also moved the timestamps would replace one untrue record with
another.

Nothing is written without `--apply`, and the file it replaces is kept. Re-running is
harmless: the restatement is a function of `fills` and the bar series, not of the
figures already in the file.

    python tools/restate_paper_ledger.py             # report only
    python tools/restate_paper_ledger.py --apply     # back up, then write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos.config import CFG  # noqa: E402
from kairos.data import STORE  # noqa: E402
from kairos.execution import (  # noqa: E402
    JOURNAL, KILL, OrderIntent, PaperBroker, _instrument_currency, _paper_currency,
)
from kairos.store import atomic_write, dumps_strict  # noqa: E402

FILLS_KEPT = 500        # PaperBroker._save truncates to this: at it, history is lost


def _bars() -> pd.DataFrame | None:
    df = STORE.load("USDPKR")
    if df is None or not len(df) or not {"date", "close"} <= set(df.columns):
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values("date")


def _fx_on(bars: pd.DataFrame | None, day, frm: str, to: str) -> float | None:
    """The cross on or before `day` — the print that was live when the leg struck."""
    f, t = str(frm or "").upper(), str(to or "").upper()
    if not f or not t or f == t:
        return 1.0
    if bars is None:
        return None
    hit = bars[bars["date"] <= day]
    if not len(hit):
        return None
    try:
        close = float(hit.iloc[-1]["close"])
    except (TypeError, ValueError):
        return None
    if close <= 0 or close != close:      # NaN never compares equal to itself
        return None
    if (f, t) == ("USD", "PKR"):
        return close
    if (f, t) == ("PKR", "USD"):
        return 1.0 / close
    return None


class _Replay(PaperBroker):
    """The production ledger, shown the cross of each leg's own day."""

    day = None
    bars: pd.DataFrame | None = None

    def rate(self, frm: str) -> float | None:
        return _fx_on(self.bars, self.day, frm, self.currency)

    def _save(self) -> None:
        """A replay is a measurement; the file it reproduces figures for is not
        written by it. Overriding this is what makes `--apply` the only path that can
        touch the ledger, and it means a report run leaves no state file anywhere."""


def _legs(state: dict) -> list[dict]:
    return sorted(state.get("fills") or [], key=lambda f: str(f["ts"]))


def _replay(state: dict, bars, scratch: Path):
    """Run the recorded legs on a throwaway book and report its money.

    `PaperBroker.__init__` reads `<data_dir>/paper_account.json`, and a file it
    cannot read renames itself and engages the kill switch. So the replay book is
    built with the data dir pointed at an empty scratch directory: a report-only run
    must never have a path to the live ledger it is here only to measure.
    """
    currency = str(state.get("currency") or _paper_currency()).upper()
    # A replayed leg calls JOURNAL.log like any fill, and the journal is the audit
    # trail of the book that really traded. Replaying the same legs to *measure* them
    # must not record them a second time, so the logger is silent for the run.
    real_log, real_dir = JOURNAL.log, CFG.data_dir
    stamps: list = []
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        CFG.data_dir = scratch
        # Built before the logger is replaced: a constructor that found a damaged
        # ledger would have engaged the kill switch, and that must happen against the
        # empty scratch directory, never the live one.
        book = _Replay()
        book.state_path = scratch / "replay_account.json"
        book.currency = currency
        book.capital = book.cash = float(state.get("capital") or 0.0)
        book.positions, book.fills = {}, []
        book.realized_pnl = book.day_pnl = 0.0
        book._day_stamp = ""
        book.bars = bars
        JOURNAL.log = lambda event, payload=None: None
        for leg in _legs(state):
            price, qty = float(leg["price"]), float(leg["quantity"])
            if not (price > 0) or not (qty > 0):
                return book, stamps, f"leg {leg.get('ts')} has no usable price or size"
            fee = float(leg.get("commission") or 0.0)
            book.day = pd.Timestamp(leg["ts"]).date()
            before = book.day_pnl
            # Slippage is not re-applied: the recorded price already carries it, and
            # the commission is stated back in bps from the recorded fee, so the
            # replay's fill is the fill that happened in every figure the file holds.
            fill = book.submit(OrderIntent(symbol=leg["symbol"], side=leg["side"],
                                           quantity=qty, venue="PAPER"), price,
                               slippage_bps=0.0,
                               commission_bps=fee / (price * qty) * 10_000.0)
            if fill is None:
                return book, stamps, (
                    f"the restated book refused its own {leg['symbol']} "
                    f"leg — the ledger cannot be rebuilt from it")
            stamps.append((book.day, float(book.day_pnl) - float(before)))
    finally:
        CFG.data_dir = real_dir
        JOURNAL.log = real_log
    return book, stamps, None


def _shape(row: dict) -> tuple:
    """What a recorded position must reproduce: its size and its average entry.

    `last` is deliberately left out — it is the mark the desk puts on a holding
    after the fill, not a number the fills can regenerate.
    """
    return (round(float(row["qty"]), 8), round(float(row["entry"]), 6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="back up the ledger and write the restated figures")
    args = ap.parse_args()

    path = CFG.data_dir / "paper_account.json"
    if not path.exists():
        print(f"no paper ledger at {path}")
        return 1
    state = json.loads(path.read_text(encoding="utf-8"))
    if len(state.get("fills") or []) >= FILLS_KEPT:
        print(f"{FILLS_KEPT} fills kept: the history is truncated, and replaying it "
              "would drop the trades before the window. Refusing.")
        return 1
    bars = _bars()
    if bars is None:
        print("no USDPKR series to restate a dollar leg with")
        return 1

    # The replay book is constructed with `data_dir` pointed here, so it finds no
    # ledger of its own and never reads or renames the one being measured.
    scratch = Path(tempfile.mkdtemp(prefix="kairos-restate-"))
    try:
        return _restate(state, path, bars, scratch, args.apply)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _restate(state: dict, path: Path, bars, scratch: Path, apply: bool) -> int:
    book, stamps, problem = _replay(state, bars, scratch)
    if problem:
        print(f"restatement refused: {problem}")
        return 1
    replayed = {k: _shape(v) for k, v in book.positions.items()}
    recorded = {k: _shape(v) for k, v in (state.get("positions") or {}).items()}
    if replayed != recorded:
        print(f"the fills do not rebuild the book: replayed {replayed}, "
              f"ledger says {recorded}")
        return 1

    day_stamp = str(state.get("day_stamp") or "")
    day_pnl = sum(delta for (d, delta) in stamps if str(d) == day_stamp)
    # A closed book has nowhere to hide a rupee: the two realised lines must add
    # up to the move in cash. If they do not, the replay disagrees with the very
    # arithmetic that wrote the ledger, and writing it would publish the gap.
    if not book.positions and abs((book.realized_pnl + book.fx_pnl)
                                 - (book.cash - book.capital)) > 0.01:
        print("restatement refused: realised P&L and the cash move do not agree "
              f"({book.realized_pnl + book.fx_pnl:.2f} vs "
              f"{book.cash - book.capital:.2f})")
        return 1
    rows = [("currency", str(state.get("currency") or "PKR"), book.currency),
            ("cash", state.get("cash"), book.cash),
            ("realized_pnl", state.get("realized_pnl"), book.realized_pnl),
            ("fx_pnl", state.get("fx_pnl"), book.fx_pnl),
            ("realized_total", None, book.realized_pnl + book.fx_pnl),
            ("day_pnl", state.get("day_pnl"), day_pnl),
            ("open positions", len(state.get("positions") or {}), len(book.positions))]
    print(f"{'figure':<14} {'on disk':>18} {'restated':>18}")
    for name, old, new in rows:
        flag = "   (unchanged)" if (isinstance(old, (int, float))
                                    and isinstance(new, (int, float))
                                    and abs(old - new) < 0.005) else ""
        print(f"{name:<14} {old!s:>18} {new!s:>18}{flag}")
    for leg, (d, delta) in zip(_legs(state), stamps):
        ccy = _instrument_currency(leg["symbol"], book.currency)
        fx = _fx_on(bars, d, ccy, book.currency)
        print(f"  {leg['ts']} {leg['side']:<4} {leg['quantity']:>10} "
              f"{leg['symbol']:<8} {ccy}->{book.currency} at {fx} -> {delta:,.2f}")

    if not apply:
        print("\nreport only — nothing written. Re-run with --apply.")
        return 0

    stamp = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(f"{path.name}.pre-restate-{stamp}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    # The restated book is written in the shape the live book writes its own state,
    # through the same serializer — with its fills left exactly as they were.
    atomic_write(path, dumps_strict({
        "currency": book.currency, "capital": book.capital, "cash": book.cash,
        "positions": book.positions, "fills": _legs(state)[-FILLS_KEPT:],
        "realized_pnl": book.realized_pnl, "fx_pnl": book.fx_pnl,
        "day_pnl": day_pnl, "day_stamp": day_stamp}, indent=1))
    JOURNAL.log("PAPER_LEDGER_RESTATED", {
        "path": str(path), "kept_as": backup.name,
        "cash_before": state.get("cash"), "cash_after": book.cash,
        "realized_before": state.get("realized_pnl"),
        "realized_after": book.realized_pnl, "fx_after": book.fx_pnl,
        "legs": len(stamps)})
    print(f"restated — previous ledger kept as {backup.name}")
    if KILL.engaged():
        print("note: the kill switch is engaged; release it before the next cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
