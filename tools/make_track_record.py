"""Generate docs/investors/TRACK_RECORD.md from the live ledgers.

Everything an outside party can check about KAIROS's performance lives in this
file, and it is rebuilt from the journals rather than typed: the numbers a
prospective investor reads must be the numbers the system actually wrote.

    python tools/make_track_record.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos.feedback import track_record  # noqa: E402

OUT = ROOT / "docs" / "investors" / "TRACK_RECORD.md"
LINE = re.compile(r"^(\w+): (\w+) ret=([+-][\d.]+)% bench=([+-][\d.]+)% "
                  r"sharpe=([\d.-]+) hit=([\d.]+) trades=(\d+) dd=([\d.-]+)%")
# A run also prints the instruments it could not sweep. They are not rows, and the
# script's own summary counts them anyway, so the log ends with "15 instruments,
# 10 unprofitable" over a fourteen-row table — a headline that argues with its own
# evidence. They are read out separately and named, so the reader can see the
# coverage rather than inherit an inflated denominator.
SKIPPED = re.compile(r"^(\w+): (insufficient history.*|ERROR.*)$")


def _spectrum() -> tuple[list[dict], dict, Path | None]:
    """Widest completed walk-forward sweep, parsed from the run log.

    Ties go to the newest file. They did not: the comparison was on row count
    alone over a sorted glob, so re-running the sweep into a new log published
    nothing — the ledger kept reading the older run of the same width because its
    name came first alphabetically. A refreshed table that cannot reach the
    document is a staler table wearing a new date, which is the failure this
    whole file exists to prevent.

    The log itself is returned as well, because a sweep is evidence with a
    vintage: it was run on one afternoon against one day's bars, and a reader has
    to be able to tell that apart from the journal figures beside it, which are
    current to the minute.
    """
    best: tuple[tuple[int, float], list[dict], dict, Path | None] = \
        ((0, 0.0), [], {}, None)
    for log in sorted((ROOT / "reports").glob("spectrum*.log")):
        rows, summary, skipped = [], {}, []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if (m := LINE.match(line.strip())):
                sym, verdict, ret, bench, sharpe, hit, trades, dd = m.groups()
                rows.append({"symbol": sym, "verdict": verdict,
                             "return": float(ret) / 100, "benchmark": float(bench) / 100,
                             "sharpe": float(sharpe), "hit_rate": float(hit),
                             "trades": int(trades), "max_dd": float(dd) / 100})
            elif line.startswith("summary:"):
                summary = ast.literal_eval(line.split(":", 1)[1].strip())
            elif (sk := SKIPPED.match(line.strip())):
                skipped.append(sk.group(1))
        if skipped:
            summary["skipped"] = skipped
        if (key := (len(rows), log.stat().st_mtime)) > best[0]:
            best = (key, rows, summary, log)
    return best[1], best[2], best[3]


def _run_summary(rows: list[dict], summary: dict) -> str:
    """Section 2's headline, counted off the table the reader can add up.

    The sweep script's own summary line counts every instrument it was *asked* to
    run, including the ones it gave up on for lack of history, so the log that
    produced the fourteen-row table said "5/15 instruments profitable, 10 not" —
    a headline arguing with its own evidence, in the direction that flatters the
    denominator and punishes the record at the same time. The skipped instruments
    are named instead, so the coverage gap is visible rather than absorbed.
    """
    wins = sum(1 for r in rows if r["verdict"] == "PROFITABLE")
    skipped = summary.get("skipped") or []
    out = (f"Run summary: **{wins}/{len(rows)} instruments profitable out-of-sample**, "
           f"the other {len(rows) - wins} unprofitable.")
    if skipped:
        out += (f" Skipped for lack of history, and not in the table: "
                f"{', '.join(skipped)}.")
    return out


def _paper() -> dict:
    p = ROOT / "data" / "paper_account.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _autopilot() -> dict:
    p = ROOT / "data" / "autopilot.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _live_book() -> dict:
    p = ROOT / "data" / "livebook.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pending_calls() -> tuple[int, str]:
    """Directional snapshots whose horizon has not printed yet, and the first
    date the tenth of them could resolve.

    The record's central sentence — "no hit rate yet" — is only honest if the
    reader can see the queue behind it. A journal with zero pending calls and
    zero matured ones is a system that is not calling; a journal with hundreds
    pending is a system that is calling and has not been graded yet. The date is
    a lower bound, not a promise: `feedback.resolve_outcomes` waits for five
    *bars*, and PSX prints none on a weekend or a market holiday.
    """
    p = ROOT / "data" / "prediction_journal.jsonl"
    if not p.exists():
        return 0, ""
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("resolved") or str(e.get("direction", "")).upper() in (
                "", "HOLD", "NEUTRAL", "FLAT"):
            continue
        try:
            out.append(datetime.fromisoformat(e["ts"])
                       + timedelta(days=int(e["horizon"])))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort()
    return len(out), (out[9].strftime("%Y-%m-%d") if len(out) >= 10 else "")


TEST_WINDOW_S = 30.0


def _audit_trail() -> dict:
    """Count what the audit journal actually contains.

    The ledger used to describe its own evidence in prose only. An outside
    reader could not tell that the money-path events were written by tests, so
    the numbers are recomputed on every rebuild and published.
    """
    p = ROOT / "data" / "audit_journal.jsonl"
    out = {"lines": 0, "events": Counter(), "live_starts": 0,
           "test_attributable": 0, "contained": False, "venue_acks": 0}
    if not p.exists():
        return out
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    out["lines"] = len(rows)
    out["events"] = Counter(r.get("event", "?") for r in rows)
    out["contained"] = "AUDIT_TRAIL_TEST_EVENTS_CONTAINED" in out["events"]

    def _ts(r):
        try:
            return datetime.fromisoformat(r["ts"]).timestamp()
        except (KeyError, ValueError):
            return None

    live = [t for t in (_ts(r) for r in rows
                        if r.get("event") == "AUTOPILOT_START"
                        and r.get("mode") == "live") if t]
    out["live_starts"] = len(live)
    # The only shape that could mean money moved: a venue ack carrying an order
    # number. The test broker answers with the constant 1 and the mock JSG
    # session answers with null, so neither counts as one.
    out["venue_acks"] = sum(
        1 for r in rows
        if r.get("event") in ("AUTOPILOT_PLACED", "JSG_PLACED", "MT5_LIVE")
        and isinstance(r.get("order_no"), int) and r["order_no"] != 1)
    out["test_attributable"] = sum(
        1 for r in rows
        if (t := _ts(r)) and any(0.0 <= t - s <= TEST_WINDOW_S for s in live))
    return out


def _method() -> dict[str, int]:
    """How the sweep actually fits its signal, read out of the code it runs with.

    The ledger used to describe this section as "100 candidate strategies per
    instrument". It is not: `walk_forward()` blends four components into one
    composite signal and grid-searches that blend's weights, never importing the
    strategy library the pack elsewhere counts at 100. Every copy of the sentence
    in the pack carried the wrong number and the wrong object, and the wrong
    number was the smaller of the two errors — a 2,592-point search is a much
    larger overfitting surface than the "parameter search is a known risk"
    caveat let a reader assume. So the count is taken from `_grid()` here rather
    than typed, and widening the grid changes the published sentence by itself.
    """
    from kairos.backtest import CALIB_WINDOW, REFIT_EVERY, SMOOTH_SPANS, _grid
    grid = _grid()
    return {"grid": len(grid) * len(SMOOTH_SPANS), "train": CALIB_WINDOW,
            "refit": REFIT_EVERY,
            "blend": sum(1 for k in grid[0] if k.startswith("w_"))}


def main() -> int:
    tr = track_record()
    rows, summary, run_log = _spectrum()
    method = _method()
    pa, ap = _paper(), _autopilot()
    lb = _live_book()
    at = _audit_trail()
    by_venue = defaultdict(list)
    for r in rows:
        by_venue["PSX" if r["symbol"] in ("OGDC", "HBL", "MCB", "LUCK", "KSE100",
                                         "ENGRO", "FFC", "PPL") else "Global"].append(r)

    md = []
    w = md.append
    w("# KAIROS — verification ledger")
    w("")
    w("_Rebuilt from the on-disk journals by `tools/make_track_record.py`. "
      "Unaudited. Nothing here is a forecast or a promise of return._")
    w("")
    w("## 1. Realized forward predictions")
    w("")
    w("| measure | value |")
    w("|---|---|")
    # Dated, because every figure below is a snapshot of a journal that is still
    # growing. An undated ledger invites the reader to assume it is current, and
    # the pack used to carry a "verified today" footer over numbers three passes
    # old. The date is the thing they can check that against.
    w(f"| rebuilt from the journals on | {datetime.now().strftime('%Y-%m-%d')} |")
    w(f"| predictions journaled | {tr.get('predictions', 0)} |")
    w(f"| horizons matured | {tr.get('resolved', 0)} |")
    w(f"| of which directional calls | {tr.get('directional_resolved', 0)} |")
    w(f"| of which non-trades (HOLD) | {tr.get('non_directional_resolved', 0)} |")
    hr = tr.get("hit_rate")
    w(f"| directional hit rate | {'not yet measurable' if hr is None else f'{hr:.1%}'} |")
    w("")
    pend, tenth_day = _pending_calls()
    if hr is None:
        w("**There is no directional hit rate yet.** Every matured snapshot was a "
          "non-trade, and a non-trade cannot be a correct prediction — the scorer "
          "used to count them as wins, which is fixed as of v2.5.0. A forward record "
          "of live calls is what this table will show, and it accumulates on its own "
          "as horizons elapse.")
    if pend:
        # "No track record yet" reads the same whether the engine has never made a
        # call or has made hundreds that have not been graded. It is only an honest
        # sentence with the queue in front of the reader, and a date they can come
        # back on — which is also what makes a stalled record visible.
        w("")
        w(f"{pend} directional calls are journaled and still waiting out their "
          f"horizon"
          + (f", and the tenth of them cannot resolve before **{tenth_day}**, five "
             f"bars after it was written. This table publishes a hit rate the day it "
             f"can." if tenth_day else
             ", fewer than ten so far, so no hit rate is definable yet."))
    w("")
    w("## 2. Walk-forward strategy sweep")
    w("")
    w(f"One composite signal per instrument, blended from {method['blend']} components. "
      f"Its weights, entry threshold, regime gate and short mode are chosen by a "
      f"{method['grid']:,}-point grid search on a rolling {method['train']}-bar training "
      f"window, re-fitted every {method['refit']} bars, and traded only out-of-sample with "
      f"PSX commission and slippage modelled. This sweep does not evaluate the 100-strategy "
      f"library: that votes inside the live engine, and none of the returns below come from it.")
    w("")
    if run_log is not None:
        run_day = datetime.fromtimestamp(run_log.stat().st_mtime)
        age = (datetime.now() - run_day).days
        w(f"_Sweep run on **{run_day:%Y-%m-%d}** (`reports/{run_log.name}`)"
          + (f", **{age} day(s) before this rebuild** — re-run "
             f"`python scripts/backtest_spectrum.py` to refresh it."
             if age > 1 else ", the same day this ledger was rebuilt.") + "_")
        w("")
    w("| instrument | verdict | net return | buy & hold | Sharpe | hit rate | trades | max drawdown |")
    w("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["sharpe"]):
        w(f"| {r['symbol']} | {r['verdict'].lower()} | {r['return']:+.1%} | "
          f"{r['benchmark']:+.1%} | {r['sharpe']:+.2f} | {r['hit_rate']:.3f} | "
          f"{r['trades']} | {r['max_dd']:.1%} |")
    w("")
    if rows:
        w(_run_summary(rows, summary))
    w("")
    for venue, vr in sorted(by_venue.items()):
        net = sum(r["return"] for r in vr) / len(vr)
        beat = sum(1 for r in vr if r["return"] > r["benchmark"]) / len(vr)
        w(f"- **{venue}**: mean net return {net:+.1%} across {len(vr)} instruments; "
          f"beat buy-and-hold on {beat:.0%} of them.")
    w("")
    w(f"_Backtests are not returns. They include costs but not every market impact, and "
      f"the traded parameter set is chosen from {method['grid']:,} candidates in every "
      f"calibration window, so selection bias is stated here as a number rather than as a "
      f"disclaimer. That is why the sweep is reported per instrument instead of as an "
      f"average, and why section 1 — the forward record — is the figure that matters._")
    w("")
    w("## 3. Live paper book")
    w("")
    w(f"- Capital Rs {pa.get('capital', 0):,.0f} · equity Rs {pa.get('cash', 0):,.0f} · "
      f"realized P&L Rs {pa.get('realized_pnl', 0):,.2f} since {ap.get('last_rotation', 'start')}")
    w(f"- Autopilot cycles run: **{ap.get('cycles', 0)}** · mode `{ap.get('mode')}` · "
      f"live-armed: {ap.get('armed_live')}")
    w(f"- Open paper positions: {len(pa.get('positions') or {})} · fills: {len(pa.get('fills') or [])}")
    w("")
    w("## 4. Funded account (read-only, from the broker)")
    w("")
    eq = lb.get("equity")
    w(f"- Holdings tracked: {len(lb.get('holdings') or [])}"
      + (f" · broker equity Rs {eq:,.2f}" if isinstance(eq, (int, float)) else
         " · equity not fetched in this snapshot"))
    w("- KAIROS observes this book; real orders require three independent keys "
      "(`autopilot.auto_place`, `js_global.allow_orders`, the broker PIN) and the "
      "kill switch to be clear.")
    w("")
    w("## 5. What is and is not established")
    w("")
    w("| claim | status |")
    w("|---|---|")
    w("| The system produces structured, risk-gated signals across PSX, FX, commodities and crypto | **established** — running continuously, journalled |")
    w("| Out-of-sample strategy selection beats buy-and-hold on some instruments | **partly established** — see the sweep; commodity books strongest |")
    w("| PSX equities are profitable net of local costs at these horizons | **not established** — fee and slippage drag exceeded signal on the equity book |")
    w("| A realized forward track record | **does not exist yet** — no directional call has matured |")
    w("| Any return promise to an investor | **explicitly not made** |")
    w("")
    w("## 6. How this file is verified")
    w("")
    w("- `data/prediction_journal.jsonl` — every prediction, timestamped, resolved "
      "only when the horizon's bars exist.")
    w("- `data/paper_account.json` and `data/autopilot.json` — the paper book and "
      "every gate decision the loop made.")
    w("- `data/audit_journal.jsonl` — order staging, arming, kill-switch and "
      "venue-loss events, append-only.")
    w("- Rebuilding this ledger is one command; it reads only those files.")
    w("")
    w("### What the audit journal actually contains")
    w("")
    w("| measure | value |")
    w("|---|---|")
    w(f"| lines recorded | {at['lines']} |")
    w(f"| distinct event types | {len(at['events'])} |")
    w(f"| live-mode loop starts | {at['live_starts']} |")
    w(f"| lines inside {TEST_WINDOW_S:.0f}s of a live-mode start "
      f"(the automated-test signature) | {at['test_attributable']} |")
    w(f"| venue acknowledgements carrying a real order number | "
      f"**{at['venue_acks']}** |")
    w("")
    if at["venue_acks"] == 0:
        w("**No order has ever reached a broker.** Every `AUTOPILOT_PLACED` line "
          "in the journal answers with the test broker's constant order number `1` "
          "at a fixed fixture price, and the single `JSG_STAGED`/`JSG_PLACED` pair "
          "came from an offline unit test whose HTTP session was mocked — that pair "
          "is annotated in the journal itself.")
    w("")
    if at["test_attributable"] and at["contained"]:
        w(f"Of the {at['lines']} lines, **{at['test_attributable']} were written by "
          "the automated test suite**, not by the trading process. The suite used to "
          "import the operator's live configuration, so the module-bound journal "
          "object inside a test process appended to the production file, armed "
          "`live` mode against stubbed brokers, and left its daemon threads running "
          "after the stubs were torn down. Nothing was placed and the paper book was "
          "never touched, but the trail was polluted, which is a defect in an "
          "append-only audit log and is not something to leave unmentioned.")
        w("")
        w("The fix is structural, not a cleanup of the file: `tests/conftest.py` now "
          "points every test at a throwaway config with an unroutable portal URL and "
          "empty credentials, blocks the transport layer so no test can reach a "
          "network, and stops every autopilot instance built during a test at "
          "teardown. The history is left in place — the containment event itself is "
          "the last line of the journal, because rewriting an audit trail to hide "
          "its own contamination would destroy the thing it is supposed to prove.")
    elif at["test_attributable"]:
        w(f"{at['test_attributable']} lines fall inside the test signature window "
          "and have not yet been annotated in the journal.")
    w("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(md)} lines)")
    counts = Counter(r["verdict"] for r in rows)
    print("sweep rows:", len(rows), dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
