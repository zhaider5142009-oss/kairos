"""Measure every strategy in the library against the market, per horizon and per regime.

    python tools/strategy_attribution.py

Writes `data/strategy_attribution.json`, which `kairos/attribution.py` turns into
the per-vote multipliers `ensemble.fuse` applies. This is a research run: it
reads history, writes one file, and never touches a broker, a gate or the book.

Why it exists: the 100-strategy library has always been polled at face value,
so a component that is right only in a trending tape carried the same weight in
a range, and the one place the system learns from the market — the realized
record — only ever updated one global confidence multiplier. Attribution needs
out-of-sample counts per strategy, and the only honest way to get them before
ten real calls have matured is to replay the library across history exactly the
way a live cycle would have seen it.

That last clause is the whole difficulty, and the design follows it:

* Each slice is the series **as it stood on that date**. Features, regime and
  patterns are recomputed on the slice, never on the full panel and then read
  backwards, because a 200-day moving average computed once over the whole file
  and indexed at the past is look-ahead wearing a history costume.
* Intermarket context is truncated by date too — it is a separate series with
  its own index, and passing the live dictionary here would hand every slice the
  last nine months of DXY, VIX and USDPKR.
* Context this run cannot supply is declared in the artifact (`ctx_supplied`),
  so a strategy that never voted because it waits for a PSX factor row or a news
  feed is recorded as *no coverage*, not as a strategy with no edge.
* A strategy that raises is counted and quarantined from weighting. Before this
  run there was no way to tell "abstained" from "broken"; the first thing it
  prints is how many of the library were silently throwing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# This run's output is usually redirected into reports/, and the Windows console
# encoding writes the em-dashes in the refusal messages as bytes that arrive as
# mojibake — in the one file an auditor reads to check what the run said.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kairos import attribution as A                                    # noqa: E402
from kairos.data import MarketData                                     # noqa: E402
from kairos.features import compute_features, detect_patterns, detect_regime  # noqa: E402
from kairos.macro import (MIN_COVERAGE, condition_label, macro_state)  # noqa: E402
from kairos.store import atomic_write                                  # noqa: E402
from kairos.strategies import STRATEGIES, run_strategy_library          # noqa: E402
from kairos.universe import resolve                                     # noqa: E402

# The sweep's own coverage set, so the attribution and the published track
# record describe the same markets rather than two different accidents.
DEFAULT_SYMBOLS = ["GOLD", "SILVER", "WTI", "BRENT", "NATGAS", "COPPER", "WHEAT",
                   "EURUSD", "BTC", "KSE100", "OGDC", "HBL", "LUCK", "ENGRO", "MCB"]

MIN_SLICE_BARS = 140        # the same floor the walk-forward refuses to trade below


def _slices(df: pd.DataFrame, step: int, horizons: tuple[int, ...]) -> list[int]:
    """Evaluation indices, newest-bar-last, each one leaving every horizon room
    to realize. A slice that cannot be graded is not a slice."""
    last = len(df) - 1 - max(horizons)
    return list(range(MIN_SLICE_BARS, last + 1, step))


def _truncated(series_by_key: dict[str, pd.DataFrame], cutoff) -> dict[str, pd.DataFrame]:
    """Context as it would have looked on `cutoff`, not as it looks today."""
    out = {}
    for key, s in series_by_key.items():
        try:
            sub = s[s["date"] <= cutoff]
        except TypeError:
            continue
        if len(sub) >= 40:
            out[key] = sub
    return out


def measure(md: MarketData, symbols: list[str], step: int,
            horizons: tuple[int, ...], max_bars: int,
            quiet: bool = False) -> dict:
    counts = A.new_counts(len(STRATEGIES), horizons)
    counts["generated"] = date.today().isoformat()
    counts["step"] = step
    counts["min_slice_bars"] = MIN_SLICE_BARS
    counts["ctx_supplied"] = ["regime", "macro (same date-truncated panel)",
                              "intermarket (truncated by date)",
                              "date_today", "patterns", "tags"]
    counts["ctx_unavailable"] = ["factor_row (needs a live PSX snapshot)",
                                 "news_sentiment (the news desk scores headlines it "
                                 "can still reach, which is today's window: a feed "
                                 "query is bounded by when:N d from the day the "
                                 "measure runs, so no historical slice can be shown "
                                 "the tape its bar actually had. The strategy that "
                                 "reads it therefore stays ungraded and keeps weight "
                                 "1.0 tagged 'no coverage' rather than being "
                                 "credited or demoted on evidence it never saw)",
                                 "psx_breadth (live cross-section only, so it is "
                                 "reported beside the macro state, never scored "
                                 "into its bucket tag)"]
    intermarket = {}
    try:
        from kairos.engine import ENGINE
        intermarket = ENGINE.intermarket()
    except Exception as exc:
        print(f"intermarket context unavailable: {type(exc).__name__}: {exc}")
    counts["intermarket_keys"] = sorted(intermarket)
    tags: dict[str, int] = {}
    thin_coverage = 0

    for sym in symbols:
        inst = resolve(sym)
        if inst is None:
            print(f"{sym}: not in the universe — skipped")
            continue
        df = md.history(inst)
        if df is None or len(df) < MIN_SLICE_BARS + max(horizons) + 10:
            print(f"{sym}: insufficient history "
                  f"({0 if df is None else len(df)} bars) — skipped")
            continue
        if max_bars and len(df) > max_bars:
            df = df.iloc[-int(max_bars):].reset_index(drop=True)
        idx = _slices(df, step, horizons)
        if not idx:
            print(f"{sym}: no gradeable slice in {len(df)} bars — skipped")
            continue
        close = df["close"].to_numpy(dtype=float)
        t0 = time.perf_counter()
        graded_before = counts["graded"]
        used = 0
        for i in idx:
            sl = df.iloc[:i + 1]
            cutoff = sl["date"].iloc[-1]
            outcomes = {}
            for h in horizons:
                j = i + h
                if j <= len(df) - 1 and close[i] > 0:
                    outcomes[h] = float(close[j] / close[i] - 1)
            if not outcomes:
                continue
            try:
                feats = compute_features(sl)
                regime = detect_regime(sl)
            except Exception as exc:
                print(f"{sym} @{i}: features failed "
                      f"({type(exc).__name__}: {str(exc)[:80]}) -- slice dropped")
                continue
            micro = str(regime.get("regime") or "Unknown")
            # The macro state is read from the same date-truncated panel the
            # strategies are handed, so a slice is classified by the world as it
            # stood on that bar — not by today's VIX wearing yesterday's label.
            panel = _truncated(intermarket, cutoff)
            macro = macro_state(panel)
            label = condition_label(micro, macro)
            tags[macro.tag] = tags.get(macro.tag, 0) + 1
            if macro.coverage < MIN_COVERAGE:
                thin_coverage += 1
            ctx = {"regime": micro,
                   "macro": macro.tag,
                   "macro_score": round(macro.score, 3),
                   "intermarket": panel,
                   "date_today": cutoff,
                   "patterns": detect_patterns(sl),
                   "tags": inst.tags}
            report: dict = {}
            votes = run_strategy_library(sl, feats, ctx, report=report)
            A.note_slice(counts, report.get("seen", [n for n, _c, _f in STRATEGIES]),
                         report.get("errors", {}))
            A.note_outcome(counts, label, outcomes, horizons)
            used += 1
            for v in votes:
                A.grade(counts, v.name, v.category, v.direction, v.strength,
                        label, outcomes)
        counts["slices"] += used
        elapsed = round(time.perf_counter() - t0, 1)
        counts["instruments"].append(
            {"symbol": inst.key, "bars": int(len(df)), "slices": used,
             "slices_dropped": len(idx) - used,
             "graded_votes": counts["graded"] - graded_before, "seconds": elapsed})
        if not quiet:
            print(f"{inst.key}: {used}/{len(idx)} slices graded, "
                  f"{counts['graded'] - graded_before} directional votes "
                  f"({elapsed}s)")
    counts["errors_total"] = sum(int(s.get("errors", 0))
                                 for s in counts["strategies"].values())
    counts["macro_slices"] = dict(sorted(tags.items()))
    counts["macro_thin_coverage"] = thin_coverage
    return counts


def _bad_buckets(counts: dict) -> list[str]:
    """Buckets that contradict themselves: hits above votes, or votes that went
    missing from the long/short split.

    The first published run did exactly that — a graded vote incremented the hit
    column and the long column at once, so the strongest strategy on the tape
    read 136% correct. That is not a rounding error, it is an arithmetic
    impossibility, and it reached the report because nothing between `grade` and
    the file asked whether the numbers could have come from real votes. An
    artifact that fails here is refused, not published with a warning.
    """
    bad: list[str] = []
    for name, st in counts["strategies"].items():
        rows = [(f"horizon {h}", b) for h, b in (st.get("by_horizon") or {}).items()]
        rows += [(f"bucket {k}", b) for k, b in (st.get("by_bucket") or {}).items()]
        for where, b in rows:
            if len(b) != 4:
                bad.append(f"{name} @ {where}: not a four-column bucket ({b})")
                continue
            votes, hits, longs, shorts = (int(x) for x in b)
            if hits > votes:
                bad.append(f"{name} @ {where}: {hits} hits on {votes} votes")
            if longs + shorts != votes:
                bad.append(f"{name} @ {where}: split {longs}/{shorts} != {votes} votes")
    return bad


def _report(counts: dict, out_path: Path) -> tuple[A.Attribution | None, str]:
    """Print what was measured, then the loader's verdict on what was written.

    A raw hit rate is the number that misled the first version of this report:
    in a bull sample every long-biased strategy reads 60-75% correct, so the
    leaderboard became a census of who was pointed up. Each row is therefore
    stated as votes minus the hits the strategy's own long/short mix would have
    collected against that horizon's up-frequency — the same subtraction the
    weights use, printed next to the base rate that makes it interpretable.
    """
    base = counts.get("base") or {}
    rows = []
    for name, st in counts["strategies"].items():
        for h, bucket in (st.get("by_horizon") or {}).items():
            n = int(bucket[0])
            if not n:
                continue
            p_up = A._p_up((base.get(str(h)) or {}).get("_all"))
            exp = A._expected(bucket, p_up) if p_up is not None else n * 0.5
            rows.append(((int(bucket[1]) - exp) / (n + 1.0), n, int(h), name,
                         st["category"], int(bucket[1]), p_up))
    # discount by evidence: a thin bucket's excess is mostly noise, so cap how
    # much a handful of votes can buy it on the leaderboard
    rows.sort(key=lambda r: -r[0] * min(r[1], 300))
    print(f"\nartifact: {out_path}")
    print(f"graded directional votes: {counts['graded']}   "
          f"slices: {counts['slices']}   "
          f"instruments: {len(counts['instruments'])}")
    print(f"strategies raising during measurement: "
          f"{sum(1 for s in counts['strategies'].values() if s.get('errors'))}"
          f"  (errors counted {counts['errors_total']})")
    print(f"strategies never voting: "
          f"{sum(1 for s in counts['strategies'].values() if not s['votes'])}"
          f" of {len(counts['strategies'])}")
    ms = counts.get("macro_slices") or {}
    print("slices by macro state: " +
          (", ".join(f"{k} {v}" for k, v in ms.items()) or "none") +
          f"  (panel too thin to classify: "
          f"{counts.get('macro_thin_coverage', 0)})")
    # Whether the macro axis is armed or decorative is a countable question, not
    # a claim: a bucket has to clear MIN_REGIME_CALLS before it may move a vote,
    # and splitting the sample by a new dimension always thins the buckets, so
    # this number is the price of the extra resolution and it gets printed.
    fired = sum(1 for st in counts["strategies"].values()
                for b in (st.get("by_bucket") or {}).values()
                if int(b[0]) >= A.MIN_REGIME_CALLS)
    # the key is "horizon|micro :: macro" — splitting on the LAST separator
    # would report three macro tags and call them conditions, which is the
    # wrong question: what matters is how many joint conditions survived
    seen_conds = {k.split("|", 1)[1] for st in counts["strategies"].values()
                  for k in (st.get("by_bucket") or {})}
    fired_conds = {k.split("|", 1)[1] for st in counts["strategies"].values()
                   for k, b in (st.get("by_bucket") or {}).items()
                   if int(b[0]) >= A.MIN_REGIME_CALLS}
    print(f"condition buckets with enough votes to move a weight: {fired} "
          f"across {len(fired_conds)} of the {len(seen_conds)} conditions seen")
    for horizon in counts["horizons"]:
        sel = [r for r in rows if r[2] == int(horizon)]
        pair = (base.get(str(horizon)) or {}).get("_all") or [0, 0]
        rate = f"{pair[1] / pair[0]:.1%}" if pair[0] else "unknown"
        print(f"\n{horizon}d — tape rose {rate} over {int(pair[0])} slices; "
              f"{sum(r[1] for r in sel)} directional votes "
              f"({sum(r[5] for r in sel)} hits, raw "
              f"{sum(r[5] for r in sel) / max(sum(r[1] for r in sel), 1):.1%})")
        if not sel:
            print("  nothing graded at this horizon")
            continue
        print("  ahead of the tape: " + ", ".join(
            f"{n} ({int(nn)} votes {hr / nn:.0%} vs {pr:.0%}, {exc:+.1%}/vote)"
            for exc, nn, _h, n, _c, hr, pr in sel[:5] if exc > 0))
        print("  behind the tape:  " + ", ".join(
            f"{n} ({int(nn)} votes {hr / nn:.0%} vs {pr:.0%}, {exc:+.1%}/vote)"
            for exc, nn, _h, n, _c, hr, pr in sel[-5:][::-1] if exc < 0))
    attr, reason = A.load(out_path, library_size=int(counts["library_size"]))
    if attr is None:
        print(f"\nTHE ENGINE WILL NOT USE THIS ARTIFACT: {reason}")
        print("It is written to disk, but `attribution.load` refuses it, so the "
              "fusion stays a flat poll. That is the fail-closed behaviour, not a "
              "success — fix the coverage or the counts before calling it done.")
    else:
        print(f"\nthe engine will apply it: {attr.counts['graded']} graded votes, "
              f"{len(attr.counts['strategies'])} strategies measured, "
              f"generated {attr.generated}")
    return attr, reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--step", type=int, default=5,
                    help="bars between evaluation slices (default 5)")
    ap.add_argument("--horizons", default="5,21",
                    help="comma-separated thesis lengths to grade (default 5,21)")
    ap.add_argument("--max-bars", type=int, default=1500,
                    help="history depth per instrument (default 1500 bars)")
    ap.add_argument("--out", default=str(A.ARTIFACT_PATH))
    args = ap.parse_args()

    horizons = tuple(int(x) for x in str(args.horizons).split(",") if x.strip())
    if not horizons:
        print("no horizons given")
        return 2
    md = MarketData()
    t0 = time.perf_counter()
    counts = measure(md, list(args.symbols), max(1, int(args.step)), horizons,
                     int(args.max_bars))
    counts["elapsed_s"] = round(time.perf_counter() - t0, 1)
    counts["ran_at"] = datetime.now().isoformat(timespec="seconds")

    bad = _bad_buckets(counts)
    if bad:
        print(f"refusing to publish: {len(bad)} self-contradictory bucket(s)")
        for line in bad[:12]:
            print("  " + line)
        print("Nothing was written, so the previous artifact — or a flat poll, if "
              "there is none — stays in force. A bucket whose hits outnumber its "
              "votes is not thin evidence, it is a broken counter.")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out_path, json.dumps(counts, indent=1, sort_keys=True))
    attr, reason = _report(counts, out_path)
    if attr is None:
        return 1
    try:
        from kairos.execution import JOURNAL
        JOURNAL.log("ATTRIBUTION_PUBLISHED", {
            "generated": counts["generated"], "graded_votes": counts["graded"],
            "slices": counts["slices"],
            "instruments": [i["symbol"] for i in counts["instruments"]],
            "horizons": list(counts["horizons"]),
            "path": str(out_path)})
    except Exception as exc:
        print(f"audit journal event ATTRIBUTION_PUBLISHED could not be appended: "
              f"{type(exc).__name__}: {str(exc)[:120]}")
    print(f"\nfull run took {counts['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
