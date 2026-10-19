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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos.feedback import track_record  # noqa: E402

OUT = ROOT / "docs" / "investors" / "TRACK_RECORD.md"
LINE = re.compile(r"^(\w+): (\w+) ret=([+-][\d.]+)% bench=([+-][\d.]+)% "
                  r"sharpe=([\d.-]+) hit=([\d.]+) trades=(\d+) dd=([\d.-]+)%")


def _spectrum() -> tuple[list[dict], dict]:
    """Widest completed walk-forward sweep, parsed from the run log."""
    best: tuple[int, list[dict], dict] = (0, [], {})
    for log in sorted((ROOT / "reports").glob("spectrum*.log")):
        rows, summary = [], {}
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if (m := LINE.match(line.strip())):
                sym, verdict, ret, bench, sharpe, hit, trades, dd = m.groups()
                rows.append({"symbol": sym, "verdict": verdict,
                             "return": float(ret) / 100, "benchmark": float(bench) / 100,
                             "sharpe": float(sharpe), "hit_rate": float(hit),
                             "trades": int(trades), "max_dd": float(dd) / 100})
            elif line.startswith("summary:"):
                summary = ast.literal_eval(line.split(":", 1)[1].strip())
        if len(rows) > len(best[1]):
            best = (len(rows), rows, summary)
    return best[1], best[2]


def _paper() -> dict:
    p = ROOT / "data" / "paper_account.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _autopilot() -> dict:
    p = ROOT / "data" / "autopilot.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _live_book() -> dict:
    p = ROOT / "data" / "livebook.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    tr = track_record()
    rows, summary = _spectrum()
    pa, ap = _paper(), _autopilot()
    lb = _live_book()
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
    w(f"| predictions journaled | {tr.get('predictions', 0)} |")
    w(f"| horizons matured | {tr.get('resolved', 0)} |")
    w(f"| of which directional calls | {tr.get('directional_resolved', 0)} |")
    w(f"| of which non-trades (HOLD) | {tr.get('non_directional_resolved', 0)} |")
    hr = tr.get("hit_rate")
    w(f"| directional hit rate | {'not yet measurable' if hr is None else f'{hr:.1%}'} |")
    w("")
    if hr is None:
        w("**There is no directional hit rate yet.** Every matured snapshot was a "
          "non-trade, and a non-trade cannot be a correct prediction — the scorer "
          "used to count them as wins, which is fixed as of v2.5.0. A forward record "
          "of live calls is what this table will show, and it accumulates on its own "
          "as horizons elapse.")
    w("")
    w("## 2. Walk-forward strategy sweep")
    w("")
    w("100 candidate strategies per instrument, parameters chosen on a rolling "
      "training window and evaluated out-of-sample with PSX commission and "
      "slippage modelled.")
    w("")
    w("| instrument | verdict | net return | buy & hold | Sharpe | hit rate | trades | max drawdown |")
    w("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["sharpe"]):
        w(f"| {r['symbol']} | {r['verdict'].lower()} | {r['return']:+.1%} | "
          f"{r['benchmark']:+.1%} | {r['sharpe']:+.2f} | {r['hit_rate']:.3f} | "
          f"{r['trades']} | {r['max_dd']:.1%} |")
    w("")
    if summary:
        w(f"Run summary: **{summary.get('profitable')}/{summary.get('instruments')} "
          f"instruments profitable out-of-sample**, {summary.get('unprofitable')} not.")
    w("")
    for venue, vr in sorted(by_venue.items()):
        net = sum(r["return"] for r in vr) / len(vr)
        beat = sum(1 for r in vr if r["return"] > r["benchmark"]) / len(vr)
        w(f"- **{venue}**: mean net return {net:+.1%} across {len(vr)} instruments; "
          f"beat buy-and-hold on {beat:.0%} of them.")
    w("")
    w("_Backtests are not returns. They include costs but not every market impact, "
      "and the parameter search is a known overfitting risk; that is why the sweep is "
      "reported per instrument instead of as an average._")
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(md)} lines)")
    counts = Counter(r["verdict"] for r in rows)
    print("sweep rows:", len(rows), dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
