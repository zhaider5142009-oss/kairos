"""One-time merge of bar files the registry used to split by spelling.

Until 2026-10-24 the registry derived every venue symbol from the KAIROS key,
while the whole-board sweep filed bars under the *board's* spelling. For names
where the two differ (ENGROH/ENGRO, BOP/BANK, NESTLE/NESP) that produced two
files: the keyed one fed by Yahoo, and the board one fed by the sweep. Bank of
Punjab's keyed file stopped at 2026-10-22 while the board file carried 10-24,
and Engro's history existed only under ENGROH.

The registry now files both streams under the key. This script folds the
orphaned board files into the keyed ones, board rows winning a shared date (the
exchange print is the fresher source) and removes the emptied file. It is
idempotent: a second run finds nothing to merge.

    python tools/migrate_psx_store.py            # merge and report
    python tools/migrate_psx_store.py --dry-run  # report only
    python tools/migrate_psx_store.py --dir PATH # another machine's cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos import universe as U  # noqa: E402


def _pairs() -> list[tuple[str, str]]:
    return [(i.key, i.psx_ticker) for i in U.psx_universe()
            if i.psx_ticker and i.psx_ticker.upper() != i.key.upper()]


def merge(store, key: str, alias: str, dry_run: bool) -> dict:
    keyed = store.load(key)
    board = store.load(alias)
    if board is None:
        return {"key": key, "alias": alias, "skipped": "no board file"}
    frames = [f for f in (keyed, board) if f is not None]
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = (merged.drop_duplicates(subset="date", keep="last")
              .sort_values("date").reset_index(drop=True))
    report = {
        "key": key, "alias": alias,
        "keyed_rows": 0 if keyed is None else len(keyed),
        "board_rows": len(board),
        "merged_rows": len(merged),
        "first": str(merged["date"].iloc[0].date()),
        "last": str(merged["date"].iloc[-1].date()),
    }
    if dry_run:
        return report
    store.save(key, merged)
    back = store.load(key)
    # refuse to delete the board file unless every one of its dates survived
    if back is None or set(board["date"].dt.normalize()) - set(back["date"].dt.normalize()):
        raise SystemExit(f"{key}: merge did not keep every {alias} bar — "
                         f"{alias} left in place")
    store._path(alias).unlink()
    report["removed"] = store._path(alias).name
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None, help="bar cache directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from kairos.config import CFG
    from kairos.data import Store

    store = Store(Path(args.dir) if args.dir else CFG.cache_dir)
    pairs = _pairs()
    if not pairs:
        print("registry has no key/ticker aliases — nothing to do")
        return 0
    print(f"cache: {store.dir}")
    print(f"aliases: {', '.join(f'{k}->{t}' for k, t in pairs)}")
    changed = 0
    for key, alias in pairs:
        r = merge(store, key, alias, args.dry_run)
        if "skipped" in r:
            print(f"  {key:6} {r['skipped']} ({alias}.csv)")
            continue
        changed += 1
        verb = "would merge" if args.dry_run else "merged"
        print(f"  {key:6} {verb} {alias}.csv: keyed {r['keyed_rows']} + board "
              f"{r['board_rows']} -> {r['merged_rows']} rows "
              f"({r['first']}..{r['last']})"
              + (f"; removed {r['removed']}" if r.get("removed") else ""))
    print(f"{changed} alias file(s) {'to merge' if args.dry_run else 'merged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
