"""Repair bar files the sweep wrote with an impossible range or a fake session.

Two defects in the PSX accumulators filled the store before 2026-10-24:

1. The snapshot carries only a close and a percentage change, and the bar was
   synthesised as high=close, low=previous close. On a down day that is a high
   below its own low, so the file held bars no session could produce. The read
   path widens them, which is why the models never saw the inversion — but the
   file, which is the record, kept it.
2. The sweep ran on the calendar day, not on a session. A weekend or a holiday
   snapshot re-serves the last session's print, so every Saturday and Sunday
   added a flat bar with Friday's close and Friday's volume.

This script rewrites the store to what the exchange printed: every PSX row on a
Saturday or Sunday is removed — the exchange is closed, so the row is the sweep
re-stamping the last session — and every surviving bar is widened until it
contains its own open and close. Crypto and FX keep their weekend rows, because
a bar there is a session on a day the venue trades.

It is idempotent and keeps a backup of every file it changes.

    python tools/repair_bar_store.py              # repair and report
    python tools/repair_bar_store.py --dry-run    # report only
    python tools/repair_bar_store.py --dir PATH   # another machine's cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos import universe as U  # noqa: E402


def _is_psx(key: str) -> bool:
    inst = U.resolve(key)
    return inst is not None and inst.market == "PSX"


def repair_frame(df: pd.DataFrame, psx: bool) -> tuple[pd.DataFrame, int, int]:
    """Drop PSX weekend rows and widen bars until they contain themselves.

    PSX does not trade on Saturday or Sunday, so a row there is the sweep, not
    the exchange — whichever prices it repeats. (An earlier version of this
    script also required the row to repeat the previous close and volume, which
    missed every weekend written after a bar that was last stored mid-session:
    the final volume differed, and the fabricated row survived.)
    """
    df = df.sort_values("date").reset_index(drop=True)
    weekend = pd.to_datetime(df["date"]).dt.dayofweek >= 5
    keep = ~weekend if psx else pd.Series(True, index=df.index)
    dropped = int((~keep).sum())
    out = df.loc[keep].reset_index(drop=True)
    before = out[["open", "high", "low", "close"]].astype(float)
    out["high"] = before[["open", "high", "close"]].max(axis=1)
    out["low"] = before[["open", "low", "close"]].min(axis=1)
    widened = int(((out["high"] != before["high"])
                   | (out["low"] != before["low"])).sum())
    return out, dropped, widened


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None, help="bar cache directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from kairos.config import CFG
    from kairos.data import Store

    store = Store(Path(args.dir) if args.dir else CFG.cache_dir)
    backup = store.dir / "repair-backup-20261024"
    files = sorted(store.dir.glob("*.csv"))
    print(f"cache: {store.dir} ({len(files)} file(s))")
    changed = total_dropped = total_widened = 0
    for path in files:
        try:
            raw = pd.read_csv(path, parse_dates=["date"])
        except (OSError, ValueError):
            continue
        if raw.empty or not {"open", "high", "low", "close"} <= set(raw.columns):
            continue
        fixed, dropped, widened = repair_frame(raw, _is_psx(path.stem))
        if not dropped and not widened:
            continue
        changed += 1
        total_dropped += dropped
        total_widened += widened
        print(f"  {path.stem:10} dropped {dropped:3d} fake session(s), "
              f"widened {widened:3d} bar(s): {len(raw)} -> {len(fixed)} rows")
        if args.dry_run:
            continue
        backup.mkdir(exist_ok=True)
        shutil.copy2(path, backup / path.name)
        store.save(path.stem, fixed)
    verb = "would change" if args.dry_run else "changed"
    print(f"{changed} file(s) {verb}: {total_dropped} fake row(s), "
          f"{total_widened} inverted bar(s)"
          + ("" if args.dry_run else f"; backups in {backup}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
