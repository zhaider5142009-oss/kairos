"""Atomic overwrite for the files that *are* the track record.

The prediction journal and the paper book are rewritten whole, not appended to,
so an interrupted write (a crash, a full disk on this machine, a forced power
cut) used to leave a truncated file behind. Every loader below them treats an
unreadable file as "no state yet", which for the paper book means starting the
next cycle at full capital with an empty position list — a reset that looks
exactly like a healthy account.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from pathlib import Path

import numpy as _np


def _json_default(obj):
    """Convert the types this system actually produces; refuse everything else.

    Every journal, state file and report used `json.dumps(..., default=str)`,
    which converts *anything* — and a string is the one thing a reader cannot
    read back. `numpy.bool_(False)` became `"False"`, which is truthy, so a
    quiet market was saved as a loud one (that is the incident recorded in
    `features.py` and `macro.py`), and a `numpy.float64` equity became `"1000000"`:
    still parseable by the desk's coercing `_num`, silently wrong in any
    arithmetic that forgot to coerce first. A type with no conversion here is a
    bug in the caller, so it raises and the caller's own error surface reports
    it, instead of the value being quietly re-typed on the way to disk.
    """
    if isinstance(obj, _np.generic):        # numpy bool_/int64/float64 scalars
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, _dt.timedelta):      # `isoformat()` is not defined on this one
        return obj.total_seconds()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} has no JSON conversion — give the "
                    f"serializer plain types, not a str-cast that the reader "
                    f"cannot recover")


def dumps_strict(obj, indent: int | None = None) -> str:
    """JSON for the files that *are* the record. Raises rather than stringify."""
    return json.dumps(obj, default=_json_default, indent=indent)


def atomic_write(path: Path, text: str) -> None:
    # a per-writer temp name: the bar store now fills several symbols at once,
    # and two threads sharing "<name>.tmp" would truncate each other mid-write
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        # Windows refuses the rename while the destination is momentarily held
        # by another writer's replace (or by whatever scans a new file), and it
        # clears in milliseconds. Retrying is the difference between a slow
        # write and a bar file that silently never lands.
        for attempt in range(4):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)
