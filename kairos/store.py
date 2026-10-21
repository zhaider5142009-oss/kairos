"""Atomic overwrite for the files that *are* the track record.

The prediction journal and the paper book are rewritten whole, not appended to,
so an interrupted write (a crash, a full disk on this machine, a forced power
cut) used to leave a truncated file behind. Every loader below them treats an
unreadable file as "no state yet", which for the paper book means starting the
next cycle at full capital with an empty position list — a reset that looks
exactly like a healthy account.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


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
