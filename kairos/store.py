"""Atomic overwrite for the files that *are* the track record.

The prediction journal and the paper book are rewritten whole, not appended to,
so an interrupted write (a crash, a full disk on this machine, a forced power
cut) used to leave a truncated file behind. Every loader below them treats an
unreadable file as "no state yet", which for the paper book means starting the
next cycle at full capital with an empty position list — a reset that looks
exactly like a healthy account.
"""

from __future__ import annotations

from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
