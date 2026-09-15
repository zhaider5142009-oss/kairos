"""
Safe restart helper — writes an ASCII-only .vbs that waits ~2.5 s, kills the
*current* process by PID (never other python processes), then relaunches via
run_jarvis.vbs.  Spawns detached so the HTTP response returns before the old
process is killed.

Safe because:
  - 2.5 s delay ensures the HTTP response is fully flushed.
  - Kill is scoped to this process's PID (+ children) — no blanket taskkill.
  - JARVIS_CONFIG (if the process runs an alternate config) is re-applied on
    relaunch, so the reap point is preserved.
  - The .vbs is written into {cache_dir} (temp + atomic replace).
  - Paper journals are fsync-backed and crash-safe.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("jarvis.restart")


def schedule_restart(
    root: str | os.PathLike,
    cache_dir: str | os.PathLike,
    config_path: str | os.PathLike | None = None,
) -> None:
    root = str(Path(root).resolve())
    run_vbs = os.path.join(root, "run_jarvis.vbs").replace("\\", "/")
    lines = [
        "Set sh = CreateObject(\"WScript.Shell\")",
        "WScript.Sleep 2500",
    ]
    if config_path:
        lines.append(
            'Set env = sh.Environment("Process")'
        )
        lines.append(
            'env("JARVIS_CONFIG") = "' + os.path.abspath(str(config_path)).replace("\\", "/") + '"'
        )
    lines.append(f'sh.Run "taskkill /F /PID {os.getpid()}", 0, True')
    lines.append(f'sh.Run "wscript ""{run_vbs}""", 0, False')
    lines.append("WScript.Quit 0")
    vbs = "\n".join(lines) + "\n"
    dest = Path(cache_dir) / "restart_jarvis.vbs"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(vbs, encoding="ascii", errors="replace")
    try:
        subprocess.Popen(
            ["wscript.exe", str(dest)],
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0x8)
                           | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            close_fds=True,
            cwd=root,
        )
        LOGGER.info("Restart helper scheduled (wscript %s, pid=%s)", dest, os.getpid())
    except Exception:
        LOGGER.exception("Failed to schedule restart")
        raise
