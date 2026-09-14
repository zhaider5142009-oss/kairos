#!/usr/bin/env python3
"""
Jarvis — entry point.

Run this directly to start the predictive trader:
    python jarvis_main.py

Use run_jarvis.bat / run_jarvis.vbs for hidden console launches.
Use install_task.ps1 to register a Windows Scheduled Task for autostart.

Single-instance: a named mutex prevents a second console from piling up —
if Jarvis is already running, the second launch just opens the Web UI and
exits immediately.
"""

from __future__ import annotations

import ctypes
import sys
import webbrowser
from pathlib import Path

_ROOT_MUTEX: int | None = None


def _already_running() -> bool:
    """Return True if another Jarvis instance holds the single-instance lock."""
    global _ROOT_MUTEX
    try:
        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, False, "Global\\JarvisPredictor_SingleInstance")
        _ROOT_MUTEX = int(mutex)  # keep the handle alive for the process lifetime
        return bool(kernel32.GetLastError())  # ERROR_ALREADY_EXISTS means held elsewhere
    except Exception:
        return False  # non-Windows / non-admin → never block startup


# Make the local package importable when launched from any CWD.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis_predict.config import JarvisConfig, setup_logging
from jarvis_predict.runtime import JarvisRuntime


def main() -> int:
    if _already_running():
        print("[Jarvis] Already running — opening http://127.0.0.1:8765 and exiting.")
        try:
            webbrowser.open("http://127.0.0.1:8765")
        except Exception:
            pass
        return 0

    config = JarvisConfig.load()
    setup_logging(config)
    runtime = JarvisRuntime(config)
    runtime.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
