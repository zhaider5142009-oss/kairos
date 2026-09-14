"""
Global keyboard hotkeys — system-wide push-to-talk without window focus.

Uses the ``keyboard`` library (Windows). When the combo fires (default
``ctrl+alt+j``) the callback runs on a single-worker thread so the low-level
hook thread never blocks on Whisper / record / LLM work.

Graceful degradation: if ``keyboard`` is not installed, or the process lacks
the privileges (admin) global hooks need on Windows, the listener logs a
warning and disables itself — the UI voice button still works.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

LOGGER = logging.getLogger("jarvis.hotkey")


class GlobalHotkeys:
    """Globally-registered hotkey → worker-thread callback. Best-effort."""

    def __init__(self, on_push_to_talk: Callable[[], None], combo: str = "ctrl+alt+j") -> None:
        self.combo = str(combo).strip().lower() or "ctrl+alt+j"
        self.on_push_to_talk = on_push_to_talk
        self.enabled = True
        self._hook = None
        self._pool = None
        self._stop = threading.Event()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._hook is not None:
            return
        try:
            import keyboard  # type: ignore
        except ImportError:
            LOGGER.warning(
                "Global hotkeys disabled: 'keyboard' package not installed. "
                "pip install keyboard  (voice button in the UI still works)")
            self.enabled = False
            return
        try:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-hotkey")
            self._hook = keyboard.add_hotkey(self.combo, self._on_fire, suppress=False)
        except Exception as e:
            LOGGER.warning(
                "Global hotkeys disabled: %s (run Jarvis as admin, or "
                "pip install keyboard). UI voice button still works.", e)
            self.enabled = False
            self._hook = None
            return
        LOGGER.info("Global hotkey %r armed (push-to-talk)", self.combo)

    def stop(self) -> None:
        self._stop.set()
        if self._hook is not None:
            try:
                import keyboard  # type: ignore
                keyboard.remove_hotkey(self._hook)
            except Exception:
                pass
            self._hook = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # --- internals -----------------------------------------------------------

    def _on_fire(self) -> None:
        """Runs on the keyboard hook thread — must return instantly."""
        if self._pool is None:  # pragma: no cover
            return
        self._pool.submit(self._worker)

    def _worker(self) -> None:
        if self._stop.is_set():
            return
        try:
            self.on_push_to_talk()
        except Exception:
            LOGGER.exception("push-to-talk handler failed")


__all__ = ["GlobalHotkeys"]