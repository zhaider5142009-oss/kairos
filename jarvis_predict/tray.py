"""
System-tray UI — pystray icon + menu.

Shows live status (equity, pending count, MT5 connection state) and
provides one-click confirm/cancel for the latest pending signal.

Runs on a worker thread; the main thread owns the prediction loop.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import webbrowser

from .config import JarvisConfig
from .listener.dispatcher import Dispatcher, EventType

LOGGER = logging.getLogger("jarvis.tray")

try:
    import pystray
    from PIL import Image, ImageDraw
    _HAS_TRAY = True
except ImportError:  # pragma: no cover
    _HAS_TRAY = False


class TrayUI:
    def __init__(
        self,
        *,
        dispatcher: Dispatcher,
        config: JarvisConfig,
        enabled: bool = True,
        refresh_seconds: float = 3.0,
    ) -> None:
        self.dispatcher = dispatcher
        self.config = config
        self.enabled = enabled
        self.refresh_seconds = max(1.0, float(refresh_seconds))

        self._icon: "pystray.Icon | None" = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_pending_count: int = 0
        self._last_equity: float = 0.0

    # --- public --------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("Tray disabled in config")
            return
        if not _HAS_TRAY:
            LOGGER.warning("pystray/Pillow not installed; tray disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-tray", daemon=True)
        self._thread.start()
        LOGGER.info("TrayUI started")

    def stop(self) -> None:
        self._stop.set()
        if self._icon:
            try:
                self._icon.stop()
            except Exception:  # pragma: no cover
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    # --- internals -----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._icon = pystray.Icon(
                name="Jarvis",
                icon=_build_icon(connected=False, pending=0),
                title="Jarvis Reborn — MT5 Algo Trader",
                menu=self._menu_factory(),
            )
            self._icon.run()
        except Exception:  # pragma: no cover
            LOGGER.exception("TrayUI crashed")

    def _menu_factory(self):
        """Build menu each time it's opened so status is fresh."""
        icon = self._icon

        def status_text() -> str:
            try:
                status = self.dispatcher.status()
            except Exception:  # pragma: no cover
                return "Status unavailable"
            return (
                f"Equity: {status['equity']:.2f}\n"
                f"Balance: {status['balance']:.2f}\n"
                f"Open positions: {status['open_positions']}\n"
                f"Pending signals: {len(status['pending_signals'])}\n"
                f"Uptime: {status['uptime_s']:.0f}s"
            )

        def confirm_latest(icon, item):
            pending = self.dispatcher.pending_snapshot()
            if not pending:
                LOGGER.info("Tray confirm: no pending signals")
                return
            latest = max(pending, key=lambda p: p["staged_at"])
            self.dispatcher.confirm(latest["signal_id"], source="tray")

        def cancel_latest(icon, item):
            pending = self.dispatcher.pending_snapshot()
            if not pending:
                return
            latest = max(pending, key=lambda p: p["staged_at"])
            self.dispatcher.cancel(latest["signal_id"], source="tray", reason="tray:cancel")

        def cancel_all(icon, item):
            self.dispatcher.cancel_all(source="tray", reason="tray:cancel-all")

        def refresh(icon, item):
            try:
                icon.update_menu()
            except Exception:  # pragma: no cover
                pass

        def open_dashboard(icon, item):
            try:
                webbrowser.open("http://127.0.0.1:8765")
            except Exception:  # pragma: no cover
                LOGGER.exception("tray open-ui failed")

        def shutdown(icon, item):
            self.dispatcher.publish(EventType.SHUTDOWN, {"source": "tray"}, source="tray")
            icon.stop()

        return pystray.Menu(
            pystray.MenuItem(lambda item: status_text(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Dashboard", open_dashboard, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Confirm latest", confirm_latest, visible=lambda item: bool(self.dispatcher.pending_snapshot())),
            pystray.MenuItem("Cancel latest", cancel_latest, visible=lambda item: bool(self.dispatcher.pending_snapshot())),
            pystray.MenuItem("Cancel ALL pending", cancel_all, visible=lambda item: bool(self.dispatcher.pending_snapshot())),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh", refresh),
            pystray.MenuItem("Shutdown Jarvis", shutdown),
        )


def _build_icon(*, connected: bool, pending: int) -> "Image.Image":
    """Generate a simple 64x64 icon programmatically (no asset files needed)."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg = (16, 24, 40) if connected else (90, 30, 30)
    d.rounded_rectangle((4, 4, 60, 60), radius=12, fill=bg)
    # Stylised 'J'
    d.rectangle((22, 14, 38, 18), fill=(220, 230, 255))
    d.rectangle((28, 14, 32, 42), fill=(220, 230, 255))
    d.rounded_rectangle((18, 40, 42, 50), radius=4, fill=(220, 230, 255))
    # Pending dot
    if pending > 0:
        d.ellipse((46, 6, 60, 20), fill=(255, 200, 40))
    return img


__all__ = ["TrayUI"]
