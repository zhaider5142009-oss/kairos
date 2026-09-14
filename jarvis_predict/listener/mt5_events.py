"""
MT5 event listener — watches equity / margin / drawdown and publishes alerts.

Runs on a worker thread, polls every N seconds. Does NOT execute anything
itself; only emits events to the dispatcher. The runtime decides what to do
with alerts (e.g. cancel all pending, push notification).
"""

from __future__ import annotations

import logging
import threading
import time

from ..mt5_bridge import MT5Bridge, JarvisMT5Error
from .dispatcher import Dispatcher, EventType

LOGGER = logging.getLogger("jarvis.mt5_events")


class MT5EventListener:
    def __init__(
        self,
        *,
        dispatcher: Dispatcher,
        bridge: MT5Bridge,
        enabled: bool = True,
        tick_every: int = 30,
        equity_drawdown_alert: float = 0.05,
        alert_window_seconds: int = 300,
        poll_seconds: float = 5.0,
    ) -> None:
        self.dispatcher = dispatcher
        self.bridge = bridge
        self.enabled = enabled
        self.tick_every = max(1, int(tick_every))
        self.equity_drawdown_alert = float(equity_drawdown_alert)
        self.alert_window_seconds = max(30, int(alert_window_seconds))
        self.poll_seconds = max(1.0, float(poll_seconds))

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_alert_at: float = 0.0
        self._high_water_equity: float = 0.0

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("MT5 event listener disabled in config")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-mt5-events", daemon=True)
        self._thread.start()
        LOGGER.info("MT5EventListener started (poll=%.1fs, dd_alert=%.2f%%)",
                    self.poll_seconds, self.equity_drawdown_alert * 100)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Initial high-water mark once connected.
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                LOGGER.exception("MT5 event loop error")
            self._stop.wait(timeout=self.poll_seconds)

    def _poll_once(self) -> None:
        if not self.bridge.ensure_connected():
            return
        equity = self.bridge.equity()
        if equity <= 0:
            return

        # Track high-water mark.
        if equity > self._high_water_equity:
            self._high_water_equity = equity

        # Drawdown alert.
        if self._high_water_equity > 0:
            dd = (self._high_water_equity - equity) / self._high_water_equity
            now = time.time()
            if (
                dd >= self.equity_drawdown_alert
                and (now - self._last_alert_at) >= self.alert_window_seconds
            ):
                self._last_alert_at = now
                LOGGER.warning(
                    "EQUITY DRAWDOWN %.2f%% (high=%.2f now=%.2f)",
                    dd * 100, self._high_water_equity, equity,
                )
                self.dispatcher.publish(
                    EventType.EQUITY_ALERT,
                    {
                        "drawdown_pct": dd * 100,
                        "high_water": self._high_water_equity,
                        "equity": equity,
                    },
                    source="mt5_events",
                )

        # Tick fan-out — let the runtime re-poll forecasts on each tick.
        for sym in self.bridge.symbols:
            try:
                t = self.bridge.tick(sym)
                self.dispatcher.publish(
                    EventType.TICK,
                    {
                        "symbol": sym,
                        "bid": t.bid, "ask": t.ask, "last": t.last,
                        "time_msc": t.time_msc,
                    },
                    source="mt5_events",
                )
            except JarvisMT5Error:
                # Tick errors are non-fatal; just skip this symbol this cycle.
                pass


__all__ = ["MT5EventListener"]
