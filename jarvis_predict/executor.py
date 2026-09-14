"""
Trade executor — auto-execute-on-confirm flow.

Flow:
    1. Signal arrives from runtime (signal_queue).
    2. Executor checks risk gates (max open positions, daily loss).
    3. If allowed, executor "stages" a pending order (no MT5 call yet).
    4. Stage is announced via the dispatcher so all listeners see it.
    5. Any listener can confirm within `confirm_timeout` seconds.
    6. On confirm → MT5 market order is sent.
    7. On timeout → stage is cancelled and logged.

Listeners communicate back through the same dispatcher:
    - Voice listener: parses "yes" / "confirm" / "no" intents
    - HTTP listener: POST /confirm with the signal_id
    - Tray listener: menu click
    - MT5 event listener: emergency override (e.g. equity drawdown → auto-cancel)
"""

from __future__ import annotations

import logging
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Callable

from .config import JarvisConfig
from .mt5_bridge import MT5Bridge, SendResult, JarvisMT5Error
from .signals import TradeSignal

LOGGER = logging.getLogger("jarvis.executor")


@dataclass
class StagedOrder:
    signal: TradeSignal
    staged_at: float
    expires_at: float
    confirmed: bool = False
    cancelled: bool = False
    result: SendResult | None = None
    event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_pending(self) -> bool:
        return not self.confirmed and not self.cancelled and time.time() < self.expires_at

    @property
    def signal_id(self) -> str:
        return self.signal.signal_id


class TradeExecutor:
    """Thread-safe executor with a stage-then-confirm-then-send flow."""

    def __init__(
        self,
        *,
        bridge: MT5Bridge,
        confirm_timeout: float = 60.0,
        max_open_positions: int = 3,
        max_daily_loss: float = 50.0,
        max_per_symbol: int = 1,
        on_stage: Callable[[StagedOrder], None] | None = None,
        on_result: Callable[[StagedOrder, SendResult], None] | None = None,
    ) -> None:
        self.bridge = bridge
        self.confirm_timeout = float(confirm_timeout)
        self.max_open_positions = int(max_open_positions)
        self.max_daily_loss = float(max_daily_loss)
        self.max_per_symbol = max(1, int(max_per_symbol))
        self.on_stage = on_stage
        self.on_result = on_result

        self._staged: dict[str, StagedOrder] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._risk_pause = False

    def set_risk_pause(self, paused: bool) -> None:
        """Kill-switch: when active, no new signals are staged."""
        with self._lock:
            self._risk_pause = bool(paused)
        LOGGER.info("Risk pause set to %s", self._risk_pause)

    def risk_paused(self) -> bool:
        return self._risk_pause

    @classmethod
    def from_config(
        cls,
        config: JarvisConfig,
        bridge: MT5Bridge,
        on_stage: Callable | None = None,
        on_result: Callable | None = None,
    ) -> "TradeExecutor":
        s = config.section("mt5", "order")
        return cls(
            bridge=bridge,
            confirm_timeout=config.section("mt5").get("confirm_timeout", 60.0),
            max_open_positions=s["max_open_positions"],
            max_daily_loss=s["max_daily_loss"],
            max_per_symbol=int(s.get("max_per_symbol", 1)),
            on_stage=on_stage,
            on_result=on_result,
        )

    # ------------------------------------------------------------------ public

    def stage(self, signal: TradeSignal) -> StagedOrder | None:
        """Stage a signal for confirmation.

        Returns the staged order, or None if blocked by risk gates
        (max open positions, daily loss limit).
        """
        if signal.direction == 0:
            LOGGER.debug("Skipping HOLD signal %s", signal.signal_id)
            return None

        with self._lock:
            if self._risk_pause:
                LOGGER.warning("Kill switch active — staging blocked for %s", signal.signal_id)
                return None
            if self._risk_blocked():
                LOGGER.warning("Risk gate blocked signal %s", signal.signal_id)
                return None

            now = time.time()
            order = StagedOrder(
                signal=signal,
                staged_at=now,
                expires_at=now + self.confirm_timeout,
            )
            self._staged[signal.signal_id] = order

        LOGGER.info(
            "STAGED %s %s conf=%.2f entry=%.5f sl=%.5f tp=%.5f (expires in %.0fs)",
            signal.symbol, "BUY" if signal.direction == 1 else "SELL",
            signal.confidence, signal.entry_price, signal.stop_loss, signal.take_profit,
            self.confirm_timeout,
        )
        if self.on_stage:
            try:
                self.on_stage(order)
            except Exception:  # pragma: no cover
                LOGGER.exception("on_stage callback failed")
        return order

    def confirm(self, signal_id: str, source: str = "unknown") -> SendResult | None:
        """Mark a staged order as confirmed and send it to MT5."""
        with self._lock:
            order = self._staged.get(signal_id)
            if order is None:
                LOGGER.warning("confirm(%s) — no such staged order", signal_id)
                return None
            if order.cancelled:
                LOGGER.info("confirm(%s) — order already cancelled", signal_id)
                return None
            if order.confirmed:
                LOGGER.info("confirm(%s) — already confirmed (idempotent)", signal_id)
                return order.result
            if time.time() > order.expires_at:
                order.cancelled = True
                LOGGER.info("confirm(%s) — order expired", signal_id)
                return None
            order.confirmed = True
            order.event.set()

        sig = order.signal
        try:
            result = self.bridge.send_market_order(
                symbol=sig.symbol,
                direction=sig.direction,
                sl=sig.stop_loss,
                tp=sig.take_profit,
                comment=f"jarvis:{sig.signal_id[:8]}",
            )
        except JarvisMT5Error as e:
            LOGGER.exception("MT5 error sending order %s", sig.signal_id)
            result = SendResult(False, None, None, str(e))

        with self._lock:
            order.result = result

        if result.success:
            LOGGER.info("EXECUTED %s %s id=%s retcode=%s",
                        sig.symbol, "BUY" if sig.direction == 1 else "SELL",
                        result.order_id, result.retcode)
        else:
            LOGGER.error("FAILED %s: retcode=%s comment=%s",
                         sig.symbol, result.retcode, result.comment)
        if self.on_result:
            try:
                self.on_result(order, result)
            except Exception:  # pragma: no cover
                LOGGER.exception("on_result callback failed")
        return result

    def cancel(self, signal_id: str, reason: str = "") -> None:
        with self._lock:
            order = self._staged.get(signal_id)
            if order is None or order.cancelled or order.confirmed:
                return
            order.cancelled = True
            order.event.set()
        LOGGER.info("CANCELLED %s — %s", signal_id, reason or "manual")

    def cancel_all(self, reason: str = "") -> int:
        with self._lock:
            ids = [o.signal_id for o in self._staged.values() if not o.confirmed and not o.cancelled]
        for sid in ids:
            self.cancel(sid, reason)
        return len(ids)

    def pending(self) -> list[StagedOrder]:
        with self._lock:
            return [o for o in self._staged.values() if o.is_pending]

    def expire_old(self) -> int:
        """Cancel any staged orders past their deadline. Idempotent."""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, o in self._staged.items()
                if not o.confirmed and not o.cancelled and now > o.expires_at
            ]
        for sid in expired:
            self.cancel(sid, reason="expired")
        return len(expired)

    # ----------------------------------------------------------------- helpers

    def _risk_blocked(self) -> bool:
        try:
            open_positions = self.bridge.open_positions()
        except JarvisMT5Error:
            open_positions = []
        if len(open_positions) >= self.max_open_positions:
            LOGGER.warning("Max open positions reached (%d/%d)",
                           len(open_positions), self.max_open_positions)
            return True
        if self.max_per_symbol > 0:
            counts: dict[str, int] = {}
            for p in open_positions:
                sym = getattr(p, "symbol", "?")
                counts[sym] = counts.get(sym, 0) + 1
                if counts[sym] > self.max_per_symbol:
                    LOGGER.warning("Max positions per symbol reached for %s (%d)",
                                   sym, self.max_per_symbol)
                    return True
        try:
            pnl = self.bridge.daily_pnl()
        except JarvisMT5Error:
            pnl = 0.0
        if pnl <= -abs(self.max_daily_loss):
            LOGGER.warning("Daily loss limit reached (%.2f)", pnl)
            return True
        return False


__all__ = ["TradeExecutor", "StagedOrder"]
