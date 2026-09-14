"""
Central dispatcher — fan-out pub/sub for Jarvis events.

Listeners register callbacks for event types. The runtime publishes events.
Listeners can also call dispatcher.confirm(signal_id, source=...) which is
forwarded to the executor.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

LOGGER = logging.getLogger("jarvis.dispatcher")


class EventType(str, Enum):
    STAGE = "stage"               # A new staged order is waiting for confirmation
    CONFIRMED = "confirmed"       # User confirmed a staged order
    EXECUTED = "executed"         # MT5 acknowledged the order
    CANCELLED = "cancelled"       # Staged order was cancelled
    FAILED = "failed"             # Order send failed
    STATUS = "status"             # Status snapshot requested
    TICK = "tick"                 # New tick for a symbol
    EQUITY_ALERT = "equity_alert" # Equity drawdown alert
    RISK = "risk"                 # Kill-switch / risk status changed
    VOICE_COMMAND = "voice"       # Voice intent parsed
    HTTP_COMMAND = "http"         # HTTP request received
    NEWS = "news"                 # New high-relevance news article
    FORCES = "forces"             # Macro regime change
    CALENDAR = "calendar"         # Blackout window entered/exited
    PATTERN = "pattern"           # New pattern detected
    QUANTUM = "quantum"           # Quantum layer decision update
    AGENTS = "agents"             # Agent swarm produced a decision
    REPORT = "report"             # One agent finished (live telemetry stream)
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    source: str = "runtime"


class Dispatcher:
    """Thread-safe fan-out for Jarvis events."""

    def __init__(self) -> None:
        self._subs: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        # Injected references (set by runtime during startup).
        self.executor = None
        self.bridge = None
        self.runtime = None

    # --- subscriptions -----------------------------------------------------

    def subscribe(self, event_type: EventType, cb: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._subs[event_type].append(cb)

        def _unsubscribe() -> None:
            with self._lock:
                if cb in self._subs[event_type]:
                    self._subs[event_type].remove(cb)

        return _unsubscribe

    def publish(self, event_type: EventType, payload: dict[str, Any], source: str = "runtime") -> None:
        with self._lock:
            cbs = list(self._subs.get(event_type, ()))
        if not cbs:
            return
        event = Event(type=event_type, payload=payload, source=source)
        for cb in cbs:
            try:
                cb(event)
            except Exception:  # pragma: no cover
                LOGGER.exception("Subscriber for %s raised", event_type.value)

    # --- intents (called by listeners) -------------------------------------

    def confirm(self, signal_id: str, source: str) -> Any:
        if self.executor is None:
            LOGGER.warning("confirm(%s) — executor not bound", signal_id)
            return None
        result = self.executor.confirm(signal_id, source=source)
        if result is not None:
            if result.success:
                self.publish(EventType.EXECUTED, {"signal_id": signal_id, "result": _sr(result)}, source)
            else:
                self.publish(EventType.FAILED, {"signal_id": signal_id, "result": _sr(result)}, source)
        else:
            self.publish(EventType.CANCELLED, {"signal_id": signal_id, "reason": "confirm returned None"}, source)
        return result

    def cancel(self, signal_id: str, source: str, reason: str = "") -> None:
        if self.executor is None:
            return
        self.executor.cancel(signal_id, reason=reason)
        self.publish(EventType.CANCELLED, {"signal_id": signal_id, "reason": reason or source}, source)

    def cancel_all(self, source: str, reason: str = "") -> int:
        if self.executor is None:
            return 0
        n = self.executor.cancel_all(reason=reason)
        self.publish(EventType.CANCELLED, {"bulk": True, "count": n, "reason": reason or source}, source)
        return n

    def confirm_all(self, source: str, reason: str = "") -> int:
        """Confirm every pending staged order. Returns count of successes."""
        if self.executor is None:
            return 0
        n = 0
        for o in list(self.executor.pending()):
            r = self.confirm(o.signal_id, source)
            if r is not None and getattr(r, "success", False):
                n += 1
        return n

    def pending_snapshot(self) -> list[dict]:
        if self.executor is None:
            return []
        out = []
        for o in self.executor.pending():
            out.append({
                "signal_id": o.signal_id,
                "symbol": o.signal.symbol,
                "direction": o.signal.direction,
                "confidence": o.signal.confidence,
                "current_price": o.signal.current_price,
                "entry_price": o.signal.entry_price,
                "stop_loss": o.signal.stop_loss,
                "take_profit": o.signal.take_profit,
                "expires_in_s": max(0.0, o.expires_at - __import__("time").time()),
                "staged_at": o.staged_at,
            })
        return out

    def status(self) -> dict:
        """Aggregate status. Safe even when MT5 is offline."""
        equity = 0.0
        balance = 0.0
        open_count = 0
        if self.bridge:
            try:
                equity = self.bridge.equity()
            except Exception:
                equity = 0.0
            try:
                balance = self.bridge.balance()
            except Exception:
                balance = 0.0
            try:
                open_count = len(self.bridge.open_positions())
            except Exception:
                open_count = 0
        pending = self.pending_snapshot()
        return {
            "equity": equity,
            "balance": balance,
            "open_positions": open_count,
            "pending_signals": pending,
            "uptime_s": _uptime(),
        }


def _sr(result) -> dict:
    return {
        "success": result.success,
        "order_id": result.order_id,
        "retcode": result.retcode,
        "comment": result.comment,
    }


_BOOT_TS = __import__("time").time()


def _uptime() -> float:
    return __import__("time").time() - _BOOT_TS


__all__ = ["Dispatcher", "Event", "EventType"]
