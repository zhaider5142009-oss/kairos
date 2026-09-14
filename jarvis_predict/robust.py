"""
Robustness utilities — retry with backoff, circuit breaker, graceful
degradation.

Used by every external call (HTTP, MT5, OpenRouter, news feeds) so a
single broken endpoint can't cascade into a Jarvis-wide failure.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

LOGGER = logging.getLogger("jarvis.robust")


# ----- retry with exponential backoff ----------------------------------------

def retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 4,
    backoff_base: float = 0.4,
    backoff_max: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    label: str = "op",
) -> Any:
    """Synchronous retry with exponential backoff + jitter."""
    last_err: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except exceptions as e:
            last_err = e
            if attempt == max_attempts - 1:
                break
            wait = min(backoff_max, backoff_base * (2 ** attempt))
            LOGGER.debug("%s attempt %d failed: %s — retry in %.2fs",
                         label, attempt + 1, e, wait)
            time.sleep(wait)
    raise last_err if last_err else RuntimeError(f"{label} failed")


async def aretry(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 4,
    backoff_base: float = 0.4,
    backoff_max: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    label: str = "op",
) -> Any:
    """Async retry with exponential backoff + jitter."""
    last_err: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except exceptions as e:
            last_err = e
            if attempt == max_attempts - 1:
                break
            wait = min(backoff_max, backoff_base * (2 ** attempt))
            LOGGER.debug("%s attempt %d failed: %s — retry in %.2fs",
                         label, attempt + 1, e, wait)
            await asyncio.sleep(wait)
    raise last_err if last_err else RuntimeError(f"{label} failed")


# ----- circuit breaker -------------------------------------------------------

@dataclass
class CircuitBreaker:
    """Trips after `failure_threshold` consecutive failures and stays open
    for `cooldown_seconds`. Half-open after that — one trial allowed.

    Use to protect calls against a downstream that's known to be flaky
    (e.g. RSS feeds during a partial outage). When open, the protected
    function raises CircuitOpenError immediately instead of hammering
    the downstream.
    """

    name: str
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    state: str = "closed"            # closed | open | half_open
    failures: int = 0
    opened_at: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def call(self, fn: Callable[[], Any], *args, **kwargs) -> Any:
        with self._lock:
            self._maybe_close()
            if self.state == "open":
                raise CircuitOpenError(f"{self.name}: circuit open, retry in "
                                       f"{self.cooldown_seconds - (time.time() - self.opened_at):.1f}s")
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            self._on_failure(e)
            raise
        self._on_success()
        return out

    def _maybe_close(self) -> None:
        if self.state == "open" and time.time() - self.opened_at >= self.cooldown_seconds:
            self.state = "half_open"

    def _on_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def _on_failure(self, err: Exception) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.time()
                LOGGER.warning("circuit %s opened after %d failures (last: %s)",
                               self.name, self.failures, err)


class CircuitOpenError(RuntimeError):
    pass


# ----- feature-flag graceful degradation ------------------------------------

class FeatureFlag:
    """Boolean gate that disables a feature after N consecutive failures
    and re-enables after `cooldown_seconds`."""

    def __init__(self, name: str, *, cooldown_seconds: float = 300.0,
                 failures_to_disable: int = 10) -> None:
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self.failures_to_disable = failures_to_disable
        self.enabled = True
        self.failures = 0
        self._disabled_at: float = 0.0
        self._lock = threading.RLock()

    def allow(self) -> bool:
        with self._lock:
            if not self.enabled:
                if time.time() - self._disabled_at >= self.cooldown_seconds:
                    LOGGER.info("feature %s re-enabled after cooldown", self.name)
                    self.enabled = True
                    self.failures = 0
            return self.enabled

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failures_to_disable and self.enabled:
                self.enabled = False
                self._disabled_at = time.time()
                LOGGER.warning("feature %s disabled after %d failures; will retry in %.0fs",
                               self.name, self.failures, self.cooldown_seconds)


# ----- pre-built breakers & flags -------------------------------------------

# Circuit breakers per external service.
news_breaker    = CircuitBreaker("news_rss",        failure_threshold=3, cooldown_seconds=120.0)
forces_breaker  = CircuitBreaker("market_forces",   failure_threshold=5, cooldown_seconds=180.0)
calendar_breaker = CircuitBreaker("calendar",      failure_threshold=3, cooldown_seconds=120.0)
mt5_breaker     = CircuitBreaker("mt5",             failure_threshold=3, cooldown_seconds=30.0)
openrouter_breaker = CircuitBreaker("openrouter",   failure_threshold=3, cooldown_seconds=60.0)


__all__ = [
    "retry", "aretry",
    "CircuitBreaker", "CircuitOpenError",
    "FeatureFlag",
    "news_breaker", "forces_breaker", "calendar_breaker",
    "mt5_breaker", "openrouter_breaker",
]
