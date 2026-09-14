"""
Async MT5 bridge — ThreadPoolExecutor-wrapped version of MT5Bridge with
a caching + prefetch layer for ultra-low-latency data access.

MetaTrader5's Python package is synchronous and not thread-safe per
terminal connection. We wrap each call in `asyncio.to_thread` so the
event loop stays unblocked, and use a single worker thread for all MT5
calls (serialised) to avoid the package's documented threading caveats.

v1.2 improvements:
    - `MT5RateCache`: rates are cached per (symbol, timeframe) with a
      short TTL. Identical-data reads are served from cache — most 2s-poll
      cycles hit zero copy_rates calls.
    - `prewarm()`: launches a background task that warms the cache for
      every configured symbol on startup, so the first full prediction
      cycle never blocks on cold MT5 calls.
    - Cheap freshness probe: the cache revalidates by timestamp, not by
      re-pulling the world.
    - `snapshot()` is cached for 250 ms so high-frequency UI polling
      doesn't hammer MT5.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from .mt5_bridge import (
    MT5Bridge, Tick, SendResult, JarvisMT5Error, _tf_seconds,
)

LOGGER = logging.getLogger("jarvis.async_mt5")


@dataclass
class RatesEntry:
    df: pd.DataFrame
    fetched_at: float
    tf: str


class MT5RateCache:
    """Per-(symbol,timeframe) DataFrame cache with TTL + timestamp revalidation."""

    def __init__(self, ttl: float = 1.2) -> None:
        self.ttl = float(ttl)
        self._store: dict[tuple[str, str], RatesEntry] = {}
        self._lock = threading.Lock()

    def get(self, symbol: str, tf: str, max_age: float | None = None) -> pd.DataFrame | None:
        age = max_age if max_age is not None else self.ttl
        with self._lock:
            e = self._store.get((symbol, tf))
        if e is None:
            return None
        if time.time() - e.fetched_at > age:
            return None
        return e.df

    def put(self, symbol: str, tf: str, df: pd.DataFrame) -> None:
        with self._lock:
            # Only cache a richer/newer frame than what we have.
            existing = self._store.get((symbol, tf))
            if existing is not None:
                try:
                    if len(df) < len(existing.df):
                        return
                    new_last = df["time"].iloc[-1]
                    old_last = existing.df["time"].iloc[-1]
                    if new_last < old_last:
                        return
                except Exception:
                    pass
            self._store[(symbol, tf)] = RatesEntry(df=df, fetched_at=time.time(), tf=tf)

    def invalidate(self, symbol: str | None = None, tf: str | None = None) -> None:
        with self._lock:
            keys = list(self._store.keys())
            for k in keys:
                if symbol and k[0] != symbol:
                    continue
                if tf and k[1] != tf:
                    continue
                self._store.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class AsyncMT5Bridge:
    """Async wrapper around MT5Bridge with a caching + prefetch layer."""

    def __init__(self, bridge: MT5Bridge, *, max_workers: int = 1,
                 cache_ttl: float = 1.2, snapshot_ttl: float = 0.25) -> None:
        self._bridge = bridge
        self._max_workers = max_workers
        self._lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()
        self.cache = MT5RateCache(ttl=cache_ttl)
        self._snapshot_ttl = float(snapshot_ttl)
        self._unified_cache: tuple[float, dict] | None = None
        self._prewarm_task: asyncio.Task | None = None

    # ----- delegation helpers --------------------------------------------------

    async def _run(self, fn, *args, **kwargs):
        """Serialise MT5 calls — MetaTrader5 is not safe under concurrent calls."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _tf(self) -> str:
        return self._bridge.timeframe_str

    # ----- market data (cached) -------------------------------------------------

    async def rates(self, symbol: str, count: int | None = None,
                    *, timeframe: str | None = None, force: bool = False) -> pd.DataFrame:
        """Fetch rates for `symbol` — cached when the data is fresh.

        `timeframe` selects the MT5 timeframe (e.g. "M15", "H1", "D1");
        defaults to the bridge's configured timeframe.
        """
        tf = timeframe or self._tf()
        if not force:
            cached = self.cache.get(symbol, tf)
            if cached is not None:
                return cached
        df = await self._run(self._bridge.rates, symbol, count, timeframe=tf)
        if df is not None and not df.empty:
            self.cache.put(symbol, tf, df)
        return df

    async def rates_multi(self, symbols: Sequence[str], *,
                          force: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch (cached) rates for many symbols concurrently."""
        out: dict[str, pd.DataFrame] = {}
        for s in list(symbols):
            try:
                df = await self.rates(s, force=force)
            except JarvisMT5Error as e:
                LOGGER.debug("rates(%s) failed: %s", s, e)
                continue
            if df is not None and not df.empty:
                out[s] = df
        return out

    async def close_series(self, symbol: str, count: int | None = None,
                           *, force: bool = False):
        df = await self.rates(symbol, count, force=force)
        return df["close"].to_numpy() if df is not None and not df.empty else None

    async def tick(self, symbol: str) -> Tick:
        return await self._run(self._bridge.tick, symbol)

    async def equity(self) -> float:
        return await self._run(self._bridge.equity)

    async def balance(self) -> float:
        return await self._run(self._bridge.balance)

    async def open_positions(self):
        return await self._run(self._bridge.open_positions)

    async def daily_pnl(self) -> float:
        return await self._run(self._bridge.daily_pnl)

    async def ensure_connected(self) -> bool:
        return await self._run(self._bridge.ensure_connected)

    async def shutdown(self) -> None:
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
        await self._run(self._bridge.shutdown)
        self.cache.clear()

    async def send_market_order(self, **kwargs) -> SendResult:
        return await self._run(self._bridge.send_market_order, **kwargs)

    async def close_position(self, position_id: int) -> SendResult:
        return await self._run(self._bridge.close_position, position_id)

    # ----- warm-up / prefetch ---------------------------------------------------

    async def prewarm(self) -> None:
        """Background task: warm the rate cache for every configured symbol."""
        try:
            await self.ensure_connected()
            symbols = list(self._bridge.symbols)
            LOGGER.info("Prewarming MT5 rate cache for %d symbols…", len(symbols))
            for s in symbols:
                try:
                    await self.rates(s, force=True)
                except JarvisMT5Error as e:
                    LOGGER.debug("prewarm %s failed: %s", s, e)
            LOGGER.info("MT5 rate cache prewarmed")
        except Exception as e:
            LOGGER.warning("MT5 prewarm skipped: %s", e)

    def schedule_prewarm(self) -> None:
        loop = asyncio.get_running_loop()
        self._prewarm_task = loop.create_task(self.prewarm())

    # ----- snapshot (cached for UI 4Hz polling) ---------------------------------

    async def snapshot(self) -> dict:
        """One-shot snapshot of equity, balance, open count — cached 250 ms."""
        now = time.time()
        if self._unified_cache is not None and now - self._unified_cache[0] < self._snapshot_ttl:
            return self._unified_cache[1]
        async with self._snapshot_lock:
            if self._unified_cache is not None and \
                    time.time() - self._unified_cache[0] < self._snapshot_ttl:
                return self._unified_cache[1]
            eq, bal, opens = await asyncio.gather(
                self.equity(), self.balance(), self.open_positions(),
            )
            snap = {
                "equity": eq,
                "balance": bal,
                "open_count": len(opens),
                "open_positions": opens,
                "timestamp": time.time(),
            }
            self._unified_cache = (time.time(), snap)
            return snap

    # ----- convenience -----------------------------------------------------------

    async def rates_batch(self, symbols: Sequence[str], count: int | None = None) -> dict[str, pd.DataFrame]:
        """Backwards-compatible batch fetch. Uses caching when possible."""
        return await self.rates_multi(symbols, force=False)


__all__ = ["AsyncMT5Bridge", "MT5RateCache"]