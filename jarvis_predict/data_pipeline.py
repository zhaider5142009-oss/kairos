"""
Large-data pipeline.

Stores OHLCV candles in Parquet, partitioned by symbol + timeframe + date.
Caches rolling features per symbol so the runtime doesn't re-compute
indicators / patterns / correlations on every cycle.

Designed for "lots of data" — not high-frequency tick storage. Suitable for
5-min candles across years of history on a personal machine.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import JarvisConfig

LOGGER = logging.getLogger("jarvis.data")

_TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400,
    "D1": 86400, "W1": 604800, "MN1": 2592000,
}


@dataclass
class CandleKey:
    symbol: str
    timeframe: str
    date: str                       # YYYY-MM-DD


class DataPipeline:
    """Parquet-backed candle storage + in-memory feature cache."""

    def __init__(self, *, config: JarvisConfig, cache_size: int = 256) -> None:
        self.root = config.cache_dir / "parquet"
        self.root.mkdir(parents=True, exist_ok=True)
        self._feature_cache: dict[tuple[str, str, str], tuple[float, pd.DataFrame, dict]] = {}
        self._cache_size = max(32, int(cache_size))

    # --- storage -------------------------------------------------------------

    def _path_for(self, key: CandleKey) -> Path:
        # Layout: <root>/<symbol>/<timeframe>/<year>-<month>.parquet
        ts = pd.Timestamp(key.date)
        return self.root / key.symbol / key.timeframe / f"{ts.year:04d}-{ts.month:02d}.parquet"

    def write(self, df: pd.DataFrame, *, symbol: str, timeframe: str) -> None:
        """Append-merge a DataFrame into the per-month parquet file."""
        if df is None or df.empty:
            return
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
        df["_month"] = df["time"].dt.strftime("%Y-%m")
        for month, sub in df.groupby("_month"):
            year, m = month.split("-")
            path = self.root / symbol / timeframe / f"{year}-{m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            sub = sub.drop(columns=["_month"])
            if path.exists():
                existing = pd.read_parquet(path)
                sub = pd.concat([existing, sub]).drop_duplicates(subset=["time"]).sort_values("time")
            sub.to_parquet(path, engine="pyarrow", index=False)

    def read(self, *, symbol: str, timeframe: str, since: float | None = None,
             limit: int | None = None) -> pd.DataFrame:
        """Read concatenated candles across all monthly files for symbol/TF."""
        files = sorted((self.root / symbol / timeframe).glob("*.parquet")) if (self.root / symbol / timeframe).exists() else []
        if not files:
            return pd.DataFrame()
        parts = [pd.read_parquet(f) for f in files]
        df = pd.concat(parts).drop_duplicates(subset=["time"]).sort_values("time")
        if since is not None:
            cutoff = pd.Timestamp(since, unit="s", tz="UTC")
            df = df[df["time"] >= cutoff]
        if limit is not None and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        return df.reset_index(drop=True)

    def last_timestamp(self, *, symbol: str, timeframe: str) -> float | None:
        df = self.read(symbol=symbol, timeframe=timeframe, limit=1)
        if df.empty:
            return None
        return float(df["time"].iloc[-1].timestamp())

    # --- feature cache -------------------------------------------------------

    def cache_features(self, *, symbol: str, timeframe: str, fingerprint: str,
                       df: pd.DataFrame, features: dict) -> None:
        key = (symbol, timeframe, fingerprint)
        with self._eviction_lock():
            self._feature_cache[key] = (time.time(), df, features)
        self._evict()

    def get_features(self, *, symbol: str, timeframe: str, fingerprint: str,
                     max_age_seconds: float = 30.0) -> dict | None:
        key = (symbol, timeframe, fingerprint)
        entry = self._feature_cache.get(key)
        if entry is None:
            return None
        ts, _df, features = entry
        if time.time() - ts > max_age_seconds:
            return None
        return features

    # --- internals -----------------------------------------------------------

    def _eviction_lock(self):
        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        # No external lock — single-thread access pattern.
        return _Noop()

    def _evict(self) -> None:
        if len(self._feature_cache) <= self._cache_size:
            return
        # Sort by timestamp, drop oldest.
        items = sorted(self._feature_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in items[: len(self._feature_cache) - self._cache_size]:
            del self._feature_cache[k]


# ---- fingerprinting ----------------------------------------------------------

def candle_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of (last_timestamp, length) so cache invalidates on new bars."""
    if df is None or df.empty:
        return "empty"
    last_ts = str(df["time"].iloc[-1])
    n = len(df)
    return f"{last_ts}|n={n}"


def bars_per_year(timeframe: str) -> float:
    """Approx bars per year for the given timeframe."""
    secs = _TIMEFRAME_SECONDS.get(timeframe, 300)
    return 365.0 * 24 * 3600 / secs


__all__ = ["DataPipeline", "CandleKey", "candle_fingerprint", "bars_per_year"]
