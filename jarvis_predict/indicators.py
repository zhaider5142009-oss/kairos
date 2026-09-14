"""
Technical indicators — pure NumPy/pandas implementation.

No external TA library needed. All functions handle short / NaN inputs
gracefully and return structured IndicatorSnapshot values.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import JarvisConfig

LOGGER = logging.getLogger("jarvis.indicators")


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Snapshot of all technical indicators at the latest bar."""

    rsi: float                 # 0..100
    macd: float                # MACD line
    macd_signal: float         # signal line
    macd_hist: float           # MACD - signal
    atr: float                 # absolute price units
    volume_ratio: float        # current volume / MA(volume, n)
    trend_bias: int            # -1, 0, +1
    momentum_bias: int         # -1, 0, +1
    volatility_bias: int       # -1, 0, +1

    @property
    def composite_bias(self) -> int:
        """Net directional bias from all indicators."""
        return self.trend_bias + self.momentum_bias + self.volatility_bias

    def to_dict(self) -> dict:
        return {
            "rsi": round(self.rsi, 2),
            "macd": round(self.macd, 6),
            "macd_signal": round(self.macd_signal, 6),
            "macd_hist": round(self.macd_hist, 6),
            "atr": round(self.atr, 6),
            "volume_ratio": round(self.volume_ratio, 3),
            "trend_bias": self.trend_bias,
            "momentum_bias": self.momentum_bias,
            "volatility_bias": self.volatility_bias,
            "composite_bias": self.composite_bias,
        }


class TechnicalIndicators:
    """Stateless calculator — pass in OHLCV arrays, get IndicatorSnapshot."""

    def __init__(
        self,
        *,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        volume_ma_period: int = 20,
    ) -> None:
        self.rsi_period = max(2, int(rsi_period))
        self.macd_fast = max(1, int(macd_fast))
        self.macd_slow = max(self.macd_fast + 1, int(macd_slow))
        self.macd_signal = max(1, int(macd_signal))
        self.atr_period = max(2, int(atr_period))
        self.volume_ma_period = max(2, int(volume_ma_period))

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "TechnicalIndicators":
        s = config.section("indicators")
        return cls(
            rsi_period=s["rsi_period"],
            macd_fast=s["macd_fast"],
            macd_slow=s["macd_slow"],
            macd_signal=s["macd_signal"],
            atr_period=s["atr_period"],
            volume_ma_period=s["volume_ma_period"],
        )

    # --- public API ---------------------------------------------------------

    def compute(
        self,
        open_: Sequence[float],
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float] | None = None,
    ) -> IndicatorSnapshot:
        """Compute all indicators from OHLCV arrays.

        Arrays must be aligned and the same length. Volume is optional —
        if absent or all-zero, volume_ratio defaults to 1.0.
        """
        o = np.asarray(close, dtype=np.float64)
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        cl = np.asarray(close, dtype=np.float64)
        if volume is not None:
            v = np.asarray(volume, dtype=np.float64)
        else:
            v = None

        # Validate lengths (defensive).
        n = cl.size
        if n < self.macd_slow + self.macd_signal:
            LOGGER.debug("Series too short for indicators (n=%d); returning neutral", n)
            return self._neutral_snapshot()

        rsi = _rsi(cl, self.rsi_period)
        macd_line, macd_sig, macd_hist = _macd(
            cl, self.macd_fast, self.macd_slow, self.macd_signal
        )
        atr = _atr(h, l, cl, self.atr_period)
        vol_ratio = (
            _volume_ratio(v, self.volume_ma_period) if v is not None else 1.0
        )

        # Bias rules — conservative, designed to add evidence (not over-trade).
        # Trend: MACD hist > 0 => up; < 0 => down.
        if macd_hist > 0:
            trend_bias = 1
        elif macd_hist < 0:
            trend_bias = -1
        else:
            trend_bias = 0

        # Momentum: RSI > 60 => up; < 40 => down; else neutral.
        if rsi >= 60.0:
            momentum_bias = 1
        elif rsi <= 40.0:
            momentum_bias = -1
        else:
            momentum_bias = 0

        # Volatility: high volume + positive trend => continuation bias; otherwise neutral.
        if vol_ratio >= 1.5 and trend_bias != 0:
            volatility_bias = trend_bias
        else:
            volatility_bias = 0

        return IndicatorSnapshot(
            rsi=float(rsi),
            macd=float(macd_line),
            macd_signal=float(macd_sig),
            macd_hist=float(macd_hist),
            atr=float(atr),
            volume_ratio=float(vol_ratio),
            trend_bias=trend_bias,
            momentum_bias=momentum_bias,
            volatility_bias=volatility_bias,
        )

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _neutral_snapshot() -> IndicatorSnapshot:
        return IndicatorSnapshot(
            rsi=50.0, macd=0.0, macd_signal=0.0, macd_hist=0.0,
            atr=0.0, volume_ratio=1.0,
            trend_bias=0, momentum_bias=0, volatility_bias=0,
        )


# --- pure-NumPy primitives ---------------------------------------------------

def _rsi(close: np.ndarray, period: int) -> float:
    """Relative Strength Index (Wilder smoothing)."""
    if close.size <= period:
        return 50.0
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, gains.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Returns array of same length, initialised with NaN."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    if series.size < period:
        return out
    # Seed with SMA of first `period` values (standard EMA initialisation).
    out[period - 1] = series[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, series.size):
        out[i] = series[i] * k + out[i - 1] * (1.0 - k)
    return out


def _macd(
    close: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[float, float, float]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line[~np.isnan(macd_line)], signal)
    # Re-align signal line to match macd_line's valid range
    valid_start = slow - 1
    if sig_line.size == 0 or valid_start >= macd_line.size:
        return 0.0, 0.0, 0.0
    sig_aligned = np.full_like(macd_line, np.nan)
    sig_aligned[valid_start:] = sig_line
    last_macd = float(macd_line[-1]) if np.isfinite(macd_line[-1]) else 0.0
    last_sig = float(sig_aligned[-1]) if np.isfinite(sig_aligned[-1]) else 0.0
    return last_macd, last_sig, last_macd - last_sig


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """Average True Range (Wilder smoothing)."""
    if high.size <= period:
        return 0.0
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    atr = tr[:period].mean()
    for i in range(period, tr.size):
        atr = (atr * (period - 1) + tr[i]) / period
    return float(atr)


def _volume_ratio(volume: np.ndarray, period: int) -> float:
    if volume.size <= period or volume.sum() == 0:
        return 1.0
    ma = volume[-period:].mean()
    if ma == 0:
        return 1.0
    return float(volume[-1] / ma)


__all__ = ["TechnicalIndicators", "IndicatorSnapshot"]
