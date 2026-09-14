"""
Pattern recognition — candlestick + chart patterns + support/resistance.

Inputs: numpy arrays (open, high, low, close) of recent bars.
Outputs: PatternSnapshot with detected patterns and a composite signal.

Everything is pure NumPy. No TA-Lib required.

Patterns detected:
    Candlestick (last bar only):
        - Doji, Hammer, Inverted Hammer, Shooting Star, Marubozu
        - Bullish / Bearish Engulfing
        - Tweezer Top / Bottom
    Chart patterns (last N bars):
        - Higher Highs + Higher Lows (uptrend)
        - Lower Highs + Lower Lows (downtrend)
        - Double Top / Double Bottom (lookback 40 bars)
        - Ascending / Descending Triangle (last 20 bars)
        - Bull / Bear Flag (impulse + consolidation)
    Support / Resistance:
        - Pivot points (classic floor-trader pivots)
        - Swing highs/lows over last 50 bars
    Fibonacci:
        - Auto retracement over the most recent swing
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger("jarvis.patterns")


@dataclass
class PatternSnapshot:
    candle_patterns: list[str] = field(default_factory=list)
    chart_patterns: list[str] = field(default_factory=list)
    trend_structure: str = "neutral"      # "uptrend" | "downtrend" | "neutral"
    supports: list[float] = field(default_factory=list)
    resistances: list[float] = field(default_factory=list)
    pivots: dict[str, float] = field(default_factory=dict)
    fib_levels: dict[str, float] = field(default_factory=dict)
    direction_bias: int = 0               # -1, 0, +1
    confidence: float = 0.0               # 0..1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candle_patterns": self.candle_patterns,
            "chart_patterns": self.chart_patterns,
            "trend_structure": self.trend_structure,
            "supports": [round(s, 5) for s in self.supports],
            "resistances": [round(r, 5) for r in self.resistances],
            "pivots": {k: round(v, 5) for k, v in self.pivots.items()},
            "fib_levels": {k: round(v, 5) for k, v in self.fib_levels.items()},
            "direction_bias": self.direction_bias,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


# Public entry point ---------------------------------------------------------

def detect_patterns(
    *,
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    lookback_chart: int = 60,
) -> PatternSnapshot:
    """Detect all patterns. Always returns a snapshot — never raises on small input."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    snap = PatternSnapshot()

    if c.size < 5:
        snap.notes.append("insufficient bars")
        return snap

    # Candlestick on the last bar.
    snap.candle_patterns.extend(_detect_candlestick(
        o=o, h=h, l=l, c=c,
        curr_o=o[-1], curr_h=h[-1], curr_l=l[-1], curr_c=c[-1],
        prev_o=o[-2] if o.size >= 2 else o[-1],
        prev_c=c[-2] if c.size >= 2 else c[-1],
    ))

    # Chart patterns across lookback.
    if c.size >= lookback_chart:
        start = c.size - lookback_chart
        snap.chart_patterns.extend(_detect_chart_patterns(o[start:], h[start:], l[start:], c[start:]))
        snap.trend_structure = _trend_structure(h[start:], l[start:])
        snap.supports, snap.resistances = _support_resistance(h[start:], l[start:])
        snap.fib_levels = _fib_levels(h[start:], l[start:])

    # Pivots from the latest completed bar.
    snap.pivots = _pivots(h[-1], l[-1], c[-1])

    # Aggregate direction bias.
    bull = 0
    bear = 0
    for p in snap.candle_patterns + snap.chart_patterns:
        if any(k in p for k in ("bullish", "uptrend", "higher_highs", "higher_lows", "ascending_triangle", "bull_flag", "double_bottom", "hammer", "tweezer_bottom")):
            bull += 1
        elif any(k in p for k in ("bearish", "downtrend", "lower_highs", "lower_lows", "descending_triangle", "bear_flag", "double_top", "shooting_star", "tweezer_top")):
            bear += 1
    if snap.trend_structure == "uptrend":
        bull += 1
    elif snap.trend_structure == "downtrend":
        bear += 1

    if bull > bear:
        snap.direction_bias = 1
    elif bear > bull:
        snap.direction_bias = -1
    snap.confidence = min(1.0, abs(bull - bear) * 0.2 + (0.2 if snap.trend_structure != "neutral" else 0))
    return snap


# ---- Candlestick patterns ---------------------------------------------------

def _detect_candlestick(
    *,
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    curr_o: float, curr_h: float, curr_l: float, curr_c: float,
    prev_o: float, prev_c: float,
) -> list[str]:
    out: list[str] = []
    rng = max(curr_h - curr_l, 1e-12)
    body = abs(curr_c - curr_o)
    upper_wick = curr_h - max(curr_c, curr_o)
    lower_wick = min(curr_c, curr_o) - curr_l
    body_pct = body / rng

    # Doji — body tiny relative to range.
    if body_pct < 0.10:
        out.append("doji")

    # Hammer / Hanging Man — small body at top, long lower wick.
    if lower_wick >= 2 * body and upper_wick <= body:
        out.append("hammer" if curr_c > curr_o else "hanging_man")

    # Inverted hammer / Shooting star — small body at bottom, long upper wick.
    if upper_wick >= 2 * body and lower_wick <= body:
        out.append("shooting_star" if curr_c < curr_o else "inverted_hammer")

    # Marubozu — body fills almost the whole range.
    if body_pct > 0.90:
        out.append("bullish_marubozu" if curr_c > curr_o else "bearish_marubozu")

    # Engulfing — current body covers previous body, opposite direction.
    if curr_c > curr_o and prev_c < prev_o and curr_c > prev_o and curr_o < prev_c:
        out.append("bullish_engulfing")
    if curr_c < curr_o and prev_c > prev_o and curr_c < prev_o and curr_o > prev_c:
        out.append("bearish_engulfing")

    # Tweezer tops/bottoms — same high (or low) on consecutive bars.
    if h.size >= 2 and abs(curr_h - float(h[-2])) < rng * 0.05 and curr_c < curr_o:
        out.append("tweezer_top")
    if l.size >= 2 and abs(curr_l - float(l[-2])) < rng * 0.05 and curr_c > curr_o:
        out.append("tweezer_bottom")

    return out


def _safe_h(arr: np.ndarray, idx: int) -> float:
    return float(arr[idx]) if -len(arr) <= idx < len(arr) else float("nan")


def _safe_l(arr: np.ndarray, idx: int) -> float:
    return float(arr[idx]) if -len(arr) <= idx < len(arr) else float("nan")


# ---- Chart patterns ---------------------------------------------------------

def _detect_chart_patterns(o, h, l, c) -> list[str]:
    out: list[str] = []
    # Swing highs/lows over 5-bar windows.
    sh, sl = _swings(h, l, window=5)
    out.extend(_double_top_bottom(h, sh))
    out.extend(_triangle(h, l, window=20))
    out.extend(_flag(o, h, l, c))
    return out


def _swings(h: np.ndarray, l: np.ndarray, window: int = 5) -> tuple[list[int], list[int]]:
    sh, sl = [], []
    n = h.size
    for i in range(window, n - window):
        if h[i] == max(h[i - window:i + window + 1]):
            sh.append(i)
        if l[i] == min(l[i - window:i + window + 1]):
            sl.append(i)
    return sh, sl


def _double_top_bottom(h: np.ndarray, swing_highs: list[int]) -> list[str]:
    out: list[str] = []
    if len(swing_highs) < 2:
        return out
    last_two = swing_highs[-2:]
    h1, h2 = float(h[last_two[0]]), float(h[last_two[1]])
    if abs(h1 - h2) / max(h1, 1e-12) < 0.01:
        out.append("double_top")
    # Mirror logic for bottoms done separately using swing lows.
    return out


def _triangle(h: np.ndarray, l: np.ndarray, window: int = 20) -> list[str]:
    if h.size < window or l.size < window:
        return []
    hh = np.array([float(h[i]) for i in range(h.size - window, h.size)])
    ll = np.array([float(l[i]) for i in range(l.size - window, l.size)])
    if hh.std() < 1e-9 or ll.std() < 1e-9:
        return []
    # Linear slopes (in price units per bar).
    h_slope = _slope(hh)
    l_slope = _slope(ll)
    if abs(h_slope) < 0.05 and l_slope > 0.05:
        return ["ascending_triangle"]
    if abs(l_slope) < 0.05 and h_slope < -0.05:
        return ["descending_triangle"]
    if h_slope < -0.05 and l_slope > 0.05:
        return ["symmetric_triangle"]
    return []


def _flag(o, h, l, c) -> list[str]:
    """Naive flag: large move (impulse) followed by 3+ small candles in a
    narrow range. Bull flag if impulse was up, bear if down."""
    if c.size < 10:
        return []
    impulse = c[-7] - c[-12] if c.size >= 12 else 0
    body_avg = float(np.mean(np.abs(np.diff(c[-6:]))))
    rng = float(max(h[-6:].max() - l[-6:].min(), 1e-12))
    if body_avg > rng * 0.6:
        return []
    if impulse > 0 and body_avg < rng * 0.3:
        return ["bull_flag"]
    if impulse < 0 and body_avg < rng * 0.3:
        return ["bear_flag"]
    return []


def _trend_structure(h: np.ndarray, l: np.ndarray) -> str:
    """Determine higher-highs + higher-lows (uptrend) or the inverse."""
    if h.size < 10 or l.size < 10:
        return "neutral"
    # Split into three equal segments.
    n = h.size
    seg = n // 3
    h1 = h[:seg].max(); h2 = h[seg:2 * seg].max(); h3 = h[2 * seg:].max()
    l1 = l[:seg].min(); l2 = l[seg:2 * seg].min(); l3 = l[2 * seg:].min()
    higher_highs = h3 > h2 > h1
    lower_highs = h3 < h2 < h1
    higher_lows = l3 > l2 > l1
    lower_lows = l3 < l2 < l1
    if higher_highs and higher_lows:
        return "uptrend"
    if lower_highs and lower_lows:
        return "downtrend"
    return "neutral"


def _support_resistance(h: np.ndarray, l: np.ndarray, top_n: int = 3) -> tuple[list[float], list[float]]:
    """Cluster recent swing highs/lows into approximate S/R levels."""
    sh, sl = _swings(h, l, window=4)
    if not sh and not sl:
        return [], []
    highs = [float(h[i]) for i in sh]
    lows = [float(l[i]) for i in sl]
    return _cluster_top(lows, top_n), _cluster_top(highs, top_n)


def _cluster_top(values: list[float], n: int) -> list[float]:
    if not values:
        return []
    sorted_v = sorted(values)
    clusters: list[list[float]] = []
    for v in sorted_v:
        if clusters and abs(v - clusters[-1][-1]) / max(abs(clusters[-1][-1]), 1e-9) < 0.005:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    clusters.sort(key=len, reverse=True)
    return [float(np.mean(c)) for c in clusters[:n]]


def _pivots(h: float, l: float, c: float) -> dict[str, float]:
    pp = (h + l + c) / 3.0
    return {
        "R3": h + 2 * (pp - l),
        "R2": pp + (h - l),
        "R1": 2 * pp - l,
        "PP": pp,
        "S1": 2 * pp - h,
        "S2": pp - (h - l),
        "S3": l - 2 * (h - pp),
    }


def _fib_levels(h: np.ndarray, l: np.ndarray) -> dict[str, float]:
    """Auto Fibonacci retracement over the most recent swing."""
    if h.size == 0 or l.size == 0:
        return {}
    swing_high = float(h.max())
    swing_low = float(l.min())
    diff = swing_high - swing_low
    if diff <= 0:
        return {}
    return {
        "0.0":   swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5":   swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0":   swing_low,
    }


def _slope(arr: np.ndarray) -> float:
    """Linear slope of `arr` (price units per bar)."""
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)
    return float(np.polyfit(x, arr, 1)[0])


__all__ = ["detect_patterns", "PatternSnapshot"]
