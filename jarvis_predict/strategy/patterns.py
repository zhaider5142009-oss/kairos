"""
Advanced pattern recognition engine for JARVIS Reborn.
Detects candlestick patterns, chart patterns, and multi-timeframe confluence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math


class PatternType(Enum):
    CANDLESTICK = "candlestick"
    CHART = "chart"
    INDICATOR = "indicator"
    CONFLUENCE = "confluence"


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: float = 0.0


@dataclass
class DetectedPattern:
    name: str
    type: PatternType
    direction: PatternDirection
    confidence: float  # 0-1
    timeframe: str
    metadata: dict[str, Any] = field(default_factory=dict)
    candles_used: int = 0


class PatternRecognizer:
    """Advanced pattern recognition with candlestick, chart, and indicator patterns."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.6)

    # -------------------------------------------------------------------------
    # Candlestick Patterns
    # -------------------------------------------------------------------------

    def detect_candlestick_patterns(self, candles: list[Candle]) -> list[DetectedPattern]:
        """Detect major candlestick patterns. Returns list of detected patterns."""
        if len(candles) < 3:
            return []

        patterns = []
        # Use last 3-5 candles for pattern detection
        recent = candles[-5:]

        # Single candle patterns
        patterns.extend(self._detect_doji(recent))
        patterns.extend(self._detect_hammer(recent))
        patterns.extend(self._detect_shooting_star(recent))
        patterns.extend(self._detect_marubozu(recent))

        # Two candle patterns
        patterns.extend(self._detect_engulfing(recent))
        patterns.extend(self._detect_harami(recent))
        patterns.extend(self._detect_piercing_line(recent))
        patterns.extend(self._detect_dark_cloud_cover(recent))

        # Three candle patterns
        patterns.extend(self._detect_morning_star(recent))
        patterns.extend(self._detect_evening_star(recent))
        patterns.extend(self._detect_three_white_soldiers(recent))
        patterns.extend(self._detect_three_black_crows(recent))

        return [p for p in patterns if p.confidence >= self.min_confidence]

    def _body_size(self, c: Candle) -> float:
        return abs(c.close - c.open)

    def _is_bullish(self, c: Candle) -> bool:
        return c.close > c.open

    def _is_bearish(self, c: Candle) -> bool:
        return c.close < c.open

    def _upper_wick(self, c: Candle) -> float:
        return c.high - max(c.open, c.close)

    def _lower_wick(self, c: Candle) -> float:
        return min(c.open, c.close) - c.low

    def _detect_doji(self, candles: list[Candle]) -> list[DetectedPattern]:
        if not candles:
            return []
        c = candles[-1]
        body = self._body_size(c)
        range_ = c.high - c.low
        if range_ == 0:
            return []
        if body / range_ < 0.1:  # Very small body
            return [DetectedPattern(
                name="Doji", type=PatternType.CANDLESTICK,
                direction=PatternDirection.NEUTRAL, confidence=0.7,
                timeframe="current", candles_used=1,
                metadata={"body_ratio": body/range_}
            )]
        return []

    def _detect_hammer(self, candles: list[Candle]) -> list[DetectedPattern]:
        if not candles:
            return []
        c = candles[-1]
        body = self._body_size(c)
        lower = self._lower_wick(c)
        upper = self._upper_wick(c)
        range_ = c.high - c.low
        if range_ == 0 or body == 0:
            return []
        # Hammer: small body, long lower wick (2x body), little upper wick
        if lower >= 2 * body and upper < 0.1 * range_:
            direction = PatternDirection.BULLISH if self._is_bullish(c) else PatternDirection.BULLISH
            return [DetectedPattern(
                name="Hammer", type=PatternType.CANDLESTICK,
                direction=direction, confidence=0.75,
                timeframe="current", candles_used=1,
                metadata={"lower_wick_ratio": lower/body}
            )]
        return []

    def _detect_shooting_star(self, candles: list[Candle]) -> list[DetectedPattern]:
        if not candles:
            return []
        c = candles[-1]
        body = self._body_size(c)
        lower = self._lower_wick(c)
        upper = self._upper_wick(c)
        range_ = c.high - c.low
        if range_ == 0 or body == 0:
            return []
        if upper >= 2 * body and lower < 0.1 * range_:
            return [DetectedPattern(
                name="Shooting Star", type=PatternType.CANDLESTICK,
                direction=PatternDirection.BEARISH, confidence=0.75,
                timeframe="current", candles_used=1,
                metadata={"upper_wick_ratio": upper/body}
            )]
        return []

    def _detect_marubozu(self, candles: list[Candle]) -> list[DetectedPattern]:
        if not candles:
            return []
        c = candles[-1]
        body = self._body_size(c)
        range_ = c.high - c.low
        if range_ == 0:
            return []
        if body / range_ > 0.9:  # Almost no wicks
            direction = PatternDirection.BULLISH if self._is_bullish(c) else PatternDirection.BEARISH
            return [DetectedPattern(
                name="Marubozu", type=PatternType.CANDLESTICK,
                direction=direction, confidence=0.8,
                timeframe="current", candles_used=1,
                metadata={"body_ratio": body/range_}
            )]
        return []

    def _detect_engulfing(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 2:
            return []
        prev, curr = candles[-2], candles[-1]
        prev_body = self._body_size(prev)
        curr_body = self._body_size(curr)
        if prev_body == 0 or curr_body == 0:
            return []
        # Bullish engulfing: prev bearish, curr bullish, curr body engulfs prev
        if self._is_bearish(prev) and self._is_bullish(curr):
            if curr.open <= prev.close and curr.close >= prev.open:
                return [DetectedPattern(
                    name="Bullish Engulfing", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BULLISH, confidence=0.85,
                    timeframe="current", candles_used=2
                )]
        # Bearish engulfing
        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr.open >= prev.close and curr.close <= prev.open:
                return [DetectedPattern(
                    name="Bearish Engulfing", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BEARISH, confidence=0.85,
                    timeframe="current", candles_used=2
                )]
        return []

    def _detect_harami(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 2:
            return []
        prev, curr = candles[-2], candles[-1]
        prev_body = self._body_size(prev)
        curr_body = self._body_size(curr)
        if prev_body == 0:
            return []
        # Bullish harami: prev bearish large, curr bullish small inside prev
        if self._is_bearish(prev) and self._is_bullish(curr):
            if curr.open > prev.close and curr.close < prev.open and curr_body < prev_body * 0.5:
                return [DetectedPattern(
                    name="Bullish Harami", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BULLISH, confidence=0.7,
                    timeframe="current", candles_used=2
                )]
        # Bearish harami
        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr.open < prev.close and curr.close > prev.open and curr_body < prev_body * 0.5:
                return [DetectedPattern(
                    name="Bearish Harami", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BEARISH, confidence=0.7,
                    timeframe="current", candles_used=2
                )]
        return []

    def _detect_piercing_line(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 2:
            return []
        prev, curr = candles[-2], candles[-1]
        if self._is_bearish(prev) and self._is_bullish(curr):
            # Piercing: opens below prev close, closes above prev midpoint
            if curr.open < prev.close and curr.close > (prev.open + prev.close) / 2:
                return [DetectedPattern(
                    name="Piercing Line", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BULLISH, confidence=0.75,
                    timeframe="current", candles_used=2
                )]
        return []

    def _detect_dark_cloud_cover(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 2:
            return []
        prev, curr = candles[-2], candles[-1]
        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr.open > prev.close and curr.close < (prev.open + prev.close) / 2:
                return [DetectedPattern(
                    name="Dark Cloud Cover", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BEARISH, confidence=0.75,
                    timeframe="current", candles_used=2
                )]
        return []

    def _detect_morning_star(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 3:
            return []
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        # Morning star: bearish, doji/small, bullish closing above c1 midpoint
        if self._is_bearish(c1) and self._is_bullish(c3):
            c2_body = self._body_size(c2)
            c1_body = self._body_size(c1)
            if c2_body < c1_body * 0.3 and c3.close > (c1.open + c1.close) / 2:
                return [DetectedPattern(
                    name="Morning Star", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BULLISH, confidence=0.85,
                    timeframe="current", candles_used=3
                )]
        return []

    def _detect_evening_star(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 3:
            return []
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if self._is_bullish(c1) and self._is_bearish(c3):
            c2_body = self._body_size(c2)
            c1_body = self._body_size(c1)
            if c2_body < c1_body * 0.3 and c3.close < (c1.open + c1.close) / 2:
                return [DetectedPattern(
                    name="Evening Star", type=PatternType.CANDLESTICK,
                    direction=PatternDirection.BEARISH, confidence=0.85,
                    timeframe="current", candles_used=3
                )]
        return []

    def _detect_three_white_soldiers(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 3:
            return []
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if all(self._is_bullish(c) for c in [c1, c2, c3]):
            # Each opens within previous body, closes higher
            if c2.open > c1.open and c2.open < c1.close and c2.close > c1.close:
                if c3.open > c2.open and c3.open < c2.close and c3.close > c2.close:
                    return [DetectedPattern(
                        name="Three White Soldiers", type=PatternType.CANDLESTICK,
                        direction=PatternDirection.BULLISH, confidence=0.9,
                        timeframe="current", candles_used=3
                    )]
        return []

    def _detect_three_black_crows(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 3:
            return []
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if all(self._is_bearish(c) for c in [c1, c2, c3]):
            if c2.open < c1.open and c2.open > c1.close and c2.close < c1.close:
                if c3.open < c2.open and c3.open > c2.close and c3.close < c2.close:
                    return [DetectedPattern(
                        name="Three Black Crows", type=PatternType.CANDLESTICK,
                        direction=PatternDirection.BEARISH, confidence=0.9,
                        timeframe="current", candles_used=3
                    )]
        return []

    # -------------------------------------------------------------------------
    # Chart Patterns (simplified - need more candles)
    # -------------------------------------------------------------------------

    def detect_chart_patterns(self, candles: list[Candle], lookback: int = 50) -> list[DetectedPattern]:
        """Detect chart patterns like triangles, flags, head & shoulders."""
        if len(candles) < lookback:
            return []

        patterns = []
        recent = candles[-lookback:]

        # Support/Resistance levels
        levels = self._find_support_resistance(recent)
        patterns.extend(self._detect_triangle(recent, levels))
        patterns.extend(self._detect_flag_pennant(recent))
        patterns.extend(self._detect_head_and_shoulders(recent))
        patterns.extend(self._detect_double_top_bottom(recent))

        return [p for p in patterns if p.confidence >= self.min_confidence]

    def _find_support_resistance(self, candles: list[Candle], window: int = 10) -> dict[str, list[float]]:
        """Find local highs/lows as potential S/R levels."""
        highs, lows = [], []
        for i in range(window, len(candles) - window):
            c = candles[i]
            # Local high
            if all(c.high >= candles[j].high for j in range(i-window, i+window+1) if j != i):
                highs.append(c.high)
            # Local low
            if all(c.low <= candles[j].low for j in range(i-window, i+window+1) if j != i):
                lows.append(c.low)

        # Cluster nearby levels
        def cluster(levels: list[float], threshold: float = 0.002) -> list[float]:
            if not levels:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]
            for l in levels[1:]:
                if abs(l - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(l)
                else:
                    clusters.append([l])
            return [sum(c)/len(c) for c in clusters]

        return {"resistance": cluster(highs), "support": cluster(lows)}

    def _detect_triangle(self, candles: list[Candle], levels: dict) -> list[DetectedPattern]:
        # Simplified: check if highs descending and lows ascending (symmetrical)
        if len(candles) < 20:
            return []
        recent_highs = [c.high for c in candles[-20:]]
        recent_lows = [c.low for c in candles[-20:]]

        # Linear regression slope
        high_slope = self._slope(recent_highs)
        low_slope = self._slope(recent_lows)

        patterns = []
        if high_slope < -0.0001 and low_slope > 0.0001:
            patterns.append(DetectedPattern(
                name="Symmetrical Triangle", type=PatternType.CHART,
                direction=PatternDirection.NEUTRAL, confidence=0.7,
                timeframe="multi", candles_used=20,
                metadata={"high_slope": high_slope, "low_slope": low_slope}
            ))
        elif high_slope < -0.0001 and abs(low_slope) < 0.0001:
            patterns.append(DetectedPattern(
                name="Descending Triangle", type=PatternType.CHART,
                direction=PatternDirection.BEARISH, confidence=0.75,
                timeframe="multi", candles_used=20,
                metadata={"high_slope": high_slope}
            ))
        elif abs(high_slope) < 0.0001 and low_slope > 0.0001:
            patterns.append(DetectedPattern(
                name="Ascending Triangle", type=PatternType.CHART,
                direction=PatternDirection.BULLISH, confidence=0.75,
                timeframe="multi", candles_used=20,
                metadata={"low_slope": low_slope}
            ))
        return patterns

    def _detect_flag_pennant(self, candles: list[Candle]) -> list[DetectedPattern]:
        # Simplified: strong move followed by consolidation
        if len(candles) < 15:
            return []
        # Check for pole (strong move in first 5 candles)
        pole_candles = candles[-15:-10]
        flag_candles = candles[-10:]

        pole_move = abs(pole_candles[-1].close - pole_candles[0].open)
        pole_range = max(c.high for c in pole_candles) - min(c.low for c in pole_candles)
        flag_range = max(c.high for c in flag_candles) - min(c.low for c in flag_candles)

        if pole_range > 0 and flag_range / pole_range < 0.5:  # Consolidation < 50% of pole
            direction = PatternDirection.BULLISH if pole_candles[-1].close > pole_candles[0].open else PatternDirection.BEARISH
            return [DetectedPattern(
                name="Flag/Pennant", type=PatternType.CHART,
                direction=direction, confidence=0.7,
                timeframe="multi", candles_used=15,
                metadata={"pole_range": pole_range, "flag_range": flag_range}
            )]
        return []

    def _detect_head_and_shoulders(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 30:
            return []
        # Simplified: find three peaks with middle highest
        highs = [(i, c.high) for i, c in enumerate(candles[-30:])]
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i][1] > highs[i-1][1] and highs[i][1] > highs[i-2][1] and \
               highs[i][1] > highs[i+1][1] and highs[i][1] > highs[i+2][1]:
                peaks.append(highs[i])

        if len(peaks) >= 3:
            left, head, right = peaks[-3], peaks[-2], peaks[-1]
            if head[1] > left[1] and head[1] > right[1] and abs(left[1] - right[1]) / left[1] < 0.03:
                return [DetectedPattern(
                    name="Head & Shoulders", type=PatternType.CHART,
                    direction=PatternDirection.BEARISH, confidence=0.8,
                    timeframe="multi", candles_used=30,
                    metadata={"left": left[1], "head": head[1], "right": right[1]}
                )]
        return []

    def _detect_double_top_bottom(self, candles: list[Candle]) -> list[DetectedPattern]:
        if len(candles) < 20:
            return []
        highs = [c.high for c in candles[-20:]]
        lows = [c.low for c in candles[-20:]]

        patterns = []
        # Double top: two similar highs
        max_high = max(highs)
        high_count = sum(1 for h in highs if abs(h - max_high) / max_high < 0.01)
        if high_count >= 2:
            patterns.append(DetectedPattern(
                name="Double Top", type=PatternType.CHART,
                direction=PatternDirection.BEARISH, confidence=0.75,
                timeframe="multi", candles_used=20,
                metadata={"level": max_high}
            ))

        # Double bottom
        min_low = min(lows)
        low_count = sum(1 for l in lows if abs(l - min_low) / min_low < 0.01)
        if low_count >= 2:
            patterns.append(DetectedPattern(
                name="Double Bottom", type=PatternType.CHART,
                direction=PatternDirection.BULLISH, confidence=0.75,
                timeframe="multi", candles_used=20,
                metadata={"level": min_low}
            ))
        return patterns

    # -------------------------------------------------------------------------
    # Indicator-based Patterns
    # -------------------------------------------------------------------------

    def detect_indicator_patterns(
        self,
        rsi: float,
        macd_hist: float,
        macd_signal: float,
        bb_upper: float,
        bb_lower: float,
        bb_middle: float,
        price: float,
        volume: float,
        avg_volume: float
    ) -> list[DetectedPattern]:
        """Detect patterns from technical indicators."""
        patterns = []

        # RSI extremes
        if rsi >= 80:
            patterns.append(DetectedPattern(
                name="RSI Extreme Overbought", type=PatternType.INDICATOR,
                direction=PatternDirection.BEARISH, confidence=0.8,
                timeframe="current", metadata={"rsi": rsi}
            ))
        elif rsi >= 70:
            patterns.append(DetectedPattern(
                name="RSI Overbought", type=PatternType.INDICATOR,
                direction=PatternDirection.BEARISH, confidence=0.6,
                timeframe="current", metadata={"rsi": rsi}
            ))
        elif rsi <= 20:
            patterns.append(DetectedPattern(
                name="RSI Extreme Oversold", type=PatternType.INDICATOR,
                direction=PatternDirection.BULLISH, confidence=0.8,
                timeframe="current", metadata={"rsi": rsi}
            ))
        elif rsi <= 30:
            patterns.append(DetectedPattern(
                name="RSI Oversold", type=PatternType.INDICATOR,
                direction=PatternDirection.BULLISH, confidence=0.6,
                timeframe="current", metadata={"rsi": rsi}
            ))

        # MACD crossover
        if macd_hist > 0 and macd_signal <= 0:
            patterns.append(DetectedPattern(
                name="MACD Bullish Crossover", type=PatternType.INDICATOR,
                direction=PatternDirection.BULLISH, confidence=0.75,
                timeframe="current", metadata={"hist": macd_hist}
            ))
        elif macd_hist < 0 and macd_signal >= 0:
            patterns.append(DetectedPattern(
                name="MACD Bearish Crossover", type=PatternType.INDICATOR,
                direction=PatternDirection.BEARISH, confidence=0.75,
                timeframe="current", metadata={"hist": macd_hist}
            ))

        # Bollinger Band squeeze/expansion
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle else 0
        if bb_width < 0.05:  # Very narrow bands
            patterns.append(DetectedPattern(
                name="Bollinger Band Squeeze", type=PatternType.INDICATOR,
                direction=PatternDirection.NEUTRAL, confidence=0.7,
                timeframe="current", metadata={"bb_width": bb_width}
            ))
        elif price >= bb_upper:
            patterns.append(DetectedPattern(
                name="Price at Upper Bollinger Band", type=PatternType.INDICATOR,
                direction=PatternDirection.BEARISH, confidence=0.65,
                timeframe="current", metadata={"price": price, "upper": bb_upper}
            ))
        elif price <= bb_lower:
            patterns.append(DetectedPattern(
                name="Price at Lower Bollinger Band", type=PatternType.INDICATOR,
                direction=PatternDirection.BULLISH, confidence=0.65,
                timeframe="current", metadata={"price": price, "lower": bb_lower}
            ))

        # Volume spike
        if avg_volume > 0 and volume / avg_volume > 2.0:
            patterns.append(DetectedPattern(
                name="Volume Spike", type=PatternType.INDICATOR,
                direction=PatternDirection.NEUTRAL, confidence=0.7,
                timeframe="current", metadata={"volume_ratio": volume/avg_volume}
            ))

        return [p for p in patterns if p.confidence >= self.min_confidence]

    # -------------------------------------------------------------------------
    # Multi-timeframe Confluence
    # -------------------------------------------------------------------------

    def analyze_confluence(
        self,
        patterns_by_timeframe: dict[str, list[DetectedPattern]]
    ) -> list[DetectedPattern]:
        """Find confluence across timeframes."""
        confluence_patterns = []

        # Count bullish/bearish signals per timeframe
        tf_signals = {}
        for tf, patterns in patterns_by_timeframe.items():
            bullish = sum(1 for p in patterns if p.direction == PatternDirection.BULLISH)
            bearish = sum(1 for p in patterns if p.direction == PatternDirection.BEARISH)
            tf_signals[tf] = {"bullish": bullish, "bearish": bearish, "total": len(patterns)}

        # Check alignment
        all_bullish = all(s["bullish"] > s["bearish"] for s in tf_signals.values())
        all_bearish = all(s["bearish"] > s["bullish"] for s in tf_signals.values())

        if all_bullish and len(tf_signals) >= 2:
            confluence_patterns.append(DetectedPattern(
                name="Multi-Timeframe Bullish Confluence",
                type=PatternType.CONFLUENCE,
                direction=PatternDirection.BULLISH,
                confidence=0.9,
                timeframe="confluence",
                metadata={"timeframes": tf_signals}
            ))
        elif all_bearish and len(tf_signals) >= 2:
            confluence_patterns.append(DetectedPattern(
                name="Multi-Timeframe Bearish Confluence",
                type=PatternType.CONFLUENCE,
                direction=PatternDirection.BEARISH,
                confidence=0.9,
                timeframe="confluence",
                metadata={"timeframes": tf_signals}
            ))

        return confluence_patterns

    def _slope(self, values: list[float]) -> float:
        """Simple linear regression slope."""
        n = len(values)
        if n < 2:
            return 0.0
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator != 0 else 0.0