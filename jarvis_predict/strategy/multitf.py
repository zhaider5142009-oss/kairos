"""
Multi-timeframe analysis engine for JARVIS Reborn.
Analyzes confluence across M1, M5, M15, M30, H1, H4, D1 timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math


class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    def seconds(self) -> int:
        mapping = {
            Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M15: 900,
            Timeframe.M30: 1800, Timeframe.H1: 3600, Timeframe.H4: 14400,
            Timeframe.D1: 86400, Timeframe.W1: 604800, Timeframe.MN1: 2592000
        }
        return mapping[self]

    def higher(self) -> 'Timeframe | None':
        order = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
                 Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1, Timeframe.MN1]
        try:
            idx = order.index(self)
            return order[idx + 1] if idx + 1 < len(order) else None
        except ValueError:
            return None


@dataclass
class TFAnalysis:
    timeframe: Timeframe
    trend: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    rsi: float
    macd_hist: float
    ema_fast: float
    ema_slow: float
    adx: float  # Trend strength
    atr: float  # Volatility
    patterns: list[str] = field(default_factory=list)
    support: list[float] = field(default_factory=list)
    resistance: list[float] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ConfluenceResult:
    overall_trend: str
    overall_strength: float
    timeframe_alignment: dict[str, str]  # tf -> trend
    weighted_score: float  # -1 to 1
    entry_quality: str  # "A+", "A", "B", "C", "D"
    risk_reward_ratio: float
    key_levels: dict[str, list[float]]
    recommendation: str  # "long", "short", "wait", "hedge"
    details: dict[str, Any] = field(default_factory=dict)


class MultiTimeframeAnalyzer:
    """Analyzes market structure across multiple timeframes for confluence."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.timeframes = self.config.get("timeframes", [
            Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1
        ])
        self.weights = self.config.get("weights", {
            Timeframe.M15: 0.15, Timeframe.H1: 0.25,
            Timeframe.H4: 0.30, Timeframe.D1: 0.30
        })
        self.min_alignment = self.config.get("min_alignment", 0.65)

    def analyze(
        self,
        tf_data: dict[Timeframe, dict[str, Any]],
        current_price: float,
        account_equity: float,
        open_positions: list[dict]
    ) -> ConfluenceResult:
        """
        Main entry point. tf_data contains per-timeframe indicator data.
        Returns comprehensive confluence analysis.
        """
        analyses = {}
        for tf in self.timeframes:
            if tf in tf_data:
                analyses[tf] = self._analyze_timeframe(tf, tf_data[tf])

        # Calculate weighted confluence
        alignment = {tf.value: a.trend for tf, a in analyses.items()}
        weighted_score = self._calculate_weighted_score(analyses)
        overall_trend = self._determine_overall_trend(weighted_score)
        overall_strength = abs(weighted_score)

        # Key levels from all timeframes
        key_levels = self._aggregate_key_levels(analyses, current_price)

        # Risk/reward estimation
        rr_ratio = self._estimate_risk_reward(analyses, current_price, key_levels)

        # Entry quality
        entry_quality = self._grade_entry(analyses, weighted_score, rr_ratio)

        # Recommendation
        recommendation = self._make_recommendation(
            overall_trend, overall_strength, entry_quality,
            account_equity, open_positions, current_price, key_levels
        )

        return ConfluenceResult(
            overall_trend=overall_trend,
            overall_strength=overall_strength,
            timeframe_alignment=alignment,
            weighted_score=weighted_score,
            entry_quality=entry_quality,
            risk_reward_ratio=rr_ratio,
            key_levels=key_levels,
            recommendation=recommendation,
            details={
                "analyses": {tf.value: a.__dict__ for tf, a in analyses.items()},
                "weights_used": {tf.value: w for tf, w in self.weights.items()},
                "alignment_score": self._alignment_score(analyses)
            }
        )

    def _analyze_timeframe(self, tf: Timeframe, data: dict) -> TFAnalysis:
        """Analyze a single timeframe's data."""
        rsi = data.get("rsi", 50)
        macd_hist = data.get("macd_hist", 0)
        ema_fast = data.get("ema_fast", data.get("close", 0))
        ema_slow = data.get("ema_slow", data.get("close", 0))
        adx = data.get("adx", 20)
        atr = data.get("atr", 0)
        patterns = data.get("patterns", [])
        support = data.get("support", [])
        resistance = data.get("resistance", [])

        # Trend determination
        trend_score = 0
        if ema_fast > ema_slow:
            trend_score += 1
        if macd_hist > 0:
            trend_score += 1
        if rsi > 55:
            trend_score += 0.5
        elif rsi < 45:
            trend_score -= 0.5

        if trend_score >= 1.5:
            trend = "bullish"
        elif trend_score <= -1.5:
            trend = "bearish"
        else:
            trend = "neutral"

        # Strength from ADX and trend alignment
        strength = min(1.0, (adx / 50) * (abs(trend_score) / 2))

        return TFAnalysis(
            timeframe=tf,
            trend=trend,
            strength=strength,
            rsi=rsi,
            macd_hist=macd_hist,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            adx=adx,
            atr=atr,
            patterns=patterns,
            support=support,
            resistance=resistance,
            timestamp=data.get("timestamp", 0)
        )

    def _calculate_weighted_score(self, analyses: dict[Timeframe, TFAnalysis]) -> float:
        """Calculate weighted trend score across timeframes."""
        total_weight = 0
        weighted_sum = 0

        for tf, analysis in analyses.items():
            weight = self.weights.get(tf, 0)
            if weight == 0:
                continue

            # Convert trend to numeric
            trend_val = 1 if analysis.trend == "bullish" else (-1 if analysis.trend == "bearish" else 0)
            weighted_sum += trend_val * analysis.strength * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _determine_overall_trend(self, score: float) -> str:
        if score >= 0.4:
            return "strongly_bullish"
        elif score >= 0.15:
            return "bullish"
        elif score <= -0.4:
            return "strongly_bearish"
        elif score <= -0.15:
            return "bearish"
        return "neutral"

    def _alignment_score(self, analyses: dict[Timeframe, TFAnalysis]) -> float:
        """How aligned are the timeframes? 1 = perfectly aligned."""
        if not analyses:
            return 0
        trends = [1 if a.trend == "bullish" else (-1 if a.trend == "bearish" else 0) for a in analyses.values()]
        non_zero = [t for t in trends if t != 0]
        if not non_zero:
            return 0
        return abs(sum(non_zero)) / len(non_zero)

    def _aggregate_key_levels(self, analyses: dict, current_price: float) -> dict:
        """Aggregate support/resistance across timeframes."""
        all_support, all_resistance = [], []

        for analysis in analyses.values():
            all_support.extend(analysis.support)
            all_resistance.extend(analysis.resistance)

        # Cluster nearby levels
        def cluster(levels: list[float]) -> list[float]:
            if not levels:
                return []
            levels = sorted(set(round(l, 5) for l in levels))
            clusters = [[levels[0]]]
            for l in levels[1:]:
                if abs(l - clusters[-1][-1]) / clusters[-1][-1] < 0.003:
                    clusters[-1].append(l)
                else:
                    clusters.append([l])
            return [round(sum(c)/len(c), 5) for c in clusters]

        support = cluster(all_support)
        resistance = cluster(all_resistance)

        # Filter relevant levels near current price
        support = [s for s in support if s < current_price * 1.05][:5]
        resistance = [r for r in resistance if r > current_price * 0.95][:5]

        return {"support": support, "resistance": resistance}

    def _estimate_risk_reward(self, analyses: dict, price: float, levels: dict) -> float:
        """Estimate risk/reward based on nearest key levels."""
        support = levels.get("support", [])
        resistance = levels.get("resistance", [])

        nearest_support = max([s for s in support if s < price], default=price * 0.98)
        nearest_resistance = min([r for r in resistance if r > price], default=price * 1.02)

        # Use ATR for stop distance if available
        avg_atr = sum(a.atr for a in analyses.values()) / len(analyses) if analyses else 0
        atr_stop = price - avg_atr * 1.5 if avg_atr > 0 else nearest_support

        stop = max(nearest_support, atr_stop)
        target = nearest_resistance

        risk = price - stop
        reward = target - price

        return reward / risk if risk > 0 else 0

    def _grade_entry(self, analyses: dict, score: float, rr: float) -> str:
        """Grade entry quality A+ through D."""
        alignment = self._alignment_score(analyses)
        adx_avg = sum(a.adx for a in analyses.values()) / len(analyses) if analyses else 0

        grade_score = (
            abs(score) * 40 +          # Trend strength
            alignment * 30 +           # Timeframe alignment
            min(rr / 3, 1) * 20 +      # Risk/reward (capped at 3:1)
            min(adx_avg / 40, 1) * 10  # Trend strength via ADX
        )

        if grade_score >= 85: return "A+"
        elif grade_score >= 75: return "A"
        elif grade_score >= 65: return "B"
        elif grade_score >= 50: return "C"
        return "D"

    def _make_recommendation(
        self,
        trend: str,
        strength: float,
        entry_grade: str,
        equity: float,
        positions: list,
        price: float,
        levels: dict
    ) -> str:
        """Make trading recommendation."""
        max_positions = self.config.get("max_positions", 5)
        current_positions = len(positions)

        # If we have too many positions, don't add more unless hedging
        if current_positions >= max_positions:
            return "hedge" if self.config.get("hedge_enabled", False) else "wait"

        # Strong trend + good entry grade = trade
        if trend in ("strongly_bullish", "bullish") and strength > 0.3 and entry_grade in ("A+", "A", "B"):
            return "long"
        elif trend in ("strongly_bearish", "bearish") and strength > 0.3 and entry_grade in ("A+", "A", "B"):
            return "short"

        # Neutral or weak = wait
        return "wait"


def create_default_analyzer() -> MultiTimeframeAnalyzer:
    """Create analyzer with default JARVIS Reborn settings."""
    return MultiTimeframeAnalyzer({
        "timeframes": [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "weights": {
            Timeframe.M15: 0.15,
            Timeframe.H1: 0.25,
            Timeframe.H4: 0.30,
            Timeframe.D1: 0.30
        },
        "min_alignment": 0.65,
        "max_positions": 5,
        "hedge_enabled": True
    })