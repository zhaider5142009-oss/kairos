"""
Multi-timeframe hierarchical predictor.

The most robust trading signals come from aligning multiple timeframes:
    - Higher timeframe (HTF): H4 / D1 / W1 → trend direction (long horizon)
    - Lower timeframe (LTF): M1 / M5 → entry timing (short horizon)
    - Mid timeframe (MTF): M15 / H1 → confirmation (mid horizon)

This module runs the full Jarvis pipeline at every configured timeframe
in parallel and combines the forecasts hierarchically.

Algorithm:
    1. For each TF in [M1, M5, M15, H1, H4, D1]:
        a. Pull N bars at that TF
        b. Run indicators + patterns + 12-model fleet
        c. Get a directional verdict + confidence for that TF's horizon
    2. Combine via "agreement pyramid":
        - All HTFs agree on direction → strong trend → boost confidence
        - HTFs disagree → choppy market → demote
        - LTF agrees with HTF → valid entry → stage
        - LTF disagrees with HTF → counter-trend → suppress
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .calendar import EconomicCalendar
from .core import TimesFMEngine
from .data_pipeline import DataPipeline
from .ensemble import EnsembleFusion, MarketContext
from .executor import TradeExecutor
from .indicators import TechnicalIndicators
from .market_forces import MarketForcesService
from .models import ModelForecast
from .mt5_bridge import MT5Bridge, JarvisMT5Error
from .news import NewsService
from .parallel import ParallelFleet
from .patterns import detect_patterns
from .quantum import QuantumDecisionEngine

LOGGER = logging.getLogger("jarvis.multi_tf")


# TF hierarchy — lower index = lower timeframe = more granular.
DEFAULT_TIMEFRAMES: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1"]
_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400,
    "D1": 86400, "W1": 604800, "MN1": 2592000,
}

# Hierarchy weights — higher TFs dominate for trend, lower for timing.
# (tf_name, weight_for_trend)
HTF_SET = {"H4": 0.3, "D1": 0.4, "W1": 0.3}      # weights sum to 1.0
LTF_SET = {"M1": 0.3, "M5": 0.4, "M15": 0.3}      # weights sum to 1.0
HTF_HORIZON = {"H4": 32, "D1": 16, "W1": 8}       # bars to forecast
LTF_HORIZON = {"M1": 16, "M5": 16, "M15": 16}
CONF_BOOST_AGREEMENT = 1.5    # multiplier when HTF + LTF agree
CONF_DAMP_DISAGREE = 0.3      # multiplier when HTF + LTF disagree


@dataclass
class TFForecast:
    timeframe: str
    horizon: int
    fusion_direction: int
    fusion_confidence: float
    expected_return_pct: float
    ensemble_components: dict = field(default_factory=dict)
    models: list[ModelForecast] = field(default_factory=list)


@dataclass
class HierarchicalDecision:
    symbol: str
    tf_forecasts: dict[str, TFForecast]
    trend_direction: int          # direction agreed by HTF
    trend_confidence: float       # confidence from HTF aggregate
    timing_direction: int         # direction from LTF
    final_direction: int          # combined
    final_confidence: float
    agreement: bool               # HTF + LTF agree?
    regime: str = "neutral"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "tf_forecasts": {
                tf: {
                    "horizon": f.horizon,
                    "fusion_direction": f.fusion_direction,
                    "fusion_confidence": round(f.fusion_confidence, 3),
                    "expected_return_pct": round(f.expected_return_pct, 3),
                }
                for tf, f in self.tf_forecasts.items()
            },
            "trend_direction": self.trend_direction,
            "trend_confidence": round(self.trend_confidence, 3),
            "timing_direction": self.timing_direction,
            "final_direction": self.final_direction,
            "final_confidence": round(self.final_confidence, 3),
            "agreement": self.agreement,
            "regime": self.regime,
        }


class MultiTimeframePredictor:
    """Runs the Jarvis pipeline at multiple timeframes, fuses hierarchically."""

    def __init__(
        self,
        *,
        bridge: MT5Bridge,
        engine: TimesFMEngine,
        indicators: TechnicalIndicators,
        quantum: QuantumDecisionEngine,
        fusion: EnsembleFusion,
        parallel: ParallelFleet,
        news: NewsService,
        forces: MarketForcesService,
        calendar: EconomicCalendar,
        executor: TradeExecutor,
        data: DataPipeline,
        timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    ) -> None:
        self.bridge = bridge
        self.engine = engine
        self.indicators = indicators
        self.quantum = quantum
        self.fusion = fusion
        self.parallel = parallel
        self.news = news
        self.forces = forces
        self.calendar = calendar
        self.executor = executor
        self.data = data
        self.timeframes = list(timeframes)

    async def predict_async(self, symbol: str) -> HierarchicalDecision:
        """Run every TF in parallel and combine hierarchically."""
        tasks = [self._predict_one_tf(symbol, tf) for tf in self.timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tf_forecasts: dict[str, TFForecast] = {}
        for tf, res in zip(self.timeframes, results):
            if isinstance(res, Exception):
                LOGGER.debug("TF %s prediction failed: %s", tf, res)
                continue
            tf_forecasts[tf] = res

        return self._combine(symbol, tf_forecasts)

    async def _predict_one_tf(self, symbol: str, timeframe: str) -> TFForecast:
        loop = asyncio.get_running_loop()
        # Pull candles at this TF.
        try:
            df = await loop.run_in_executor(
                None, self._rates_for_tf, symbol, timeframe,
            )
        except JarvisMT5Error as e:
            raise RuntimeError(f"rates({symbol}@{timeframe}) failed: {e}") from e
        if df is None or df.empty or len(df) < 64:
            raise RuntimeError(f"insufficient bars for {symbol}@{timeframe}")

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64)
        tick_vol = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else None

        # Determine horizon for this TF.
        horizon = HTF_HORIZON.get(timeframe) or LTF_HORIZON.get(timeframe, 16)

        # Indicators + patterns.
        indicators_snap = self.indicators.compute(
            open_=open_, high=high, low=low, close=close, volume=tick_vol)
        pattern_snap = detect_patterns(
            open_=open_, high=high, low=low, close=close,
            lookback_chart=min(60, len(close)))

        # TimesFM forecast at the TF horizon.
        forecast = await loop.run_in_executor(
            None, lambda: self.engine.forecast(close, horizon=horizon),
        )
        # The fleet forecasts at the same horizon — use a quick parallel run.
        try:
            model_forecasts = await self.parallel.forecast_all(
                close, horizon=horizon, symbol=symbol,
            )
        except Exception as e:
            LOGGER.debug("parallel fleet failed for %s@%s: %s", symbol, timeframe, e)
            model_forecasts = []

        # News / forces / calendar (shared across all TFs).
        news_sent = self.news.sentiment_for_symbol(symbol) if self.news.enabled else 0.0
        items = self.news.items_for_symbol(symbol) if self.news.enabled else []
        news_rel = min(1.0, len(items) / 5.0)
        i_hint = self.forces.force_direction_for(symbol) if self.forces.enabled else 0
        in_blackout, _ = self.calendar.is_blackout_now()
        cal_impact = self.calendar.impact_for_symbol(symbol)

        # Quantum decision (compact).
        q = self.quantum.decide(
            per_asset_signal=[
                float(forecast.direction) * forecast.confidence,
                indicators_snap.composite_bias / 3.0,
                float(pattern_snap.direction_bias) * pattern_snap.confidence,
            ],
            per_asset_phase=[0, 0, 0],
            correlation_matrix=np.eye(3),
        )

        ctx = MarketContext(
            symbol=symbol, forecast=forecast, indicators=indicators_snap,
            quantum=q, pattern=pattern_snap,
            news_sentiment=news_sent, news_relevance=news_rel,
            intermarket_hint=i_hint, calendar_impact=cal_impact,
            in_blackout=in_blackout, model_forecasts=model_forecasts,
        )
        fusion = self.fusion.fuse(ctx)

        return TFForecast(
            timeframe=timeframe, horizon=horizon,
            fusion_direction=fusion.direction,
            fusion_confidence=fusion.confidence,
            expected_return_pct=forecast.expected_return_pct,
            ensemble_components=fusion.components,
            models=model_forecasts,
        )

    def _combine(self, symbol: str, tf_forecasts: dict[str, TFForecast]) -> HierarchicalDecision:
        """Hierarchical fusion across timeframes."""
        # Aggregate HTF trend.
        htf_signed = 0.0
        htf_conf = 0.0
        htf_n = 0
        for tf, f in tf_forecasts.items():
            if tf in HTF_SET:
                w = HTF_SET[tf]
                htf_signed += w * float(f.fusion_direction) * f.fusion_confidence
                htf_conf += w * f.fusion_confidence
                htf_n += 1
        # Aggregate LTF timing.
        ltf_signed = 0.0
        ltf_conf = 0.0
        ltf_n = 0
        for tf, f in tf_forecasts.items():
            if tf in LTF_SET:
                w = LTF_SET[tf]
                ltf_signed += w * float(f.fusion_direction) * f.fusion_confidence
                ltf_conf += w * f.fusion_confidence
                ltf_n += 1
        trend_direction = 1 if htf_signed > 0 else (-1 if htf_signed < 0 else 0)
        trend_confidence = min(1.0, abs(htf_signed))
        timing_direction = 1 if ltf_signed > 0 else (-1 if ltf_signed < 0 else 0)
        timing_confidence = min(1.0, abs(ltf_signed))
        agreement = (trend_direction != 0
                     and timing_direction != 0
                     and trend_direction == timing_direction)
        # Final.
        if agreement:
            final_direction = trend_direction
            final_confidence = min(1.0, trend_confidence * timing_confidence * CONF_BOOST_AGREEMENT)
        else:
            final_direction = 0
            final_confidence = min(1.0, trend_confidence * CONF_DAMP_DISAGREE)
        regime = self.forces.regime() if self.forces else "neutral"
        return HierarchicalDecision(
            symbol=symbol,
            tf_forecasts=tf_forecasts,
            trend_direction=trend_direction,
            trend_confidence=trend_confidence,
            timing_direction=timing_direction,
            final_direction=final_direction,
            final_confidence=final_confidence,
            agreement=agreement,
            regime=regime,
        )

    # ----- sync wrapper for non-async callers --------------------------

    def predict(self, symbol: str) -> HierarchicalDecision:
        return asyncio.run(self.predict_async(symbol))

    def _rates_for_tf(self, symbol: str, timeframe: str):
        # Temporarily swap timeframe on the bridge.
        original = self.bridge.timeframe_str
        try:
            self.bridge.timeframe_str = timeframe
            return self.bridge.rates(symbol)
        finally:
            self.bridge.timeframe_str = original


__all__ = [
    "MultiTimeframePredictor", "HierarchicalDecision", "TFForecast",
    "DEFAULT_TIMEFRAMES",
]
