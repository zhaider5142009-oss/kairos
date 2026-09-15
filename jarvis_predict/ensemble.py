"""
Multi-source ensemble fusion.

Sources combined (each contributes a [-1, +1] signed vote + [0,1] confidence):
    1. Models     — N independent predictive models (TimesFM, Chronos, qlib-*,
                    FinGPT, FinRobot, …). Aggregated as weighted consensus,
                    with per-model provenance preserved.
    2. TA         — RSI / MACD / ATR / volume composite bias
    3. Quantum    — amplitude-encoded + QAOA-optimised + entangled vote
    4. News       — VADER sentiment across relevant recent articles
    5. Intermarket— DXY, US10Y, VIX, SPX regime hints
6. Patterns   — chart + candlestick pattern composite bias
     7. Calendar   — blackout + impact-aware volatility expectation
     8. Desk       — commodity research desk (gold/metals/energy macro panel)

Final vote:
    vote     = Σ (weight_i × source_i_signed × source_i_confidence)
    direction = sign(vote) when |vote| >= min_confidence, else HOLD
    confidence = |vote|

Weights are regime-adaptive: in a "risk_off" regime we double-count
intermarket + patterns; during FOMC blackouts we suppress everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import JarvisConfig
from .core import ForecastResult
from .indicators import IndicatorSnapshot
from .market_forces import MarketForcesService
from .models import ModelForecast
from .news import NewsService
from .patterns import PatternSnapshot
from .quantum import QuantumDecision

LOGGER = logging.getLogger("jarvis.ensemble")


@dataclass
class MarketContext:
    """Everything the ensemble needs to make one decision for one symbol."""

    symbol: str
    forecast: ForecastResult
    indicators: IndicatorSnapshot
    quantum: QuantumDecision | None = None
    pattern: PatternSnapshot | None = None
    news_sentiment: float = 0.0        # -1..+1
    news_relevance: float = 0.0        # 0..1 (how much we trust the news signal)
    intermarket_hint: int = 0          # -1/0/+1
    calendar_impact: int = 0           # 0..3
    in_blackout: bool = False
    model_forecasts: list[ModelForecast] = field(default_factory=list)
    desk_hint: Any = None              # DeskVote | None — commodity desk read


@dataclass
class FusionResult:
    direction: int
    confidence: float
    per_source: dict[str, dict] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    regime: str = "neutral"
    model_consensus: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "per_source": self.per_source,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "model_consensus": self.model_consensus,
        }


# Default weights (must sum to 1.0 in the neutral regime).
DEFAULT_WEIGHTS = {
    "models":     0.28,
    "ta":         0.14,
    "quantum":    0.14,
    "news":       0.08,
    "intermarket":0.09,
    "patterns":   0.09,
    "calendar":   0.10,
    "desk":       0.08,
}

# Regime-adaptive weight modifiers.
_RISK_OFF_BOOST = {"intermarket": 1.6, "patterns": 1.4, "news": 1.3}
_RISK_ON_BOOST = {"models": 1.2, "patterns": 1.2}
_NEWS_SPIKE_BOOST = {"news": 2.0}
_MODEL_AGREEMENT_BOOST = {"models": 1.5}     # bonus when models strongly agree


class EnsembleFusion:
    """Fuses N model consensus + 6 non-model sources into one directional decision."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        min_confidence: float = 0.45,
        news_service: NewsService | None = None,
        forces_service: MarketForcesService | None = None,
        model_multipliers: dict[str, float] | None = None,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.min_confidence = float(min_confidence)
        self.news_service = news_service
        self.forces_service = forces_service
        # Adaptive per-model weight multipliers (from ModelLearner). Set at
        # runtime; pure dict, keeps this class stateless otherwise.
        self.model_multipliers = dict(model_multipliers or {})

    def set_model_multipliers(self, multipliers: dict[str, float]) -> None:
        self.model_multipliers = dict(multipliers)

    @classmethod
    def from_config(
        cls,
        config: JarvisConfig,
        *,
        news_service: NewsService | None = None,
        forces_service: MarketForcesService | None = None,
    ) -> "EnsembleFusion":
        section = config.section("ensemble")
        return cls(
            weights=section.get("weights", DEFAULT_WEIGHTS),
            min_confidence=section.get("min_confidence", 0.45),
            news_service=news_service,
            forces_service=forces_service,
        )

    # --- main API ----------------------------------------------------------

    def fuse(self, ctx: MarketContext) -> FusionResult:
        per_source: dict[str, dict] = {}

        # 1. Models — consensus over every available ModelForecast.
        model_signed, model_conf, model_meta = self._aggregate_models(ctx.model_forecasts)
        per_source["models"] = {
            "signed": model_signed,
            "raw_consensus": model_signed,
            "confidence": model_conf,
            "count": len(ctx.model_forecasts),
            "available": sum(1 for f in ctx.model_forecasts if f.status == "ok"),
            "models": model_meta,
        }

        # 2. Technical indicators
        ta_signed = ctx.indicators.composite_bias / 3.0
        ta_conf = min(1.0, abs(ctx.indicators.composite_bias) / 3.0)
        per_source["ta"] = {
            "signed": ta_signed,
            "raw_composite": ctx.indicators.composite_bias,
            "confidence": ta_conf,
        }

        # 3. Quantum
        if ctx.quantum is not None:
            q_signed = float(ctx.quantum.direction) * ctx.quantum.confidence
            per_source["quantum"] = {
                "signed": q_signed,
                "raw_direction": ctx.quantum.direction,
                "confidence": ctx.quantum.confidence,
                "magnitude": ctx.quantum.magnitude,
            }
        else:
            per_source["quantum"] = {"signed": 0.0, "confidence": 0.0, "skipped": True}

        # 4. News
        n_signed = float(ctx.news_sentiment) * ctx.news_relevance
        per_source["news"] = {
            "signed": n_signed,
            "raw_sentiment": ctx.news_sentiment,
            "relevance": ctx.news_relevance,
            "confidence": ctx.news_relevance,
        }

        # 5. Intermarket
        i_signed = float(ctx.intermarket_hint)
        i_conf = 1.0 if ctx.intermarket_hint != 0 else 0.0
        per_source["intermarket"] = {
            "signed": i_signed,
            "raw_hint": ctx.intermarket_hint,
            "confidence": i_conf,
        }

        # 6. Patterns
        if ctx.pattern is not None:
            p_signed = float(ctx.pattern.direction_bias) * ctx.pattern.confidence
            per_source["patterns"] = {
                "signed": p_signed,
                "raw_bias": ctx.pattern.direction_bias,
                "confidence": ctx.pattern.confidence,
                "candle": ctx.pattern.candle_patterns,
                "chart": ctx.pattern.chart_patterns,
            }
        else:
            per_source["patterns"] = {"signed": 0.0, "confidence": 0.0, "skipped": True}

        # 7. Calendar — penalty (not a vote).
        cal_modifier = 0.0
        if ctx.in_blackout:
            cal_modifier = -1.0
        elif ctx.calendar_impact >= 3:
            cal_modifier = -0.5
        elif ctx.calendar_impact == 2:
            cal_modifier = -0.2
        per_source["calendar"] = {
            "signed": cal_modifier,
            "confidence": 1.0,
            "in_blackout": ctx.in_blackout,
            "impact": ctx.calendar_impact,
        }

        # 8. Commodities desk — gold/metals/energy macro panel.
        if ctx.desk_hint is not None and not ctx.desk_hint.stale:
            d_signed = float(ctx.desk_hint.direction) * ctx.desk_hint.confidence
            per_source["desk"] = {
                "signed": d_signed,
                "raw_direction": ctx.desk_hint.direction,
                "score": ctx.desk_hint.score,
                "confidence": ctx.desk_hint.confidence,
                "hedge_hint": ctx.desk_hint.hedge_hint,
                "narrative": ctx.desk_hint.narrative,
            }
        else:
            per_source["desk"] = {"signed": 0.0, "confidence": 0.0, "skipped": True}

        # --- Regime-adaptive weighting ---------------------------------------
        weights = self._adapt_weights(ctx, model_meta)

        # --- Weighted vote ---------------------------------------------------
        weighted = 0.0
        components: dict[str, float] = {}
        for src, info in per_source.items():
            if src == "calendar":
                continue
            w = weights.get(src, 0.0)
            contribution = w * info["signed"] * max(0.05, info["confidence"])
            weighted += contribution
            components[src] = contribution

        if cal_modifier < 0:
            weighted *= (1.0 + cal_modifier)
            components["calendar_penalty"] = weighted * cal_modifier

        confidence = min(1.0, abs(weighted))
        if confidence < self.min_confidence or cal_modifier <= -0.9:
            direction = 0
        else:
            direction = 1 if weighted > 0 else -1

        regime = self.forces_service.regime() if self.forces_service else "neutral"

        return FusionResult(
            direction=direction,
            confidence=confidence,
            per_source=per_source,
            components=components,
            regime=regime,
            model_consensus={
                "n": len(ctx.model_forecasts),
                "ok": sum(1 for f in ctx.model_forecasts if f.status == "ok"),
                "signed_vote": round(model_signed, 4),
                "agreement": model_meta.get("agreement", 0.0),
                "directions": model_meta.get("directions", {}),
            },
        )

    # --- helpers ------------------------------------------------------------

    def _aggregate_models(
        self, forecasts: list[ModelForecast]
    ) -> tuple[float, float, dict]:
        """Aggregate N ModelForecasts into a single signed vote + confidence.

        - signed:     Σ (direction * confidence * base_weight) / Σ weights, clamped [-1, 1]
        - confidence: agreement * average confidence, boosted when most models agree
        - meta:       dict of per-model contribution for provenance
        """
        if not forecasts:
            return 0.0, 0.0, {"directions": {}, "agreement": 0.0}

        signed = 0.0
        total_weight = 0.0
        confs: list[float] = []
        dirs: dict[str, int] = {}
        contribs: dict[str, float] = {}

        for f in forecasts:
            if f.status != "ok":
                continue
            # Skip "uncertain" forecasts (direction == 0 and conf near 0).
            base_w = max(0.0, f.confidence) + 0.05   # 0.05 floor so a vote always counts
            # Adaptive multiplier from ModelLearner (recent track record).
            mult = self.model_multipliers.get(f.model_name, 1.0)
            signed += float(f.direction) * f.confidence * base_w * mult
            total_weight += base_w * max(0.25, mult)
            confs.append(f.confidence)
            dirs[f.model_name] = f.direction
            contribs[f.model_name] = round(float(f.direction) * f.confidence * mult, 4)

        if total_weight <= 0:
            return 0.0, 0.0, {"directions": dirs, "agreement": 0.0, "contribs": contribs}

        signed = max(-1.0, min(1.0, signed / total_weight))

        # Agreement: fraction of models that agree on the dominant direction.
        if dirs:
            counts = {1: 0, -1: 0, 0: 0}
            for d in dirs.values():
                counts[d] = counts.get(d, 0) + 1
            total = sum(counts.values())
            dom = max(counts.values()) if total > 0 else 0
            agreement = dom / total
        else:
            agreement = 0.0

        # Confidence: agreement * mean confidence, boosted.
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        confidence = min(1.0, agreement * 0.6 + mean_conf * 0.4 + (0.2 if agreement >= 0.8 else 0.0))

        meta = {
            "directions": dirs,
            "agreement": round(agreement, 3),
            "mean_confidence": round(mean_conf, 3),
            "contribs": contribs,
        }
        return signed, confidence, meta

    def _adapt_weights(self, ctx: MarketContext, model_meta: dict) -> dict[str, float]:
        weights = dict(self.weights)
        if self.forces_service:
            regime = self.forces_service.regime()
            if regime == "risk_off":
                for k, mult in _RISK_OFF_BOOST.items():
                    weights[k] = weights.get(k, 0.0) * mult
            elif regime == "risk_on":
                for k, mult in _RISK_ON_BOOST.items():
                    weights[k] = weights.get(k, 0.0) * mult
        if ctx.news_relevance >= 0.7:
            for k, mult in _NEWS_SPIKE_BOOST.items():
                weights[k] = weights.get(k, 0.0) * mult
        # Model agreement bonus.
        if model_meta.get("agreement", 0.0) >= 0.7:
            for k, mult in _MODEL_AGREEMENT_BOOST.items():
                weights[k] = weights.get(k, 0.0) * mult
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights


__all__ = ["EnsembleFusion", "FusionResult", "MarketContext"]
