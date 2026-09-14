"""
Signal generator — full feature vector.

TradeSignal now carries the entire provenance of the decision so the user
(tray, HTTP, voice, logs) can see exactly which source voted what.

Backwards compatible: all existing fields preserved; new fields added.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import JarvisConfig
from .ensemble import FusionResult, MarketContext
from .indicators import IndicatorSnapshot
from .patterns import PatternSnapshot
from .quantum import QuantumDecision

LOGGER = logging.getLogger("jarvis.signals")


@dataclass(frozen=True)
class TradeSignal:
    signal_id: str
    symbol: str
    direction: int              # -1 (sell), 0 (hold), +1 (buy)
    confidence: float           # 0..1
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    r_multiple: float
    timestamp: float
    timestamp_iso: str

    # Full provenance
    fusion: dict
    indicators: dict
    timesfm: dict
    quantum: dict | None
    patterns: dict | None
    news: dict
    intermarket: dict
    calendar: dict
    regime: str = "neutral"
    # NEW: per-model forecasts (every model that ran)
    models: list[dict] = field(default_factory=list)
    model_consensus: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.confidence > 0

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "current_price": round(self.current_price, 6),
            "entry_price": round(self.entry_price, 6),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
            "atr": round(self.atr, 6),
            "r_multiple": round(self.r_multiple, 3),
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "regime": self.regime,
            "fusion": self.fusion,
            "indicators": self.indicators,
            "timesfm": self.timesfm,
            "quantum": self.quantum,
            "patterns": self.patterns,
            "news": self.news,
            "intermarket": self.intermarket,
            "calendar": self.calendar,
            "models": self.models,
            "model_consensus": self.model_consensus,
        }


class SignalGenerator:
    """Stateless. Pass in MarketContext + fusion + current_price, get a TradeSignal."""

    def __init__(
        self,
        *,
        sl_atr_multiple: float = 1.5,
        tp_atr_multiple: float = 2.5,
    ) -> None:
        self.sl_atr_multiple = max(0.1, float(sl_atr_multiple))
        self.tp_atr_multiple = max(0.1, float(tp_atr_multiple))

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "SignalGenerator":
        s = config.section("mt5", "order")
        return cls(
            sl_atr_multiple=s["sl_atr_multiple"],
            tp_atr_multiple=s["tp_atr_multiple"],
        )

    def generate(
        self,
        *,
        symbol: str,
        fusion: FusionResult,
        ctx: MarketContext,
        current_price: float,
    ) -> TradeSignal:
        atr = max(0.0, ctx.indicators.atr)
        sl_distance = atr * self.sl_atr_multiple
        tp_distance = atr * self.tp_atr_multiple
        r_mult = (tp_distance / sl_distance) if sl_distance > 0 else 0.0

        if fusion.direction == 1:
            entry = float(current_price)
            sl = float(current_price - sl_distance)
            tp = float(current_price + tp_distance)
        elif fusion.direction == -1:
            entry = float(current_price)
            sl = float(current_price + sl_distance)
            tp = float(current_price - tp_distance)
        else:
            entry = float(current_price)
            sl = float(current_price)
            tp = float(current_price)

        now = time.time()
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        return TradeSignal(
            signal_id=str(uuid.uuid4()),
            symbol=symbol,
            direction=fusion.direction,
            confidence=fusion.confidence,
            current_price=float(current_price),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            atr=atr,
            r_multiple=float(r_mult),
            timestamp=now,
            timestamp_iso=iso,
            fusion=fusion.to_dict(),
            indicators=ctx.indicators.to_dict(),
            timesfm=ctx.forecast.to_dict(),
            quantum=ctx.quantum.to_dict() if ctx.quantum else None,
            patterns=ctx.pattern.to_dict() if ctx.pattern else None,
            news={
                "sentiment": round(ctx.news_sentiment, 3),
                "relevance": round(ctx.news_relevance, 3),
            },
            intermarket={
                "hint": ctx.intermarket_hint,
            },
            calendar={
                "in_blackout": ctx.in_blackout,
                "impact": ctx.calendar_impact,
            },
            regime=fusion.regime,
            models=[f.to_dict() for f in ctx.model_forecasts],
            model_consensus=fusion.model_consensus,
        )


__all__ = ["SignalGenerator", "TradeSignal"]
