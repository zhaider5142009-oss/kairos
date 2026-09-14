"""
JARVIS‑Reborn strategy engine — ultra‑sophisticated MT5‑only algo trader.

Public API
----------
assess_market(forecast, rsi, macd, account, cfg)   → dict with trend, decision, hedge
should_enter(position, decision, risk)               → bool
should_exit(position, decision, equity_curve)        → bool (tp/sl/hedge)
StrategyEngine                                       → main orchestration
PatternRecognizer                                    → advanced pattern detection
MultiTimeframeAnalyzer                               → MTF confluence
PositionManager                                      → position sizing, hedging, risk
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Config dataclass — extend from your jarvis_predict.config if you wish
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Knobs the user can tweak (read from config.yaml or override at runtime)."""

    # ——— entry / exit ———————————————————————————————————————————————————
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ——— position sizing —————————————————————————————————————————————
    max_open_positions: int = 5
    max_daily_loss: float = 50.0        # % of equity; if exceeded → hedge
    max_drawdown_pct: float = 10.0      # hard stop

    # ——— hedge ———————————————————————————————————————————————————————
    hedge_enabled: bool = False
    hedge_threshold_pct: float = 5.0    # open opposite side when drawdown ≥ this

    # ——— time‑frame selection ———————————————————————————————————————
    use_mtf: bool = True                # aggregate M5 / M15 / H1 signals
    confidence_weight: float = 0.6      # how much TimesFM forecast drives decision


# ---------------------------------------------------------------------------
# Core assessment – pure functions, no jarvis imports (avoid cycles)
# ---------------------------------------------------------------------------

def _rsi_label(rsi: float, ov: float, os_: float) -> str | None:
    if rsi >= ov:
        return "overbought"
    if rsi <= os_:
        return "oversold"
    return None


def _macd_signal(fast: float, slow: float, sig: float) -> str | None:
    """Return 'bullish' if fast > slow and crossing up, etc. Simplified."""
    if fast > slow:
        return "bullish"
    if fast < slow:
        return "bearish"
    return None


def assess_market(
    *,
    forecast: float,                # TimesFM forecast sign: >0 bull, <0 bear, ≈0 neutral
    rsi: float,
    macd_fast: float,
    macd_slow: float,
    macd_signal: float,
    account_equity: float,
    account_drawdown_pct: float,
    cfg: StrategyConfig | None = None,
) -> dict[str, Any]:
    """Return a market‑assessment dict used by the decision layer."""

    cfg = cfg or StrategyConfig()

    rsi_status = _rsi_label(rsi, cfg.rsi_overbought, cfg.rsi_oversold)
    macd_status = _macd_signal(macd_fast, macd_slow, cfg.macd_signal)

    # ——— TimesFM bias —————————————————————————————————————————————
    fm_bull = forecast > 0
    fm_bear = forecast < 0

    # ——— Composite signal ———————————————————————————————————————
    # Weight forecast against technicals; default 60/40 split.
    score = 0.0
    if fm_bull:
        score += cfg.confidence_weight
    elif fm_bear:
        score -= cfg.confidence_weight

    if macd_status == "bullish":
        score += (1 - cfg.confidence_weight) * 0.5
    elif macd_status == "bearish":
        score -= (1 - cfg.confidence_weight) * 0.5

    if rsi_status == "overbought":
        score -= 0.2          # caution: price possibly too high
    elif rsi_status == "oversold":
        score += 0.2          # caution: price possibly too low

    # ——— Decision ——————————————————————————————————————————
    if score >= 0.3:
        decision = "buy"
    elif score <= -0.3:
        decision = "sell"
    else:
        decision = "hold"

    # ——— Hedge logic ———————————————————————————————————————
    hedge = bool(
        cfg.hedge_enabled
        and account_drawdown_pct >= cfg.hedge_threshold_pct
        and abs(score) > 0.1   # only hedge if we have a decent edge
    )

    # ——— Trend label for UI —————————————————————————————————
    if score > 0.5:
        trend = "strong bull"
    elif score > 0.1:
        trend = "bull"
    elif score < -0.5:
        trend = "strong bear"
    elif score < -0.1:
        trend = "bear"
    else:
        trend = "neutral"

    return {
        "forecast": forecast,
        "rsi": rsi,
        "rsi_status": rsi_status,
        "macd_status": macd_status,
        "score": round(score, 3),
        "decision": decision,
        "trend": trend,
        "hedge": hedge,
        "hedge_threshold_pct": cfg.hedge_threshold_pct,
        "max_daily_loss_pct": cfg.max_daily_loss,
    }


def should_enter(
    *,
    position_side: str | None,        # None, "long", "short", or "hedge"
    decision: str,                      # "buy"/"sell"/"hold" from assess_market
    hedge: bool,
    cfg: StrategyConfig,
) -> bool:
    """Return True if a new position (or hedge) should be opened."""

    # Max‑open‑positions guard
    if cfg.max_open_positions is not None:
        # (the caller must count open positions; here we just guard the concept)
        pass

    # If hedge is needed and we have capacity, allow it
    if hedge and position_side != "long" and position_side != "short":
        return True

    # Normal entry logic
    if decision == "buy" and position_side != "short":
        return True
    if decision == "sell" and position_side != "long":
        return True
    return False


def should_exit(
    *,
    position_side: str,
    decision: str,
    hedge: bool,
    entry_price: float,
    current_price: float,
    cfg: StrategyConfig,
    trailing_stop_pct: float | None = None,
) -> dict[str, Any]:
    """Return exit action: {'action': 'close'|'trail'|'hedge', 'reason': str, 'price': float}."""

    reason = ""
    action = "close"

    # ——— Hard drawdown stop —————————————————————————————————
    if cfg.max_drawdown_pct is not None:
        # (caller provides current drawdown %; simplified here)
        pass

    # ——— Take‑profit / stop‑loss based on price move —————————
    if position_side == "long":
        tp_price = entry_price * (1 + cfg.max_daily_loss / 100)  # placeholder: use config
        sl_price = entry_price * (1 - 2.0)                       # 2% initial SL placeholder
        if current_price >= tp_price:
            action = "take_profit"
            reason = "TP hit"
        elif current_price <= sl_price:
            action = "stop_loss"
            reason = "SL hit"
    elif position_side == "short":
        tp_price = entry_price * (1 - cfg.max_daily_loss / 100)
        sl_price = entry_price * (1 + 2.0)
        if current_price <= tp_price:
            action = "take_profit"
            reason = "TP hit"
        elif current_price >= sl_price:
            action = "stop_loss"
            reason = "SL hit"

    # ——— Hedge trigger —————————————————————————————————
    if hedge and action == "close":
        action = "hedge"
        reason = "drawdown hedge"

    return {"action": action, "reason": reason, "price": current_price}


# ---------------------------------------------------------------------------
# Re-export submodules
# ---------------------------------------------------------------------------
from .patterns import PatternRecognizer, Candle, DetectedPattern, PatternType, PatternDirection
from .multitf import MultiTimeframeAnalyzer, Timeframe, ConfluenceResult, create_default_analyzer
from .position_mgr import PositionManager, create_default_manager, Position, RiskMetrics, HedgeConfig, PositionSizingConfig
from .engine import StrategyEngine, StrategySignal, create_engine

__all__ = [
    "StrategyConfig",
    "assess_market",
    "should_enter",
    "should_exit",
    "PatternRecognizer",
    "Candle",
    "DetectedPattern",
    "PatternType",
    "PatternDirection",
    "MultiTimeframeAnalyzer",
    "Timeframe",
    "ConfluenceResult",
    "create_default_analyzer",
    "PositionManager",
    "create_default_manager",
    "Position",
    "RiskMetrics",
    "HedgeConfig",
    "PositionSizingConfig",
    "StrategyEngine",
    "StrategySignal",
    "create_engine",
]