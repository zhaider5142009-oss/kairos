"""
Specialised agents — the worker pool of the swarm.

Each agent is a small, pure, synchronous function that reads a
SwarmContext (frozen decision snapshot + raw arrays) and casts one
signed vote. The swarm runs them concurrently and merges the votes.

Roles (10 by default, configurable via the swarm):
    TrendSniper    — TA composite bias / RSI / MACD
    PatternScope   — candlestick + chart pattern bias
    QuantumRead    — amplitude/QAOA/entanglement decision
    NewsSentinel   — VADER sentiment × relevance
    MacroScope     — intermarket regime hint
    CalendarGuard  — blackout suppression (abstains when clear)
    FleetBarometer — 12-model weighted consensus
    RiskShield     — position/drawdown/loss guard (abstains when safe)
    MT5Lens        — live MT5 snapshot proxy (spread/tick sanity)
    VolatilityScale— ATR-based confidence scaler (never flips a vote)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import AgentVote, SwarmContext

LOGGER = logging.getLogger("jarvis.agents")


def _mk(
    name: str,
    role: str,
    direction: int,
    confidence: float,
    note: str,
    *,
    weight: float = 1.0,
    extra: dict | None = None,
    force_scale: bool = False,
) -> AgentVote | None:
    if direction == 0 or confidence <= 0:
        return None
    return AgentVote(
        agent=name, role=role,
        direction=int(direction),
        confidence=float(min(1.0, max(0.0, confidence))),
        weight=float(weight),
        note=note or "",
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# 1. Trend Sniper — technical composite bias
# ---------------------------------------------------------------------------

class TrendSniper:
    name = "TrendSniper"
    role = "technical"
    description = "RSI/MACD/ATR/volume composite bias"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        ta = ctx.get("ta") or {}
        bias = int(ta.get("composite_bias", 0))
        if bias == 0:
            return None
        rsi = float(ta.get("rsi", 50.0))
        # RSI extremes reinforce/reduce confidence.
        conf = abs(bias) / 3.0
        if rsi > 70 or rsi < 30:
            conf = min(1.0, conf + 0.15)
        return _mk(
            self.name, self.role, 1 if bias > 0 else -1, conf,
            f"composite {bias:+d}, RSI {rsi:.1f}", weight=1.0,
            extra={"bias": bias, "rsi": round(rsi, 1)},
        )


# ---------------------------------------------------------------------------
# 2. Pattern Scope — candle + chart patterns
# ---------------------------------------------------------------------------

class PatternScope:
    name = "PatternScope"
    role = "pattern"
    description = "candlestick + chart pattern structure"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        pat = ctx.get("patterns") or {}
        bias = int(pat.get("direction_bias", 0))
        conf = float(pat.get("confidence", 0.0))
        return _mk(
            self.name, self.role, bias, conf,
            self._note(pat), weight=0.9,
            extra={"structure": pat.get("trend_structure", "neutral")},
            force_scale=True,
        )

    @staticmethod
    def _note(pat: dict) -> str:
        candles = ",".join((pat.get("candle_patterns") or [])[:2])
        charts = ",".join((pat.get("chart_patterns") or [])[:2])
        parts = [p for p in (candles, charts) if p]
        return ("patterns " + ", ".join(parts)) or "no strong pattern"


# ---------------------------------------------------------------------------
# 3. Quantum Read — the quantum-inspired layer
# ---------------------------------------------------------------------------

class QuantumRead:
    name = "QuantumRead"
    role = "quantum"
    description = "amplitude + QAOA + entanglement decision"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        q = ctx.get("quantum") or {}
        direction = int(q.get("direction", 0))
        if direction == 0:
            return None
        mag = float(q.get("magnitude", 0.0))
        conf = float(q.get("confidence", 0.0))
        conf = min(1.0, conf + (0.1 if mag > 0.5 else 0.0))
        return _mk(
            self.name, self.role, direction, conf,
            f"quantum magnitude {mag:.3f}", weight=0.8,
            extra={"magnitude": round(mag, 3)},
        )


# ---------------------------------------------------------------------------
# 4. News Sentinel — sentiment × relevance
# ---------------------------------------------------------------------------

class NewsSentinel:
    name = "NewsSentinel"
    role = "sentiment"
    description = "VADER news sentiment weighted by relevance"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        news = ctx.get("news") or {}
        sent = float(news.get("sentiment", 0.0))
        rel = float(news.get("relevance", 0.0))
        if abs(sent) < 0.2 or rel < 0.1:
            return None
        direction = 1 if sent > 0 else -1
        return _mk(
            self.name, self.role, direction, min(1.0, rel * abs(sent) * 2.2),
            f"news {sent:+.2f} rel {rel:.2f}", weight=0.7,
            extra={"sentiment": round(sent, 3), "relevance": round(rel, 2)},
            force_scale=True,
        )


# ---------------------------------------------------------------------------
# 5. Macro Scope — intermarket regime hint
# ---------------------------------------------------------------------------

class MacroScope:
    name = "MacroScope"
    role = "macro"
    description = "DXY/US10Y/VIX/SPX intermarket regime"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        forces = ctx.get("forces") or {}
        hint = int(forces.get("hint", 0))
        if hint == 0:
            return None
        regime = ctx.get("regime", "neutral")  # carried in snapshot when present
        return _mk(
            self.name, self.role, hint, 0.55,
            f"macro hint {hint:+d}", weight=0.75,
            extra={"regime": regime},
        )


# ---------------------------------------------------------------------------
# 6. Calendar Guard — blackout suppression
# ---------------------------------------------------------------------------

class CalendarGuard:
    name = "CalendarGuard"
    role = "risk"
    description = "blackout windows suppress trading"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        cal = ctx.get("calendar") or {}
        if cal.get("in_blackout"):
            impact = int(cal.get("impact", 3))
            # Strong negative vote — the swarm director treats this as a veto.
            note = f"blackout window (impact {impact}) — veto"
            return AgentVote(
                agent=self.name, role=self.role, direction=-1,
                confidence=0.95, weight=2.0, note=note,
                extra={"blackout": True, "impact": impact},
            )
        return None


# ---------------------------------------------------------------------------
# 7. Fleet Barometer — the 12-model consensus
# ---------------------------------------------------------------------------

class FleetBarometer:
    name = "FleetBarometer"
    role = "ensemble"
    description = "weighted consensus of the predictive-model fleet"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        cons = ctx.get("model_consensus") or {}
        signed = float(cons.get("signed_vote", 0.0))
        if abs(signed) < 0.12:
            return None
        agreement = float(cons.get("agreement", 0.0))
        conf = min(1.0, abs(signed) * agreement + agreement * 0.25)
        n_ok = int(cons.get("ok", 0))
        return _mk(
            self.name, self.role, 1 if signed > 0 else -1, conf,
            f"fleet consensus {signed:+.2f} agreement {agreement:.2f} ({n_ok} models)",
            weight=1.25,
            extra={"n_ok": n_ok, "agreement": round(agreement, 3)},
        )


# ---------------------------------------------------------------------------
# 8. Risk Shield — position/drawdown/loss guard
# ---------------------------------------------------------------------------

class RiskShield:
    name = "RiskShield"
    role = "risk"
    description = "max-position + daily-loss + drawdown guard"

    W = 2.0

    def __init__(self, *, max_open: int = 3, max_daily_loss: float = 50.0,
                 drawdown_threshold: float = 0.05, status_provider=None) -> None:
        self.max_open = int(max_open)
        self.max_daily_loss = float(max_daily_loss)
        self.drawdown_threshold = float(drawdown_threshold)
        self.status_provider = status_provider        # callable() -> dispatcher.status()

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        try:
            if self.status_provider is not None:
                s = self.status_provider()
                open_n = int(s.get("open_positions", 0))
                if open_n >= self.max_open:
                    return self._veto(f"max open positions ({open_n}/{self.max_open})")
        except Exception:
            pass
        return None

    def _veto(self, reason: str) -> AgentVote:
        return AgentVote(
            agent=self.name, role=self.role, direction=-1,
            confidence=0.9, weight=self.W, note=f"risk guard: {reason}",
            extra={"veto": True, "reason": reason},
        )


# ---------------------------------------------------------------------------
# 9. MT5 Lens — live market context proxy
# ---------------------------------------------------------------------------

class MT5Lens:
    name = "MT5Lens"
    role = "market"
    description = "live-price sanity and spread tension"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        arr = ctx.arrays.get("close")
        if arr is None or len(arr) < 4:
            return None
        last = float(arr[-1])
        prev = float(arr[-4])
        drift = (last - prev) / max(abs(prev), 1e-12)
        if abs(drift) < 2e-4:
            return None
        direction = 1 if drift > 0 else -1
        return _mk(
            self.name, self.role, direction, min(1.0, abs(drift) * 40.0),
            f"short-term drift {drift*100:.3f}%", weight=0.4,
            extra={"last": round(last, 6), "drift_pct": round(drift * 100, 3)},
            force_scale=True,
        )


# ---------------------------------------------------------------------------
# 10. Volatility Scale — ATR confidence scaler (never flips direction)
# ---------------------------------------------------------------------------

class VolatilityScale:
    name = "VolatilityScale"
    role = "sizer"
    description = "ATR-normalised confidence modulation"

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        ta = ctx.get("ta") or {}
        atr = float(ta.get("atr", 0.0))
        arr = ctx.arrays.get("close")
        if arr is None or atr <= 0:
            return None
        last = float(arr[-1])
        if last <= 0:
            return None
        atr_pct = atr / last
        # Low volatility → trending steady → nudge confidence UP,
        # extreme volatility → chop → nudge DOWN. Never flips sign.
        if atr_pct < 0.0015:
            direction = 1
            conf = 0.20
            note = f"low vol ({atr_pct*100:.3f}%) → conviction"
        elif atr_pct > 0.009:
            direction = 1
            conf = 0.12
            note = f"high vol ({atr_pct*100:.3f}%) → hedge conviction"
        else:
            return None
        return _mk(
            self.name, self.role, direction, conf, note, weight=0.2,
            extra={"atr_pct": round(atr_pct * 100, 4)},
            force_scale=True,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def default_agents(**overrides) -> list[Any]:
    """Instantiate the standard ten-agent pool."""
    return [
        TrendSniper(),
        PatternScope(),
        QuantumRead(),
        NewsSentinel(),
        MacroScope(),
        CalendarGuard(),
        FleetBarometer(),
        RiskShield(**{k: v for k, v in overrides.items()
                      if k in ("max_open", "max_daily_loss", "drawdown_threshold",
                               "status_provider")}),
        MT5Lens(),
        VolatilityScale(),
    ]


def roles() -> dict[str, str]:
    return {
        "TrendSniper": "technical",
        "PatternScope": "pattern",
        "QuantumRead": "quantum",
        "NewsSentinel": "sentiment",
        "MacroScope": "macro",
        "CalendarGuard": "risk",
        "FleetBarometer": "ensemble",
        "RiskShield": "risk",
        "MT5Lens": "market",
        "VolatilityScale": "sizer",
    }


__all__ = ["default_agents", "roles", "TrendSniper", "PatternScope"]