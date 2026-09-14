"""
Meta-orchestrator — uses OpenRouter-hosted LLMs (DeepSeek-V3, V4.1,
GPT-4o, Claude, Llama) as the reasoning layer over all 12+ model outputs.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Per-symbol per-cycle, the runtime produces:                       │
    │   - 12+ ModelForecast objects (one per predictive model)          │
    │   - TA snapshot (RSI / MACD / ATR / volume)                       │
    │   - Pattern snapshot (candles + chart + S/R + pivots + fib)       │
    │   - QuantumDecision (amplitude + QAOA + entanglement)             │
    │   - News sentiment (RSS + VADER)                                   │
    │   - Intermarket hint (DXY/US10Y/VIX/SPX)                          │
    │   - Calendar impact + blackout flag                                │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ MetaOrchestrator.explain(payload)                                 │
    │   1. Serialises the above into a compact prompt.                  │
    │   2. If OpenRouter is configured → sends to the model chain       │
    │      (DeepSeek-V4.1 → GPT-4o → Claude → Gemini free).             │
    │   3. Else → falls back to the deterministic reasoner.              │
    └─────────────────────────────────────────────────────────────────┘

The OpenRouter path is the *production* path — it's the user's
"maximise token limit and intelligence" wiring. Each model can use up
to 2048 output tokens, with prompt context up to 128k for Claude / 200k
for Gemini / 64k for GPT-4o. Cost per call is tracked via the
OpenRouterClient.usage property.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("jarvis.meta")


@dataclass
class MetaVerdict:
    direction: int          # -1 / 0 / +1
    confidence: float       # 0..1
    reasoning: str
    source: str             # "openrouter:<model>" | "deterministic"


# ----- prompt construction ---------------------------------------------------

_SYSTEM_PROMPT = """You are a senior quantitative trader with 25 years of experience.
Your job is to evaluate the signals below and produce a precise directional verdict
on the next 16 bars of an instrument. Output EXACTLY these four lines, no preamble:

VERDICT: BUY | SELL | HOLD
CONFIDENCE: <0.00 to 1.00, two decimals>
EXPECTED_RETURN_PCT: <signed percent, e.g. +0.42 or -0.18>
REASONING: <one short sentence explaining the dominant factor>
"""


def _build_user_prompt(payload: dict) -> str:
    sym = payload.get("symbol", "?")
    parts = [f"Symbol: {sym}"]
    models = payload.get("models") or []
    if models:
        rows = []
        for m in models:
            if not isinstance(m, dict):
                continue
            rows.append(f"  - {m.get('model','?'):20s} dir={m.get('direction',0):+d} conf={m.get('confidence',0):.2f} status={m.get('status','?')}")
        parts.append("Predictive models:\n" + "\n".join(rows))
    fusion = payload.get("fusion") or {}
    if fusion:
        parts.append(f"Ensemble: dir={fusion.get('direction',0):+d} conf={fusion.get('confidence',0):.2f} regime={fusion.get('regime','neutral')}")
    ta = payload.get("ta") or {}
    if ta:
        parts.append(f"Technical: composite_bias={ta.get('composite_bias',0)} ATR={ta.get('atr',0):.5f} RSI={ta.get('rsi',0):.1f}")
    quantum = payload.get("quantum") or {}
    if quantum:
        parts.append(f"Quantum: dir={quantum.get('direction',0):+d} conf={quantum.get('confidence',0):.2f} magnitude={quantum.get('magnitude',0):.3f}")
    patterns = payload.get("patterns") or {}
    if patterns:
        parts.append(f"Patterns: trend={patterns.get('trend_structure','neutral')} candles={patterns.get('candle_patterns',[])} charts={patterns.get('chart_patterns',[])}")
    news = payload.get("news") or {}
    if news:
        parts.append(f"News: sentiment={news.get('sentiment',0):+.2f} relevance={news.get('relevance',0):.2f}")
    forces = payload.get("forces") or {}
    if forces:
        parts.append(f"Macro: hint={forces.get('hint',0):+d}")
    calendar = payload.get("calendar") or {}
    if calendar:
        parts.append(f"Calendar: blackout={calendar.get('in_blackout',False)} impact={calendar.get('impact',0)}")
    return "\n".join(parts)


def _parse_verdict(text: str) -> tuple[int, float, float, str]:
    """Extract (direction, confidence, expected_return_pct, reasoning)."""
    direction = 0
    confidence = 0.5
    expected_return = 0.0
    reasoning = ""
    text_l = text.lower()
    if "verdict: buy" in text_l or "verdict: long" in text_l or "verdict: bullish" in text_l:
        direction = 1
    elif "verdict: sell" in text_l or "verdict: short" in text_l or "verdict: bearish" in text_l:
        direction = -1
    else:
        direction = 0
    import re
    m = re.search(r"confidence\s*:?\s*([01](?:\.\d+)?)", text_l)
    if m:
        confidence = float(m.group(1))
    m = re.search(r"expected_return_pct\s*:?\s*(-?\d+(?:\.\d+)?)", text_l)
    if m:
        expected_return = float(m.group(1))
    m = re.search(r"reasoning\s*:?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()[:300]
    if not reasoning:
        reasoning = text.strip()[:300]
    return direction, max(0.0, min(1.0, confidence)), expected_return, reasoning


# ----- orchestrator -----------------------------------------------------------

class MetaOrchestrator:
    """OpenRouter-backed (preferred) or deterministic (fallback) reasoner."""

    def __init__(
        self,
        *,
        client=None,                       # OpenRouterClient | None
        model: str = "deepseek/deepseek-chat",
        fallback_models: list[str] | None = None,
        max_tokens: int = 512,
        prefer_openrouter: bool = True,
    ) -> None:
        self._client = client
        self.model = model
        self.fallback_models = fallback_models or [
            "deepseek/deepseek-chat",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-70b-instruct",
            "openai/gpt-4o-mini",
        ]
        self.max_tokens = max_tokens
        self.prefer_openrouter = bool(prefer_openrouter and client is not None)

    def set_client(self, client) -> None:
        self._client = client
        self.prefer_openrouter = bool(client is not None)

    def explain(self, payload: dict) -> MetaVerdict:
        """Produce a verdict + reasoning for a single decision."""
        if self.prefer_openrouter and self._client is not None:
            v = self._try_openrouter(payload)
            if v is not None:
                return v
        return self._deterministic_reason(payload)

    # ----- OpenRouter path ---------------------------------------------------

    def _try_openrouter(self, payload: dict) -> MetaVerdict | None:
        try:
            from .openrouter import (
                ChatMessage, ChatRequest, OpenRouterError, chat_with_fallback,
            )
        except ImportError:
            return None
        user_prompt = _build_user_prompt(payload)
        req = ChatRequest(
            model=self.model,
            messages=[
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        try:
            resp = chat_with_fallback(
                self._client, req, fallback_models=self.fallback_models,
            )
        except OpenRouterError as e:
            LOGGER.warning("OpenRouter meta failed: %s", e)
            return None
        direction, confidence, expected_return, reasoning = _parse_verdict(resp.content)
        return MetaVerdict(
            direction=direction,
            confidence=confidence,
            reasoning=f"{reasoning}  (tokens={resp.total_tokens}, cost=${resp.cost_usd:.4f})",
            source=f"openrouter:{resp.model}",
        )

    # ----- deterministic fallback (always works) --------------------------

    def _deterministic_reason(self, payload: dict) -> MetaVerdict:
        models = payload.get("models") or []
        fusion = payload.get("fusion") or {}
        ta = payload.get("ta") or {}
        quantum = payload.get("quantum") or {}
        patterns = payload.get("patterns") or {}
        news = payload.get("news") or {}
        forces = payload.get("forces") or {}
        calendar = payload.get("calendar") or {}

        bullish = bearish = 0.0
        reasons: list[str] = []

        for m in models:
            if not isinstance(m, dict):
                continue
            d = m.get("direction")
            c = m.get("confidence") or 0.0
            if d == 1:
                bullish += c
                if c > 0.5:
                    reasons.append(f"<b>{m.get('model')}</b> bullish ({c:.2f})")
            elif d == -1:
                bearish += c
                if c > 0.5:
                    reasons.append(f"<b>{m.get('model')}</b> bearish ({c:.2f})")

        ta_bias = ta.get("composite_bias", 0)
        if ta_bias > 0:
            bullish += 0.2; reasons.append(f"TA composite +{ta_bias}")
        elif ta_bias < 0:
            bearish += 0.2; reasons.append(f"TA composite {ta_bias}")

        q_dir = quantum.get("direction", 0)
        q_conf = quantum.get("confidence", 0.0) or 0.0
        if q_dir == 1:
            bullish += q_conf; reasons.append(f"Quantum bullish ({q_conf:.2f})")
        elif q_dir == -1:
            bearish += q_conf; reasons.append(f"Quantum bearish ({q_conf:.2f})")

        if patterns.get("trend_structure") == "uptrend":
            bullish += 0.2; reasons.append("Pattern uptrend")
        elif patterns.get("trend_structure") == "downtrend":
            bearish += 0.2; reasons.append("Pattern downtrend")

        sent = news.get("sentiment", 0)
        if sent > 0.3:
            bullish += abs(sent) * 0.3; reasons.append(f"News +{sent:.2f}")
        elif sent < -0.3:
            bearish += abs(sent) * 0.3; reasons.append(f"News {sent:.2f}")

        if forces.get("hint") == 1:
            bullish += 0.15; reasons.append("Macro hint bullish")
        elif forces.get("hint") == -1:
            bearish += 0.15; reasons.append("Macro hint bearish")

        if calendar.get("in_blackout"):
            reasons.append("⚠ Blackout — would suppress")
            bullish *= 0.1; bearish *= 0.1

        if bullish > bearish:
            direction = 1
            confidence = min(1.0, bullish / (bullish + bearish + 1e-9))
        elif bearish > bullish:
            direction = -1
            confidence = min(1.0, bearish / (bullish + bearish + 1e-9))
        else:
            direction = 0
            confidence = 0.0

        text = "; ".join(reasons[:8]) if reasons else "no strong signals"
        return MetaVerdict(
            direction=direction,
            confidence=confidence,
            reasoning=text,
            source="deterministic",
        )


__all__ = ["MetaOrchestrator", "MetaVerdict"]
