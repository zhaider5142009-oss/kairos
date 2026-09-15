"""
DeepSeek-V4.1 adapter — uses OpenRouter's hosted DeepSeek-V4.1-Flash model
as a "reasoning forecaster" in the 12-model fleet.

Why OpenRouter and not local DGX-Sparks?
    DeepSeek-V4.1-Flash-DGX-Sparks is the GPU serving stack for the V4.1
    weights — it needs an NVIDIA DGX box (multi-GPU, ~600 GB host memory).
    On a personal PC it won't run. Instead we hit OpenRouter's
    `deepseek/deepseek-chat` endpoint, which runs the same
    weights on their cluster.

The adapter treats DeepSeek-V4.1 as a **contextual forecaster**: it gets
the last 512 bars + the indicator snapshot + recent news sentiment and
asks the LLM for a directional verdict + reasoning. The LLM's
`expected_return_pct` and `confidence` are computed from the magnitude of
its text response (numeric token) or set to 0 if not parseable.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from ._interp import build_forecast
from .base import ModelForecast, PredictiveModel
from ..openrouter import (
    ChatMessage, ChatRequest, OpenRouterClient, OpenRouterError,
    DEFAULT_MODEL_CHAIN, chat_with_fallback,
)

LOGGER = logging.getLogger("jarvis.models.deepseek_v41")

_VERDICT_RE = re.compile(
    r"verdict\s*:\s*(buy|sell|long|short|hold|bullish|bearish|neutral)",
    re.IGNORECASE,
)
_RETURN_RE = re.compile(
    r"expected\s*(?:return|move)\s*:?\s*(-?\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_CONF_RE = re.compile(
    r"confidence\s*:?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class DeepSeekV41Adapter(PredictiveModel):
    """DeepSeek-V4.1 reasoning forecaster via OpenRouter."""

    name = "deepseek_v41"
    family = "deepseek"
    description = "DeepSeek-V4.1-Flash via OpenRouter — LLM reasoning forecaster."

    def __init__(
        self,
        *,
        client: OpenRouterClient | None = None,
        model: str = "deepseek/deepseek-chat",
        fallback_models: list[str] | None = None,
        context_bars: int = 256,
    ) -> None:
        super().__init__()
        self._client = client
        self.model = model
        self.fallback_models = fallback_models or DEFAULT_MODEL_CHAIN
        self.context_bars = max(64, int(context_bars))

    def _load_impl(self) -> None:
        if self._client is None:
            self._client = OpenRouterClient()
        self._status = "ready"

    def _build_prompt(self, values: np.ndarray) -> str:
        # Take the last N closes, summarise, and ask for a verdict.
        n = min(self.context_bars, values.size)
        last = values[-n:]
        # Crude normalisation — pass raw prices; the LLM can read them.
        bars_text = ", ".join(f"{v:.5f}" for v in last[::max(1, n // 32)][-32:])
        if len(bars_text) > 1500:
            bars_text = bars_text[-1500:]
        return (
            "You are a senior quantitative trader. Given the recent close prices "
            f"of an instrument ({n} bars, most recent 32 shown):\n[{bars_text}]\n\n"
            "Predict the next 16 bars. Output EXACTLY three lines:\n"
            "VERDICT: BUY | SELL | HOLD\n"
            "EXPECTED_RETURN: <signed percent, e.g. +0.42 or -0.18>\n"
            "CONFIDENCE: <0.00 to 1.00>\n"
            "REASONING: <one sentence why>\n"
        )

    def _parse_response(self, text: str) -> tuple[int, float, float]:
        direction = 0
        mv = _VERDICT_RE.search(text)
        if mv:
            v = mv.group(1).lower()
            if v in ("buy", "long", "bullish"):
                direction = 1
            elif v in ("sell", "short", "bearish"):
                direction = -1
            else:
                direction = 0
        ret_match = _RETURN_RE.search(text)
        ret = float(ret_match.group(1)) if ret_match else 0.0
        conf_match = _CONF_RE.search(text)
        conf = float(conf_match.group(1)) if conf_match else 0.5
        conf = max(0.0, min(1.0, conf))
        # Apply a 0.02% dead-zone on expected return, mirroring TimesFM logic.
        if abs(ret) < 0.02:
            ret = 0.0
            direction = 0
        return direction, ret, conf

    def _deterministic_fallback(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        """No-API-key fallback — plain multi-factor directional vote, labelled
        honestly in extras.mode so the ensemble provenance stays transparent."""
        arr = values.astype(np.float64)
        n = arr.size
        if n < 32:
            point = np.zeros(horizon, dtype=np.float32)
            return ModelForecast(
                model_name=self.name, family=self.family,
                point=point, quantiles=None, horizon=horizon,
                direction=0, expected_return_pct=0.0, confidence=0.0,
                inference_ms=1.0, status="ok",
                extras={"mode": "no_api_fallback", "reason": "insufficient bars"},
            )
        ma_s = arr[-min(16, n):].mean()
        ma_l = arr[-min(64, n):].mean()
        momentum = (ma_s - ma_l) / max(abs(ma_l), 1e-9)
        rets = np.diff(arr[-min(32, n):]) / arr[-min(32, n):-1]
        vol = float(np.std(rets))
        persistence = float(np.mean(np.sign(np.diff(arr[-min(11, n):]))))
        composite = float(np.clip(
            0.5 * momentum * 100 + 0.3 * persistence + 0.2 * np.tanh(1.0 / max(vol, 1e-6) * momentum),
            -1.0, 1.0,
        ))
        last = float(arr[-1])
        target = last * (1.0 + composite * 0.01)
        point = np.linspace(last, target, horizon, dtype=np.float32)
        direction = 1 if composite > 0.1 else (-1 if composite < -0.1 else 0)
        confidence = float(np.clip(abs(composite) * 0.8 + 0.1, 0.0, 0.9))
        expected_return_pct = float((point[-1] - last) / last * 100.0)
        return ModelForecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None, horizon=horizon,
            direction=direction, expected_return_pct=expected_return_pct,
            confidence=confidence, inference_ms=1.0, status="ok",
            extras={
                "mode": "no_api_fallback",
                "composite": round(composite, 4),
                "volatility": round(vol, 6),
            },
        )

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        assert self._client is not None, "client not bound"
        if not (self._client.api_keys or []):
            return self._deterministic_fallback(values, horizon, **kwargs)
        prompt = self._build_prompt(values)
        req = ChatRequest(
            model=self.model,
            messages=[
                ChatMessage(role="system",
                            content="You are a precise quant trader. Follow the exact output format."),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0.1,
            max_tokens=512,
        )
        t0 = time.perf_counter()
        try:
            resp = chat_with_fallback(
                self._client, req, fallback_models=self.fallback_models,
            )
        except OpenRouterError as e:
            LOGGER.debug("deepseek_v41 forecast failed: %s", e)
            return ModelForecast(
                model_name=self.name, family=self.family,
                point=values[-horizon:] if values.size else np.zeros(horizon, dtype=np.float32),
                quantiles=None, horizon=horizon,
                direction=0, expected_return_pct=0.0, confidence=0.0,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                status=f"error: {e}"[:60],
            )
        direction, ret, conf = self._parse_response(resp.content)
        # Synthesize a point forecast from the verdict + magnitude.
        last = float(values[-1])
        target = last * (1.0 + ret / 100.0)
        point = np.linspace(last, target, horizon, dtype=np.float32)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return ModelForecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None, horizon=horizon,
            direction=direction, expected_return_pct=ret,
            confidence=conf, inference_ms=inference_ms,
            status="ok",
            extras={
                "tokens_used": resp.total_tokens,
                "cost_usd": round(resp.cost_usd, 5),
                "model_used": resp.model,
                "snippet": resp.content[:120].replace("\n", " "),
            },
        )


__all__ = ["DeepSeekV41Adapter"]
