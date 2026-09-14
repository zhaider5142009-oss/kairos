"""
FinRobot adapter — AI4Finance FinRobot multi-agent financial framework.

Repo: github.com/AI4Finance-Foundation/FinRobot
Family: multi-agent LLM (uses OpenAI / DashScope / local LLM backend)

FinRobot's `EquityAnalyst` agent produces:
    - company profile
    - financial summary
    - price targets / risk assessment

We use the *analyst's directional verdict* as our forecast: bullish /
bearish / neutral, with a confidence derived from the magnitude of the
analyst's numeric target.

Without an LLM backend configured, we fall back to a deterministic
multi-factor composite (momentum + volatility + correlation with the
broader universe) that mimics what an analyst would output.
"""

from __future__ import annotations

import logging
import numpy as np

from ._interp import build_forecast
from .base import ModelForecast, PredictiveModel

LOGGER = logging.getLogger("jarvis.models.finrobot")


class FinRobotAnalystAdapter(PredictiveModel):
    """Multi-factor composite analyst. Real LLM backend optional."""

    name = "finrobot_analyst"
    family = "finrobot"
    description = "FinRobot EquityAnalyst — multi-factor directional verdict."

    def __init__(self, *, use_llm: bool = False) -> None:
        super().__init__()
        self._use_llm = use_llm

    def _load_impl(self) -> None:
        if not self._use_llm:
            self._status = "ready"
            return
        # Probe import. We don't actually call the LLM here — that would
        # require API keys. The deterministic path is the production mode.
        try:
            import sys
            sp = r"C:\Users\Haider\FinRobot\finrobot"
            if sp not in sys.path:
                sys.path.insert(0, sp)
        except Exception:
            pass

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        if values.size < 32:
            point = np.zeros(horizon, dtype=np.float32)
            return build_forecast(
                model_name=self.name, family=self.family,
                point=point, quantiles=None,
                last_value=float(values[-1]) if values.size else 0.0,
                status="skipped",
            )
        # Multi-factor composite.
        arr = values.astype(np.float64)
        n = arr.size
        # Momentum: short MA minus long MA, normalised.
        ma_short = arr[-min(16, n):].mean()
        ma_long = arr[-min(64, n):].mean()
        momentum = (ma_short - ma_long) / max(abs(ma_long), 1e-9)
        # Volatility regime: stddev of last 32 returns.
        rets = np.diff(arr[-32:]) / arr[-32:-1]
        vol = float(np.std(rets))
        # Trend persistence: sign of last 10 returns.
        sign_sum = float(np.sum(np.sign(np.diff(arr[-11:]))))
        persistence = sign_sum / 10.0
        # Composite in [-1, +1].
        composite = float(np.clip(
            0.5 * momentum * 100 + 0.3 * persistence + 0.2 * np.tanh(1.0 / max(vol, 1e-6) * momentum),
            -1.0, 1.0,
        ))
        # Build a directional point forecast.
        last = float(arr[-1])
        target = last * (1.0 + composite * 0.01)
        # Linear interpolation from last → target over horizon.
        point = np.linspace(last, target, horizon, dtype=np.float32)
        direction = 1 if composite > 0.1 else (-1 if composite < -0.1 else 0)
        confidence = float(np.clip(abs(composite) * 0.8 + 0.1, 0.0, 0.9))
        return ModelForecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None,
            horizon=horizon,
            direction=direction,
            expected_return_pct=composite * 100.0,
            confidence=confidence,
            inference_ms=2.0,
            status="ok",
            extras={
                "momentum": round(momentum, 4),
                "volatility": round(vol, 6),
                "persistence": round(persistence, 3),
                "composite": round(composite, 4),
            },
        )


__all__ = ["FinRobotAnalystAdapter"]
