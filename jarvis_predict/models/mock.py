"""
Mock model — used for testing, smoke runs, and as fallback when the real
model fails to load (e.g. weights missing, network down).

Always available. Always returns a deterministic forecast from the last
value + linear extrapolation.
"""

from __future__ import annotations

import numpy as np

from ._interp import build_forecast
from .base import ModelForecast, PredictiveModel


class MockModel(PredictiveModel):
    """Deterministic linear-extrapolation forecaster. Never fails."""

    name = "mock"
    family = "mock"
    description = "Deterministic fallback — linear extrapolation of the last N points."

    def __init__(self, *, slope: float = 0.0, noise_amp: float = 0.0, seed: int = 0) -> None:
        super().__init__()
        self.slope = float(slope)
        self.noise_amp = float(noise_amp)
        self.seed = int(seed)
        self._status = "ready"

    def _load_impl(self) -> None:  # nothing to load
        return

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        if values.size == 0:
            return build_forecast(
                model_name=self.name, family=self.family,
                point=np.zeros(horizon, dtype=np.float32), quantiles=None,
                last_value=0.0, status="mock",
            )
        # Linear extrapolation from last 32 points.
        n = min(32, values.size)
        x = np.arange(n, dtype=np.float32)
        y = values[-n:]
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(n, n + horizon, dtype=np.float32)
        point = intercept + slope * future_x
        # Optional deterministic noise.
        if self.noise_amp > 0:
            rng = np.random.default_rng(self.seed + values.size)
            point = point + self.noise_amp * rng.standard_normal(horizon).astype(np.float32)
        return build_forecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None,
            last_value=float(values[-1]), status="mock",
        )


__all__ = ["MockModel"]
