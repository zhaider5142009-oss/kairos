"""
TimesFM adapter — wraps the existing TimesFMEngine so it exposes the
PredictiveModel interface.

Repo: github.com/google-research/timesfm
Family: foundation time-series model (decoder-only transformer)
Variants exposed:
    - timesfm_2.5  (Apache-2.0 weights)
    - timesfm_3.0  (non-commercial weights — disabled by default)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import ModelForecast, PredictiveModel
from ..core import TimesFMEngine

LOGGER = logging.getLogger("jarvis.models.timesfm")


class TimesFMAdapter(PredictiveModel):
    name = "timesfm"
    family = "timesfm"
    description = "Google TimesFM 2.5 — decoder-only time-series foundation model."

    def __init__(
        self,
        *,
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cpu",
        context_len: int = 512,
        horizon: int = 16,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._engine = TimesFMEngine(
            checkpoint=checkpoint,
            device=device,
            per_core_batch_size=16,
            context_len=context_len,
            horizon=horizon,
            cache_dir=Path(cache_dir) if cache_dir else Path("C:/Users/Haider/jarvis/cache/models"),
        )

    def _load_impl(self) -> None:
        self._engine.load()

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        if horizon != self._engine.horizon:
            from ..core import TimesFMEngine as _E
            eng = _E(
                checkpoint=self._engine.checkpoint,
                device=self._engine.device,
                per_core_batch_size=self._engine.per_core_batch_size,
                context_len=self._engine.context_len,
                horizon=horizon,
                cache_dir=self._engine.cache_dir,
            )
            eng.load()
            result = eng.forecast(values)
        else:
            result = self._engine.forecast(values)
        return ModelForecast(
            model_name=self.name,
            family=self.family,
            point=result.point,
            quantiles=result.quantiles,
            horizon=result.horizon,
            direction=result.direction,
            expected_return_pct=result.expected_return_pct,
            confidence=result.confidence,
            inference_ms=result.inference_ms,
            status="ok",
            extras={"checkpoint": self._engine.checkpoint, "device": self._engine.device},
        )


__all__ = ["TimesFMAdapter"]
