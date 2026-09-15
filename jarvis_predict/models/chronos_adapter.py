"""
Chronos adapter — Amazon Chronos / Chronos-2 foundation time-series model.

Repo: github.com/amazon-science/chronos-forecasting
Family: foundation time-series model (T5-based encoder/decoder)

The chronos package provides:
    BaseChronosPipeline.predict(inputs, prediction_length)
        - inputs: torch.Tensor [batch, context_len] or list thereof
        - prediction_length: int

Auto-downloads model weights from HuggingFace on first load (one-shot
network call; afterwards cached locally). Falls back to mock if download
or import fails.
"""

from __future__ import annotations

import logging
import os
import numpy as np

from .base import ModelForecast, PredictiveModel
from .mock import MockModel

LOGGER = logging.getLogger("jarvis.models.chronos")


# HuggingFace repos for Chronos variants.
_CHRONOS_REPOS = {
    "tiny":   "amazon/chronos-t5-tiny",
    "mini":   "amazon/chronos-t5-mini",
    "small":  "amazon/chronos-t5-small",
    "base":   "amazon/chronos-t5-base",
    "large":  "amazon/chronos-t5-large",
    "bolt-tiny":  "amazon/chronos-bolt-tiny",
    "bolt-mini":  "amazon/chronos-bolt-mini",
    "bolt-small": "amazon/chronos-bolt-small",
    "bolt-base":  "amazon/chronos-bolt-base",
}


class ChronosAdapter(PredictiveModel):
    """Amazon Chronos pipeline. Falls back to MockModel if loading fails."""

    name = "chronos"
    family = "chronos"
    description = "Amazon Chronos — T5-based time-series foundation model."

    def __init__(
        self,
        *,
        variant: str = "bolt-tiny",
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant if variant in _CHRONOS_REPOS else "bolt-tiny"
        self.device = device
        self.cache_dir = cache_dir
        self._pipeline = None
        # Fallback is always available, so a failed load doesn't NPE on forecast.
        self._fallback: PredictiveModel = MockModel(seed=42)

    def _load_impl(self) -> None:
        if self.cache_dir:
            os.environ.setdefault("HF_HOME", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
        try:
            from chronos import BaseChronosPipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "chronos not installed. Run: pip install chronos-forecasting"
            ) from e
        repo = _CHRONOS_REPOS[self.variant]
        try:
            self._pipeline = BaseChronosPipeline.from_pretrained(
                repo, device_map=self.device,
            )
        except Exception as e:
            raise RuntimeError(f"failed to download {repo}: {e}") from e

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        if self._pipeline is None:
            return self._fallback.forecast(values, horizon, **kwargs)
        try:
            import torch
        except ImportError:
            return self._fallback.forecast(values, horizon, **kwargs)  # type: ignore[union-attr]
        ctx = values.astype(np.float32)
        if ctx.ndim == 1:
            ctx = ctx[None, :]            # [1, T]
        t = torch.from_numpy(ctx)
        with torch.no_grad():
            quantiles, mean = self._pipeline.predict_quantiles(
                inputs=t,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
        # quantiles shape: [batch, prediction_length, num_quantiles]
        q = quantiles[0].cpu().numpy().astype(np.float32)
        point = mean[0].cpu().numpy().astype(np.float32)
        from ._interp import build_forecast
        return build_forecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=q,
            last_value=float(values[-1]),
            status="ok",
        )


__all__ = ["ChronosAdapter"]
