"""
TimesFM predictive engine.

Wraps Google TimesFM 2.5 (200M params, PyTorch) behind a small, opinionated
interface that the rest of Jarvis consumes.

Usage:
    engine = TimesFMEngine.from_config(config)
    result = engine.forecast(close_prices_np)            # numpy array
    direction = result.direction                         # -1, 0, +1
    confidence = result.confidence                       # 0..1
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import JarvisConfig

LOGGER = logging.getLogger("jarvis.core")

# TimesFM patch size = 32 tokens. Inputs shorter than this still work but
# accuracy suffers; the model pads short inputs internally.
_MIN_CONTEXT = 32


@dataclass(frozen=True)
class ForecastResult:
    """Result of one TimesFM inference call."""

    point: np.ndarray            # shape (horizon,)
    quantiles: np.ndarray | None  # shape (horizon, 9) when quantiles are produced
    horizon: int
    context_len: int
    direction: int               # -1, 0, +1
    expected_return_pct: float   # signed, e.g. 0.12 = +0.12%
    confidence: float            # 0..1
    inference_ms: float

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "context_len": self.context_len,
            "direction": self.direction,
            "expected_return_pct": round(self.expected_return_pct, 4),
            "confidence": round(self.confidence, 4),
            "inference_ms": round(self.inference_ms, 2),
        }


class TimesFMEngine:
    """Lazy-loaded, cached TimesFM predictor.

    The underlying model is heavy (~200M params). We load it once at process
    start and reuse it across symbols and ticks.
    """

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        per_core_batch_size: int,
        context_len: int,
        horizon: int,
        cache_dir: Path,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.per_core_batch_size = per_core_batch_size
        self.context_len = max(_MIN_CONTEXT, int(context_len))
        self.horizon = max(1, int(horizon))
        self.cache_dir = cache_dir
        self._model = None  # loaded lazily on first forecast()

    # --- factory ------------------------------------------------------------

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "TimesFMEngine":
        section = config.section("timesfm")
        return cls(
            checkpoint=section["checkpoint"],
            device=section.get("device", "cpu"),
            per_core_batch_size=int(section.get("per_core_batch_size", 16)),
            context_len=int(section.get("context_len", 512)),
            horizon=int(section.get("horizon", 16)),
            cache_dir=Path(
                config.raw["paths"].get("model_cache") or config.cache_dir / "models"
            ).expanduser().resolve(),
        )

    # --- model lifecycle ---------------------------------------------------

    def load(self) -> None:
        """Eagerly load + compile the model. Idempotent."""
        if self._model is not None:
            return
        LOGGER.info(
            "Loading TimesFM checkpoint=%s device=%s context=%d horizon=%d",
            self.checkpoint, self.device, self.context_len, self.horizon,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir))

        # Import here so module import time stays light if user only wants config.
        from timesfm import TimesFM_2p5_200M_torch, configs  # type: ignore

        t0 = time.perf_counter()
        # torch_compile=False avoids needing a Triton toolchain on Windows.
        model = TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint,
            torch_compile=False,
        )

        fc = configs.ForecastConfig(
            per_core_batch_size=self.per_core_batch_size,
            max_context=self.context_len,
            max_horizon=self.horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
            return_backcast=False,
        )
        model.compile(fc)
        self._model = model
        LOGGER.info("TimesFM ready in %.1fs", time.perf_counter() - t0)

    # --- inference ---------------------------------------------------------

    def forecast(self, values: Sequence[float] | np.ndarray) -> ForecastResult:
        """Forecast a single univariate series.

        Args:
            values: 1-D sequence of floats (close prices, returns, etc.).
                    NaN is handled by TimesFM via internal linear interpolation.

        Returns:
            ForecastResult with point, quantiles, direction, confidence.
        """
        self.load()

        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D input, got shape {arr.shape}")
        if arr.size < _MIN_CONTEXT:
            raise ValueError(
                f"Need at least {_MIN_CONTEXT} points, got {arr.size}"
            )

        # TimesFM expects inputs as a list of 1-D arrays for batched inference.
        t0 = time.perf_counter()
        # Pad/trim to context_len (TimesFM does this internally too, but we
        # do it explicitly to make `context_len` reporting accurate).
        if arr.size > self.context_len:
            ctx = arr[-self.context_len:]
        else:
            ctx = arr
        point, quantile = self._model.forecast(
            horizon=self.horizon,
            inputs=[ctx],
        )
        # Outputs are (1, horizon) and (1, horizon, 9) — squeeze the batch dim.
        point = np.asarray(point).reshape(-1)[: self.horizon]
        quantile = np.asarray(quantile).reshape(self.horizon, -1)

        direction, expected_return_pct, confidence = _interpret_forecast(
            last_value=float(ctx[-1]),
            forecast_point=point,
            forecast_quantiles=quantile,
        )

        return ForecastResult(
            point=point,
            quantiles=quantile,
            horizon=self.horizon,
            context_len=int(ctx.size),
            direction=direction,
            expected_return_pct=expected_return_pct,
            confidence=confidence,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # --- convenience -------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"TimesFMEngine(checkpoint={self.checkpoint!r}, device={self.device!r}, "
            f"context_len={self.context_len}, horizon={self.horizon}, loaded={self.is_loaded})"
        )


def _interpret_forecast(
    *,
    last_value: float,
    forecast_point: np.ndarray,
    forecast_quantiles: np.ndarray,
) -> tuple[int, float, float]:
    """Convert raw TimesFM output to a directional trade signal.

    Returns (direction, expected_return_pct, confidence).

    direction: -1 (sell), 0 (hold), +1 (buy)
    expected_return_pct: signed percent change vs last_value (last forecast step)
    confidence: 0..1 — based on quantile spread + monotonicity of forecast path.
    """
    if last_value == 0 or not np.isfinite(last_value):
        return 0, 0.0, 0.0

    # Final-step move is the simplest direction proxy.
    last_forecast = float(forecast_point[-1])
    move_pct = (last_forecast - last_value) / abs(last_value) * 100.0

    # Quantile spread as confidence proxy: tighter spread -> higher confidence.
    # Quantile matrix is (horizon, 9) with quantiles 0.1..0.9 in canonical order.
    # Median is the 5th column (index 4).
    if forecast_quantiles.ndim == 2 and forecast_quantiles.shape[-1] >= 9:
        q_low = float(forecast_quantiles[-1, 1])    # 0.2 quantile
        q_high = float(forecast_quantiles[-1, 7])   # 0.8 quantile
        spread_pct = abs(q_high - q_low) / max(abs(last_value), 1e-9) * 100.0
        # Confidence = 1 / (1 + spread_pct * k).  k=4 -> 0.05% spread → 0.83
        confidence = 1.0 / (1.0 + spread_pct * 4.0)
    else:
        confidence = 0.5

    # Path monotonicity bonus: if the path is consistently moving in one direction,
    # bump confidence.
    diffs = np.diff(forecast_point)
    if diffs.size > 0:
        sign = np.sign(diffs)
        consistency = float(np.mean(sign == np.sign(move_pct))) if move_pct != 0 else 0.5
        confidence = float(np.clip(confidence * (0.7 + 0.6 * consistency), 0.0, 0.99))

    # Direction classification with a tiny dead-zone (avoid trading noise).
    # 0.02% threshold on absolute move.
    if move_pct > 0.02:
        direction = 1
    elif move_pct < -0.02:
        direction = -1
    else:
        direction = 0

    return direction, move_pct, float(confidence)


__all__ = ["TimesFMEngine", "ForecastResult"]
