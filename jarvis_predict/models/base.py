"""
Predictive-model adapter base class.

Every forecasting source — TimesFM, Chronos, qlib LSTM, FinGPT, FinRobot,
even a mock — implements `PredictiveModel`. The ensemble talks to all of
them through this interface, so adding a 13th, 14th model is one
config-line + one module.

Contract:
    name        — stable identifier (e.g. "timesfm", "qlib_lstm")
    family      — repo family ("timesfm", "chronos", "qlib", "fingpt", "finrobot", "mock")
    is_available()  -> bool     # can we run this right now?
    status           -> str      # "ready" | "loading" | "unavailable" | "mock" | "error: ..."
    forecast(values, horizon) -> ModelForecast
        Always returns a ModelForecast — never raises on bad input.
        On unrecoverable error, returns a forecast with confidence=0 and
        status="error: <reason>" so the ensemble can skip it cleanly.

ModelForecast fields:
    point          — shape (horizon,)
    quantiles      — shape (horizon, 9) or None
    direction      — -1, 0, +1
    expected_return_pct — signed percent move
    confidence     — 0..1
    inference_ms   — wall-clock latency
    status         — "ok" | "skipped" | "error: ..." | "mock"
    extras         — free-form dict for per-model metadata
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger("jarvis.models")


@dataclass(frozen=True)
class ModelForecast:
    """Uniform forecast result from any PredictiveModel."""

    model_name: str
    family: str
    point: np.ndarray
    quantiles: np.ndarray | None
    horizon: int
    direction: int                  # -1, 0, +1
    expected_return_pct: float      # signed, percent
    confidence: float               # 0..1
    inference_ms: float
    status: str = "ok"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "family": self.family,
            "horizon": self.horizon,
            "direction": self.direction,
            "expected_return_pct": round(self.expected_return_pct, 4),
            "confidence": round(self.confidence, 4),
            "inference_ms": round(self.inference_ms, 2),
            "status": self.status,
            "extras": {k: (round(v, 4) if isinstance(v, float) else str(v)[:60])
                       for k, v in (self.extras or {}).items()},
        }


class PredictiveModel(ABC):
    """Abstract base for every forecasting model Jarvis can use."""

    name: str = "abstract"
    family: str = "abstract"
    description: str = ""

    def __init__(self) -> None:
        self._status: str = "uninitialised"
        self._last_error: str = ""

    # ----- introspection -----------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    def is_available(self) -> bool:
        """True if the model is loaded and ready to forecast."""
        return self._status == "ready" or self._status == "mock"

    # ----- lifecycle ---------------------------------------------------------

    def load(self) -> None:  # noqa: D401 — overridable
        """Lazy-load model weights / state. Idempotent."""
        if self._status in ("ready", "mock"):
            return
        self._status = "loading"
        try:
            self._load_impl()
            if self._status == "loading":
                self._status = "ready"
        except Exception as e:  # pragma: no cover
            self._last_error = f"{type(e).__name__}: {e}"
            LOGGER.warning("Model %s failed to load: %s", self.name, self._last_error)
            self._status = "unavailable"

    @abstractmethod
    def _load_impl(self) -> None:
        """Override to actually load weights / connect to repo."""
        raise NotImplementedError

    # ----- inference ---------------------------------------------------------

    def forecast(
        self,
        values: Sequence[float] | np.ndarray,
        horizon: int,
        **kwargs: Any,
    ) -> ModelForecast:
        """Public entry — wraps _forecast_impl with timing + error capture."""
        self.load()
        t0 = time.perf_counter()
        try:
            arr = np.asarray(values, dtype=np.float32)
            result = self._forecast_impl(arr, horizon, **kwargs)
            if not isinstance(result, ModelForecast):
                # Allow subclasses to return a tuple / dict.
                result = self._normalise(result, arr, horizon, t0)
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            LOGGER.debug("Model %s forecast failed: %s", self.name, self._last_error)
            result = ModelForecast(
                model_name=self.name,
                family=self.family,
                point=np.zeros(horizon, dtype=np.float32),
                quantiles=None,
                horizon=horizon,
                direction=0,
                expected_return_pct=0.0,
                confidence=0.0,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                status=f"error: {self._last_error[:60]}",
            )
        return result

    @abstractmethod
    def _forecast_impl(
        self,
        values: np.ndarray,
        horizon: int,
        **kwargs: Any,
    ) -> ModelForecast | tuple[np.ndarray, np.ndarray | None]:
        """Override to do the actual forecasting.

        May return either a ModelForecast or (point, quantiles). The base
        class normalises whatever the subclass returns.
        """
        raise NotImplementedError

    # ----- helpers -----------------------------------------------------------

    @staticmethod
    def _interpret_simple(
        *,
        last_value: float,
        forecast_point: np.ndarray,
        forecast_quantiles: np.ndarray | None = None,
        model_name: str = "?",
        family: str = "?",
        inference_ms: float = 0.0,
        status: str = "ok",
    ) -> ModelForecast:
        """Convenience: build a ModelForecast from raw arrays + auto-interpret."""
        from . import _interp
        return _interp.build_forecast(
            model_name=model_name,
            family=family,
            point=forecast_point,
            quantiles=forecast_quantiles,
            last_value=last_value,
            inference_ms=inference_ms,
            status=status,
        )

    def _normalise(
        self,
        result: Any,
        values: np.ndarray,
        horizon: int,
        t0: float,
    ) -> ModelForecast:
        """Convert a raw tuple / dict return into a ModelForecast."""
        if isinstance(result, tuple) and len(result) >= 1:
            point = np.asarray(result[0], dtype=np.float32).reshape(-1)[:horizon]
            q = np.asarray(result[1], dtype=np.float32).reshape(horizon, -1) if len(result) > 1 and result[1] is not None else None
        elif isinstance(result, dict):
            point = np.asarray(result.get("point", np.zeros(horizon)), dtype=np.float32)
            q = result.get("quantiles")
        else:
            point = np.asarray(result, dtype=np.float32).reshape(-1)[:horizon]
            q = None
        last = float(values[-1]) if values.size > 0 else 0.0
        return self._interpret_simple(
            last_value=last,
            forecast_point=point,
            forecast_quantiles=q,
            model_name=self.name,
            family=self.family,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            status="ok",
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} status={self._status!r}>"


__all__ = ["PredictiveModel", "ModelForecast"]
