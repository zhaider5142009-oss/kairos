"""
Shared helper — convert raw forecast arrays into a ModelForecast with
direction / expected-return / confidence auto-derived.

Same logic used by the existing TimesFMEngine._interpret_forecast — kept
separate so the model adapters don't depend on core.py.

Improvements in v1.2:
    - Noise-floor awareness: dynamic direction threshold based on the
      forecast's interquartile spread (forecast uncertainty), rather than
      a fixed 0.02% cut-off. This prevents micro-moves from being
      classified as actionable and lets strong moves through at high
      confidence.
    - Monotonicity boost: a clean monotonic trend across the forecast
      horizon signals conviction and elevates confidence.
    - Quantile-spread confidence: uses full 9-quantile band for tight
      estimates instead of relying on a single quartile pair.
"""

from __future__ import annotations

import numpy as np

from .base import ModelForecast


def _iqr_spread(point: np.ndarray, last_value: float) -> float:
    """Estimate the interquartile range spread as % of price."""
    if point.size < 4 or last_value == 0:
        return 0.5
    q25 = float(np.percentile(point, 25))
    q75 = float(np.percentile(point, 75))
    return abs(q75 - q25) / max(abs(last_value), 1e-9) * 100.0


def _monotonicity(arr: np.ndarray) -> float:
    """0..1. 1.0 = perfectly monotonic (all steps same sign)."""
    if arr.size < 2:
        return 0.0
    signs = np.sign(np.diff(arr))
    if signs.size == 0:
        return 0.0
    dominant = np.sign(np.sum(signs))
    if dominant == 0:
        return 0.0
    return float(np.mean(signs == dominant))


def _quantile_confidence(quantiles: np.ndarray | None, last_value: float) -> float | None:
    """Use the 9-quantile band to derive a noise-aware confidence.
    Returns None if quantiles are missing/invalid."""
    if quantiles is None:
        return None
    if quantiles.ndim != 2 or quantiles.shape[-1] < 9:
        return None
    # Use the central 80% band (q10..q90) which is less noisy than IQR.
    q10 = float(quantiles[-1, 1])
    q90 = float(quantiles[-1, 7])
    q30 = float(quantiles[-1, 3])
    q70 = float(quantiles[-1, 5])
    spread_outer = abs(q90 - q10) / max(abs(last_value), 1e-9) * 100.0
    spread_inner = abs(q70 - q30) / max(abs(last_value), 1e-9) * 100.0
    if spread_outer < 1e-9:
        return 0.85
    # Tighter bands = higher confidence; use a sigmoid-like map.
    base = 1.0 / (1.0 + spread_outer * 2.0)
    # Inner/outer agreement: if inner is a subset, bands are well-behaved.
    ratio = spread_inner / max(spread_outer, 1e-9)
    if ratio > 0.9:
        base *= 0.85
    return max(0.05, min(0.98, base))


def build_forecast(
    *,
    model_name: str,
    family: str,
    point: np.ndarray,
    quantiles: np.ndarray | None,
    last_value: float,
    inference_ms: float = 0.0,
    status: str = "ok",
    extras: dict | None = None,
) -> ModelForecast:
    """Derive direction + confidence from raw forecast arrays."""
    point = np.asarray(point, dtype=np.float32).reshape(-1)
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float32)
    horizon = int(point.size)

    if last_value == 0 or not np.isfinite(last_value):
        return ModelForecast(
            model_name=model_name, family=family,
            point=point, quantiles=quantiles, horizon=horizon,
            direction=0, expected_return_pct=0.0, confidence=0.0,
            inference_ms=inference_ms, status=status, extras=extras or {},
        )

    last_forecast = float(point[-1])
    expected_return_pct = (last_forecast - last_value) / abs(last_value) * 100.0

    # --- confidence from quantile bands ---
    quant_conf = _quantile_confidence(quantiles, last_value)
    if quant_conf is not None:
        confidence = quant_conf
    else:
        confidence = 0.5

    # --- dynamic noise floor: depends on forecast uncertainty ---
    iqr_pct = _iqr_spread(point, last_value)
    noise_floor = max(0.005, min(0.06, iqr_pct * 0.04))

    # --- monotonicity boost (a clean trend through the horizon = conviction) ---
    mono = _monotonicity(point)
    if mono > 0.65 and confidence > 0.3:
        confidence = min(0.98, confidence * (1.0 + (mono - 0.65) * 0.8))

    # --- direction consistency through the path ---
    diffs = np.diff(point)
    if diffs.size > 0 and expected_return_pct != 0:
        sign = np.sign(diffs)
        consistency = float(np.mean(sign == np.sign(expected_return_pct)))
        confidence = float(np.clip(confidence * (0.7 + 0.6 * consistency), 0.0, 0.99))

    # --- dynamic direction assignment against noise floor ---
    if abs(expected_return_pct) > noise_floor:
        direction = 1 if expected_return_pct > 0 else -1
    else:
        direction = 0

    return ModelForecast(
        model_name=model_name,
        family=family,
        point=point,
        quantiles=quantiles,
        horizon=horizon,
        direction=direction,
        expected_return_pct=expected_return_pct,
        confidence=float(confidence),
        inference_ms=inference_ms,
        status=status,
        extras=extras or {},
    )


__all__ = ["build_forecast"]
