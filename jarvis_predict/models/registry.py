"""
Model registry — config-driven discovery, instantiation, lifecycle.

Given a `models:` section in `config.yaml`, this module:
    1. Walks each declared family + variant
    2. Builds the matching PredictiveModel instance
    3. Returns a ModelFleet that can be queried for forecasts in parallel

Failed models stay in the fleet with `status="unavailable"` — the ensemble
just skips them with a low weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import PredictiveModel, ModelForecast

LOGGER = logging.getLogger("jarvis.models.registry")


@dataclass
class ModelSpec:
    """Configuration entry for one model."""
    name: str               # e.g. "qlib_lstm"
    family: str             # e.g. "qlib"
    enabled: bool = True
    weight: float = 1.0     # base weight in ensemble
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelFleet:
    """Holds the live PredictiveModel instances for the runtime."""

    models: list[PredictiveModel]
    specs: dict[str, ModelSpec]

    def names(self) -> list[str]:
        return [m.name for m in self.models]

    def get(self, name: str) -> PredictiveModel | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def available(self) -> list[PredictiveModel]:
        return [m for m in self.models if m.is_available()]

    def forecast_all(
        self,
        values,
        horizon: int,
        *,
        symbol: str | None = None,
        fail_fast: bool = False,
    ) -> list[ModelForecast]:
        """Call every model. Returns a list of ModelForecast (errors included)."""
        results: list[ModelForecast] = []
        for m in self.models:
            try:
                res = m.forecast(values, horizon, symbol=symbol)
                # Config weight participates in the ensemble via extras.weight
                # (read by EnsembleFusion._aggregate_models).  Defaults stay
                # backward compatible: weight absent -> confidence-driven.
                spec = self.specs.get(m.name)
                if spec is not None:
                    weight = float(getattr(spec, "weight", 1.0))
                    if res.extras is None:
                        object.__setattr__(res, "extras", {"weight": weight})
                    else:
                        res.extras.setdefault("weight", weight)
                results.append(res)
            except Exception as e:  # pragma: no cover
                LOGGER.exception("Model %s crashed", m.name)
                if fail_fast:
                    raise
                results.append(ModelForecast(
                    model_name=m.name, family=m.family,
                    point=values[-horizon:] if hasattr(values, "__getitem__") else [],
                    quantiles=None, horizon=horizon,
                    direction=0, expected_return_pct=0.0, confidence=0.0,
                    inference_ms=0.0,
                    status=f"error: {type(e).__name__}: {e}"[:60],
                ))
        return results

    def status_table(self) -> list[dict]:
        return [
            {
                "name": m.name,
                "family": m.family,
                "status": m.status,
                "available": m.is_available(),
                "error": m.last_error or "",
            }
            for m in self.models
        ]


def _build_one(spec: ModelSpec, client: Any = None) -> PredictiveModel:
    """Instantiate one PredictiveModel from a spec. Falls back to MockModel on error."""
    family = spec.family.lower()
    # `cadence` is a fleet-scheduler hint, not an adapter kwarg.
    spec.kwargs = {k: v for k, v in spec.kwargs.items() if k != "cadence"}
    _model: PredictiveModel | None = None
    try:
        if family == "timesfm":
            from .timesfm_adapter import TimesFMAdapter
            _model = TimesFMAdapter(**spec.kwargs)
        elif family == "chronos":
            from .chronos_adapter import ChronosAdapter
            _model = ChronosAdapter(**spec.kwargs)
        elif family == "qlib":
            from .qlib_adapter import QlibAdapter
            algo = spec.kwargs.pop("algo", spec.name.removeprefix("qlib_"))
            _model = QlibAdapter(algo=algo, **spec.kwargs)
        elif family == "fingpt":
            from .fingpt_adapter import (
                FinGPTForecasterAdapter, FinGPTSentimentAdapter,
            )
            kind = spec.kwargs.pop("kind", spec.name.removeprefix("fingpt_"))
            _model = FinGPTSentimentAdapter(**spec.kwargs) if kind == "sentiment" else FinGPTForecasterAdapter(**spec.kwargs)
        elif family == "finrobot":
            from .finrobot_adapter import FinRobotAnalystAdapter
            _model = FinRobotAnalystAdapter(**spec.kwargs)
        elif family == "deepseek":
            from .deepseek_v41_adapter import DeepSeekV41Adapter
            if client is not None:
                spec.kwargs = dict(spec.kwargs)
                spec.kwargs.setdefault("client", client)
            _model = DeepSeekV41Adapter(**spec.kwargs)
        elif family in ("arima", "ets", "auto_arima", "svr", "forest", "boost"):
            from .stats_adapters import (
                ArimaAdapter, EtsAdapter, AutoArimaAdapter,
                SvrAdapter, ForestAdapter, BoostAdapter,
            )
            kind = spec.kwargs.pop("algo", None)
            if family == "arima":
                _model = ArimaAdapter(**spec.kwargs)
            elif family == "ets":
                _model = EtsAdapter(**spec.kwargs)
            elif family == "auto_arima":
                _model = AutoArimaAdapter(**spec.kwargs)
            elif family == "svr":
                _model = SvrAdapter(**spec.kwargs)
            elif family == "forest":
                _model = ForestAdapter(algo=kind or "extra", **spec.kwargs)
            else:
                _model = BoostAdapter(algo=kind or "xgboost", **spec.kwargs)
        elif family == "mock":
            from .mock import MockModel
            _model = MockModel(**spec.kwargs)
        else:
            raise ValueError(f"unknown model family: {spec.family}")
    except Exception as e:
        LOGGER.warning("Failed to build model %s (%s): %s — falling back to MockModel",
                       spec.name, spec.family, e)
        from .mock import MockModel
        _model = MockModel(seed=hash(spec.name) & 0xFFFFFFFF)
    # Instance identity = config identity (distinct per variant for the
    # ModelLearner per-model multiplier and ensemble weight stamping).
    _model.name = spec.name
    return _model


def build_fleet(specs: list[ModelSpec], client: Any = None) -> ModelFleet:
    """Build a ModelFleet from a list of ModelSpec.

    `client` is an optional OpenRouter client injected into LLM-backed
    adapters (deepseek family) so they reuse the runtime's configured keys.
    """
    models: list[PredictiveModel] = []
    spec_map: dict[str, ModelSpec] = {}
    for s in specs:
        if not s.enabled:
            continue
        spec_map[s.name] = s
        models.append(_build_one(s, client=client))
    LOGGER.info("Fleet built: %d models", len(models))
    return ModelFleet(models=models, specs=spec_map)


def specs_from_config(config_section: dict[str, Any]) -> list[ModelSpec]:
    """Parse the `models:` section of config.yaml into ModelSpec list.

    Config schema:
        models:
          timesfm_2.5:
            family: timesfm
            enabled: true
            weight: 1.5
            kwargs: { checkpoint: "google/timesfm-2.5-200m-pytorch", device: "cpu" }
          chronos_bolt_tiny:
            family: chronos
            enabled: true
            weight: 1.0
            kwargs: { variant: "bolt-tiny" }
          qlib_lstm:
            family: qlib
            enabled: true
            weight: 0.8
            kwargs: { algo: "lstm", epochs: 5 }
          ...
    """
    out: list[ModelSpec] = []
    for name, entry in config_section.items():
        if not isinstance(entry, dict):
            continue
        out.append(ModelSpec(
            name=name,
            family=entry.get("family", name.split("_", 1)[0]),
            enabled=bool(entry.get("enabled", True)),
            weight=float(entry.get("weight", 1.0)),
            kwargs=dict(entry.get("kwargs", {})),
        ))
    return out


__all__ = ["ModelSpec", "ModelFleet", "build_fleet", "specs_from_config"]
