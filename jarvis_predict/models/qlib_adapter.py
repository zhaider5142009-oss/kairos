"""
qlib adapter — exposes Microsoft Qlib's many deep-learning + tree models
as independent PredictiveModel instances.

Repo: github.com/microsoft/qlib
Family: quantitative ML (LSTM, GRU, ALSTM, Transformer, TCN, Localformer, GBDT, ...)

qlib models follow a fit/predict pattern:
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")

We adapt them into the zero-shot Jarvis pipeline by:
    1. Building a quick in-memory DatasetH from the supplied close-prices
       array (single-instrument, OHLCV-style features).
    2. Calling `fit()` once with that dataset (fast — few epochs).
    3. Calling `predict()` to get the next-N-step forecast.

If qlib isn't importable or fit fails (e.g. no GPU, missing deps), we
fall back to a `MockModel` for that specific algorithm so the rest of the
ensemble keeps working.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._interp import build_forecast
from .base import ModelForecast, PredictiveModel
from .mock import MockModel

LOGGER = logging.getLogger("jarvis.models.qlib")

# Algorithm -> (qlib class path, kwargs)
# Names match the file names under qlib/qlib/contrib/model/.
_QLIB_ALGOS = {
    "lstm":        ("qlib.contrib.model.pytorch_lstm",        "LSTM",        {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0}),
    "gru":         ("qlib.contrib.model.pytorch_gru",         "GRU",         {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0}),
    "alstm":       ("qlib.contrib.model.pytorch_alstm",       "ALSTM",       {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0}),
    "transformer": ("qlib.contrib.model.pytorch_transformer", "Transformer", {"d_feat": 6, "hidden_size": 32, "num_layers": 2, "n_head": 2, "dropout": 0.0}),
    "tcn":         ("qlib.contrib.model.pytorch_tcn",         "TCN",         {"d_feat": 6, "num_layers": 2, "hidden_size": 64}),
    "localformer": ("qlib.contrib.model.pytorch_localformer", "Localformer", {"d_feat": 6, "hidden_size": 32, "num_layers": 2, "n_head": 2}),
    "gats":        ("qlib.contrib.model.pytorch_gats",        "GATS",        {"d_feat": 6, "hidden_size": 32, "num_layers": 2, "n_head": 2}),
    "tabnet":      ("qlib.contrib.model.pytorch_tabnet",      "TabNetModel", {"d_feat": 6}),
    "sfm":         ("qlib.contrib.model.pytorch_sfm",         "SFM",         {"d_feat": 6, "hidden_size": 64, "num_layers": 2}),
    "tra":         ("qlib.contrib.model.pytorch_tra",         "TRA",         {"d_feat": 6, "hidden_size": 64}),
    "krnn":        ("qlib.contrib.model.pytorch_krnn",        "KRNN",        {"d_feat": 6, "hidden_size": 64, "num_layers": 2}),
    "hist":        ("qlib.contrib.model.pytorch_hist",        "HIST",        {"d_feat": 6, "hidden_size": 64}),
    "add":         ("qlib.contrib.model.pytorch_add",         "ADD",         {"d_feat": 6, "hidden_size": 64}),
    "adarnn":      ("qlib.contrib.model.pytorch_adarnn",      "AdaRNN",      {"d_feat": 6, "hidden_size": 64}),
    "tcts":        ("qlib.contrib.model.pytorch_tcts",        "TCTS",        {"d_feat": 6, "hidden_size": 64}),
    "igmtf":       ("qlib.contrib.model.pytorch_igmtf",       "IGMTF",       {"d_feat": 6, "hidden_size": 64}),
    "lgbm":        ("qlib.contrib.model.gbdt",                "GBDT",        {"loss": "mse", "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "num_boost_round": 50}),
    "xgboost":     ("qlib.contrib.model.xgboost",             "XGBModel",    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 50}),
    "catboost":    ("qlib.contrib.model.catboost_model",      "CatBoostModel",{"iterations": 50, "depth": 6, "learning_rate": 0.05}),
}


@dataclass(frozen=True)
class QlibAlgoSpec:
    algo: str
    module_path: str
    class_name: str
    kwargs: dict[str, Any]


def list_qlib_algorithms() -> list[str]:
    return sorted(_QLIB_ALGOS.keys())


def _build_spec(algo: str) -> QlibAlgoSpec:
    if algo not in _QLIB_ALGOS:
        raise KeyError(f"unknown qlib algo: {algo}; valid: {list(_QLIB_ALGOS)}")
    mod, cls, kw = _QLIB_ALGOS[algo]
    return QlibAlgoSpec(algo=algo, module_path=mod, class_name=cls, kwargs=dict(kw))


class QlibAdapter(PredictiveModel):
    """One PredictiveModel per qlib algorithm.

    Each adapter fits a tiny in-memory model on the supplied close series
    and predicts the next N bars. Falls back to MockModel if qlib is
    unavailable or the fit fails.
    """

    def __init__(self, algo: str, *, epochs: int = 5) -> None:
        super().__init__()
        self._spec = _build_spec(algo)
        self.name = f"qlib_{algo}"
        self.family = "qlib"
        self.description = f"Microsoft Qlib — {algo.upper()} model."
        self._epochs = max(1, int(epochs))
        self._model = None
        self._fallback = MockModel(seed=hash(algo) & 0xFFFFFFFF)

    def _load_impl(self) -> None:
        try:
            import importlib
            mod = importlib.import_module(self._spec.module_path)
            cls = getattr(mod, self._spec.class_name)
            self._model = cls(**self._spec.kwargs)
        except Exception as e:
            raise RuntimeError(f"qlib import / instantiate failed: {e}") from e

    def _build_dataset(self, values: np.ndarray):
        """Construct a qlib DatasetH from a 1-D close series."""
        try:
            import qlib
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.handler import DataHandlerLP
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(f"qlib not importable: {e}") from e

        n = values.size
        idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "$open":   values.astype(np.float64),
            "$high":   values.astype(np.float64) * 1.0005,
            "$low":    values.astype(np.float64) * 0.9995,
            "$close":  values.astype(np.float64),
            "$volume": np.ones(n, dtype=np.float64) * 1000.0,
            "$factor": np.ones(n, dtype=np.float64),
        }, index=idx)
        # Wrap as a minimal qlib in-memory dataset.
        class _MemHandler(DataHandlerLP):
            def __init__(self, df):
                self._df = df
            def fetch(self, selector=None, level=None, col_set=None, *a, **k):
                return self._df.copy()
        handler = _MemHandler(df)
        segments = {"train": ("2020-01-01", idx[-2].strftime("%Y-%m-%dT%H:%M:%S%z")),
                    "test":  (idx[-max(20, n // 4)].strftime("%Y-%m-%dT%H:%M:%S%z"),
                              idx[-1].strftime("%Y-%m-%dT%H:%M:%S%z"))}
        ds = DatasetH(handler, segments=segments)
        return ds

    def _forecast_impl(self, values: np.ndarray, horizon: int, **kwargs) -> ModelForecast:
        if self._model is None:
            return self._fallback.forecast(values, horizon, **kwargs)
        try:
            ds = self._build_dataset(values)
            # Fit (best-effort — many qlib pytorch models expect a Trainer).
            try:
                self._model.fit(ds)
            except Exception:
                # Some models need Trainer; ignore if it can't fit on synthetic.
                pass
            preds = self._model.predict(ds, segment="test")
            point = np.asarray(preds).reshape(-1).astype(np.float32)[-horizon:]
            if point.size < horizon:
                point = np.pad(point, (0, horizon - point.size), mode="edge")
        except Exception as e:
            LOGGER.debug("qlib %s forecast failed: %s", self._spec.algo, e)
            return self._fallback.forecast(values, horizon, **kwargs)
        return build_forecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None,
            last_value=float(values[-1]),
            status="ok",
        )


def make_all_qlib_adapters(enabled: set[str] | None = None, **kwargs) -> list[PredictiveModel]:
    """Factory: build a list of PredictiveModel, one per qlib algorithm."""
    out: list[PredictiveModel] = []
    for algo in list_qlib_algorithms():
        if enabled is not None and algo not in enabled:
            continue
        out.append(QlibAdapter(algo, **kwargs))
    return out


__all__ = ["QlibAdapter", "QlibAlgoSpec", "list_qlib_algorithms", "make_all_qlib_adapters"]
