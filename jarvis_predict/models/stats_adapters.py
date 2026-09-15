"""
Statistical + gradient-boosted forecasting families — installed top-20
forecasters wired straight into the fleet:

    arima       — statsmodels ARIMA  (BIC order selection)
    ets         — statsmodels Holt-Winters exponential smoothing
    auto_arima  — pmdarima auto_arima (seasonal-capable model search)
    svr         — scikit-learn support vector regression
    forest      — scikit-learn extra / random forest regression
    boost       — xgboost | lightgbm | catboost gradient boosting

Every family follows the canonical degradation contract used across the
fleet: `_load_impl()` never raises — a missing dependency flips the model to
status "mock" and forecasting falls back to a labelled momentum backfill, so
the ensemble always receives this model's vote.  Provenance stays honest via
extras.mode (e.g. "statsmodels.arima", "sklearn.svr", "xgboost").
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from .base import PredictiveModel, ModelForecast
from ._interp import build_forecast

LOGGER = logging.getLogger("jarvis.models.stats")


# ---------------------------------------------------------------- features

def _logsafe(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float64), 1e-9)


def _dataset(values: np.ndarray, window: int, steps: int = 1):
    """Windowed feature rows + aligned forward log-return targets.

    Returns (X, y) where row i holds features of the closes up to time i and
    y[i] is the log-return from i to i+steps (multi-bar momentum target).
    Rows whose target would run past the series end are dropped.  All rows
    are scale-free and built only from the close series, so the same feature
    vector exists at the final prediction instant (see `_last_row`).
    """
    x = _logsafe(values)
    n = x.size
    s = max(1, int(steps))
    if n < window + s + 2:
        return None
    lr = np.log(x[1:]) - np.log(x[:-1])
    rows: list[list[float]] = []
    targets: list[float] = []
    for i in range(window, n - s):
        seg = x[i - window + 1: i + 1]
        lr_seg = lr[i - window: i]
        i3, i5, i10 = max(0, i - 3), max(0, i - 5), max(0, i - 10)
        r1 = float(lr[i - 1])
        r3 = float(np.log(x[i]) - np.log(x[i3]))
        r5 = float(np.log(x[i]) - np.log(x[i5]))
        r10 = float(np.log(x[i]) - np.log(x[i10]))
        mom = float(np.log(seg[-1]) - np.log(seg[0]))
        vol = float(np.std(lr_seg)) if lr_seg.size > 1 else 0.0
        mu, sd = float(np.mean(seg)), float(np.std(seg)) or 1e-9
        z = (seg[-1] - mu) / sd
        tail4 = x[i - 4: i + 1] if i >= 4 else seg
        tt = np.arange(tail4.size, dtype=np.float64)
        slope = float(np.polyfit(tt, tail4, 1)[0]) / max(1e-9, float(np.mean(np.abs(tail4))))
        rows.append([r1, r3, r5, r10, mom, vol, z, slope])
        targets.append(float(np.log(x[i + s]) - np.log(x[i])))
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _last_row(values, window: int) -> np.ndarray | None:
    """Single feature row for the most recent close — predicts the next step."""
    x = _logsafe(values)
    n = x.size
    if n < window + 2:
        return None
    lr = np.log(x[1:]) - np.log(x[:-1])
    seg = x[n - window: n]
    lr_seg = lr[n - window - 1: n - 1]
    i = n - 1
    i3, i5, i10 = max(0, i - 3), max(0, i - 5), max(0, i - 10)
    r1 = float(lr[i - 1])
    r3 = float(np.log(x[i]) - np.log(x[i3]))
    r5 = float(np.log(x[i]) - np.log(x[i5]))
    r10 = float(np.log(x[i]) - np.log(x[i10]))
    mom = float(np.log(seg[-1]) - np.log(seg[0]))
    vol = float(np.std(lr_seg)) if lr_seg.size > 1 else 0.0
    mu, sd = float(np.mean(seg)), float(np.std(seg)) or 1e-9
    z = (seg[-1] - mu) / sd
    tail4 = x[i - 4: i + 1] if i >= 4 else seg
    tt = np.arange(tail4.size, dtype=np.float64)
    slope = float(np.polyfit(tt, tail4, 1)[0]) / max(1e-9, float(np.mean(np.abs(tail4))))
    return np.asarray([[r1, r3, r5, r10, mom, vol, z, slope]], dtype=np.float64)


def _residual_fan(resid: np.ndarray, last_level: float, horizon: int) -> np.ndarray:
    """9-level probabilistic cone from the residual distribution.

    Forecast errors grow roughly with sqrt(horizon) under an i.i.d. residual
    assumption — produce a (horizon, 9) quantile fan for the noise-aware
    confidence logic in `build_forecast`.
    """
    h = max(1, int(horizon))
    out = np.zeros((h, 9), dtype=np.float32)
    if resid is None or resid.size < 2:
        out[:] = float(last_level)
        return out
    quants = np.quantile(resid, np.linspace(0.1, 0.9, 9))
    for t in range(1, h + 1):
        out[t - 1] = float(last_level) * np.exp(quants * np.sqrt(t))
    return out


def _level_from_ret(level: float, step_ret: float) -> float:
    return level * math.exp(step_ret)


def _momentum_backfill(model: "PredictiveModel", values, horizon: int) -> ModelForecast:
    """Deterministic fallback — trend-extrapolated path, honestly labelled."""
    x = _logsafe(values)
    h = max(1, int(horizon))
    last = float(x[-1]) if x.size else 0.0
    pt_levels: list[float] = []
    if x.size >= 8:
        seg = x[-min(24, x.size):]
        mom = np.log(seg[-1]) - np.log(seg[0])
        slope = mom / max(seg.size - 1, 1.0)
        for t in range(1, h + 1):
            pt_levels.append(seg[-1] * math.exp(slope * t))
    else:
        pt_levels = [last] * h
    point = np.asarray(pt_levels, dtype=np.float32)
    return build_forecast(
        model_name=model.name, family=model.family,
        point=point, quantiles=None, last_value=last,
        extras={"mode": "stats_backfill", "fallback": model.family},
    )


def _need(module: str) -> None:
    __import__(module)


def _require(*mods: str) -> None:
    for m in mods:
        _need(m)


# ------------------------------------------------------------- shared bases

class _TrendFamilyAdapter(PredictiveModel):
    """Base for classical statistical families: fit → level path + residuals.

    Subclasses provide `_dep_mods` (modules to import) and `_fit_path(values,
    horizon) -> (path, resid, desc)`.
    """

    family = "stats_trend"
    _dep_mods: tuple[str, ...] = ("statsmodels",)
    _dep_label = "statsmodels"

    def __init__(self, lookback: int = 200, seed: int = 11, **kwargs: Any):
        super().__init__()
        self.lookback = max(40, int(lookback))
        self.seed = int(seed)
        self._deps_ok = False

    # -- lifecycle ---------------------------------------------------------
    def _load_impl(self) -> None:
        try:
            _require(*self._dep_mods)
            self._deps_ok = True
            self._status = "ready"
        except Exception as e:
            self._deps_ok = False
            self._last_error = (f"{self._dep_label} unavailable — running "
                                f"momentum backfill: {e}")
            LOGGER.warning("Model %s %s", self.name, self._last_error)
            self._status = "mock"

    def _forecast_impl(self, values, horizon, **kwargs) -> ModelForecast:
        if not self._deps_ok:
            return _momentum_backfill(self, values, horizon)
        t0 = time.perf_counter()
        seg = _logsafe(values)[-self.lookback:]
        try:
            path, resid, desc = self._fit_path(seg, int(horizon))
        except Exception as e:
            LOGGER.debug("Model %s forecast failed: %s", self.name, e)
            return _momentum_backfill(self, values, horizon)
        anchor = float(seg[-1])
        extras = {"mode": self._dep_label, "spec": desc}
        return build_forecast(
            model_name=self.name, family=self.family,
            point=np.asarray(path, dtype=np.float32),
            quantiles=_residual_fan(np.asarray(resid), anchor, int(horizon)),
            last_value=anchor, inference_ms=(time.perf_counter() - t0) * 1000.0,
            extras=extras,
        )

    def _fit_path(self, values: np.ndarray, horizon: int):
        raise NotImplementedError


class _WindowMLAdapter(PredictiveModel):
    """Base for feature-window probabilistic classifiers (SVC / forests /
    boosting): learn the *direction* of the next step, convert the class
    probability edge into a scaled expected move, and produce an honest
    level path + quantile cone.
    """

    family = "ml_window"
    _dep_mods: tuple[str, ...] = ("sklearn",)
    _dep_label = "sklearn"

    def __init__(self, window: int = 32, seed: int = 7, mode: str = "blend",
                 steps: int = 4, **kwargs: Any):
        super().__init__()
        self.window = max(12, int(window))
        self.seed = int(seed)
        self.mode = str(mode)
        self.steps = max(1, int(steps))
        self._deps_ok = False

    # -- lifecycle ---------------------------------------------------------
    def _load_impl(self) -> None:
        try:
            _require(*self._dep_mods)
            self._deps_ok = True
            self._status = "ready"
        except Exception as e:
            self._deps_ok = False
            self._last_error = (f"{self._dep_label} unavailable — running "
                                f"momentum backfill: {e}")
            LOGGER.warning("Model %s %s", self.name, self._last_error)
            self._status = "mock"

    def _make_classifier(self, seed: int):
        raise NotImplementedError

    def _forecast_impl(self, values, horizon, **kwargs) -> ModelForecast:
        if not self._deps_ok:
            return _momentum_backfill(self, values, horizon)
        t0 = time.perf_counter()
        x = _logsafe(values)
        try:
            ds = _dataset(x, self.window, self.steps)
            if ds is None or ds[1].size < 10:
                return _momentum_backfill(self, values, horizon)
            X, y_ret = ds
            # Binary direction flags — every classifier accepts {0,1}.
            y_flag = np.where(y_ret >= 0, 1, 0).astype(int)
            clf, desc = self._make_classifier(self.seed)
            clf.fit(X, y_flag)
            classes = [int(c) for c in np.asarray(clf.classes_)]
            if set(classes) != {0, 1}:
                return _momentum_backfill(self, values, horizon)
            sigma_h = float(np.std(y_ret)) or 1e-6
            resid = np.asarray(y_ret, dtype=np.float64)

            # Recursive multi-bar path: each prediction advances `steps` bars.
            work = list(x)
            seg_levels: list[float] = []
            dirs: list[int] = []
            edges: list[float] = []
            n_steps = int(math.ceil(int(horizon) / self.steps))
            guard = 0
            while len(seg_levels) < n_steps and guard < n_steps + 64:
                guard += 1
                row = _last_row(work, self.window)
                if row is None:
                    break
                proba = clf.predict_proba(row)[0]
                p_up = float(proba[classes.index(1)])
                p_down = float(proba[classes.index(0)])
                edge = abs(p_up - p_down)
                step_dir = 1 if p_up >= p_down else -1
                step_ret = step_dir * max(0.0, edge) * sigma_h
                work.append(_level_from_ret(work[-1], step_ret))
                seg_levels.append(work[-1])
                dirs.append(step_dir)
                edges.append(edge)

            # Interpolate the coarse seg path to the exact horizon.
            hh = int(horizon)
            if not seg_levels:
                seg_levels = [float(x[-1])] * max(1, n_steps)
            if len(seg_levels) < n_steps:
                while len(seg_levels) < n_steps:
                    seg_levels.append(seg_levels[-1])
            base = float(x[-1])
            if n_steps == 1:
                point = np.full(hh, float(seg_levels[0]), dtype=np.float32)
            else:
                grid = np.linspace(0.0, 1.0, n_steps)
                wanted = np.linspace(0.0, 1.0, hh)
                interp = np.interp(wanted, grid, np.asarray(seg_levels))
                point = np.asarray(interp, dtype=np.float32)

            anchor = base
            dom = int(np.sign(np.sum(dirs))) if dirs else 0
            extras = {
                "mode": self._dep_label, "spec": desc, "features": 8,
                "window": self.window, "personality": self.mode,
                "steps": self.steps,
                "step_agreement": round(float(
                    np.mean([d == dom for d in dirs])) if dom != 0 else 0.0, 3),
                "mean_edge": round(float(np.mean(edges)), 3),
            }
            return build_forecast(
                model_name=self.name, family=self.family,
                point=point, quantiles=_residual_fan(resid, anchor, hh),
                last_value=anchor, inference_ms=(time.perf_counter() - t0) * 1000.0,
                extras=extras,
            )
        except Exception as e:
            LOGGER.debug("Model %s forecast failed: %s", self.name, e)
            return _momentum_backfill(self, values, horizon)


# ------------------------------------------------------------- ARIMA (statsmodels)
class ArimaAdapter(_TrendFamilyAdapter):
    name = "arima"
    family = "arima"
    description = "statsmodels ARIMA with BIC-selected (p, q) order"
    _dep_mods = ("statsmodels",)
    _dep_label = "statsmodels.arima"

    def _fit_path(self, values: np.ndarray, horizon: int):
        import warnings
        import statsmodels.api as sm

        best_fit: Any = None
        best_order = (1, 1)
        best_bic = math.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, q in ((0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (2, 2)):
                try:
                    fit = sm.tsa.ARIMA(values, order=(p, 1, q)).fit()
                    bic = float(fit.bic)
                    if bic < best_bic:
                        best_bic, best_fit, best_order = bic, fit, (p, q)
                except Exception:
                    continue
        if best_fit is None:
            raise RuntimeError("ARIMA fit failed")
        path = np.asarray(best_fit.forecast(steps=horizon), dtype=np.float64)
        resid = np.asarray(best_fit.resid, dtype=np.float64)
        return path, resid, f"ARIMA{best_order}"


# ------------------------------------------------- ETS (Holt-Winters, statsmodels)
class EtsAdapter(_TrendFamilyAdapter):
    name = "ets"
    family = "ets"
    description = "statsmodels Holt-Winters exponential smoothing (AIC trend)"
    _dep_mods = ("statsmodels",)
    _dep_label = "statsmodels.ets"

    def _fit_path(self, values: np.ndarray, horizon: int):
        import warnings
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        best_fit: Any = None
        best_aic = math.inf
        best_trend = "add"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trend in ("add", "mul"):
                try:
                    fit = ExponentialSmoothing(
                        values, trend=trend, damped_trend=(trend == "add"),
                        seasonal=None, initialization_method="estimated",
                    ).fit()
                    aic = float(fit.aic)
                    if aic < best_aic:
                        best_aic, best_fit, best_trend = aic, fit, trend
                except Exception:
                    continue
        if best_fit is None:
            raise RuntimeError("ETS fit failed")
        path = np.asarray(best_fit.forecast(steps=horizon), dtype=np.float64)
        resid = np.asarray(values - best_fit.fittedvalues, dtype=np.float64)
        return path, resid, f"ETS-{best_trend}"


# ------------------------------------------------- auto_arima (pmdarima)
class AutoArimaAdapter(_TrendFamilyAdapter):
    name = "auto_arima"
    family = "auto_arima"
    description = "pmdarima auto_arima — automated (p,d,q) search"
    _dep_mods = ("pmdarima",)
    _dep_label = "pmdarima.autoarima"

    def _fit_path(self, values: np.ndarray, horizon: int):
        from pmdarima import auto_arima

        fit = auto_arima(
            values, start_p=0, max_p=3, d=None, max_d=1,
            start_q=0, max_q=3, seasonal=False, stepwise=True,
            error_action="ignore", suppress_warnings=True,
        )
        path = np.asarray(fit.predict(n_periods=horizon), dtype=np.float64)
        resid = np.asarray(fit.resid(), dtype=np.float64)
        order = tuple(getattr(fit, "order", (1, 1, 0)))
        return path, resid, f"AutoARIMA{order}"


# ------------------------------------------------- SVR (scikit-learn)
class SvrAdapter(_WindowMLAdapter):
    name = "svr"
    family = "svr"
    description = "scikit-learn rbf SVC (Platt probabilities) on horizon features"
    _dep_mods = ("sklearn",)
    _dep_label = "sklearn.svc"

    def _make_classifier(self, seed: int):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        svc = SVC(kernel="rbf", C=1.0, class_weight="balanced")
        pipe = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(svc, method="sigmoid", ensemble=False),
        )
        return pipe, "SVC-rbf"


# ------------------------------------------------- forest (scikit-learn)
class ForestAdapter(_WindowMLAdapter):
    name = "forest"
    family = "forest"
    description = "scikit-learn extra/random forest classifier on windowed features"
    _dep_mods = ("sklearn",)
    _dep_label = "sklearn.forest"

    def __init__(self, algo: str = "extra", n_estimators: int = 120,
                 max_depth: int = 6, **kwargs: Any):
        super().__init__(**kwargs)
        self.algo = str(algo).lower()
        self.n_estimators = max(20, int(n_estimators))
        self.max_depth = max(2, int(max_depth))

    def _make_classifier(self, seed: int):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        if self.algo == "rf":
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1)
            desc = f"RFC-{self.n_estimators}"
        else:
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1)
            desc = f"XTC-{self.n_estimators}"
        return clf, desc


# ------------------------------------------------- boost (xgboost / lightgbm / catboost)
class BoostAdapter(_WindowMLAdapter):
    name = "boost"
    family = "boost"
    description = "gradient-boosted tree classifier (xgboost / lightgbm / catboost)"
    _dep_mods = ("sklearn",)
    _dep_label = "boost"

    def __init__(self, algo: str = "xgboost", n_estimators: int = 90,
                 max_depth: int = 4, learning_rate: float = 0.06, **kwargs: Any):
        super().__init__(**kwargs)
        self.algo = str(algo).lower()
        self.n_estimators = max(30, int(n_estimators))
        self.max_depth = max(2, int(max_depth))
        self.learning_rate = min(0.5, max(0.01, float(learning_rate)))

    def _load_impl(self) -> None:
        mods = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
        mod = mods.get(self.algo, "xgboost")
        try:
            _require("sklearn", mod)
            self._deps_ok = True
            self._status = "ready"
        except Exception as e:
            self._deps_ok = False
            self._last_error = f"{self._dep_label}.{self.algo} unavailable — {e}"
            LOGGER.warning("Model %s %s", self.name, self._last_error)
            self._status = "mock"

    def _make_classifier(self, seed: int):
        if self.algo == "lightgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, random_state=seed,
                n_jobs=1, verbose=-1, class_weight="balanced")
            desc = f"LGBc-{self.n_estimators}"
        elif self.algo == "catboost":
            import catboost as cb
            clf = cb.CatBoostClassifier(
                iterations=self.n_estimators, depth=self.max_depth,
                learning_rate=self.learning_rate, random_state=seed,
                verbose=False, allow_writing_files=False,
                auto_class_weights="Balanced")
            desc = f"CATc-{self.n_estimators}"
        else:
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, random_state=seed,
                n_jobs=1, verbosity=0,
                scale_pos_weight=1.0)
            desc = f"XGBc-{self.n_estimators}"
        return clf, desc


_PREDICTOR_STACK: list[tuple[str, str, str]] = [
    ("statsmodels.arima", "statsmodels", "ARIMA / ETS statistical forecasting"),
    ("pmdarima",          "pmdarima",    "auto_arima model search"),
    ("sklearn.svr",       "sklearn",     "support vector regression"),
    ("sklearn.forest",    "sklearn",     "extra / random forest regression"),
    ("xgboost",           "xgboost",     "gradient boosted tree regression"),
    ("lightgbm",          "lightgbm",    "gradient boosted tree regression"),
    ("catboost",          "catboost",    "gradient boosted tree regression"),
]


def discover() -> dict[str, str]:
    """Probe which installed predictor families are importable right now."""
    out: dict[str, str] = {}
    for label, mod, _desc in _PREDICTOR_STACK:
        try:
            _need(mod)
            out[label] = "ready"
        except Exception:
            out[label] = "missing"
    return out


__all__ = [
    "ArimaAdapter", "EtsAdapter", "AutoArimaAdapter",
    "SvrAdapter", "ForestAdapter", "BoostAdapter",
    "discover",
]