"""Statistical model battery: the classical quant forecasting stack.

Each model returns a ModelForecast with direction, expected return over the
prediction horizon, and a confidence derived from in-sample fit quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelForecast:
    model: str
    direction: int                      # -1 / 0 / +1
    expected_return: float              # over horizon (log-ish simple return)
    horizon: int                        # trading days
    confidence: float                   # 0..1
    detail: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _direction(exp_ret: float, thr: float = 0.0008) -> int:
    return 1 if exp_ret > thr else -1 if exp_ret < -thr else 0


def _fit_confidence(scale: float) -> float:
    return float(np.clip(0.35 + scale, 0.1, 0.95))


# ---------------------------------------------------------------------------
# 1. ARIMA — auto-order via AIC grid
# ---------------------------------------------------------------------------
def arima_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return None
    y = np.log(close.values.astype(float))
    if len(y) < 60:
        return None
    y = y[-400:]
    # choose d by ADF test (AIC comparisons across differencing orders are invalid)
    try:
        d = 0 if adfuller(y, autolag="AIC")[1] < 0.05 else 1
    except Exception:
        d = 1
    best_aic, best_order, best_res = np.inf, (1, d, 1), None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in (0, 1, 2):
            for q in (0, 1, 2):
                try:
                    res = ARIMA(y, order=(p, d, q)).fit(method_kwargs={"maxiter": 60})
                    if res.aic < best_aic:
                        best_aic, best_order, best_res = res.aic, (p, d, q), res
                except Exception:
                    continue
    if best_res is None:
        return None
    try:
        fc = best_res.get_forecast(steps=horizon)
        pred = np.asarray(fc.predicted_mean)
        se = np.asarray(fc.se_mean)
        exp_ret = float(np.exp(pred[-1]) / close.iloc[-1] - 1)
        sigma_h = float(se[-1]) if np.size(se) else 0.02
        # honest confidence: signal-to-noise of the forecast move
        snr = abs(pred[-1] - y[-1]) / (sigma_h or 1e-9)
        conf = _fit_confidence(0.15 + min(0.45, snr * 0.35))
        return ModelForecast("ARIMA", _direction(exp_ret), exp_ret, horizon, conf,
                             f"order={best_order} aic={best_aic:.1f} "
                             f"forecast_se={sigma_h:.4f} snr={snr:.2f}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. GARCH(1,1) — variance forecast + drift (manual MLE, no arch dependency)
# ---------------------------------------------------------------------------
def _garch_mle(rets: np.ndarray) -> tuple[float, float, float] | None:
    r2 = rets ** 2
    n = len(r2)
    var0 = float(r2.mean()) or 1e-10

    def neg_ll(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha * r2[t - 1] + beta * sig2[t - 1]
        return float(np.mean(np.log(2 * np.pi * sig2) + r2 / sig2))

    try:
        res = minimize(neg_ll, x0=[var0 * 0.05, 0.08, 0.90],
                       method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-8, "fatol": 1e-8})
        w, a, b = res.x
        if w > 0 and a >= 0 and b >= 0 and a + b < 0.999:
            return float(w), float(a), float(b)
    except Exception:
        pass
    return None


def garch_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    rets = np.diff(np.log(close.values.astype(float)))[-400:]
    if len(rets) < 80:
        return None
    est = _garch_mle(rets)
    if est is None:
        return None
    omega, alpha, beta = est
    r2 = rets ** 2
    sig2 = float(r2[-1])
    for _ in range(horizon):
        sig2 = omega + alpha * r2[-1] + beta * sig2
    vol_h = float(np.sqrt(sig2))
    persistence = alpha + beta
    drift = float(rets[-60:].mean()) * horizon
    # GARCH is primarily a variance model: signal = vol-adjusted drift tilt
    exp_ret = float(drift - 0.5 * sig2)
    conf = _fit_confidence(0.2 + 0.35 * (1 - persistence))
    return ModelForecast("GARCH(1,1)", _direction(exp_ret), exp_ret, horizon, conf,
                         f"omega={omega:.2e} a={alpha:.3f} b={beta:.3f} "
                         f"vol_h={vol_h:.4f} persistence={persistence:.3f}",
                         extras={"vol_forecast": vol_h, "persistence": persistence})


# ---------------------------------------------------------------------------
# 3. Kalman filter — local level + local linear trend
# ---------------------------------------------------------------------------
def kalman_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    y = np.log(close.values.astype(float))
    if len(y) < 40:
        return None
    y = y[-300:]
    n = len(y)
    # state: [level, slope]
    x = np.array([y[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([1e-5, 1e-7])
    R = np.array([[np.var(np.diff(y)) * 0.25]]) if n > 2 else np.array([[1e-4]])
    innovation_vars: list[float] = []
    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y_hat = float((H @ x_pred)[0])
        S = float((H @ P_pred @ H.T)[0, 0]) + R[0][0]
        innovation = y[t] - y_hat
        K = (P_pred @ H.T) / S
        x = x_pred + (K.flatten() * innovation)
        P = (np.eye(2) - K @ H) @ P_pred
        innovation_vars.append(S)
    level, slope = float(x[0]), float(x[1])
    pred_log = level + slope * horizon
    exp_ret = float(np.exp(pred_log) / close.iloc[-1] - 1)
    innov_sd = float(np.sqrt(np.mean(innovation_vars[-60:]))) if len(innovation_vars) > 10 else 0.02
    conf = _fit_confidence(0.3 - innov_sd * 4)
    return ModelForecast("Kalman", _direction(exp_ret), exp_ret, horizon, conf,
                          f"level={level:.4f} slope={slope * 252 * 100:.2f}%/yr "
                          f"innov_sd={innov_sd:.4f}",
                          extras={"slope_annual": slope * 252})


# ---------------------------------------------------------------------------
# 4. Ornstein-Uhlenbeck mean reversion with half-life
# ---------------------------------------------------------------------------
def ou_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    y = np.log(close.values.astype(float))[-300:]
    if len(y) < 40:
        return None
    y_lag, y_now = y[:-1], y[1:]
    if np.var(y_lag) < 1e-14:
        return None
    phi = float(np.polyfit(y_lag, y_now, 1)[0])
    if phi >= 0.999 or phi <= 0:
        phi = 0.99
    half_life = abs(float(np.log(0.5) / np.log(phi)))
    mu = float(np.mean(y))
    theta = -np.log(phi)
    y_t = y[-1]
    revert = (mu - y_t) * (1 - np.exp(-theta * horizon))
    exp_ret = float(np.exp(y_t + revert) / np.exp(y_t) - 1) if revert != 0 else 0.0
    dist_sd = float((y_t - mu) / (np.std(y) or 1e-9))
    # stronger confidence when far from mean and fast reversion
    conf = _fit_confidence(0.25 + min(0.3, abs(dist_sd) * 0.12) + 0.1 * (half_life < 20))
    return ModelForecast("OU-MeanRev", _direction(exp_ret), exp_ret, horizon, conf,
                          f"phi={phi:.4f} half_life={half_life:.1f}d z={dist_sd:.2f}",
                          extras={"half_life_days": half_life, "z_score": dist_sd})


# ---------------------------------------------------------------------------
# 5. Holt-Winters exponential smoothing (damped trend)
# ---------------------------------------------------------------------------
def holtwinters_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return None
    y = close.values.astype(float)[-400:]
    if len(y) < 60:
        return None
    try:
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
        pred = float(fit.forecast(horizon)[-1])
        exp_ret = pred / close.iloc[-1] - 1
        resid_sd = float(np.std(fit.resid)) / float(np.mean(y) or 1.0)
        conf = _fit_confidence(0.3 - resid_sd * 8)
        return ModelForecast("Holt-Winters", _direction(exp_ret), float(exp_ret),
                              horizon, conf, f"damped trend, resid_rel_sd={resid_sd:.5f}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 6. Momentum-drift + Fourier seasonality hybrid
# ---------------------------------------------------------------------------
def fourier_momentum_forecast(close: pd.Series, horizon: int = 5) -> ModelForecast | None:
    y = np.log(close.values.astype(float))
    if len(y) < 120:
        return None
    y = y[-360:]
    t = np.arange(len(y), dtype=float)
    try:
        coeffs = np.polyfit(t, y, 1)
        trend = coeffs[0]
        resid = y - np.polyval(coeffs, t)
        harmonics = 3
        fft = np.fft.rfft(resid)
        amps = np.abs(fft) / len(resid)
        top = np.argsort(amps[1:harmonics * 6])[-harmonics:] + 1
        seasonality = 0.0
        for k in top:
            seasonality += (2 / len(resid)) * np.real(
                fft[k] * np.exp(2j * np.pi * k * (len(resid) + horizon) / len(resid)))
        # seasonal term must stay sub-noise: cap at half a residual sd, scaled by horizon
        cap = 0.5 * float(np.std(resid)) * min(1.0, horizon / 20.0)
        seasonality = float(np.clip(seasonality, -cap if cap > 0 else -0.01,
                                    cap if cap > 0 else 0.01))
        pred_log = y[-1] + trend * horizon + seasonality
        exp_ret = float(np.exp(pred_log) / np.exp(y[-1]) - 1)
        strength = min(1.0, abs(seasonality) / (np.std(resid) or 1e-9))
        conf = _fit_confidence(0.2 + 0.2 * strength)
        return ModelForecast("Fourier-Momentum", _direction(exp_ret), exp_ret, horizon,
                              conf, f"trend={trend * 252 * 100:.2f}%/yr "
                              f"seasonal_lift={seasonality:.4f}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7. Ensemble entry point
# ---------------------------------------------------------------------------
def run_statistical_battery(close: pd.Series, horizon: int = 5) -> list[ModelForecast]:
    forecasts: list[ModelForecast] = []
    for fn in (arima_forecast, garch_forecast, kalman_forecast, ou_forecast,
               holtwinters_forecast, fourier_momentum_forecast):
        try:
            out = fn(close, horizon)
            if out is not None:
                forecasts.append(out)
        except Exception:
            continue
    return forecasts
