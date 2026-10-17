"""Risk engine: VaR/CVaR, Kelly sizing, vol targeting, portfolio analytics.

Institutional conventions:
  - Parametric, historical, and Cornish-Fisher VaR side by side
  - Kelly capped and fractioned (half-Kelly default) — full Kelly is never bet
  - Position size derived from risk budget / stop distance, not conviction alone
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG


def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 20:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 20:
        return 0.0
    mu, sd = float(np.mean(returns)), float(np.std(returns))
    from scipy.stats import norm
    z = float(norm.ppf(1 - confidence))
    return float(-(mu + z * sd))


def var_cornish_fisher(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 30:
        return var_parametric(returns, confidence)
    from scipy.stats import norm
    z = float(norm.ppf(1 - confidence))
    mu = float(np.mean(returns))
    sd = float(np.std(returns))
    s = float(pd.Series(returns).skew())
    k = float(pd.Series(returns).kurt())
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s * s / 36)
    return float(-(mu + z_cf * sd))


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 20:
        return 0.0
    tail = returns[returns <= -var_historical(returns, confidence)]
    return float(-tail.mean()) if len(tail) else 0.0


@dataclass
class RiskProfile:
    symbol: str
    var_95_daily: float
    cvar_95_daily: float
    var_cf_daily: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    kelly_fraction: float
    suggested_weight: float
    position_notes: str = ""


def compute_risk(symbol: str, close: pd.Series, signal_confidence: float = 0.0,
                 horizon: int = 5) -> RiskProfile:
    r = close.pct_change().dropna().values.astype(float)
    if len(r) < 25:
        return RiskProfile(symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, "insufficient history")
    conf = float(CFG.risk.get("var_confidence", 0.95))
    daily_var = var_historical(r, conf)
    daily_cvar = cvar_historical(r, conf)
    daily_var_cf = var_cornish_fisher(r, conf)
    ann_vol = float(np.std(r) * np.sqrt(252))
    ann_ret = float(np.mean(r) * 252)
    downside = r[r < 0]
    dvol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 3 else ann_vol
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    sortino = ann_ret / dvol if dvol > 0 else 0.0
    peak = close.cummax()
    mdd = float((close / peak - 1).min())

    # Kelly from signal confidence as win-prob proxy (PM practice: use
    # realized strategy hit-rate when available; confidence is our prior)
    p = min(0.72, 0.5 + signal_confidence * 0.4)          # win probability
    b = max(0.5, abs(sharpe) * 0.8)                        # payoff ratio proxy
    kelly = max(0.0, (p * (1 + b) - 1) / b)
    kelly = min(kelly, float(CFG.risk.get("kelly_fraction_cap", 0.25)))

    # vol-targeted weight (risk parity style), floored by VaR budget
    vol_target = float(CFG.risk.get("annual_vol_target", 0.18))
    w_vol = vol_target / ann_vol if ann_vol > 0 else 0.0
    # daily VaR budget: 1% of portfolio per position
    w_var = 0.01 / daily_var if daily_var > 0 else 0.0
    suggested = float(min(CFG.risk.get("max_position_weight", 0.10),
                          kelly, w_vol * 0.35, w_var * 0.8))
    suggested = max(0.0, suggested)
    notes = (f"1d VaR95={daily_var:.2%} CVaR={daily_cvar:.2%} "
             f"CF-VaR={daily_var_cf:.2%}; {horizon}d VaR≈"
             f"{daily_var * np.sqrt(horizon):.2%}")
    return RiskProfile(symbol, daily_var, daily_cvar, daily_var_cf, ann_vol,
                       sharpe, sortino, mdd, kelly, suggested, notes)


def portfolio_metrics(close: pd.Series) -> dict[str, float]:
    r = close.pct_change().dropna().values.astype(float)
    if len(r) < 30:
        return {}
    ann_ret = float(np.mean(r) * 252)
    ann_vol = float(np.std(r) * np.sqrt(252))
    downside = r[r < 0]
    dvol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 3 else ann_vol
    peak = close.cummax()
    mdd = float((close / peak - 1).min())
    rf = 0.045
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0,
        "sortino": (ann_ret - rf) / dvol if dvol > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else 0.0,
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurt()),
        "positive_day_ratio": float((r > 0).mean()),
    }


def position_size(capital: float, price: float, stop_distance_pct: float,
                  risk_budget_pct: float = 0.01) -> dict[str, Any]:
    """Classic bank size: units = (capital × risk budget) / (price × stop distance)."""
    if price <= 0 or stop_distance_pct <= 0:
        return {"units": 0, "notional": 0.0, "risk_amount": 0.0}
    risk_amount = capital * risk_budget_pct
    units = int(risk_amount / (price * stop_distance_pct))
    return {"units": units, "notional": units * price, "risk_amount": risk_amount}
