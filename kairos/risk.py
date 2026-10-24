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


# -------------------------------------------------------------- cluster budget
# `rules.max_correlation` was declared as a portfolio correlation cap and read by
# nothing: every entry was sized on its own conviction, so one cycle could put
# eight PSX scrips on the book whose five-day windows overlap and whose returns
# move together — one macro shock, eight stops. This is what the key means now.
#
# A pair whose histories do not overlap is not evidence of independence. Two
# same-market names with a short store are assumed 0.75, which is inside the
# cluster; a cross-market pair gets 0.40, which is not. Refusing on unknown would
# freeze a fresh deploy, and treating unknown as zero is how a correlation cap
# becomes decoration.
UNKNOWN_RHO_SAME_MARKET = 0.75
UNKNOWN_RHO_CROSS_MARKET = 0.40
MIN_RETURNS = 15


def daily_returns(close: pd.Series, n: int = 90) -> pd.Series | None:
    """Simple daily returns for correlation, or None when too few are printed."""
    try:
        s = pd.to_numeric(pd.Series(close), errors="coerce").dropna().astype(float)
    except (TypeError, ValueError):
        return None
    if len(s) < MIN_RETURNS + 1:
        return None
    r = s.tail(n + 1).pct_change().dropna()
    return r if len(r) >= MIN_RETURNS else None


def returns_from_bars(df: pd.DataFrame | None, n: int = 90) -> pd.Series | None:
    """Daily returns from a stored bar frame, indexed by date.

    The index is deduplicated and sorted before the returns are taken: a store
    that has ever written the same session twice would otherwise align two
    series into a cartesian pair, and a correlation computed on that is noise
    wearing a number.
    """
    if df is None or len(df) < 2 or "close" not in getattr(df, "columns", []):
        return None
    try:
        idx = (pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns
               else pd.RangeIndex(len(df)))
        close = pd.Series(
            pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float),
            index=idx)
        close = close[~close.index.duplicated(keep="last")].sort_index()
    except (TypeError, ValueError):
        return None
    return daily_returns(close, n)


def correlation(a: pd.Series | None, b: pd.Series | None) -> float | None:
    """Pearson correlation on the dates both series share, or None.

    None means "not enough common history to answer", not zero: the caller
    decides what an unknown pair is assumed to be.
    """
    if a is None or b is None:
        return None
    joined = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    if len(joined) < MIN_RETURNS:
        return None
    if joined["a"].std() == 0 or joined["b"].std() == 0:
        return None
    rho = float(joined["a"].corr(joined["b"]))
    return None if rho != rho else rho


def cluster_budget(candidate: dict, book: list[dict], equity: float,
                   rho_cap: float = 0.7,
                   cluster_cap_pct: float = 30.0) -> dict:
    """How much of `candidate['notional']` one correlated cluster can still take.

    Membership is the declared `max_correlation`: a same-side holding joins the
    cluster when the return correlation is at or above the cap. Opposite sides
    are not summed — a long and a short that move together offset rather than
    compound — and a holding whose notional cannot be read is treated as having
    spent the whole cluster cap, because "unreadable" is the conservative
    answer and not the free one.

    Returns `allowed_notional` (never more than the candidate asked for), the
    cluster's share of equity after the add, and a reason when nothing fits.
    """
    out = {"allowed_notional": 0.0, "rho_max": None, "cluster_pct": 0.0,
           "members": [], "scaled": False, "reason": ""}
    try:
        eq = float(equity)
    except (TypeError, ValueError):
        eq = 0.0
    try:
        want = max(0.0, float(candidate.get("notional") or 0.0))
    except (TypeError, ValueError):
        want = 0.0
    if eq <= 0:
        out["reason"] = "no usable equity figure — cluster cap not evaluated"
        return out
    if want <= 0:
        out["reason"] = "no notional to size"
        return out
    try:
        cap = eq * max(0.0, float(cluster_cap_pct)) / 100.0
    except (TypeError, ValueError):
        cap = 0.0
    if cap <= 0:
        out["reason"] = "cluster cap is not a usable percentage"
        return out

    side = str(candidate.get("side", "")).upper()
    market = str(candidate.get("market", "")).upper()
    held, rho_max, unreadable = 0.0, None, False
    for pos in book or []:
        if str(pos.get("side", "")).upper() != side:
            continue
        rho = correlation(candidate.get("returns"), pos.get("returns"))
        if rho is None:
            rho = (UNKNOWN_RHO_SAME_MARKET
                   if market and market == str(pos.get("market", "")).upper()
                   else UNKNOWN_RHO_CROSS_MARKET)
        rho_max = rho if rho_max is None else max(rho_max, rho)
        if rho < rho_cap:
            continue
        name = str(pos.get("symbol") or pos.get("key") or "?")
        out["members"].append(name)
        try:
            held += max(0.0, float(pos.get("notional") or 0.0))
        except (TypeError, ValueError):
            unreadable = True
    if unreadable:
        held = cap

    allowed = max(0.0, min(want, cap - held))
    out.update(allowed_notional=round(allowed, 2),
               rho_max=None if rho_max is None else round(rho_max, 3),
               cluster_pct=round((held + allowed) / eq * 100, 1),
               scaled=allowed + 1e-9 < want)
    if allowed <= 0 and out["members"]:
        shown = ", ".join(out["members"][:3]) + ("…" if len(out["members"]) > 3 else "")
        out["reason"] = (f"correlated cluster ({shown}) already at "
                         f"{held / eq * 100:.0f}% of equity, at its "
                         f"{cluster_cap_pct:g}% cap")
    return out
