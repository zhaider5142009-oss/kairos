"""Adaptive walk-forward backtesting + calibration + paper trading loop.

Institutional doctrine:
  1. Every instrument gets its OWN parameters, calibrated on a trailing
     window by grid search over the signal-blend space (trend / momentum /
     mean-reversion / breakout mix, entry threshold, regime gate, short mode).
  2. Trading happens ONLY out-of-sample: parameters fitted on the trailing
     window, re-calibrated every `refit_every` days, never on future bars.
  3. Risk overlays: volatility-targeted sizing, drawdown throttle, cost
     hurdle on every flip, hysteresis entry/exit bands.
  4. The same calibrated signal feeds probability calibration for live
     dossiers — the bridge between backtest and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG

VOL_TARGET = 0.20
CALIB_WINDOW = 252          # trailing bars used for parameter calibration
REFIT_EVERY = 30            # re-calibrate every N bars
SMOOTH_SPANS = (1, 3)       # raw and EMA-conditioned blends compete per instrument


@dataclass
class BacktestResult:
    symbol: str
    bars: int
    trades: int
    hit_rate: float
    total_return: float
    benchmark_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    exposure: float
    equity_curve: list[float]
    annual_return: float
    verdict: str
    params: dict[str, Any] = field(default_factory=dict)
    calibrations: int = 0


# ---------------------------------------------------------------------------
# Signal components — computed once per instrument, blended per parameters
# ---------------------------------------------------------------------------
def _components(df: pd.DataFrame) -> dict[str, pd.Series]:
    from .features import feature_panel, _adx
    p = feature_panel(df)
    rsi_str = (p["rsi14"] - 50).div(50).abs().clip(0, 1)
    trend = (0.65 * np.sign(p["sma50_d"]) * (0.5 + 0.5 * rsi_str) +
             0.35 * np.sign(p["macd_h_n"])).clip(-1, 1)
    mom = np.sign(p["ret_63"]) * 0.7 + np.sign(p["ret_21"]) * 0.3
    mr = -np.sign(p["bb_b"].sub(0.5).where(p["bb_b"].abs() > 1.0, 0.0))
    brk = np.sign(p["don_p"].sub(0.5).where(
        (p["don_p"] > 0.95) | (p["don_p"] < 0.05), 0.0))
    return {
        "trend": trend.clip(-1, 1),
        "mom": mom.clip(-1, 1),
        "mr": mr,
        "brk": brk,
        "adx": pd.Series(_adx(df).values, index=df.index),
        "sma200_d": p["sma200_d"],
        "ret63": p["ret_63"],
        "ret63_abs": p["ret_63"].abs(),
        "vol20": p["vol20"],
        "bb_b_abs": p["bb_b"].abs(),
    }


def _blended(comps: dict[str, pd.Series],
             w_t: float, w_m: float, w_r: float, w_b: float,
             adx_gate: float, shorts: str, span: int = 1) -> pd.Series:
    denom = w_t + w_m + w_r + w_b
    if denom <= 0:
        return pd.Series(0.0, index=comps["trend"].index)
    sig = (w_t * comps["trend"] + w_m * comps["mom"] +
           w_r * comps["mr"] + w_b * comps["brk"]) / denom
    # stand aside when neither trend nor dislocation is present
    gate = (comps["adx"] > adx_gate) | (comps["ret63_abs"] > 0.08) | \
           (comps["bb_b_abs"] > 1.5)
    sig = sig.where(gate, 0.0)
    if shorts == "off":
        sig = sig.clip(lower=0.0)
    else:  # "bear": shorts only in confirmed bear structure
        bear_ok = (comps["sma200_d"] < 0) & (comps["ret63"] < 0.08)
        sig = sig.where(~((sig < 0) & ~bear_ok), 0.0)
    if span > 1:
        sig = sig.ewm(span=span, adjust=False).mean()
    return sig.clip(-1, 1)


def _simulate(close: np.ndarray, sig: pd.Series, vol20: np.ndarray,
              start: int, end: int, cost: float,
              entry: float) -> dict[str, Any]:
    """Equity loop with hysteresis, vol targeting, and drawdown throttle."""
    s = sig.values
    position = 0.0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    exposed = 0
    days = 0
    daily: list[float] = []
    curve: list[float] = [1.0]
    exit_band = max(0.35 * entry, 0.05)
    throttle = 1.0
    for i in range(start, end - 1):
        v = float(vol20[i]) if np.isfinite(vol20[i]) and vol20[i] > 0 else 0.5
        size = float(np.clip(VOL_TARGET / v, 0.25, 1.0)) * throttle
        sv = float(s[i])
        if sv > entry and position <= 0:
            new_pos = size
        elif sv < -entry and position <= 0:
            new_pos = -size
        elif position != 0 and np.sign(position) * sv < exit_band:
            new_pos = 0.0
        else:
            new_pos = position
        if new_pos != position:
            if abs(new_pos) > abs(position) and equity < peak * 0.90:
                new_pos = 0.5 * new_pos      # drawdown throttle: half size
            equity *= (1 - cost)
            trades += 1
            position = new_pos
        r = close[i + 1] / close[i] - 1
        equity *= (1 + position * r)
        if position != 0:
            exposed += 1
        days += 1
        daily.append(position * r)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
        # release the throttle once equity recovers
        throttle = 1.0 if equity > peak * 0.97 else max(throttle, 0.5)
        curve.append(equity)
    dr = np.array(daily) if daily else np.array([0.0])
    ann = float(dr.mean() * 252) if len(dr) else 0.0
    vol = float(dr.std() * np.sqrt(252)) if len(dr) > 5 else 0.0
    sharpe = ann / vol if vol > 0 else 0.0
    downside = dr[dr < 0]
    dvol = float(downside.std() * np.sqrt(252)) if len(downside) > 3 else vol
    return {"equity": equity, "ann": ann, "sharpe": sharpe,
            "sortino": ann / dvol if dvol > 0 else 0.0, "mdd": mdd,
            "trades": trades, "exposure": exposed / max(days, 1),
            "daily": dr, "curve": curve}


def _grid() -> list[dict[str, Any]]:
    out = []
    for w_t in (0.20, 0.40, 0.60):
        for w_m in (0.10, 0.25, 0.40):
            for w_r in (0.05, 0.25, 0.45):
                for w_b in (0.05, 0.20):
                    for entry in (0.12, 0.22, 0.32, 0.45):
                        for adx_gate in (10, 20, 30):
                            for shorts in ("bear", "off"):
                                out.append({"w_t": w_t, "w_m": w_m, "w_r": w_r,
                                            "w_b": w_b, "entry": entry,
                                            "adx_gate": adx_gate, "shorts": shorts})
    return out


def _calibrate(close: np.ndarray, comps: dict[str, pd.Series],
               vol20: np.ndarray, start: int, end: int,
               cost: float) -> dict[str, Any]:
    """Grid search on [start,end): maximize a Sharpe/turnover/drawdown
    objective. Purely in-window — no future leakage."""
    best, best_obj = None, -1e9
    for prm in _grid():
        for span in SMOOTH_SPANS:
            sig = _blended(comps, prm["w_t"], prm["w_m"], prm["w_r"], prm["w_b"],
                           prm["adx_gate"], prm["shorts"], span)
            sim = _simulate(close, sig, vol20, start, end, cost, prm["entry"])
            if sim["trades"] < 6:
                continue                   # not statistically meaningful
            tpy = sim["trades"] * 252 / max(end - start - 1, 1)
            obj = (sim["sharpe"]
                   + 2.0 * (sim["equity"] - 1.0)               # reward real money
                   + 1.5 * min(sim["mdd"] + 1.0, 1.0)          # reward shallow drawdowns
                   - 0.015 * tpy)                              # turnover penalty
            if obj > best_obj:
                best_obj, best = obj, {**prm, "span": span}
    return best or {"w_t": 0.40, "w_m": 0.25, "w_r": 0.05, "w_b": 0.05,
                    "entry": 0.22, "adx_gate": 20, "shorts": "bear", "span": 1}


# ---------------------------------------------------------------------------
# Public walk-forward entry point
# ---------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, symbol: str, train_window: int = CALIB_WINDOW,
                 step: int = 1, cost_bps: float | None = None,
                 test_days: int | None = None) -> BacktestResult:
    if df is None or len(df) < 140:
        raise ValueError("insufficient history for backtest (need >= 140 bars)")
    inst_cost = cost_bps
    if inst_cost is None:
        inst_cost = float(CFG.section("backtest").get("psx_cost_bps", 60) if
                          symbol in {s.upper() for s in _psx_keys()} else
                          CFG.section("backtest").get("global_cost_bps", 12))
    cost = inst_cost / 10_000

    comps = _components(df)
    close = df["close"].values.astype(float)
    vol20 = comps["vol20"].values
    n = len(df)

    # true out-of-sample: trade only the final segment. test_days (from the
    # --days flag) sets that segment length directly; otherwise fall back to
    # the trailing ~45% of the series.
    if test_days:
        test_start = max(60, n - int(test_days))
    else:
        test_start = max(60, n - max(train_window, int(n * 0.55)))
    if n - test_start < 60:
        test_start = n - 60

    # ---- rolling walk-forward with periodic re-calibration -----------------
    position = 0.0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    exposed = 0
    daily: list[float] = []
    curve: list[float] = [1.0]
    calibrations = 0
    params = _calibrate(close, comps, vol20, 0, test_start, cost)
    calibrations += 1
    exit_band = max(0.35 * params["entry"], 0.05)
    throttle = 1.0
    sig = _blended(comps, params["w_t"], params["w_m"], params["w_r"],
                   params["w_b"], params["adx_gate"], params["shorts"],
                   params.get("span", 1)).values
    fwd5: dict[int, float] = {}

    for i in range(test_start, n - 1):
        if i > test_start and (i - test_start) % REFIT_EVERY == 0:
            w_start = max(0, i - train_window)
            params = _calibrate(close, comps, vol20, w_start, i, cost)
            calibrations += 1
            exit_band = max(0.35 * params["entry"], 0.05)
            sig = _blended(comps, params["w_t"], params["w_m"], params["w_r"],
                           params["w_b"], params["adx_gate"],
                           params["shorts"], params.get("span", 1)).values
        v = float(vol20[i]) if np.isfinite(vol20[i]) and vol20[i] > 0 else 0.5
        size = float(np.clip(VOL_TARGET / v, 0.25, 1.0)) * throttle
        sv = float(sig[i])
        if abs(sv) > params["entry"]:
            fwd5[i] = sv
        if sv > params["entry"] and position <= 0:
            new_pos = size
        elif sv < -params["entry"] and position <= 0:
            new_pos = -size
        elif position != 0 and np.sign(position) * sv < exit_band:
            new_pos = 0.0
        else:
            new_pos = position
        if new_pos != position:
            if abs(new_pos) > abs(position) and equity < peak * 0.90:
                new_pos = 0.5 * new_pos
            equity *= (1 - cost)
            trades += 1
            position = new_pos
        r = close[i + 1] / close[i] - 1
        equity *= (1 + position * r)
        if position != 0:
            exposed += 1
        daily.append(position * r)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
        throttle = 1.0 if equity > peak * 0.97 else max(throttle, 0.5)
        curve.append(equity)

    # directional hit-rate at 5d horizon on strong-signal days
    aligned = checked = 0
    for i, sv in fwd5.items():
        j = min(i + 5, n - 1)
        checked += 1
        if np.sign(sv) == np.sign(close[j] / close[i] - 1):
            aligned += 1
    hit_rate = aligned / checked if checked else 0.5

    dr = np.array(daily) if daily else np.array([0.0])
    total_ret = equity - 1
    bench = close[-1] / close[test_start] - 1
    ann = float(dr.mean() * 252)
    vol = float(dr.std() * np.sqrt(252)) if len(dr) > 5 else 0.0
    sharpe = ann / vol if vol > 0 else 0.0
    downside = dr[dr < 0]
    dvol = float(downside.std() * np.sqrt(252)) if len(downside) > 3 else vol
    sortino = ann / dvol if dvol > 0 else 0.0
    verdict = ("PROFITABLE" if total_ret > 0 and sharpe > 0.5 else
               "MARGINAL" if total_ret > 0 else "UNPROFITABLE")
    return BacktestResult(
        symbol=symbol, bars=n, trades=trades, hit_rate=round(hit_rate, 3),
        total_return=round(total_ret, 4), benchmark_return=round(float(bench), 4),
        sharpe=round(sharpe, 3), sortino=round(sortino, 3),
        max_drawdown=round(mdd, 4),
        exposure=round(exposed / max(len(curve) - 1, 1), 3),
        equity_curve=[round(x, 5) for x in curve[-200:]],
        annual_return=round(ann, 4), verdict=verdict,
        params=params, calibrations=calibrations)


# ---------------------------------------------------------------------------
# Empirical probability calibration — bridges backtest evidence to live P(up)
# ---------------------------------------------------------------------------
def empirical_calibration(df: pd.DataFrame, horizon: int = 5) -> dict[str, Any] | None:
    """Bucket historical composite-signal scores, measure realized 5d outcomes,
    and return the empirical P(up) ladder + hit-rate. Used to calibrate the
    live probability_up so the system's confidence tracks reality."""
    if df is None or len(df) < 220:
        return None
    comps = _components(df)
    sig = _blended(comps, 0.40, 0.25, 0.05, 0.05, 20, "bear").values
    close = df["close"].values.astype(float)
    n = len(df)
    buckets = {k: [0, 0] for k in range(-2, 3)}      # [wins, total] per bucket
    for i in range(60, n - horizon):
        sv = float(sig[i])
        if abs(sv) < 0.15:
            continue
        b = int(np.clip(round(sv * 2), -2, 2))
        j = min(i + horizon, n - 1)
        up = close[j] > close[i]
        buckets[b][1] += 1
        if (b > 0 and up) or (b < 0 and not up) or (b == 0):
            buckets[b][0] += 1
    total = sum(t for _, t in buckets.values())
    if total < 40:
        return None
    ladder = {str(k): round(w / t, 3) for k, (w, t) in buckets.items() if t >= 8}
    hits = sum(w for k, (w, t) in buckets.items() if k != 0)
    checks = sum(t for k, (w, t) in buckets.items() if k != 0)
    return {"hit_rate": round(hits / checks, 3) if checks else None,
            "observations": total, "p_up_ladder": ladder,
            "horizon_days": horizon}


def _psx_keys() -> list[str]:
    from .universe import psx_universe
    return [i.key for i in psx_universe()]


def run_paper_session(md, symbols: list[str], engine, days: int = 60,
                      capital: float = 1_000_000.0) -> dict[str, Any]:
    """Walk-forward validation over the watchlist + one live demo paper order
    to prove the execution path end-to-end (gate -> broker -> journal)."""
    from .execution import OrderIntent, PaperBroker, RiskGate
    from .universe import resolve
    broker = PaperBroker(capital=capital)
    gate = RiskGate()
    results = {}
    for sym in symbols:
        inst = resolve(sym)
        if inst is None:
            continue
        df = md.history(inst)
        if df is None or len(df) < 140:
            results[inst.key] = {"error": "insufficient history"}
            continue
        try:
            bt = walk_forward(df, inst.key, test_days=days)
            results[inst.key] = {
                "verdict": bt.verdict, "trades": bt.trades,
                "hit_rate": bt.hit_rate, "return": bt.total_return,
                "benchmark": bt.benchmark_return, "sharpe": bt.sharpe,
                "max_dd": bt.max_drawdown, "params": bt.params,
                "calibrations": bt.calibrations}
        except Exception as exc:
            results[inst.key] = {"error": str(exc)}
    demo_order = None
    try:
        strong = engine.predict(symbols[0], horizon=5)
        if strong.get("ok") and strong["signal"]["score"] > 0.18:
            side = "BUY" if strong["signal"]["score"] > 0 else "SELL"
            # A Rs 10,000 demonstration order at a dollar price is not 10,000/price
            # units: the book counts one money and the tape prints the other, and
            # no cross means no size rather than a size at par.
            units = broker.units_for(symbols[0], 10_000, strong["price"])
            qty = max(1.0, round(units, 2)) if units else 0.0
            intent = OrderIntent(symbol=symbols[0], side=side, quantity=qty, venue="PAPER")
            acct = broker.account()
            allowed, why = gate.check(intent, acct, broker.open_positions())
            if allowed:
                demo_order = broker.submit(intent, strong["price"])
    except Exception:
        pass
    profitable = sum(1 for r in results.values() if r.get("verdict") == "PROFITABLE")
    marginal = sum(1 for r in results.values() if r.get("verdict") == "MARGINAL")
    return {"results": results,
            "summary": {"instruments": len(results), "profitable": profitable,
                        "marginal": marginal,
                        "unprofitable": len(results) - profitable - marginal},
            "demo_paper_order": demo_order.__dict__ if demo_order else None,
            "account": broker.account()}
