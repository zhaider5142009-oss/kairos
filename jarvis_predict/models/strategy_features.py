"""
FastFeatures — one shared indicator pass per symbol per cycle.

The strategy fleet (~30 rule-based models) is only fast because they never
recompute indicators: `StrategyContext.compute()` derives every feature
bundle once (vectorised numpy, sub-millisecond on 600 bars) and the models
read from it.  This is the latency backbone of the "act in milliseconds"
path — the full slice call below is the only indicator math the fleet pays.

Indicators (banks/CTA toolbox):
    ema7/ema14/ema30/ema60/ema100      exponential averages
    macd / macd_sig / macd_hist
    rsi2 / rsi7 / rsi14                 Wilder RSI
    atr14 / atr_norm                    ATR + %ATR of price
    bb_mid/bb_up/bb_lo (20,2σ), bb_z
    kelt_up/kelt_lo (20, 1.5×ATR)
    dch_hi/dch_lo (20/40)               Donchian channel
    adx14 / di_plus / di_minus
    cci20
    stoch_k/stoch_d (14,3,3)
    z20 / z60                           z-score of close over 20/60
    mom1/mom5/mom10/mom20/mom60         log-returns
    pct20/pct60                         percentile rank of last close
    vol20 / vol60                       rolling std dev of returns
    range_ratio                         (high-low)/atr — expansion proxy
    ATR(band_norm)                      price position inside ATR envelope
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    d = np.diff(close)
    up = np.clip(d, 0, None)
    dn = -np.clip(d, None, 0)
    out = np.full_like(close, 50.0, dtype=np.float64)
    if close.shape[0] <= period + 1:
        return out
    up_avg = up[:period].mean()
    dn_avg = dn[:period].mean()
    for i in range(period, close.shape[0] - 1):
        up_avg = (up_avg * (period - 1) + up[i]) / period
        dn_avg = (dn_avg * (period - 1) + dn[i]) / period
        rs = up_avg / (dn_avg + _EPS)
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    out[:period + 1] = 50.0
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if n < 2:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    s = c[n:] - c[:-n]
    s2 = c2[n:] - c2[:-n]
    out[n - 1:] = np.sqrt(np.clip((s2 - s ** 2 / n) / (n - 1), 0, None))
    return out


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if n < 1:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    s = c[n:] - c[:-n]
    out[n - 1:] = s / n
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(n - 1, x.shape[0]):
        out[i] = x[i - n + 1:i + 1].max()
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(n - 1, x.shape[0]):
        out[i] = x[i - n + 1:i + 1].min()
    return out


def _ffill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs so one corrupt MT5 bar can't poison indicators."""
    out = np.array(arr, dtype=np.float64, copy=True)
    n = out.shape[0]
    for i in range(1, n):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    if np.isnan(out[0]):
        first = out[np.isfinite(out)]
        out[0] = first[0] if first.size else 0.0
    return out


class StrategyContext:
    """A single, precomputed indicator bundle for one symbol snapshot."""

    __slots__ = ("feat", "prices")

    def __init__(self, feat: dict[str, np.ndarray], prices: dict[str, np.ndarray]):
        self.feat = feat
        self.prices = prices          # raw OHLC arrays (for direct strategy math)

    @classmethod
    def compute(cls, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                close: np.ndarray, volume: np.ndarray | None = None) -> "StrategyContext":
        o = _ffill(np.asarray(open_, dtype=np.float64))
        h = _ffill(np.asarray(high, dtype=np.float64))
        l = _ffill(np.asarray(low, dtype=np.float64))
        c = _ffill(np.asarray(close, dtype=np.float64))
        n = c.shape[0]

        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.full(n, 0.0, dtype=np.float64)
        if n > 1:
            atr[0] = 0.0
            atr[1] = tr[0]
            for i in range(1, n - 1):
                atr[i + 1] = (atr[i] * 13 + tr[i]) / 14.0
        atr = np.nan_to_num(atr)

        f = {
            "ema7": _ema(c, 7), "ema14": _ema(c, 14), "ema30": _ema(c, 30),
            "ema60": _ema(c, 60), "ema100": _ema(c, 100),
            "rsi2": _wilder_rsi(c, 2), "rsi7": _wilder_rsi(c, 7),
            "rsi14": _wilder_rsi(c, 14),
            "atr14": atr,
            "atr_norm": atr / (np.abs(c) + _EPS),
            "bb_mid": _rolling_mean(c, 20),
            "bb_z": _rolling_std(c, 20),
            "z20": np.nan, "z60": np.nan,
            "pct20": np.nan, "pct60": np.nan,
            "mom1": np.nan, "mom5": np.nan, "mom10": np.nan,
            "mom20": np.nan, "mom60": np.nan,
            "vol20": np.nan, "vol60": np.nan,
        }

        ema_fast = _ema(c, 12)
        ema_slow = _ema(c, 26)
        f["macd"] = ema_fast - ema_slow
        f["macd_sig"] = _ema(f["macd"], 9)
        f["macd_hist"] = f["macd"] - f["macd_sig"]

        m = _rolling_mean(c, 20)
        s = _rolling_std(c, 20)
        z = (c - m) / (s + _EPS)
        f["bb_up"] = m + 2.0 * s
        f["bb_lo"] = m - 2.0 * s
        f["bb_z"] = z

        f["kelt_up"] = m + 1.5 * atr
        f["kelt_lo"] = m - 1.5 * atr

        f["dch20_hi"], f["dch20_lo"] = _rolling_max(c, 20), _rolling_min(c, 20)
        f["dch40_hi"], f["dch40_lo"] = _rolling_max(c, 40), _rolling_min(c, 40)

        # ADX / DI — Wilder smoothed directional movement.
        up_move = np.where((h[1:] > h[:-1]) & (h[1:] - h[:-1] > l[:-1] - l[1:]), h[1:] - h[:-1], 0.0)
        dn_move = np.where((l[1:] < l[:-1]) & (l[:-1] - l[1:] > h[1:] - h[:-1]), l[:-1] - l[1:], 0.0)
        adx = np.zeros(n, dtype=np.float64)
        di_p = np.zeros(n, dtype=np.float64)
        di_m = np.zeros(n, dtype=np.float64)
        sum_up, sum_dn, sum_tr = up_move[:14].sum(), dn_move[:14].sum(), tr[:14].sum()
        for i in range(14, n - 1):
            sum_up = sum_up - sum_up / 14.0 + up_move[i]
            sum_dn = sum_dn - sum_dn / 14.0 + dn_move[i]
            sum_tr = sum_tr - sum_tr / 14.0 + tr[i]
            pdi = 100.0 * sum_up / (sum_tr + _EPS)
            mdi = 100.0 * sum_dn / (sum_tr + _EPS)
            dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + _EPS)
            di_p[i + 1], di_m[i + 1] = pdi, mdi
            adx[i + 1] = dx if adx[i] == 0.0 else (adx[i] * 13 + dx) / 14.0
        f["adx14"] = adx
        f["di_plus"] = di_p
        f["di_minus"] = di_m

        # CCI(20) — (typical - sma20) / (0.015 * mean absolute deviation).
        typ = (h + l + c) / 3.0
        typ_ma = _rolling_mean(typ, 20)
        mad_x = np.abs(typ - typ_ma)
        mad_x = np.nan_to_num(mad_x, nan=0.0)   # cumsum can't ride NaN seeds
        mad_ma = _rolling_mean(mad_x, 20)
        f["cci20"] = (typ - typ_ma) / (0.015 * mad_ma + _EPS)

        # Stochastic (14,3,3).
        ll = _rolling_min(l, 14)
        hh = _rolling_max(h, 14)
        rng = hh - ll
        k = np.where(rng > _EPS, (c - ll) / (rng + _EPS) * 100.0, 50.0)
        f["stoch_k"] = k
        f["stoch_d"] = _rolling_mean(k, 3)

        ret = np.diff(c, prepend=c[0]) / (np.abs(c[0]) + _EPS) * 1.0
        log_ret = np.diff(np.log(np.abs(c) + _EPS), prepend=0.0)
        for period, key in ((1, "mom1"), (5, "mom5"), (10, "mom10"), (20, "mom20"), (60, "mom60")):
            p = np.full(n, 0.0)
            if n > period:
                p[period:] = log_ret[period:]
                # rolling sum of last `period` log-returns
                cs = np.cumsum(np.insert(log_ret, 0, 0.0))
                p = cs[period:] - cs[:-period]
                p = np.concatenate([np.zeros(period), p])
            f[key] = p * 100.0
        f["mom1"] = np.nan_to_num(f["mom1"])

        f["z20"] = (c - _rolling_mean(c, 20)) / (_rolling_std(c, 20) + _EPS)
        f["z60"] = (c - _rolling_mean(c, 60)) / (_rolling_std(c, 60) + _EPS)
        f["pct20"] = _rolling_std(c, 20) * 0.0 + 50.0
        for i in range(19, n):
            f["pct20"][i] = (c[i] <= c[i - 19:i + 1]).mean() * 100.0
        f["pct60"] = _rolling_std(c, 60) * 0.0 + 50.0
        for i in range(59, n):
            f["pct60"][i] = (c[i] <= c[i - 59:i + 1]).mean() * 100.0

        var = np.zeros(n, dtype=np.float64)
        var[1:] = log_ret[1:]
        f["vol20"] = _rolling_std(var, 20) * 100.0
        f["vol60"] = _rolling_std(var, 60) * 100.0

        f["range_ratio"] = np.concatenate([[0.0], (h[1:] - l[1:]) / (atr[1:] + _EPS)])

        return cls(feat=f, prices={"open": o, "high": h, "low": l, "close": c,
                                   "volume": np.asarray(volume, dtype=np.float64)
                                   if volume is not None else None})


__all__ = ["StrategyContext", "StrategyContext as FastFeatures"]