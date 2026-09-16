"""
IncrementalFeatureEngine — per-bar rolling indicator bundle for walk-forward
backtests (strictly no lookahead).

The live path computes `StrategyContext.compute()` once per snapshot window
(cheap on a rolling 600-bar array).  A walk-forward backtest that repeats that
for every historical bar means O(n^2) indicator work and hangs on real data.
This engine reproduces the same indicator formulas bar-by-bar: every value at
bar ``i`` depends only on bars ``0..i``, so a full 4000-bar walk costs
O(n * window) scalar work.

Each ``context()`` call returns tail arrays (length ``_TAIL``) which is all
the strategy rules index (``[-1]..[-5]``), so the adapters run unchanged and
with identical semantics.

Mirrors `strategy_features.StrategyContext.compute`:
    ema (recursive), Wilder RSI, Wilder ATR, Wilder ADX/DI,
    BB(20,2) / keltner(20,1.5*ATR) / donchian(20/40),
    z20 / z60, pct20 / pct60, macd(12,26,9), cci(20), stoch(14,3,3),
    mom1/5/10/20/60 (log-return sums), vol20/60, range_ratio, atr_norm.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_EPS = 1e-9
_TAIL = 8                      # adapters index [-5], [-3], [-2], [-1]
_LONG = 60                     # widest rolling window
FILL = 50.0                    # neutral value for warm-up percentile rows


def _wmean(values, w):
    values = list(values)
    if len(values) < w:
        return None
    return sum(values[-w:]) / w


def _wstd(values, w):
    xs = list(values)[-w:]
    if len(xs) < w:
        return None
    m = sum(xs) / w
    var = sum((x - m) * (x - m) for x in xs) / (w - 1)
    return var ** 0.5


class IncrementalFeatureEngine:
    """One instance per backtest symbol; pushes bars as the walk goes."""

    _KEYS = (
        "ema7", "ema14", "ema30", "ema60", "ema100",
        "rsi2", "rsi7", "rsi14",
        "atr14", "atr_norm",
        "bb_mid", "bb_up", "bb_lo", "bb_z", "z20", "z60",
        "pct20", "pct60",
        "macd", "macd_sig", "macd_hist",
        "kelt_up", "kelt_lo",
        "dch20_hi", "dch20_lo", "dch40_hi", "dch40_lo",
        "adx14", "di_plus", "di_minus",
        "cci20", "stoch_k", "stoch_d",
        "mom1", "mom5", "mom10", "mom20", "mom60",
        "vol20", "vol60", "range_ratio",
    )

    def __init__(self) -> None:
        self._ops: deque = deque(maxlen=_LONG)
        self._cls: deque = deque(maxlen=_LONG)
        self._his: deque = deque(maxlen=_LONG)
        self._los: deque = deque(maxlen=_LONG)
        self._tps: deque = deque(maxlen=_LONG)     # typical prices (CCI)
        self._lret: deque = deque(maxlen=_LONG)    # log returns (incl. bar 0)
        self._diffs: deque = deque(maxlen=14)      # close diffs (RSI seeding)
        self._moves: deque = deque(maxlen=14)      # (up_mv, dn_mv, tr) / bar
        self._ks: deque = deque(maxlen=3)          # raw stochastic for %D

        self._ema: dict[int, float] = {}
        self._rsi: dict[int, tuple[float, float]] = {}   # period -> (up_avg, dn_avg)
        self._atr: float = 0.0
        self._sum_up = self._sum_dn = self._sum_tr = 0.0
        self._adx_prev = 0.0
        self._adx_ready = False
        self._macd_raw: deque = deque(maxlen=70)

        self._val: dict[str, list[float]] = {}
        for key in self._KEYS:
            self._val[key] = []

    # ------------------------------------------------------------------ state

    def push(self, o, h, l, c) -> None:
        i = len(self._cls)                       # bar index being added
        cv = float(c)
        self._ops.append(float(o))
        self._cls.append(cv)
        self._his.append(float(h))
        self._los.append(float(l))
        self._tps.append((float(h) + float(l) + cv) / 3.0)

        if i == 0:
            lr = 0.0
            diff = 0.0
        else:
            lr = np.log(abs(cv) + _EPS) - np.log(abs(self._cls[-2]) + _EPS)
            diff = cv - self._cls[-2]
        self._lret.append(lr)
        if i > 0:
            self._diffs.append(diff)

        # --- EMA (matches _ema: seeded at first value) ---
        for span in (12, 26, 7, 14, 30, 60, 100):
            a = 2.0 / (span + 1.0)
            prev = self._ema.get(span, cv)
            self._ema[span] = a * cv + (1.0 - a) * prev

        # --- Wilder RSI (matches _wilder_rsi) ---
        for period in (2, 7, 14):
            if i <= period:
                r = FILL
            elif i == period + 1:
                # seed averages from the first `period` diffs (bars 0..period)
                closes = list(self._cls)
                ups = [max(closes[j + 1] - closes[j], 0.0) for j in range(period)]
                dns = [max(closes[j] - closes[j + 1], 0.0) for j in range(period)]
                up_avg = (sum(ups) / period * (period - 1) + max(diff, 0.0)) / period
                dn_avg = (sum(dns) / period * (period - 1) + max(-diff, 0.0)) / period
                self._rsi[period] = (up_avg, dn_avg)
                r = FILL if dn_avg <= _EPS else 100.0 - 100.0 / (1.0 + up_avg / dn_avg)
            else:
                up_avg, dn_avg = self._rsi[period]
                up_avg = (up_avg * (period - 1) + max(diff, 0.0)) / period
                dn_avg = (dn_avg * (period - 1) + max(-diff, 0.0)) / period
                self._rsi[period] = (up_avg, dn_avg)
                r = FILL if dn_avg <= _EPS else 100.0 - 100.0 / (1.0 + up_avg / dn_avg)
            self._val[f"rsi{period}"].append(r)

        # --- Wilder ATR(14): atr[0]=tr[0]; then (atr*13+tr)/14 ---
        if i == 0:
            tr_cur = 0.0
        else:
            prev_c = self._cls[-2]
            tr_cur = max(h - l, abs(h - prev_c), abs(l - prev_c))
            self._atr = tr_cur if i == 1 else (self._atr * 13.0 + tr_cur) / 14.0
        self._val["atr14"].append(self._atr)
        self._val["atr_norm"].append(self._atr / (abs(cv) + _EPS))

        # --- Wilder ADX / DI (matches vectorized loop) ---
        up_move = dn_move = 0.0
        if i > 0:
            up = h - self._his[-2]
            dn = self._los[-2] - l
            if up > 0 and up > dn:
                up_move = up
            elif dn > 0 and dn > up:
                dn_move = dn
            self._moves.append((up_move, dn_move, tr_cur))
        if i == 14:
            # seed running sums from the first 14 moves/bars (indices 0..13)
            self._sum_up = sum(m[0] for m in self._moves)
            self._sum_dn = sum(m[1] for m in self._moves)
            self._sum_tr = sum(m[2] for m in self._moves)
            self._adx_ready = True
        if i > 14 and self._adx_ready:
            self._sum_up = self._sum_up - self._sum_up / 14.0 + up_move
            self._sum_dn = self._sum_dn - self._sum_dn / 14.0 + dn_move
            self._sum_tr = self._sum_tr - self._sum_tr / 14.0 + tr_cur
            pdi = 100.0 * self._sum_up / (self._sum_tr + _EPS)
            mdi = 100.0 * self._sum_dn / (self._sum_tr + _EPS)
            dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + _EPS)
            adx = dx if self._adx_prev == 0.0 else (self._adx_prev * 13.0 + dx) / 14.0
            self._adx_prev = adx
        else:
            pdi = mdi = dx = adx = 0.0
        self._val["adx14"].append(adx)
        self._val["di_plus"].append(pdi)
        self._val["di_minus"].append(mdi)

        # --- MACD(12,26,9): signal == EMA9 over macd values ---
        macd = self._ema[12] - self._ema[26]
        self._macd_raw.append(macd)
        sig = None
        for mv in self._macd_raw:
            a = 2.0 / (9.0 + 1.0)
            sig = mv if sig is None else a * mv + (1.0 - a) * sig
        self._val["macd"].append(macd)
        self._val["macd_sig"].append(sig if sig is not None else 0.0)
        self._val["macd_hist"].append(macd - (sig if sig is not None else 0.0))

        for span, key in ((7, "ema7"), (14, "ema14"), (30, "ema30"),
                          (60, "ema60"), (100, "ema100")):
            self._val[key].append(self._ema[span])

        # --- BB(20,2) ---
        m20 = _wmean(self._cls, 20)
        s20 = _wstd(self._cls, 20)
        if m20 is not None and s20 is not None:
            bb_mid = m20
            bb_z = (cv - m20) / (s20 + _EPS)
            bb_up = m20 + 2.0 * s20
            bb_lo = m20 - 2.0 * s20
        else:
            bb_mid = bb_z = bb_up = bb_lo = 0.0
        self._val["bb_mid"].append(bb_mid)
        self._val["bb_up"].append(bb_up)
        self._val["bb_lo"].append(bb_lo)
        self._val["bb_z"].append(bb_z)
        self._val["z20"].append(bb_z)

        m60 = _wmean(self._cls, 60)
        s60 = _wstd(self._cls, 60)
        self._val["z60"].append((cv - m60) / (s60 + _EPS) if m60 is not None and s60 else 0.0)

        win20 = list(self._cls)[-20:] if len(self._cls) >= 20 else None
        win60 = list(self._cls)[-60:] if len(self._cls) >= 60 else None
        self._val["pct20"].append(float(np.mean([cv <= x for x in win20])) * 100.0 if win20 else FILL)
        self._val["pct60"].append(float(np.mean([cv <= x for x in win60])) * 100.0 if win60 else FILL)

        self._val["kelt_up"].append(bb_mid + 1.5 * self._atr)
        self._val["kelt_lo"].append(bb_mid - 1.5 * self._atr)

        for span, suf in ((20, "20"), (40, "40")):
            w = list(self._cls)[-span:] if len(self._cls) >= span else None
            self._val[f"dch{suf}_hi"].append(max(w) if w else 0.0)
            self._val[f"dch{suf}_lo"].append(min(w) if w else 0.0)

        # --- CCI(20) ---
        tps = list(self._tps)
        if len(tps) >= 20:
            tma_now = sum(tps[-20:]) / 20
            # per-element |typ[j] - sma20(j)| for the last 20 bars
            mad_sum = 0.0
            for idx in range(len(tps) - 20, len(tps)):
                window = tps[idx - 19: idx + 1]
                sma = sum(window) / 20
                mad_sum += abs(tps[idx] - sma)
            mad_ma = mad_sum / 20
            cci = (tps[-1] - tma_now) / (0.015 * mad_ma + _EPS)
        else:
            cci = 0.0
        self._val["cci20"].append(cci)

        # --- Stochastic(14,3,3) ---
        ll14 = list(self._los)[-14:] if len(self._los) >= 14 else list(self._los)
        hh14 = list(self._his)[-14:] if len(self._his) >= 14 else list(self._his)
        if len(ll14) >= 14:
            ll = min(ll14)
            hh = max(hh14)
            rng = hh - ll
            k = 50.0 if rng <= _EPS else (cv - ll) / (rng + _EPS) * 100.0
        else:
            k = 50.0
        self._ks.append(k)
        self._val["stoch_k"].append(k)
        self._val["stoch_d"].append(sum(self._ks) / len(self._ks))

        lr_list = list(self._lret)
        for period, key in ((1, "mom1"), (5, "mom5"), (10, "mom10"),
                            (20, "mom20"), (60, "mom60")):
            if len(lr_list) >= period:
                self._val[key].append(sum(lr_list[-period:]) * 100.0)
            else:
                self._val[key].append(0.0)

        for window, key in ((20, "vol20"), (60, "vol60")):
            if len(lr_list) >= window:
                v = _wstd(lr_list[-window:], window) or 0.0
            else:
                v = 0.0
            self._val[key].append(v * 100.0)

        self._val["range_ratio"].append((h - l) / (self._atr + _EPS) if self._atr else 0.0)

    # ------------------------------------------------------------------ output

    def context(self) -> tuple[dict, dict]:
        """Tail bundles for the current bar — exactly what the rules read."""
        feat = {}
        for key in self._KEYS:
            feat[key] = np.asarray(self._val[key][-_TAIL:], dtype=np.float64)
        prices = {
            "close": np.asarray(list(self._cls)[-_TAIL:], dtype=np.float64),
            "open": np.asarray(list(self._ops)[-_TAIL:], dtype=np.float64),
            "high": np.asarray(list(self._his)[-_TAIL:], dtype=np.float64),
            "low": np.asarray(list(self._los)[-_TAIL:], dtype=np.float64),
            "volume": np.zeros(_TAIL, dtype=np.float64),
        }
        return feat, prices

    @property
    def n_bars(self) -> int:
        return len(self._cls)


__all__ = ["IncrementalFeatureEngine"]