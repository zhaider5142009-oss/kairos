"""Feature engine: 60+ technical indicators, candlestick patterns, regime
detection, and cross-sectional factor computation. Pure numpy/pandas — no TA-Lib."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------
def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).mean()


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=1).mean()


def _wma(x: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return x.rolling(n, min_periods=1).apply(
        lambda v: float(np.dot(v, weights[-len(v):]) / weights[-len(v):].sum())
        if len(v) else np.nan, raw=True)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()


def _stoch(df: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    ll = df["low"].rolling(n, min_periods=1).min()
    hh = df["high"].rolling(n, min_periods=1).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k = k.fillna(50.0)
    return k, k.rolling(d, min_periods=1).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = _ema(close, fast) - _ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=1).mean()
    return line, sig, line - sig


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = _atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(close, n)
    sd = close.rolling(n, min_periods=1).std()
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, pct_b


def _log_close(close: pd.Series) -> np.ndarray:
    """log of the close, with a non-positive print left as a hole, not as -inf.

    `np.log(0)` is -inf, and one -inf is enough to make `np.histogram` raise and
    `np.polyfit` answer with a garbage slope — on the scan path that is an
    exception out of the middle of the instrument loop, ending the cycle every
    other position's stop was meant to be managed in. Dropping the bar is the
    honest reading: the series is shorter, not wrong.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(close.values.astype(float))
    return lp[np.isfinite(lp)]


def _hurst(close: pd.Series, max_lag: int = 20) -> float:
    """Rescaled-range Hurst exponent — >0.55 trending, <0.45 mean-reverting."""
    prices = _log_close(close)
    if len(prices) < 32:
        return 0.5
    lags = np.arange(2, min(max_lag + 1, len(prices) // 4))
    if len(lags) < 4:
        return 0.5
    tau = np.array([np.std(prices[lag:] - prices[:-lag]) for lag in lags])
    keep = tau > 0
    tau, lags = tau[keep], lags[keep]
    if len(tau) < 4:
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    h = poly[0]
    return float(np.clip(h, 0.05, 0.95))


def _entropy(close: pd.Series, bins: int = 8) -> float:
    rets = np.diff(_log_close(close))
    rets = rets[np.isfinite(rets)]
    if len(rets) < 20:
        return 0.0
    counts, _ = np.histogram(rets[-120:], bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _linreg_slope(close: pd.Series, n: int = 20) -> tuple[float, float]:
    # A hole inside the window is not a flat bar: fitting through it would read
    # -inf as a collapse. Short data is the honest answer, and short data is 0.
    y = _log_close(close)[-n:]
    if len(y) < n:
        return 0.0, 0.0
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - ss_res / ss_tot
    return float(slope), float(max(0.0, r2))


# ---------------------------------------------------------------------------
# Master feature computation
# ---------------------------------------------------------------------------
def doctrine_snapshot(df: pd.DataFrame,
                      trend_period: int = 50) -> dict[str, float | None]:
    """The three numbers the trading doctrine gates an entry on.

    `rules.py` declares a trend filter (`require_above_sma50` at
    `trend_sma_period`) and a heat filter (`rsi_oversold`/`rsi_overbought`) that
    the entry decision has to be able to answer, so they are computed here from
    the frame the engine is already holding — the alternative is a second history
    call per candidate on the autopilot's thread, which is a network round trip
    spent to re-derive two indicators that were in hand a moment earlier.

    Anything the frame cannot support comes back as `None`, not as a neutral
    number. That distinction is the whole point: `_rsi` fills 50.0 for a series
    too short for its own `min_periods`, which is a plausible-looking reading of
    nothing, and a gate that trusts it buys a name the trend filter never looked
    at. Two bars of history is not a trend, and the caller is told so.
    """
    out: dict[str, float | None] = {"close": None, "sma": None, "rsi14": None}
    if df is None or len(df) < 2 or "close" not in getattr(df, "columns", []):
        return out
    c = pd.to_numeric(df["close"], errors="coerce")
    px = c.iloc[-1]
    if not (np.isfinite(px) and px > 0):
        return out
    out["close"] = float(px)
    try:
        n: int | None = max(2, int(trend_period))
    except (TypeError, ValueError):
        # The operator's own period is unreadable, so there is no trend filter to
        # apply and no default that stands in for it: `sma` stays None and the
        # gate refuses the entry naming that, instead of quietly measuring the
        # trend against a 50 the configuration never said. Only the mean is
        # abandoned — the heat filter does not depend on the period, so taking it
        # down as well would report one bad knob as two blind controls.
        n = None
    if n is not None and len(c) >= n:
        s = _sma(c, n).iloc[-1]
        if np.isfinite(s):
            out["sma"] = float(s)
    if len(c) >= 15:
        r = _rsi(c, 14).iloc[-1]
        if np.isfinite(r):
            out["rsi14"] = float(r)
    return out


def compute_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute the full feature vector from the LAST row of each indicator."""
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].fillna(0)
    f: dict[str, float] = {}

    # returns over horizons
    for n in (1, 2, 3, 5, 10, 21, 42, 63, 126, 252):
        f[f"ret_{n}d"] = float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else 0.0

    # moving averages + distances
    for n in (5, 10, 20, 50, 100, 200):
        ma = _sma(c, n)
        f[f"sma{n}_dist"] = float(c.iloc[-1] / ma.iloc[-1] - 1) if len(c) >= n else 0.0
        f[f"ema{n}_dist"] = float(c.iloc[-1] / _ema(c, n).iloc[-1] - 1) if len(c) >= n else 0.0
    f["sma50_gt_200"] = 1.0 if len(c) >= 200 and _sma(c, 50).iloc[-1] > _sma(c, 200).iloc[-1] else 0.0
    f["ma_align"] = sum(
        1 for a, b in ((5, 10), (10, 20), (20, 50), (50, 100), (100, 200))
        if len(c) > b and _sma(c, a).iloc[-1] > _sma(c, b).iloc[-1]) / 5.0

    # momentum oscillators
    f["rsi14"] = float(_rsi(c).iloc[-1])
    f["rsi7"] = float(_rsi(c, 7).iloc[-1])
    f["rsi28"] = float(_rsi(c, 28).iloc[-1])
    k, d = _stoch(df)
    f["stoch_k"], f["stoch_d"] = float(k.iloc[-1]), float(d.iloc[-1])
    macd, macd_sig, macd_hist = _macd(c)
    # The last close is the denominator of every normalized feature below, so
    # it is a hole when it is not a positive price — never a substitute one.
    # The panel path already did this (`c.replace(0, np.nan)`); this one used to
    # write 1.0, which turned a halted or delisted print into a one-rupee price
    # and scaled MACD, ATR%, momentum and every candle ratio by it.
    last = float(c.iloc[-1])
    px = last if last > 0 else float("nan")
    f["macd_norm"] = float(macd.iloc[-1] / px)
    f["macd_hist_norm"] = float(macd_hist.iloc[-1] / px)
    f["adx14"] = float(_adx(df).iloc[-1])
    f["cci20"] = float(((c - _sma(c, 20)) / (0.015 * c.rolling(20, min_periods=1).std().replace(0, np.nan))).iloc[-1] or 0.0)
    hh14 = df["high"].rolling(14, min_periods=1).max().iloc[-1]
    ll14 = df["low"].rolling(14, min_periods=1).min().iloc[-1]
    f["williams_r"] = float((hh14 - c.iloc[-1]) / ((hh14 - ll14) or 1e-9) * -100)
    mom = c.diff(10)
    f["momentum10"] = float(mom.iloc[-1] / px)
    roc = c.pct_change(12)
    f["roc12"] = float(roc.iloc[-1] or 0.0)
    f["trix14"] = float(((_ema(_ema(_ema(c, 14), 14), 14).pct_change() ).iloc[-1] or 0.0) * 100)

    # volatility suite
    atr = _atr(df)
    f["atr14_pct"] = float(atr.iloc[-1] / px)
    f["atr_ratio"] = float(atr.iloc[-1] / atr.rolling(50, min_periods=1).mean().iloc[-1]) if len(df) > 50 else 1.0
    ret = c.pct_change().fillna(0.0)
    f["vol20"] = float(ret.rolling(20, min_periods=5).std().iloc[-1] or 0.0) * np.sqrt(252)
    f["vol60"] = float(ret.rolling(60, min_periods=10).std().iloc[-1] or 0.0) * np.sqrt(252)
    f["vol_ratio"] = f["vol20"] / f["vol60"] if f["vol60"] > 0 else 1.0
    # Where this bar's volatility sits in the instrument's own history. A fixed
    # threshold on an annualized volatility means different things on a cement
    # scrip and on EURUSD, so anything asking "is vol high *here*" needs a
    # rank, not a number — and the strategies are handed `features`, not the
    # regime dict, so a rank that exists only in `detect_regime` is invisible
    # to them.
    vol_hist = (ret.rolling(20, min_periods=5).std().dropna() * np.sqrt(252)).tail(252)
    f["vol_percentile"] = (round(float((vol_hist <= f["vol20"]).mean()), 3)
                           if len(vol_hist) >= 20 and np.isfinite(f["vol20"]) else 0.5)
    logret = np.diff(_log_close(c))
    if len(logret) > 30:
        ann = float(np.sqrt(252))
        win = 20
        # Garman-Klass is a per-bar *variance* estimate,
        # 0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2, and it is legitimately negative on
        # a bar that closes almost exactly where it opened. So the mean is taken
        # first and the root second; a root of the mean of squares turns every
        # negative bar into a positive contribution.
        with np.errstate(divide="ignore", invalid="ignore"):
            hl2 = np.log(h.values / lo.values) ** 2
            oc2 = np.log(c.values / df["open"].values) ** 2
            # Inside the same errstate: a bar with no printed range and a close
            # outside it makes both terms infinite, and inf - inf is invalid. The
            # nan_to_num below is what drops it; the warning is what fills the
            # daemon's log with a stack of arithmetic nobody can act on.
            gk_var = 0.5 * hl2 - (2 * np.log(2) - 1) * oc2
        mean_gk = float(np.nanmean(np.nan_to_num(
            gk_var[-win:], nan=0.0, posinf=0.0, neginf=0.0)))
        # Clamped, so 0.0 here means *this estimator has nothing to say* — not that
        # the market was still. A *valid* bar can never drive it negative, because
        # |C-O| <= H-L always; a bar whose open or close lies outside its own printed
        # range can, and the provider series for a pegged FX pair does exactly that
        # (USDPKR's last twenty bars read 0.0 for Garman-Klass while Parkinson reads
        # 0.16 — the close sat above the day's high on thirteen of them). The same
        # value also covers "not enough history" below. Any consumer must test for it;
        # `Garman-Klass vs Close Divergence` abstains on 0.0.
        f["garman_klass"] = max(mean_gk, 0.0) ** 0.5 * ann
        mean_hl = float(np.nanmean(np.nan_to_num(
            hl2[-win:], nan=0.0, posinf=0.0, neginf=0.0)))
        f["parkinson"] = max(mean_hl, 0.0) ** 0.5 / np.sqrt(4 * np.log(2)) * ann
        f["skew60"] = float(np.nan_to_num(pd.Series(logret[-60:]).skew()))
        f["kurt60"] = float(np.nan_to_num(pd.Series(logret[-60:]).kurt()))
        f["downside_vol"] = float(ret[ret < 0].tail(60).std() or 0.0) * np.sqrt(252)
    else:
        f.update(garman_klass=0.0, parkinson=0.0, skew60=0.0, kurt60=0.0, downside_vol=0.0)

    # Bollinger / Keltner / Donchian
    mid, upper, lower, width, pct_b = _bollinger(c)
    f["bb_width"] = float(width.iloc[-1] or 0.0)
    f["bb_pct_b"] = float(pct_b.iloc[-1] or 0.5)
    f["bb_squeeze"] = 1.0 if f["bb_width"] < float(width.rolling(120, min_periods=20).quantile(0.2).iloc[-1] or 1.0) else 0.0
    ema20 = _ema(c, 20)
    atr14 = _atr(df)
    f["keltner_pos"] = float((c.iloc[-1] - (ema20.iloc[-1] + 2 * atr14.iloc[-1])) /
                             (4 * atr14.iloc[-1] or 1e-9))
    f["keltner_neg"] = float((c.iloc[-1] - (ema20.iloc[-1] - 2 * atr14.iloc[-1])) /
                             (4 * atr14.iloc[-1] or 1e-9))
    don_hi = h.rolling(20, min_periods=1).max().iloc[-1]
    don_lo = lo.rolling(20, min_periods=1).min().iloc[-1]
    f["donchian_pos"] = float((c.iloc[-1] - don_lo) / ((don_hi - don_lo) or 1e-9))
    f["dist_52w_high"] = float(c.iloc[-1] / (h.rolling(252, min_periods=60).max().iloc[-1] or px) - 1)
    f["dist_52w_low"] = float(c.iloc[-1] / (lo.rolling(252, min_periods=60).min().iloc[-1] or px) - 1)

    # volume behavior
    if v.sum() > 0:
        v_ma = v.rolling(20, min_periods=1).mean()
        f["vol_z"] = float((v.iloc[-1] - v_ma.iloc[-1]) / (v.rolling(60, min_periods=5).std().iloc[-1] or 1e-9))
        f["volume_trend"] = float(v_ma.iloc[-1] / (v.rolling(60, min_periods=1).mean().iloc[-1] or 1.0))
        obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
        # the same denominator the panel's obv_sl uses: the volume the change was
        # measured over, not |OBV|, which passes through zero and turns a flat
        # balance into a number 10^16 times the rest of the feature row
        v5 = float(v.iloc[-5:].sum())
        f["obv_slope"] = float((obv.iloc[-1] - obv.iloc[-6]) / v5) if v5 > 0 else 0.0
        mfi_tp = (h + lo + c) / 3
        mf = mfi_tp.diff().fillna(0.0) * v
        pos = mf.clip(lower=0).rolling(14, min_periods=1).sum()
        neg = (-mf).clip(lower=0).rolling(14, min_periods=1).sum()
        f["mfi14"] = float((100 * pos / (pos + neg).replace(0, np.nan)).iloc[-1] or 50.0)
        f["cmf20"] = float((((c - lo) - (h - c)) / ((h - lo).replace(0, np.nan)) * v).rolling(20, min_periods=1).sum().iloc[-1] /
                           (v.rolling(20, min_periods=1).sum().iloc[-1] or 1e-9))
        dollar = c * v
        # `+ 1e-9` was silently inventing the most illiquid reading possible: a
        # bar with no volume at all divides by the epsilon and returns an
        # Amihud of order 1e16, so a missing print out-ranked a genuinely dead
        # book. Only bars that actually traded are averaged, and the veto that
        # consumes this figure reads a rank against the instrument's own
        # history — the raw value is a rupee-share-scale number whose absolute
        # threshold means nothing across venues.
        traded = dollar > 0
        illiq = (ret.abs() / dollar.where(traded)).dropna()
        f["amihud"] = float(illiq.rolling(21, min_periods=5).mean().iloc[-1] * 1e9
                            if len(illiq) >= 5 else 0.0)
        illiq_hist = (illiq.rolling(21, min_periods=5).mean() * 1e9).dropna().tail(252)
        # A rank against the instrument's own history is the only comparison
        # that means the same thing on a cement scrip and on BTC — but a rank
        # also goes blind to a book that has been dry for weeks, because every
        # window inside the drought ties with the last one. Against the median
        # it does not: a hundred-times-drier tape is a hundred times the
        # baseline however flat the drought itself is. 1.0 is the answer for
        # "no usable reading", which includes the bars nobody traded.
        baseline = float(illiq_hist.median()) if len(illiq_hist) >= 20 else 0.0
        f["amihud_relative"] = (
            round(float(f["amihud"] / baseline), 3)
            if baseline > 0 and f["amihud"] > 0 and bool((dollar.tail(5) > 0).all())
            else 1.0)
        f["vwap_dist"] = float(c.iloc[-1] / ((dollar.rolling(20, min_periods=1).sum() / v.rolling(20, min_periods=1).sum().replace(0, np.nan)).iloc[-1] or px) - 1)
    else:
        f.update(vol_z=0.0, volume_trend=1.0, obv_slope=0.0, mfi14=50.0,
                 cmf20=0.0, amihud=0.0, amihud_relative=1.0, vwap_dist=0.0)

    # structure & statistical properties
    f["hurst"] = _hurst(c)
    f["entropy"] = _entropy(c)
    slope, r2 = _linreg_slope(c, 20)
    f["linreg_slope20"] = slope * 252
    f["linreg_r2"] = r2
    slope60, r2_60 = _linreg_slope(c, 60)
    f["linreg_slope60"] = slope60 * 252
    f["linreg_r2_60"] = r2_60
    if len(logret := np.diff(_log_close(c))) > 25:
        f["autocorr1"] = float(np.corrcoef(logret[:-1], logret[1:])[0, 1])
        f["autocorr5"] = float(np.corrcoef(logret[:-5], logret[5:])[0, 1]) if len(logret) > 15 else 0.0
    else:
        f.update(autocorr1=0.0, autocorr5=0.0)

    # drawdown & run statistics
    peak = c.cummax()
    dd = c / peak - 1
    f["drawdown"] = float(dd.iloc[-1])
    f["drawdown_5d_ago"] = float(dd.iloc[-6]) if len(dd) > 6 else 0.0
    up_days = int((ret.tail(10) > 0).sum())
    f["up_day_ratio_10"] = up_days / 10.0
    gap = (df["open"] / c.shift() - 1) if len(df) > 1 else pd.Series([0.0])
    f["last_gap"] = float(gap.iloc[-1] or 0.0)
    f["gap_frequency"] = float((gap.tail(60).abs() > 0.005).mean()) if len(gap) > 60 else 0.0

    # candlestick anatomy (last bar)
    o = df["open"]
    body = float(abs(c.iloc[-1] - o.iloc[-1]) / px)
    rng = float((h.iloc[-1] - lo.iloc[-1]) / px) or 1e-9
    f["body_ratio"] = body / rng
    f["upper_shadow"] = float((h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])) / px) / rng
    f["lower_shadow"] = float((min(c.iloc[-1], o.iloc[-1]) - lo.iloc[-1]) / px) / rng

    return f


# ---------------------------------------------------------------------------
# Candlestick pattern detection (last bar / last 3 bars)
# ---------------------------------------------------------------------------
def detect_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 3:
        return []
    o, c = df["open"].values, df["close"].values
    h, lo = df["high"].values, df["low"].values
    c1, c2, c3 = c[-1], c[-2], c[-3]
    o1, o2, o3 = o[-1], o[-2], o[-3]
    h1, l1 = h[-1], lo[-1]
    body1 = abs(c1 - o1)
    rng1 = h1 - l1 or 1e-12
    body2 = abs(c2 - o2)
    pats: list[str] = []
    if body1 / rng1 < 0.1:
        pats.append("doji")
    if body1 / rng1 < 0.3 and (min(c1, o1) - l1) > 2 * body1 and (h1 - max(c1, o1)) < body1:
        pats.append("hammer")
    if body1 / rng1 < 0.3 and (h1 - max(c1, o1)) > 2 * body1 and (min(c1, o1) - l1) < body1:
        pats.append("shooting_star")
    if c1 > o1 and c2 < o2 and c1 >= o2 and o1 <= c2 and body1 > body2:
        pats.append("bullish_engulfing")
    if c1 < o1 and c2 > o2 and c1 <= o2 and o1 >= c2 and body1 > body2:
        pats.append("bearish_engulfing")
    if body1 < body2 * 0.5 and max(c1, o1) < max(c2, o2) and min(c1, o1) > min(c2, o2):
        pats.append("harami")
    if c3 < o3 and abs(c2 - o2) < body1 * 0.6 and c1 > o1 and c1 > (o2 + c2) / 2:
        pats.append("morning_star")
    if c3 > o3 and c2 > o2 and c1 < o1 and c1 < (o2 + c2) / 2:
        pats.append("evening_star")
    if c1 > o1 and c2 > o2 and c3 > o3 and c1 > c2 > c3:
        pats.append("three_white_soldiers")
    if c1 < o1 and c2 < o2 and c3 < o3 and c1 < c2 < c3:
        pats.append("three_black_crows")
    if body1 / rng1 > 0.9 and c1 > o1:
        pats.append("bullish_marubozu")
    if body1 / rng1 > 0.9 and c1 < o1:
        pats.append("bearish_marubozu")
    if abs(c1 - c2) / max(c1, c2) < 0.002 and abs(l1 - lo[-2]) / max(l1, lo[-2]) < 0.004:
        pats.append("tweezer_bottom")
    if abs(c1 - c2) / max(c1, c2) < 0.002 and abs(h1 - h[-2]) / max(h1, h[-2]) < 0.004:
        pats.append("tweezer_top")
    return pats


# ---------------------------------------------------------------------------
# Vectorized feature panel for ML training (one pass over full history)
# ---------------------------------------------------------------------------
def feature_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build the ML feature matrix: one row per bar, all indicators vectorized."""
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].fillna(0)
    ret = c.pct_change()
    panel = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 5, 10, 21, 63, 126):
        panel[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 50, 100, 200):
        panel[f"sma{n}_d"] = c / _sma(c, n) - 1
        panel[f"ema{n}_d"] = c / _ema(c, n) - 1
    panel["rsi14"] = _rsi(c)
    panel["rsi7"] = _rsi(c, 7)
    k, d = _stoch(df)
    panel["stoch_k"], panel["stoch_d"] = k, d
    macd_l, macd_s, macd_h = _macd(c)
    px = c.replace(0, np.nan)
    panel["macd_n"] = macd_l / px
    panel["macd_h_n"] = macd_h / px
    panel["adx14"] = _adx(df)
    atr = _atr(df)
    panel["atr_p"] = atr / px
    panel["atr_r"] = atr / atr.rolling(50, min_periods=1).mean()
    panel["vol20"] = ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    panel["vol60"] = ret.rolling(60, min_periods=10).std() * np.sqrt(252)
    _, _, _, width, pct_b = _bollinger(c)
    panel["bb_w"] = width
    panel["bb_b"] = pct_b
    don_hi = h.rolling(20, min_periods=1).max()
    don_lo = lo.rolling(20, min_periods=1).min()
    panel["don_p"] = (c - don_lo) / (don_hi - don_lo).replace(0, np.nan)
    panel["dd"] = c / c.cummax() - 1
    panel["gap"] = df["open"] / c.shift() - 1
    if v.sum() > 0:
        v_ma = v.rolling(20, min_periods=1).mean()
        v_sd = v.rolling(60, min_periods=5).std()
        panel["vol_z"] = (v - v_ma) / v_sd.replace(0, np.nan)
        obv = (np.sign(ret.fillna(0.0)) * v).cumsum()
        # OBV's level is an arbitrary cumulative sum that passes through zero, so
        # dividing its change by |OBV| divides by ~1e-9 on the bars where the
        # balance is flat: 4.95e16 on NIKKEI, against a panel whose next-largest
        # column is 1e2, and that is what made the Ridge solve report
        # rcond=3.8e-34 on a matrix it should have handled cleanly. The volume
        # traded over the same window is the scale the change is actually
        # measured against, and the ratio is net signed volume as a share of
        # everything that traded: bounded in [-1, 1], same sign as before.
        panel["obv_sl"] = obv.diff(5) / v.rolling(5, min_periods=1).sum().replace(0, np.nan)
    else:
        panel["vol_z"] = 0.0
        panel["obv_sl"] = 0.0
    panel["body"] = (c - df["open"]) / (h - lo).replace(0, np.nan)
    panel["up_ratio"] = ret.gt(0).rolling(10, min_periods=1).mean()
    panel["dist_h52"] = c / h.rolling(252, min_periods=60).max() - 1
    panel["dist_l52"] = c / lo.rolling(252, min_periods=60).min() - 1
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
REGIMES = ("Bull-Quiet", "Bull-Volatile", "Bear-Quiet", "Bear-Volatile", "Range",
           "MeanRev-Quiet", "MeanRev-Volatile")


def detect_regime(df: pd.DataFrame) -> dict[str, Any]:
    c = df["close"]
    hurst = _hurst(c)
    ret = c.pct_change().fillna(0.0)
    vol20 = float(ret.rolling(20, min_periods=5).std().iloc[-1] or 0.0) * np.sqrt(252)
    vol60 = float(ret.rolling(60, min_periods=10).std().iloc[-1] or 0.0) * np.sqrt(252)
    vol_percentile = 0.5
    if len(ret) > 60:
        vals = ret.rolling(20).std().dropna() * np.sqrt(252)
        vol_percentile = float((vals < vol20).mean())
    sma50 = _sma(c, 50).iloc[-1] if len(c) >= 50 else c.iloc[-1]
    sma200 = _sma(c, 200).iloc[-1] if len(c) >= 200 else sma50
    trend_up = bool(c.iloc[-1] > sma50 and sma50 >= sma200)
    trend_down = bool(c.iloc[-1] < sma50 and sma50 <= sma200)
    adx = float(_adx(df).iloc[-1])
    # `bool(...)` is not decoration. `vol20` and `vol60` are numpy floats, so the
    # second comparison answers numpy.bool_, and `or` returns that operand
    # whenever the first is false — which is the common case. The desk survives
    # only because its endpoint round-trips every value through `_safe`; the CLI
    # dossier, which writes with `default=str`, saved the string "False" for a
    # quiet market, and any reader that later asks `if regime["high_volatility"]`
    # gets True out of a saved report and False out of the live one.
    high_vol = bool(vol_percentile > 0.7 or
                    (vol20 > 1.6 * vol60 if vol60 > 0 else False))

    if adx > 25 and trend_up:
        regime = "Bull-Volatile" if high_vol else "Bull-Quiet"
    elif adx > 25 and trend_down:
        regime = "Bear-Volatile" if high_vol else "Bear-Quiet"
    elif hurst < 0.45:
        regime = "MeanRev-Volatile" if high_vol else "MeanRev-Quiet"
    else:
        regime = "Range"
    return {
        "regime": regime,
        "hurst": round(hurst, 3),
        "adx": round(adx, 1),
        "vol_annual": round(vol20, 4),
        "vol_percentile": round(vol_percentile, 2),
        "trend": "up" if trend_up else "down" if trend_down else "flat",
        "high_volatility": high_vol,
    }


# ---------------------------------------------------------------------------
# Cross-sectional factors (PSX universe from TradingView snapshot)
# ---------------------------------------------------------------------------
FACTOR_COLS = ["name", "close", "change", "change|1W", "change|1M", "Perf.3M",
               "Perf.6M", "Perf.Y", "RSI", "SMA50", "SMA200", "Volatility.D",
               "market_cap_basic", "peratio", "roe", "net_margin", "debt_to_equity",
               "beta_1_year", "average_volume_10d_calc", "Recommend.All", "sector"]


def cross_sectional_factors(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Rank the whole PSX cross-section on momentum/value/quality/low-vol/size."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame()
    df = snapshot.copy()
    for col in FACTOR_COLS[2:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def zsc(s: pd.Series, invert: bool = False) -> pd.Series:
        if s is None:
            return None
        s = s.astype(float).replace([np.inf, -np.inf], np.nan)
        mu, sd = float(s.mean()), float(s.std())
        z = (s - mu) / sd if sd and sd > 0 else s * 0
        return -z if invert else z

    mom1m = zsc(df.get("change|1M"))
    mom3m = zsc(df.get("Perf.3M"))
    mom12_1 = None
    if "Perf.Y" in df:
        mom12_1 = zsc(df["Perf.Y"]) - mom3m.fillna(0)
    value = zsc(df.get("peratio"), invert=True) if "peratio" in df else None
    quality = (zsc(df.get("roe")).fillna(0) + zsc(df.get("net_margin")).fillna(0)
               - zsc(df.get("debt_to_equity")).fillna(0)) if "roe" in df else None
    lowvol = zsc(df.get("Volatility.D"), invert=True) if "Volatility.D" in df else None
    size = zsc(df.get("market_cap_basic"), invert=True) if "market_cap_basic" in df else None
    liq = zsc(df.get("average_volume_10d_calc")) if "average_volume_10d_calc" in df else None
    tech = None
    if "RSI" in df:
        rsi = df["RSI"].astype(float)
        tech = ((rsi - 50) / 20).clip(-1.5, 1.5)
    tv_rec = None
    if "Recommend.All" in df:
        rec = df["Recommend.All"].astype(float)
        tv_rec = zsc(-rec) if rec.notna().sum() > 5 else None

    out = pd.DataFrame({"name": df.get("name"), "close": df.get("close"),
                        "volume": df.get("volume"),
                        "sector": df.get("sector"),
                        "momentum_1m": mom1m, "momentum_3m": mom3m,
                        "momentum_12_1": mom12_1, "value": value,
                        "quality": quality, "low_vol": lowvol, "size": size,
                        "liquidity": liq, "tech": tech, "tv_recommend": tv_rec})
    score = (0.22 * out["momentum_1m"].fillna(0) +
             0.18 * out["momentum_3m"].fillna(0) +
             0.10 * out["momentum_12_1"].fillna(0) +
             0.12 * out["value"].fillna(0) +
             0.12 * out["quality"].fillna(0) +
             0.08 * out["low_vol"].fillna(0) +
             0.05 * out["size"].fillna(0) +
             0.08 * out["liquidity"].fillna(0) +
             0.05 * out["tech"].fillna(0) +
             0.05 * out["tv_recommend"].fillna(0))
    out["composite_score"] = score
    out["rank"] = score.rank(ascending=False)
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)
