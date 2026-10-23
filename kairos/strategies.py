"""The 100-strategy library.

Ten families x ten-ish strategies each. Every strategy is a pure function of
(price history, computed features, context) -> optional StrategyVote.
Votes are fused by the ensemble engine; nothing here trades on its own.

Context keys (all optional):
  regime          str           detected regime label
  macro           str           cross-asset state: Risk-On / Neutral / Risk-Off /
                                Unknown (see kairos/macro.py)
  macro_score     float         -1..+1 risk-appetite score behind that label
  intermarket     dict          {key: DataFrame} of related series
  date_today      datetime      evaluation date (calendar effects)
  factor_row      pd.Series     cross-sectional factor row (PSX)
  news_sentiment  float         -1..+1 aggregate news tone
  patterns        list[str]     candlestick patterns detected
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .ensemble import StrategyVote

STRATEGIES: list[tuple[str, str, Callable]] = []


def strategy(name: str, category: str):
    def wrap(fn: Callable):
        STRATEGIES.append((name, category, fn))
        return fn
    return wrap


def _vote(direction: int, strength: float, rationale: str = "") -> StrategyVote:
    return StrategyVote(name="", category="", direction=direction,
                        strength=float(np.clip(strength, 0.0, 1.0)), rationale=rationale)


# ===========================================================================
# FAMILY 1 — TREND FOLLOWING (01-20)
# ===========================================================================
@strategy("SMA-20/50 Cross", "Trend")
def s01(df, f, ctx):
    d = f["sma20_dist"]
    return _vote(1 if d > 0 else -1, min(1.0, abs(d) * 8),
                 f"price {d:+.1%} vs SMA20, 50d {'above' if f['sma50_gt_200'] else 'below'} 200d")


@strategy("Golden/Death Cross State", "Trend")
def s02(df, f, ctx):
    return _vote(1 if f["sma50_gt_200"] else -1, 0.6,
                 "SMA50 " + ("above" if f["sma50_gt_200"] else "below") + " SMA200")


@strategy("EMA Ribbon Alignment", "Trend")
def s03(df, f, ctx):
    a = f["ma_align"]
    return _vote(1 if a > 0.6 else -1 if a < 0.4 else 0, abs(a - 0.5) * 2 * 0.8,
                 f"ribbon alignment {a:.0%}")


@strategy("MACD Signal", "Trend")
def s04(df, f, ctx):
    m = f["macd_norm"]
    h = f["macd_hist_norm"]
    direction = 1 if m > 0 else -1
    strength = min(1.0, abs(m) * 30) * (1.15 if np.sign(m) == np.sign(h) else 0.7)
    return _vote(direction, strength, f"MACD {m:+.4f} hist {h:+.4f}")


@strategy("ADX Trend Strength", "Trend")
def s05(df, f, ctx):
    adx = f["adx14"]
    if adx < 20:
        return None
    direction = 1 if f["linreg_slope20"] > 0 else -1
    return _vote(direction, min(1.0, (adx - 20) / 30),
                 f"ADX {adx:.0f} with slope {f['linreg_slope20']:+.2%}/yr")


@strategy("Donchian Position", "Trend")
def s06(df, f, ctx):
    p = f["donchian_pos"]
    if p > 0.9:
        return _vote(1, (p - 0.9) * 10 + 0.4, "at top of 20d channel")
    if p < 0.1:
        return _vote(-1, (0.1 - p) * 10 + 0.4, "at bottom of 20d channel")
    return _vote(1 if p > 0.5 else -1, abs(p - 0.5) * 0.8, f"channel position {p:.0%}")


@strategy("52-Week High Proximity", "Trend")
def s07(df, f, ctx):
    d = f["dist_52w_high"]
    if d > -0.02:
        return _vote(1, 0.7, f"within 2% of 52w high ({d:+.1%})")
    if d < -0.30:
        return _vote(-1, min(0.8, -d), f"{d:+.1%} from 52w high")
    return _vote(0, 0.2, f"{d:+.1%} from 52w high")


@strategy("Time-Series Momentum 3M", "Trend")
def s08(df, f, ctx):
    r = f["ret_63d"]
    return _vote(1 if r > 0 else -1, min(1.0, abs(r) * 3),
                 f"3m return {r:+.1%}")


@strategy("Time-Series Momentum 12M", "Trend")
def s09(df, f, ctx):
    r = f["ret_252d"] if "ret_252d" in f else f["ret_126d"]
    return _vote(1 if r > 0 else -1, min(1.0, abs(r) * 1.5),
                 f"12m return {r:+.1%}")


@strategy("Dual Momentum (abs+rel)", "Trend")
def s10(df, f, ctx):
    r3, r6 = f["ret_63d"], f["ret_126d"]
    if r3 > 0 and r6 > 0:
        return _vote(1, min(1.0, r3 * 4), f"dual momentum aligned {r3:+.1%}/{r6:+.1%}")
    if r3 < 0 and r6 < 0:
        return _vote(-1, min(1.0, -r3 * 4), f"dual momentum negative {r3:+.1%}/{r6:+.1%}")
    return _vote(0, 0.25, "mixed momentum")


@strategy("Linear Regression Trend", "Trend")
def s11(df, f, ctx):
    slope, r2 = f["linreg_slope60"], f["linreg_r2_60"]
    if r2 < 0.05:
        return None
    direction = 1 if slope > 0 else -1
    return _vote(direction, min(1.0, abs(slope) * 2 * r2 * 3),
                 f"60d trend {slope:+.1%}/yr R²={r2:.2f}")


@strategy("Higher-Highs Structure", "Trend")
def s12(df, f, ctx):
    if len(df) < 30:
        return None
    h = df["high"].tail(20).values
    hh = sum(1 for i in range(1, 10) if h[-i] > h[-i - 1])
    ll_col = df["low"].tail(20).values
    ll = sum(1 for i in range(1, 10) if ll_col[-i] > ll_col[-i - 1])
    direction = 1 if hh >= 6 else -1 if hh <= 3 else 0
    return _vote(direction, abs(hh - 4.5) / 4.5,
                 f"{hh}/9 rising highs, {ll}/9 rising lows")


@strategy("KAMA Efficiency Ratio", "Trend")
def s13(df, f, ctx):
    c = df["close"]
    if len(c) < 25:
        return None
    change = abs(c.iloc[-1] - c.iloc[-11])
    vol_path = c.diff().abs().tail(10).sum()
    er = float(change / vol_path) if vol_path > 0 else 0.0
    direction = 1 if c.iloc[-1] > c.iloc[-11] else -1
    return _vote(direction if er > 0.3 else 0, max(0.0, er - 0.25),
                 f"efficiency ratio {er:.2f}")


@strategy("Price vs VWAP-20", "Trend")
def s14(df, f, ctx):
    d = f["vwap_dist"]
    return _vote(1 if d > 0 else -1, min(1.0, abs(d) * 6),
                 f"{d:+.1%} vs 20d VWAP")


@strategy("Tenkan/Kijun (Ichimoku core)", "Trend")
def s15(df, f, ctx):
    if len(df) < 52:
        return None
    h, lo, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_b = (h.rolling(52).max() + lo.rolling(52).min()) / 2
    t, k, b = tenkan.iloc[-1], kijun.iloc[-1], span_b.iloc[-1]
    px = c.iloc[-1]
    # int() is not decoration: px, b, k and t are numpy scalars, and numpy adds
    # bool scalars as *xor*, so this sum used to be True/False, never 0..3.
    # `score >= 2` was then unreachable and the strategy voted bearish on every
    # symbol with 52+ bars — a permanent short vote inside the consensus.
    score = int(px > b) + int(px > k) + int(t > k)
    if score >= 2:
        return _vote(1, score / 3 + 0.2, "ichimoku bullish stack")
    if score <= 1:
        return _vote(-1, (3 - score) / 3 + 0.2, "ichimoku bearish stack")
    return None


@strategy("Parabolic SAR State", "Trend")
def s16(df, f, ctx):
    if len(df) < 20:
        return None
    af, ep, trend, sar = 0.02, df["high"].iloc[0], 1, df["low"].iloc[0]
    for i in range(1, len(df)):
        hi, lo = df["high"].iloc[i], df["low"].iloc[i]
        sar = sar + af * (ep - sar)
        if trend == 1:
            if lo < sar:
                trend, sar, ep, af = -1, hi, lo, 0.02
            elif hi > ep:
                ep, af = hi, min(0.2, af + 0.02)
        else:
            if hi > sar:
                trend, sar, ep, af = 1, lo, hi, 0.02
            elif lo < ep:
                ep, af = lo, min(0.2, af + 0.02)
    return _vote(trend, 0.55, f"PSAR {'long' if trend == 1 else 'short'} regime")


@strategy("SuperTrend Direction", "Trend")
def s17(df, f, ctx):
    if len(df) < 30:
        return None
    c, h, lo = df["close"], df["high"], df["low"]
    atr = (h - lo).rolling(10).mean()
    mid = (h + lo) / 2
    upper, lower = mid + 3 * atr, mid - 3 * atr
    direction, stop = 1, lower.iloc[10]
    for i in range(11, len(df)):
        if c.iloc[i] > stop and direction == -1:
            direction, stop = 1, upper.iloc[i]
        elif c.iloc[i] < stop and direction == 1:
            direction, stop = -1, lower.iloc[i]
        elif direction == 1:
            stop = max(stop, lower.iloc[i])
        else:
            stop = min(stop, upper.iloc[i])
    return _vote(direction, 0.6, f"SuperTrend(10,3) {'up' if direction == 1 else 'down'}")


@strategy("Volatility-Adjusted Momentum", "Trend")
def s18(df, f, ctx):
    r = f["ret_63d"]
    vol = f["vol60"] or 0.15
    if vol <= 0:
        return None
    sharpe_mom = r / vol
    return _vote(1 if sharpe_mom > 0 else -1, min(1.0, abs(sharpe_mom) * 2),
                 f"3m return/vol = {sharpe_mom:+.2f}")


@strategy("8/21 EMA Cross State", "Trend")
def s19(df, f, ctx):
    d8 = f["ema10_dist"]
    d21 = f["ema20_dist"]
    if d8 > 0 and d21 > 0:
        return _vote(1, 0.55, "above both fast EMAs")
    if d8 < 0 and d21 < 0:
        return _vote(-1, 0.55, "below both fast EMAs")
    return _vote(0, 0.25, "EMA cross transitioning")


@strategy("Trend Continuation Gap", "Trend")
def s20(df, f, ctx):
    g = f["last_gap"]
    if abs(g) < 0.004:
        return None
    r1 = f["ret_1d"]
    if np.sign(g) == np.sign(r1):
        return _vote(int(np.sign(g)), min(0.7, abs(g) * 25),
                     f"gap {g:+.1%} confirmed by close direction")


# ===========================================================================
# FAMILY 2 — MEAN REVERSION (21-40)
# ===========================================================================
@strategy("RSI-14 Extreme", "MeanReversion")
def s21(df, f, ctx):
    r = f["rsi14"]
    if r < 30:
        return _vote(1, (30 - r) / 15 + 0.4, f"RSI oversold {r:.0f}")
    if r > 70:
        return _vote(-1, (r - 70) / 15 + 0.4, f"RSI overbought {r:.0f}")
    return None


@strategy("RSI-7 Fast Reversal", "MeanReversion")
def s22(df, f, ctx):
    r = f["rsi7"]
    if r < 25:
        return _vote(1, 0.5, f"fast RSI oversold {r:.0f}")
    if r > 75:
        return _vote(-1, 0.5, f"fast RSI overbought {r:.0f}")
    return None


@strategy("Z-Score Reversion (20d)", "MeanReversion")
def s23(df, f, ctx):
    z = f["bb_pct_b"]
    if z > 1.0:
        return _vote(-1, min(1.0, z - 0.5), f"%B={z:.2f} above upper band")
    if z < 0.0:
        return _vote(1, min(1.0, -z + 0.5), f"%B={z:.2f} below lower band")
    return None


@strategy("Bollinger Squeeze Release", "MeanReversion")
def s24(df, f, ctx):
    if f["bb_squeeze"] and f["adx14"] < 18:
        return _vote(1 if f["ret_5d"] > 0 else -1, 0.45,
                     "squeeze coiled, first direction of release")
    return None


@strategy("Stochastic Cross in Extreme", "MeanReversion")
def s25(df, f, ctx):
    k, d = f["stoch_k"], f["stoch_d"]
    if k < 20 and k > d:
        return _vote(1, 0.55, f"stoch crossing up from {k:.0f}")
    if k > 80 and k < d:
        return _vote(-1, 0.55, f"stoch crossing down from {k:.0f}")
    return None


@strategy("Williams %R Extreme", "MeanReversion")
def s26(df, f, ctx):
    w = f["williams_r"]
    if w < -80:
        return _vote(1, 0.5, f"%R={w:.0f} oversold")
    if w > -20:
        return _vote(-1, 0.5, f"%R={w:.0f} overbought")
    return None


@strategy("CCI-20 Extreme", "MeanReversion")
def s27(df, f, ctx):
    cci = f["cci20"]
    if cci < -150:
        return _vote(1, min(1.0, (-cci - 100) / 150), f"CCI={cci:.0f} deeply oversold")
    if cci > 150:
        return _vote(-1, min(1.0, (cci - 100) / 150), f"CCI={cci:.0f} deeply overbought")
    return None


@strategy("Keltner Band Reversion", "MeanReversion")
def s28(df, f, ctx):
    if f["keltner_pos"] > 1.0:
        return _vote(-1, min(1.0, f["keltner_pos"] - 0.6), "above upper Keltner")
    if f["keltner_neg"] < 0.0:
        return _vote(1, min(1.0, -f["keltner_neg"] + 0.4), "below lower Keltner")
    return None


@strategy("3-Day Reversal", "MeanReversion")
def s29(df, f, ctx):
    r3 = f["ret_3d"]
    if abs(r3) < 0.015:
        return None
    vol = f["vol20"] or 0.15
    if abs(r3) > 1.5 * vol / np.sqrt(252) * 3:
        return _vote(-int(np.sign(r3)), min(0.8, abs(r3) * 10),
                     f"3d move {r3:+.1%} exceeds 1.5 sigma")


@strategy("Short-Term Reversal (1d)", "MeanReversion")
def s30(df, f, ctx):
    r1 = f["ret_1d"]
    if abs(r1) < 0.01:
        return None
    return _vote(-int(np.sign(r1)), min(0.6, abs(r1) * 20),
                 f"1d reversal candidate {r1:+.1%}")


@strategy("Gap Fade", "MeanReversion")
def s31(df, f, ctx):
    g = f["last_gap"]
    if abs(g) < 0.006:
        return None
    return _vote(-int(np.sign(g)), min(0.65, abs(g) * 20),
                 f"gap {g:+.1%} likely to fill")


@strategy("Drawdown Recovery Bounce", "MeanReversion")
def s32(df, f, ctx):
    dd = f["drawdown"]
    r5 = f["ret_5d"]
    if dd < -0.12 and r5 > 0.02:
        return _vote(1, min(1.0, -dd * 3), f"bouncing from {dd:+.0%} drawdown")
    return None


@strategy("Volatility Spike Reversal", "MeanReversion")
def s33(df, f, ctx):
    if f["atr_ratio"] > 1.8 and f["ret_1d"] < 0 and f["rsi14"] < 45:
        return _vote(1, min(0.8, f["atr_ratio"] - 1.2),
                     f"ATR spike {f['atr_ratio']:.1f}x on red day — capitulation setup")
    return None


@strategy("TRIX Reversal", "MeanReversion")
def s34(df, f, ctx):
    t = f["trix14"]
    if abs(t) < 0.05:
        return None
    return _vote(-int(np.sign(t)), min(0.5, abs(t) * 3), f"TRIX {t:+.2f} overextended")


@strategy("MFI Extreme Reversal", "MeanReversion")
def s35(df, f, ctx):
    m = f["mfi14"]
    if m < 20:
        return _vote(1, 0.55, f"MFI={m:.0f} money flow oversold")
    if m > 80:
        return _vote(-1, 0.55, f"MFI={m:.0f} money flow overbought")
    return None


@strategy("RSI Divergence", "MeanReversion")
def s36(df, f, ctx):
    if len(df) < 40:
        return None
    c = df["close"].tail(30).values
    delta = df["close"].diff().fillna(0)
    gain = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14).mean()
    rsi_series = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    r = rsi_series.tail(30).values
    price_ll = c[-5:].min() < c[-25:-5].min()
    rsi_hl = r[-5:].min() > r[-25:-5].min()
    price_hh = c[-5:].max() > c[-25:-5].max()
    rsi_lh = r[-5:].max() < r[-25:-5].max()
    if price_ll and rsi_hl:
        return _vote(1, 0.65, "bullish RSI divergence")
    if price_hh and rsi_lh:
        return _vote(-1, 0.65, "bearish RSI divergence")
    return None


@strategy("Volume Climax Reversal", "MeanReversion")
def s37(df, f, ctx):
    if f["vol_z"] > 2.2 and f["ret_1d"] < 0 and f["rsi14"] < 40:
        return _vote(1, min(0.75, (f["vol_z"] - 1.5) / 2), "capitulation volume climax")
    if f["vol_z"] > 2.2 and f["ret_1d"] > 0 and f["rsi14"] > 65:
        return _vote(-1, min(0.75, (f["vol_z"] - 1.5) / 2), "euphoric volume climax")
    return None


@strategy("Envelopes Reversion (3%)", "MeanReversion")
def s38(df, f, ctx):
    d = f["sma20_dist"]
    if d > 0.03:
        return _vote(-1, min(0.6, (d - 0.03) * 12), f"{d:+.1%} above 3% envelope")
    if d < -0.03:
        return _vote(1, min(0.6, (-d - 0.03) * 12), f"{d:+.1%} below 3% envelope")
    return None


@strategy("Prior-Day Range Fade", "MeanReversion")
def s39(df, f, ctx):
    if len(df) < 3:
        return None
    prev_rng = (df["high"].iloc[-2] - df["low"].iloc[-2]) / df["close"].iloc[-2]
    if prev_rng < 0.01:
        return None
    pos = (df["close"].iloc[-1] - df["low"].iloc[-2]) / (df["high"].iloc[-2] - df["low"].iloc[-2])
    if pos > 0.95:
        return _vote(-1, 0.4, "failed at prior-day high")
    if pos < 0.05:
        return _vote(1, 0.4, "held prior-day low")
    return None


@strategy("CMF Divergence", "MeanReversion")
def s40(df, f, ctx):
    cmf = f["cmf20"]
    r10 = f["ret_10d"]
    if r10 > 0.05 and cmf < -0.05:
        return _vote(-1, 0.5, "price up but money flowing out")
    if r10 < -0.05 and cmf > 0.05:
        return _vote(1, 0.5, "price down but money flowing in")
    return None


# ===========================================================================
# FAMILY 3 — BREAKOUT (41-50)
# ===========================================================================
@strategy("20-Day Breakout", "Breakout")
def s41(df, f, ctx):
    if len(df) < 21:
        return None
    hi20 = df["high"].iloc[:-1].tail(20).max()
    lo20 = df["low"].iloc[:-1].tail(20).min()
    if df["close"].iloc[-1] > hi20:
        return _vote(1, 0.7, "closed above 20d high")
    if df["close"].iloc[-1] < lo20:
        return _vote(-1, 0.7, "closed below 20d low")
    return None


@strategy("55-Day Turtle Breakout", "Breakout")
def s42(df, f, ctx):
    if len(df) < 56:
        return None
    hi = df["high"].iloc[:-1].tail(55).max()
    lo = df["low"].iloc[:-1].tail(55).min()
    if df["close"].iloc[-1] > hi:
        return _vote(1, 0.8, "turtle long breakout 55d")
    if df["close"].iloc[-1] < lo:
        return _vote(-1, 0.8, "turtle short breakdown 55d")
    return None


@strategy("NR7 Consolidation Break", "Breakout")
def s43(df, f, ctx):
    if len(df) < 10:
        return None
    rngs = (df["high"] - df["low"]).tail(7).values
    if rngs.argmin() != 6:
        return None
    mid = (df["high"].iloc[-1] + df["low"].iloc[-1]) / 2
    direction = 1 if df["close"].iloc[-1] > mid else -1
    return _vote(direction, 0.45, "NR7 narrow range coil")


@strategy("ATR Range Expansion", "Breakout")
def s44(df, f, ctx):
    if f["atr_ratio"] > 1.5 and f["adx14"] > 22:
        direction = 1 if f["ret_1d"] > 0 else -1
        return _vote(direction, min(0.7, f["atr_ratio"] - 0.8),
                     f"range expansion {f['atr_ratio']:.1f}x with trend")


@strategy("Volume-Confirmed Breakout", "Breakout")
def s45(df, f, ctx):
    if f["vol_z"] > 1.5 and f["ret_1d"] > 0.025:
        return _vote(1, min(0.85, f["vol_z"] / 2 + 0.3),
                     f"breakout on {f['vol_z']:.1f}σ volume")
    if f["vol_z"] > 1.5 and f["ret_1d"] < -0.025:
        return _vote(-1, min(0.85, f["vol_z"] / 2 + 0.3),
                     f"breakdown on {f['vol_z']:.1f}σ volume")
    return None


@strategy("Bollinger Squeeze Breakout", "Breakout")
def s46(df, f, ctx):
    if not f["bb_squeeze"]:
        return None
    direction = 1 if f["ret_5d"] > 0 else -1 if f["ret_5d"] < 0 else 0
    if direction == 0:
        return None
    return _vote(direction, 0.5, "BB squeeze — energy loading for expansion")


@strategy("Darvas Box Break", "Breakout")
def s47(df, f, ctx):
    if len(df) < 40:
        return None
    h = df["high"].tail(40)
    box_top = h.iloc[:-5].max()
    if df["close"].iloc[-1] > box_top:
        return _vote(1, 0.6, f"Darvas box top {box_top:.1f} taken out")
    return None


@strategy("52-Week Breakout", "Breakout")
def s48(df, f, ctx):
    if f["dist_52w_high"] > -0.005:
        return _vote(1, 0.85, "at all-time/52w high — blue sky breakout")
    if f["dist_52w_low"] < 0.005:
        return _vote(-1, 0.75, "at 52w low — breakdown")
    return None


@strategy("Opening Gap Continuation", "Breakout")
def s49(df, f, ctx):
    g = f["last_gap"]
    if abs(g) < 0.01:
        return None
    body = f["body_ratio"]
    if np.sign(g) > 0 and body > 0.6:
        return _vote(1, 0.5, f"gap-up {g:+.1%} with strong body")
    if np.sign(g) < 0 and body > 0.6:
        return _vote(-1, 0.5, f"gap-down {g:+.1%} with strong body")
    return None


@strategy("Pennant/Triangle Completion", "Breakout")
def s50(df, f, ctx):
    if len(df) < 30:
        return None
    rng20 = (df["high"].tail(20).max() - df["low"].tail(20).min()) / df["close"].iloc[-1]
    rng5 = (df["high"].tail(5).max() - df["low"].tail(5).min()) / df["close"].iloc[-1]
    if rng20 > 0.06 and rng5 < rng20 * 0.35:
        direction = 1 if f["ret_63d"] > 0 else -1
        return _vote(direction, 0.45, f"symmetrical coil: {rng5:.1%} vs {rng20:.1%} range")


# ===========================================================================
# FAMILY 4 — VOLATILITY (51-58)
# ===========================================================================
@strategy("Volatility Regime Shift", "Volatility")
def s51(df, f, ctx):
    vr = f["vol_ratio"]
    if vr > 1.4:
        return _vote(-1 if f["ret_1d"] < 0 else 0, min(0.6, vr - 1.0),
                     f"short vol {f['vol20']:.0%} vs long {f['vol60']:.0%}")
    if vr < 0.7:
        return _vote(1, min(0.5, 0.7 - vr), "vol compression — calm before storm, bias up")
    return None


@strategy("Vol Breakout (ATR expansion)", "Volatility")
def s52(df, f, ctx):
    if f["atr_ratio"] > 1.6:
        return _vote(1 if f["ret_1d"] > 0 else -1, 0.5,
                     f"ATR expanding {f['atr_ratio']:.1f}x")


@strategy("Garman-Klass vs Close Divergence", "Volatility")
def s53(df, f, ctx):
    gk = f["garman_klass"]
    cl = f["vol20"]
    if gk > 0 and cl > 0 and gk > 1.5 * cl:
        return _vote(-1, 0.4, "intraday vol exceeds close-to-close — distribution")
    return None


@strategy("Low-Vol Anomaly Tilt", "Volatility")
def s54(df, f, ctx):
    if f["vol60"] < 0.12 and f["ret_126d"] > 0:
        return _vote(1, 0.45, f"low-vol {f['vol60']:.0%} with positive trend — anomaly premium")
    return None


@strategy("Kurtosis Fat-Tail Alert", "Volatility")
def s55(df, f, ctx):
    if f["kurt60"] > 5:
        return _vote(0, 0.5, f"excess kurtosis {f['kurt60']:.1f} — cut size, tail risk")
    return None


@strategy("Downside Vol Asymmetry", "Volatility")
def s56(df, f, ctx):
    dv, uv = f["downside_vol"], f["vol20"]
    if uv > 0 and dv > 1.4 * uv:
        return _vote(-1, 0.45, "downside vol dominates — bearish asymmetry")
    if uv > 0 and dv < 0.85 * uv:
        return _vote(1, 0.35, "upside vol dominates")
    return None


@strategy("Beta-Vol Overlay", "Volatility")
def s57(df, f, ctx):
    if f.get("vol_percentile", 0.5) > 0.85:
        return _vote(-1, 0.4, f"vol at the {f['vol_percentile']:.0%} of its own history — de-risk")
    return None


@strategy("Choppiness Avoidance", "Volatility")
def s58(df, f, ctx):
    if f["adx14"] < 15 and f["hurst"] < 0.5:
        return _vote(0, 0.6, f"choppy (ADX {f['adx14']:.0f}, H={f['hurst']:.2f}) — stand aside")
    return None


# ===========================================================================
# FAMILY 5 — SEASONALITY / CALENDAR (59-66)
# ===========================================================================
def _calendar(ctx) -> pd.Timestamp | None:
    d = ctx.get("date_today")
    if d is None and ctx.get("df") is not None and len(ctx["df"]):
        d = ctx["df"]["date"].iloc[-1]
    return pd.Timestamp(d) if d is not None else None


@strategy("Turn-of-Month Effect", "Seasonality")
def s59(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.day <= 2 or d.day >= 28:
        return _vote(1, 0.35, "turn-of-month window — historical bullish drift")
    return None


@strategy("Friday Effect", "Seasonality")
def s60(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.weekday() == 4:
        return _vote(1, 0.25, "Friday — pre-weekend institutional positioning")
    if d.weekday() == 0:
        return _vote(-1, 0.2, "Monday — weekend-risk discount")
    return None


@strategy("January Effect", "Seasonality")
def s61(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.month == 1 and d.day <= 15:
        return _vote(1, 0.35, "first two January weeks — seasonal strength")
    return None


@strategy("Ramadan/Rabi Window (PSX)", "Seasonality")
def s62(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.month in (2, 3):
        return _vote(1, 0.2, "pre-Ramadan accumulation window historically")
    if d.month == 12:
        return _vote(1, 0.25, "year-end PSX rally window")
    return None


@strategy("Fiscal Year-End (PK June)", "Seasonality")
def s63(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.month == 6:
        return _vote(1 if f["ret_63d"] > 0 else 0, 0.3,
                     "June: Pakistan fiscal year-end, book-flattering window dressing")
    return None


@strategy("Quarter-End Window Dressing", "Seasonality")
def s64(df, f, ctx):
    d = _calendar(ctx)
    if d is None:
        return None
    if d.month in (3, 6, 9, 12) and d.day >= 25:
        return _vote(1 if f["ret_63d"] > 0 else -1, 0.25, "quarter-end momentum padding")
    return None


@strategy("Month-of-Year Momentum", "Seasonality")
def s65(df, f, ctx):
    if len(df) < 300:
        return None
    d = _calendar(ctx)
    if d is None:
        return None
    dates = pd.to_datetime(df["date"])
    same_month = df["close"][dates.dt.month == d.month]
    if len(same_month) < 15:
        return None
    mret = float(same_month.iloc[-1] / same_month.iloc[0] - 1) if len(same_month) > 1 else 0
    if abs(mret) < 0.01:
        return None
    return _vote(int(np.sign(mret)), min(0.4, abs(mret) * 4),
                 f"month-{d.month} historical drift {mret:+.1%}")


@strategy("Weekly Seasonality Pattern", "Seasonality")
def s66(df, f, ctx):
    if len(df) < 120:
        return None
    d = _calendar(ctx)
    if d is None:
        return None
    dates = pd.to_datetime(df["date"])
    fwd = df["close"].shift(-1) / df["close"] - 1
    mask = dates.dt.weekday == d.weekday()
    mean_fwd = float(fwd[mask].dropna().mean())
    if abs(mean_fwd) < 0.001:
        return None
    return _vote(int(np.sign(mean_fwd)), min(0.35, abs(mean_fwd) * 60),
                 f"weekday-{d.weekday()} next-day drift {mean_fwd:+.2%}")


# ===========================================================================
# FAMILY 6 — INTERMARKET / MACRO (67-76)
# ===========================================================================
def _corr_signal(ctx_key: str, ctx, my_ret_10d: float, invert: bool = False,
                 min_corr: float = 0.25) -> StrategyVote | None:
    inter = ctx.get("intermarket") or {}
    other = inter.get(ctx_key)
    if other is None or len(other) < 40:
        return None
    c = ctx["df"]["close"].tail(60)
    o = other["close"].tail(60)
    n = min(len(c), len(o))
    if n < 40:
        return None
    rc = np.diff(np.log(c.tail(n).values.astype(float)))
    ro = np.diff(np.log(o.tail(n).values.astype(float)))
    if np.std(ro) == 0 or np.std(rc) == 0:
        return None
    corr = float(np.corrcoef(rc, ro)[0, 1])
    if abs(corr) < min_corr:
        return None
    o_ret_10 = float(o.iloc[-1] / o.iloc[-11] - 1)
    implied = o_ret_10 * corr
    if invert:
        implied = -implied
    if abs(implied) < 0.008:
        return None
    return _vote(int(np.sign(implied)), min(0.7, abs(implied) * 8 + abs(corr) * 0.2),
                 f"{ctx_key} 10d {o_ret_10:+.1%} corr={corr:+.2f} implies {implied:+.1%}")


@strategy("Dollar Inverse Effect", "Intermarket")
def s67(df, f, ctx):
    return _corr_signal("DXY", ctx, f["ret_10d"], invert=True)


@strategy("Rates Pressure Effect", "Intermarket")
def s68(df, f, ctx):
    return _corr_signal("US10Y", ctx, f["ret_10d"], invert=False)


@strategy("VIX Fear Gauge", "Intermarket")
def s69(df, f, ctx):
    inter = ctx.get("intermarket") or {}
    vix = inter.get("VIX")
    if vix is None or len(vix) < 30:
        return None
    v = float(vix["close"].iloc[-1])
    v_prev = float(vix["close"].iloc[-11])
    if v > 28:
        return _vote(-1, min(0.8, (v - 24) / 12), f"VIX {v:.0f} fear regime")
    if v < 14 and v < v_prev:
        return _vote(1, 0.5, f"VIX {v:.0f} complacent calm")
    return None


@strategy("Risk-On/Off (SPX lead)", "Intermarket")
def s70(df, f, ctx):
    return _corr_signal("SPX", ctx, f["ret_10d"])


@strategy("Oil Synchronization", "Intermarket")
def s71(df, f, ctx):
    return _corr_signal("WTI", ctx, f["ret_10d"])


@strategy("Gold Tailwind", "Intermarket")
def s72(df, f, ctx):
    return _corr_signal("GOLD", ctx, f["ret_10d"])


@strategy("Copper-Gold Growth Barometer", "Intermarket")
def s73(df, f, ctx):
    inter = ctx.get("intermarket") or {}
    cu, au = inter.get("COPPER"), inter.get("GOLD")
    if cu is None or au is None or len(cu) < 40 or len(au) < 40:
        return None
    ratio = (cu["close"] / au["close"]).tail(60)
    if len(ratio) < 40:
        return None
    r10 = float(ratio.iloc[-1] / ratio.iloc[-11] - 1)
    if abs(r10) < 0.01:
        return None
    return _vote(int(np.sign(r10)), min(0.6, abs(r10) * 6),
                 f"copper/gold ratio {r10:+.1%} — global growth signal")


@strategy("PKR Stability Effect (PSX)", "Intermarket")
def s74(df, f, ctx):
    inter = ctx.get("intermarket") or {}
    pkr = inter.get("USDPKR")
    if pkr is None or len(pkr) < 40:
        return None
    r10 = float(pkr["close"].iloc[-1] / pkr["close"].iloc[-11] - 1)
    if r10 > 0.01:
        return _vote(-1, min(0.7, r10 * 20), f"PKR depreciating {r10:+.1%} — foreign outflow risk")
    if r10 < -0.003:
        return _vote(1, min(0.5, -r10 * 20), f"PKR stable/appreciating {r10:+.1%}")
    return None


@strategy("BTC Risk Appetite Proxy", "Intermarket")
def s75(df, f, ctx):
    return _corr_signal("BTC", ctx, f["ret_10d"])


@strategy("Wheat→Fertilizer Pass-Through (PSX)", "Intermarket")
def s76(df, f, ctx):
    is_agri = "agri-proxy" in ctx.get("tags", ())
    if not is_agri:
        return None
    return _corr_signal("WHEAT", ctx, f["ret_10d"])


# ===========================================================================
# FAMILY 7 — STATISTICAL EDGE (77-82)
# ===========================================================================
@strategy("Hurst Regime Exploit", "Statistical")
def s77(df, f, ctx):
    h = f["hurst"]
    if h > 0.60:
        return _vote(1 if f["ret_21d"] > 0 else -1, min(0.7, (h - 0.55) * 4),
                     f"H={h:.2f} persistent — follow the trend")
    if h < 0.40:
        return _vote(-1 if f["ret_5d"] > 0 else 1, min(0.7, (0.45 - h) * 4),
                     f"H={h:.2f} anti-persistent — fade the move")
    return None


@strategy("Autocorrelation Edge", "Statistical")
def s78(df, f, ctx):
    a1 = f["autocorr1"]
    if a1 > 0.15:
        return _vote(1 if f["ret_1d"] > 0 else -1, min(0.5, a1 * 2),
                     f"AC(1)={a1:+.2f} positive — continuation")
    if a1 < -0.15:
        return _vote(-1 if f["ret_1d"] > 0 else 1, min(0.5, -a1 * 2),
                     f"AC(1)={a1:+.2f} negative — alternation")
    return None


@strategy("Variance Ratio Test", "Statistical")
def s79(df, f, ctx):
    c = df["close"]
    if len(c) < 120:
        return None
    r1 = np.diff(np.log(c.values.astype(float)))
    r5 = r1[::5][:len(r1) // 5] * 0  # placeholder replaced below
    r5 = np.array([r1[i:i + 5].sum() for i in range(0, len(r1) - 5, 5)])
    v1, v5 = float(np.var(r1)), float(np.var(r5))
    if v1 == 0 or v5 == 0:
        return None
    vr = (v5 / 5) / v1
    if vr > 1.15:
        return _vote(1 if f["ret_21d"] > 0 else -1, min(0.5, (vr - 1) * 1.5),
                     f"VR={vr:.2f} trending variance")
    if vr < 0.85:
        return _vote(-1 if f["ret_5d"] > 0 else 1, min(0.5, (1 - vr) * 1.5),
                     f"VR={vr:.2f} mean-reverting variance")
    return None


@strategy("Runs Test Edge", "Statistical")
def s80(df, f, ctx):
    c = df["close"]
    if len(c) < 90:
        return None
    r = np.sign(np.diff(np.log(c.values.astype(float)))[-90:])
    r = r[r != 0]
    n = len(r)
    if n < 40:
        return None
    runs = 1 + int(np.sum(r[1:] != r[:-1]))
    n1, n2 = int((r > 0).sum()), int((r < 0).sum())
    if n1 == 0 or n2 == 0:
        return None
    exp_runs = 2 * n1 * n2 / n + 1
    vr_runs = (runs - exp_runs) / np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1)))
    if vr_runs < -1.8:
        return _vote(1 if f["ret_5d"] > 0 else -1, min(0.45, abs(vr_runs) / 4),
                     f"runs z={vr_runs:+.1f} — streaky, continuation")
    if vr_runs > 1.8:
        return _vote(-1 if f["ret_5d"] > 0 else 1, min(0.45, vr_runs / 4),
                     f"runs z={vr_runs:+.1f} — alternating, fade")
    return None


@strategy("Entropy Information Filter", "Statistical")
def s81(df, f, ctx):
    e = f["entropy"]
    if e > 1.9:
        return _vote(0, 0.5, f"entropy {e:.2f} — near-random walk, no edge")
    if e < 1.2:
        return _vote(1 if f["ret_21d"] > 0 else -1, 0.4,
                     f"entropy {e:.2f} — structured, trend exploitable")
    return None


@strategy("OU Half-Life Trader", "Statistical")
def s82(df, f, ctx):
    hl = None
    y = np.log(df["close"].values.astype(float))[-250:]
    if len(y) > 60:
        phi = float(np.polyfit(y[:-1], y[1:], 1)[0])
        if 0 < phi < 1:
            hl = abs(float(np.log(0.5) / np.log(phi)))
    if hl is None or hl > 60 or hl < 1:
        return None
    z = float((y[-1] - np.mean(y)) / (np.std(y) or 1e-9))
    if abs(z) < 1.0:
        return None
    strength = min(0.7, abs(z) * 0.25) * (1.2 if hl < 15 else 0.8)
    return _vote(-int(np.sign(z)), strength,
                 f"z={z:+.2f}, half-life {hl:.0f}d — statistical reversion")


# ===========================================================================
# FAMILY 8 — CROSS-SECTIONAL FACTORS (83-88) — PSX factor overlay
# ===========================================================================
@strategy("Factor: Momentum 1M", "Factor")
def s83(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or "momentum_1m" not in row or pd.isna(row.get("momentum_1m")):
        return None
    z = float(row["momentum_1m"])
    return _vote(int(np.sign(z)), min(0.8, abs(z) / 2.5), f"1m momentum z={z:+.2f}")


@strategy("Factor: Value (low PE)", "Factor")
def s84(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or pd.isna(row.get("value")):
        return None
    z = float(row["value"])
    return _vote(1 if z > 0 else -1, min(0.6, abs(z) / 3), f"value z={z:+.2f}")


@strategy("Factor: Quality (ROE/margin)", "Factor")
def s85(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or pd.isna(row.get("quality")):
        return None
    z = float(row["quality"])
    return _vote(1 if z > 0 else -1, min(0.6, abs(z) / 3), f"quality z={z:+.2f}")


@strategy("Factor: Low Volatility", "Factor")
def s86(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or pd.isna(row.get("low_vol")):
        return None
    z = float(row["low_vol"])
    return _vote(1 if z > 0 else 0, min(0.5, abs(z) / 3), f"low-vol z={z:+.2f}")


@strategy("Factor: Size (small-cap premium)", "Factor")
def s87(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or pd.isna(row.get("size")):
        return None
    z = float(row["size"])
    return _vote(1 if z > 0 else 0, min(0.4, abs(z) / 3), f"size z={z:+.2f}")


@strategy("Factor: Liquidity Screen", "Factor")
def s88(df, f, ctx):
    row = ctx.get("factor_row")
    if row is None or pd.isna(row.get("liquidity")):
        return None
    z = float(row["liquidity"])
    if z < -1.5:
        return _vote(0, 0.6, "illiquid — avoid full size")
    return None


# ===========================================================================
# FAMILY 9 — VOLUME / MICROSTRUCTURE (89-96)
# ===========================================================================
@strategy("Volume Surge Direction", "Volume")
def s89(df, f, ctx):
    if f["vol_z"] > 1.8:
        return _vote(1 if f["ret_1d"] > 0 else -1, min(0.75, f["vol_z"] / 2.5),
                     f"{f['vol_z']:.1f}σ volume with {f['ret_1d']:+.1%} move")
    return None


@strategy("OBV Trend Confirmation", "Volume")
def s90(df, f, ctx):
    o = f["obv_slope"]
    return _vote(1 if o > 0 else -1, min(0.6, abs(o) * 4),
                 f"OBV 5d slope {o:+.2f}")


@strategy("Up/Down Day Ratio", "Volume")
def s91(df, f, ctx):
    u = f["up_day_ratio_10"]
    if u > 0.7:
        return _vote(1, (u - 0.5) * 1.6, f"{u:.0%} up days out of 10")
    if u < 0.3:
        return _vote(-1, (0.5 - u) * 1.6, f"{u:.0%} up days out of 10")
    return None


@strategy("Amihud Illiquidity Cost", "Volume")
def s92(df, f, ctx):
    if f.get("amihud_relative", 1.0) > 5:
        return _vote(0, 0.5, f"price impact {f['amihud_relative']:.0f}x this "
                             "instrument's own median — reduce size")
    return None


@strategy("Effort vs Result Divergence", "Volume")
def s93(df, f, ctx):
    if len(df) < 25:
        return None
    vol_now = df["volume"].tail(5).mean()
    vol_prev = df["volume"].tail(25).head(20).mean()
    rng_now = (df["high"] - df["low"]).tail(5).mean()
    rng_prev = (df["high"] - df["low"]).tail(25).head(20).mean()
    if vol_prev == 0 or rng_prev == 0:
        return None
    effort = vol_now / vol_prev
    result = rng_now / rng_prev
    if effort > 1.5 and result < 0.7:
        return _vote(-1 if f["ret_5d"] > 0 else 1, 0.45,
                     "heavy volume, small range — absorption against the move")
    if effort < 0.7 and result > 1.3:
        return _vote(1 if f["ret_5d"] > 0 else -1, 0.4,
                     "light volume, large range — vacuum move, fade")
    return None


@strategy("Volume Trend Break", "Volume")
def s94(df, f, ctx):
    vt = f["volume_trend"]
    if vt > 1.5 and f["ret_10d"] > 0:
        return _vote(1, min(0.6, vt - 1.0), f"volume trend {vt:.1f}x expanding with rally")
    if vt < 0.6 and f["ret_10d"] < 0:
        return _vote(-1, 0.4, "declining volume confirms weakness")
    return None


@strategy("Aroon Direction", "Volume")
def s95(df, f, ctx):
    if len(df) < 30:
        return None
    h, lo = df["high"], df["low"]
    since_high = 25 - int(np.argmax(h.tail(25).values))
    since_low = 25 - int(np.argmin(lo.tail(25).values))
    up = (25 - since_high) / 25 * 100
    dn = (25 - since_low) / 25 * 100
    if up > 70 and up > dn:
        return _vote(1, up / 100, f"Aroon up={up:.0f}")
    if dn > 70 and dn > up:
        return _vote(-1, dn / 100, f"Aroon down={dn:.0f}")
    return None


@strategy("MFI Trend Pressure", "Volume")
def s96(df, f, ctx):
    m = f["mfi14"]
    if 40 < m < 60:
        return None
    return _vote(1 if m > 60 else -1, abs(m - 50) / 50,
                 f"MFI {m:.0f} money-flow pressure")


# ===========================================================================
# FAMILY 10 — PATTERN CONFLUENCE & SENTIMENT (97-100)
# ===========================================================================
@strategy("Candlestick Confluence", "Pattern")
def s97(df, f, ctx):
    pats = ctx.get("patterns") or []
    bull = {"hammer", "bullish_engulfing", "morning_star", "three_white_soldiers",
            "tweezer_bottom", "bullish_marubozu"}
    bear = {"shooting_star", "bearish_engulfing", "evening_star", "three_black_crows",
            "tweezer_top", "bearish_marubozu"}
    b = sum(1 for p in pats if p in bull)
    s = sum(1 for p in pats if p in bear)
    if b and not s:
        return _vote(1, min(0.7, 0.35 * b), f"bullish patterns: {', '.join(p for p in pats if p in bull)}")
    if s and not b:
        return _vote(-1, min(0.7, 0.35 * s), f"bearish patterns: {', '.join(p for p in pats if p in bear)}")
    return None


@strategy("Doji Indecision Filter", "Pattern")
def s98(df, f, ctx):
    pats = ctx.get("patterns") or []
    if "doji" in pats and f["adx14"] < 20:
        return _vote(0, 0.5, "doji in range — no edge, wait for resolution")
    return None


@strategy("News Sentiment Tilt", "Sentiment")
def s99(df, f, ctx):
    s = ctx.get("news_sentiment")
    if s is None or abs(float(s)) < 0.15:
        return None
    # The number is the same whichever estimator produced it, and the vote says
    # which one did and how many headlines stood behind it: a tone from two
    # stories and one from nine are the same sign and not the same evidence, and
    # an operator reviewing a news-driven entry should not have to go and find
    # out afterwards.
    tone = ctx.get("news_tone") or {}
    desk = tone.get("desk") or {}
    detail = f"news sentiment {float(s):+.2f}"
    if tone.get("source"):
        detail += f" [{tone['source']}"
        if desk.get("matching"):
            detail += f", {int(desk['matching'])} headlines"
        detail += "]"
    return _vote(int(np.sign(s)), min(0.6, abs(float(s))), detail)


@strategy("Multi-Strategy Confluence Master", "Sentiment")
def s100(df, f, ctx):
    """Meta-strategy: requires strong alignment of trend, momentum and volume."""
    checks = [f["sma20_dist"] > 0, f["ret_21d"] > 0, f["macd_norm"] > 0,
              f["rsi14"] > 50, f["obv_slope"] > 0]
    neg = [f["sma20_dist"] < 0, f["ret_21d"] < 0, f["macd_norm"] < 0,
           f["rsi14"] < 50, f["obv_slope"] < 0]
    if sum(checks) == 5:
        return _vote(1, 0.9, "full bullish confluence (trend+momentum+volume)")
    if sum(neg) == 5:
        return _vote(-1, 0.9, "full bearish confluence (trend+momentum+volume)")
    return None


# ---------------------------------------------------------------------------
# Registry execution
# ---------------------------------------------------------------------------
def run_strategy_library(df: pd.DataFrame, features: dict[str, float],
                         ctx: dict[str, Any] | None = None,
                         report: dict[str, Any] | None = None) -> list[StrategyVote]:
    """Poll the library.

    `report` is optional and, when given, is filled with every name that was
    asked and the ones that raised. A strategy that throws has always been
    dropped here in silence, which makes it indistinguishable from one that
    legitimately abstained — and the difference is the whole of what an
    attribution run needs: an abstention is a judgment, a `KeyError` on every
    slice is a broken component that has never once been measured.
    """
    ctx = dict(ctx or {})
    ctx["df"] = df
    votes: list[StrategyVote] = []
    if report is not None:
        report["seen"] = [name for name, _cat, _fn in STRATEGIES]
        report["errors"] = errors = {}
    for name, category, fn in STRATEGIES:
        try:
            v = fn(df, features, ctx)
        except Exception as exc:
            if report is not None:
                errors[name] = f"{type(exc).__name__}: {exc}"[:160]
            continue
        if v is None:
            continue
        v.name = name
        v.category = category
        votes.append(v)
    return votes


def library_size() -> int:
    return len(STRATEGIES)
