"""Four library members that could never speak, and the plumbing that let them.

`tools/strategy_attribution.py` grades whatever a strategy *claims*, so a
strategy that returns None on every slice is indistinguishable from one that
looked at the data and stayed quiet. Around a dozen of the hundred recorded
exactly zero votes across the walk. These tests hold up the three that were
defects rather than deliberate silences, plus the one that was reading a hole
in the data as its loudest signal.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from kairos.features import compute_features
from kairos.strategies import s53, s57, s92


def _tape(rows: int, sigma, hl: float, volume, seed: int) -> pd.DataFrame:
    """One continuous price series, so no test trips over a level jump between
    two independently generated halves. `sigma` and `volume` may be scalars or
    per-bar arrays."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, rows) * np.asarray(sigma, dtype=float)
    close = np.cumprod(1.0 + steps) * 100.0
    open_ = np.r_[close[0], close[:-1]]
    span = np.maximum(np.abs(open_), 1e-9) * np.asarray(hl, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": open_, "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span, "close": close,
        "volume": np.full(rows, 1.0) * np.asarray(volume, dtype=float),
    })


def _calm_then_wild() -> pd.DataFrame:
    """100 calm bars, then a tail four times as wild — a rank, not a level."""
    sigma = np.r_[np.full(100, 0.002), np.full(30, 0.008)]
    return _tape(130, sigma, 0.004, 1e6, 7)


def _wide_range() -> pd.DataFrame:
    """Sessions that swing far more intraday than the close-to-close tape says."""
    return _tape(140, 0.004, 0.008, 1e6, 11)


def _no_wick(rows: int = 140, seed: int = 3) -> pd.DataFrame:
    """A bar whose high and low are its own body: the range is the move, with
    nothing extra. This is the shape the divergence member must stay quiet on."""
    rng = np.random.default_rng(seed)
    close = np.cumprod(1.0 + rng.normal(0.0, 0.008, rows)) * 100.0
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": open_, "high": np.maximum(open_, close),
        "low": np.minimum(open_, close), "close": close,
        "volume": np.full(rows, 1e6),
    })


def _gapped(rows: int = 140, seed: int = 5) -> pd.DataFrame:
    """Closes open away from the previous close, the way a real session does.
    The gap is the difference between the open-to-close return Garman-Klass
    subtracts and the close-to-close return it must not."""
    rng = np.random.default_rng(seed)
    close = np.cumprod(1.0 + rng.normal(0.0, 0.008, rows)) * 100.0
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * (1.0 + rng.normal(0.0, 0.008, rows))
    body = np.abs(open_ - close)
    wick = np.maximum(body, close * 0.004) * 0.5
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": open_, "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick, "close": close,
        "volume": np.full(rows, 1e6),
    })


def _disturb(df: pd.DataFrame, end_bars_ago: int, rows: int, mult: float) -> pd.DataFrame:
    """Widen the range of a block that ends `end_bars_ago` bars from the tape."""
    out = df.copy()
    n = len(out)
    sl = slice(n - end_bars_ago - rows, n - end_bars_ago)
    out.loc[sl, "high"] = out.loc[sl, "high"] * mult
    out.loc[sl, "low"] = out.loc[sl, "low"] / mult
    return out


def _dry_book() -> pd.DataFrame:
    """The same tape, and a tail with a hundredth of the turnover."""
    volume = np.r_[np.full(120, 1e6), np.full(40, 1e4)]
    return _tape(160, 0.006, 0.004, volume, 3)


def test_vol_percentile_is_a_feature_and_not_only_a_regime_field():
    """`Beta-Vol Overlay` asked for `features["vol_percentile"]`, which only
    `detect_regime` produced — and asked for it through a conditional that
    answered False whenever the key was missing. So it never voted, in either
    path, without raising anything."""
    feats = compute_features(_calm_then_wild())
    assert "vol_percentile" in feats, "the field the overlay reads does not exist"
    pct = feats["vol_percentile"]
    assert 0.0 <= pct <= 1.0
    assert type(pct) is float, "a numpy scalar in a persisted feature dict"
    assert pct > 0.85, f"a vol tail 4x the body did not rank high: {pct}"
    calm = compute_features(_tape(140, 0.004, 0.004, 1e6, 5))
    assert calm["vol_percentile"] <= 0.85, "a flat tape ranks itself as a vol spike"

    vote = s57(_calm_then_wild(), feats, {})
    assert vote is not None and vote.direction < 0, "the overlay still never speaks"


def test_range_estimators_and_the_close_to_close_vol_are_the_same_size():
    """Garman-Klass and Parkinson were per-bar numbers in a dict whose every
    other volatility field is annualized. `Garman-Klass vs Close Divergence`
    compared them at a 1.5x gate, so it was dividing a one-day number by a
    252-day one — unreachable by a factor of sixteen."""
    df = _wide_range()
    feats = compute_features(df)
    ratio = feats["garman_klass"] / feats["vol20"]
    assert ratio > 1.5, f"intraday range far beyond the close tape scored {ratio}"
    # The same frame measured the old way: the daily estimate against the
    # annualized one. This is the assertion the fix was made for, and it is the
    # falsifiable half — if the annualization were missing, the line above fails.
    assert (feats["garman_klass"] / math.sqrt(252)) / feats["vol20"] < 1.5, \
        "the pre-fix scale no longer fails this test, so this test checks nothing"
    vote = s53(df, feats, {})
    assert vote is not None and vote.direction < 0, "the divergence strategy still abstains"


def test_the_range_estimators_agree_with_each_other_on_a_tape_with_gaps():
    """Annualizing them was the first fix, and it made the member vote — on every
    instrument in the book, at the same strength, on a calm tape. Two more errors
    were underneath, and they hid in each other: the estimator subtracted the
    *close-to-close* return instead of the open-to-close one, and it took the root
    of the mean of *squares* of a quantity that is legitimately negative on a bar
    that closes near its open, so every negative bar was paid for as a positive
    one. On a gapped session the two range estimators disagreed by forty percent.
    They are two estimates of the same volatility and must not."""
    feats = compute_features(_gapped())
    agree = feats["parkinson"] / feats["garman_klass"]
    assert 0.8 <= agree <= 1.2, \
        f"the two range estimators read {agree:.2f} apart on one tape — an operand is wrong"
    plain = compute_features(_no_wick())
    assert plain["garman_klass"] / plain["vol20"] < 1.5, \
        "a bar whose range is only its body was reported as intraday divergence"
    assert s53(_no_wick(), plain, {}) is None, "the member votes on an ordinary tape"


def test_garman_klass_looks_at_the_same_window_as_the_vol_it_is_compared_to():
    """The third error was a lookback: Garman-Klass was a 60-bar estimate beside a
    20-bar one, so on any tape whose volatility had fallen over the last quarter
    the ratio measured the two windows, not the two estimators — which is exactly
    what made it fire everywhere the afternoon the units were fixed."""
    base = _no_wick(160, 3)
    outside = _disturb(base, 30, 30, 1.06)   # bars -60..-30: in a 60-bar lookback only
    inside = _disturb(base, 0, 20, 1.06)     # the last 20 bars
    gk_base = compute_features(base)["garman_klass"]
    gk_out = compute_features(outside)["garman_klass"]
    gk_in = compute_features(inside)["garman_klass"]
    assert abs(gk_out / gk_base - 1.0) < 0.02, \
        f"a disturbance outside the window moved the estimate {100*(gk_out/gk_base-1):.0f}%"
    assert gk_in / gk_base > 1.5, \
        "the same disturbance inside the window did not move it at all"
    assert compute_features(outside)["vol20"] == pytest.approx(
        compute_features(base)["vol20"]), "vol20 no longer reads the last 20 bars either"


def test_a_missing_print_is_not_the_most_illiquid_bar_in_the_history():
    """`amihud` divided by `dollar + 1e-9`. A bar with no volume divided by the
    epsilon, and the result out-ranked every real reading on the tape by nine
    orders of magnitude — so the illiquidity veto fired hardest where the data
    was absent, which is where a screen should say nothing."""
    df = _dry_book()
    df.loc[df.index[-20]:, "volume"] = 0.0
    feats = compute_features(df)
    assert feats["amihud"] < 1e12, \
        f"zero-volume bars were read as illiquidity of {feats['amihud']:.2e}"
    # A hole in the tail must not be reported as a cost. The neutral reading is
    # the one the strategy's own gate treats as "no comment".
    assert feats["amihud_relative"] == 1.0, \
        f"absent volume was still given a cost ratio of {feats['amihud_relative']}"
    assert s92(df, feats, {}) is None, "the veto spoke on missing data"

    # A frame too short to hold a 21-bar rolling cost has no median to compare
    # against, and the neutral value is what the gate reads as silence.
    short = compute_features(_tape(20, 0.006, 0.004, 1e6, 3))
    assert short["amihud_relative"] == 1.0, "a frame with no history compared itself to nothing"

    steady = compute_features(_tape(160, 0.006, 0.004, 1e6, 3))
    assert abs(steady["amihud_relative"] - 1.0) < 0.2, \
        f"a book that never thins called itself illiquid at {steady['amihud_relative']:.2f}"


def test_a_book_that_dries_up_gets_the_size_veto():
    """The replacement for the percentile rank. A persistently dry tail ties
    with itself at the top of its own history and so never crossed a strict
    percentile gate; a ratio against the instrument's own median measures how
    far this tape is from its own norm, which is what the veto claims to do."""
    df = _dry_book()
    feats = compute_features(df)
    assert feats["amihud_relative"] > 5, \
        f"a 100x drier book ratioed itself at {feats['amihud_relative']:.1f}"
    vote = s92(df, feats, {})
    assert vote is not None and vote.direction == 0 and vote.strength > 0, \
        "the size veto still never speaks on a dry book"


def test_the_three_silent_members_vote_on_a_normal_tape_software_can_produce():
    """Silence must be a decision, not a KeyError waiting behind a `.get`."""
    from kairos.strategies import run_strategy_library

    sigma = np.r_[np.full(110, 0.002), np.linspace(0.003, 0.016, 40), np.full(10, 0.008)]
    hl = np.r_[np.full(150, 0.004), np.full(10, 0.02)]
    volume = np.r_[np.full(110, 1e6), np.full(50, 1e4)]
    df = _tape(160, sigma, hl, volume, 21)
    feats = compute_features(df)
    assert feats["vol_percentile"] > 0.85 and feats["amihud_relative"] > 5
    ctx = {"regime": "Range", "macro": "Neutral", "intermarket": {},
           "date_today": df["date"].iloc[-1], "patterns": {}, "tags": ()}
    by_name = {}
    for v in run_strategy_library(df, feats, ctx):
        by_name.setdefault(v.name, v)
    assert by_name["Beta-Vol Overlay"].direction < 0
    assert by_name["Garman-Klass vs Close Divergence"].direction < 0
    assert by_name["Amihud Illiquidity Cost"].direction == 0

    # Every feature the new strategies read survives the trip through the
    # prediction journal, which is JSON and refuses a numpy scalar.
    keys = ("vol_percentile", "garman_klass", "parkinson", "amihud", "amihud_relative")
    restored = json.loads(json.dumps({k: feats[k] for k in keys}))
    assert restored["vol_percentile"] == feats["vol_percentile"]
