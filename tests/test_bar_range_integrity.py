"""Bars that contradict their own range, and the one door that repairs them.

A provider series for a daily FX print carries a close from a later fix than the
range it sits in. The row parses, so nothing raises — but it describes a bar no
market could have traded, and every range-based reading of it is a statement
about that impossible bar. Found by asking why a volatility estimator reported
zero intraday variance on a currency that had just moved sixteen percent
annualized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.data import MarketData, coerce_bar_range, out_of_range_bars
from kairos.features import compute_features


def _peg_tape(rows: int = 140, seed: int = 9) -> pd.DataFrame:
    """A pegged-currency daily series: the range prints flat around the open and
    the close lands outside it, which is what USDPKR does on most of its bars."""
    rng = np.random.default_rng(seed)
    open_ = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.0005, rows))
    close = open_ * (1.0 + rng.normal(0.002, 0.004, rows))
    high = open_ * 1.006
    low = open_ * 0.994
    bad = close > high                      # the fix lands above the day's high
    close[bad] = high[bad] * 1.02
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.full(rows, 1e5)})


def test_the_fixture_is_the_defect_and_not_a_caricature_of_it():
    """Before asserting a repair, assert the broken shape really occurs: a close
    outside its own range makes the Garman-Klass window mean negative, and a
    negative variance clamps to zero volatility."""
    df = _peg_tape()
    assert out_of_range_bars(df) > 0
    broken = compute_features(df)
    assert broken["garman_klass"] == 0.0, "the estimator was not driven negative"
    assert broken["parkinson"] > 0.05, \
        "and it went negative on a series that was genuinely still"


def test_the_repair_makes_the_same_bars_produce_a_volatility_rather_than_none():
    df = coerce_bar_range(_peg_tape())
    f = compute_features(df)
    assert out_of_range_bars(df) == 0
    assert f["garman_klass"] > 0.0, "the estimator is still silent after the repair"
    # Range estimators describe one market, so on repaired bars they should
    # disagree by a factor of about one, not by the whole of one estimate.
    ratio = f["parkinson"] / f["garman_klass"]
    assert 0.5 < ratio < 2.0, f"the pair still disagrees by {ratio:.1f}x"


def test_the_repair_widens_and_never_moves_a_price_that_was_already_inside():
    df = _peg_tape()
    out = coerce_bar_range(df)
    pd.testing.assert_series_equal(out["open"], df["open"])
    pd.testing.assert_series_equal(out["close"], df["close"])
    # The closes the paper record and the sweep were measured on are untouched:
    # widening can only raise a high or lower a low, never the reverse.
    assert (out["high"] >= df["high"]).all()
    assert (out["low"] <= df["low"]).all()
    assert ((out["high"] - out["low"]) >= (df["high"] - df["low"])).all()
    inside = (df["open"].between(df["low"], df["high"])
              & df["close"].between(df["low"], df["high"]))
    pd.testing.assert_series_equal(out.loc[inside, "high"], df.loc[inside, "high"])
    pd.testing.assert_series_equal(out.loc[inside, "low"], df.loc[inside, "low"])


def test_history_is_the_door_every_consumer_uses_and_it_is_repaired_there():
    """`MarketData.history()` is the only way the engine, the autopilot, the
    desk, the feedback loop and the sweep get bars, so the repair belongs there
    rather than repeated at ten call sites — and a fix at the store would leave
    the live fetch unrepaired."""
    md = MarketData.__new__(MarketData)      # no __init__: no config, no network
    raw = _peg_tape()
    md._history = lambda instrument, days=None, use_cache=True: raw
    out = md.history(None)
    assert out is not raw, "history() handed the malformed frame straight through"
    assert out_of_range_bars(out) == 0
    assert out_of_range_bars(raw) > 0, "the fixture no longer demonstrates the defect"


def test_a_bar_with_an_inverted_range_is_still_made_self_consistent():
    """The same door has to catch a row where low exceeds high outright — Yahoo
    emits one on a half-holiday print, and `min`/`max` over the three prices is
    the only place it can be untangled."""
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3),
                       "open": [100.0, 101.0, 102.0],
                       "high": [100.5, 100.0, 103.0],
                       "low": [104.0, 100.0, 102.5],
                       "close": [100.2, 101.5, 102.8],
                       "volume": [1.0, 1.0, 1.0]})
    out = coerce_bar_range(df)
    assert (out["high"] >= out["low"]).all()
    assert out_of_range_bars(out) == 0
    assert out["high"].iloc[0] == pytest.approx(104.0)


def test_a_frame_that_is_not_bars_at_all_passes_through_unchanged():
    empty = pd.DataFrame()
    assert coerce_bar_range(empty) is empty
    no_ohlc = pd.DataFrame({"close": [1.0, 2.0]})
    assert coerce_bar_range(no_ohlc) is no_ohlc
    assert out_of_range_bars(None) == 0
    assert out_of_range_bars(no_ohlc) == 0
