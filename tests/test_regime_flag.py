"""A flag that is not the type it claims is a flag that reads backwards later.

`detect_regime` answered `high_volatility` with `vol_percentile > 0.7 or
vol20 > 1.6 * vol60`. Both operands are numpy floats, so the comparison yields
`numpy.bool_`, and `or` returns that operand whenever the percentile branch is
false — which is the ordinary case for an ordinary tape, so the leak sat on the
main path, not in an edge.

Nothing noticed, because every reader had a different excuse. The desk survived
by accident: its endpoint round-trips the payload through `_safe`, which coerces
numpy scalars. The CLI dossier writes with `default=str`, so a quiet market was
saved as the *string* `"False"` — and a reader asking `if
regime["high_volatility"]` of that file got True out of a report that had
recorded False, and False out of the live object that was never written. The
same market, two answers, depending on whether it had been through a disk.

`numpy.bool_` is not a subclass of `bool`, which is why this one bit. The
floats in the same dict are `numpy.float64`, which *is* a subclass of `float`,
so they serialize and compare as numbers and are harmless — a distinction the
last test pins down, because "make everything a Python type" would be a claim
this code does not need and this test would not have to enforce.
"""

from __future__ import annotations

import json

import numpy as np

import pandas as pd

from kairos.features import detect_regime


def _frame(closes) -> pd.DataFrame:
    """Real high/low spread: identical bars would make ADX a division by zero
    and the regime label would mean nothing."""
    closes = list(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1000.0] * len(closes)})


def _quiet_walk(seed: int) -> pd.DataFrame:
    """An ordinary tape: choppy early, then settling. Every one of these leaves
    the current 20-bar volatility inside the middle of its own history, so the
    percentile comparison comes out false and `or` falls through to the numpy
    comparison — the branch the flag used to leak from."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.012, 220)
    rets[:70] *= 3.0
    rets[70:180] *= 0.4
    rets[180:] *= 1.3
    return _frame(100.0 * np.exp(np.cumsum(rets)))


def _loud_tail_walk() -> pd.DataFrame:
    """A low percentile *and* a recent window far louder than the stretch it sits
    in: the only way to reach the numpy comparison and come back True."""
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0, 0.004, 210)
    rets[:120] *= 6.0
    rets[120:190] *= 0.15
    rets[190:] *= 2.0
    return _frame(100.0 * np.exp(np.cumsum(rets)))


QUIET_WALKS = [_quiet_walk(s) for s in range(12)]
LOUD_TAIL_WALK = _loud_tail_walk()


def test_the_high_volatility_flag_is_a_python_bool():
    fell_through = 0
    for walk in QUIET_WALKS + [LOUD_TAIL_WALK]:
        reg = detect_regime(walk)
        assert type(reg["high_volatility"]) is bool
        json.dumps(reg)  # no default=, no _safe(): it serializes as itself
        if reg["vol_percentile"] <= 0.7:
            fell_through += 1
    # counted, because twelve walks that never reached the branch would let this
    # test pass while proving nothing about the leak it was written for
    assert fell_through == 13, (
        f"only {fell_through} of 13 walks reached the numpy comparison, so this "
        "suite no longer exercises the branch the flag leaked from")


def test_a_saved_regime_says_the_same_thing_as_the_live_one():
    live = detect_regime(LOUD_TAIL_WALK)
    assert live["vol_percentile"] <= 0.7, (
        "the loud case must come from the fall-through branch — the percentile "
        "branch was never the broken one")
    assert live["high_volatility"] is True
    assert json.loads(json.dumps(live))["high_volatility"] is True

    quiet = detect_regime(QUIET_WALKS[0])
    assert quiet["vol_percentile"] <= 0.7
    assert quiet["high_volatility"] is False
    # this is the case the dossier used to get backwards: "False" is not false
    assert json.loads(json.dumps(quiet, default=str))["high_volatility"] is False


def test_no_value_in_the_regime_dict_needs_a_string_fallback():
    """The dossier's `default=str` is what turned a wrong type into a wrong
    answer. If it is never reached, no value can be saved as a word — and that
    catches the next numpy scalar of this kind without demanding that
    `numpy.float64`, which is a subclass of `float` and always was harmless, be
    coerced for no reason."""
    for walk in (LOUD_TAIL_WALK, QUIET_WALKS[0], QUIET_WALKS[7]):
        reg = detect_regime(walk)
        assert json.dumps(reg, default=str) == json.dumps(reg), (
            f"{reg} — a value the CLI would have written as a string")
