"""The walk-forward window the ML battery scores itself on, and what it did.

`_walk_forward_splits` computes `test_size` — 83 bars on a 500-bar series — and
then handed back a test block starting at `end - embargo`, so the window the
battery scored was 5 bars long and the *gap* was 83. Every number that came out
of it was a number about the wrong thing.

It stayed invisible because `_hit_rate` answered a hard-coded `0.5` for any
window under eight bars. A floor that returns chance does not report "unknown":
`edge = abs(hit - 0.5) * 2` reads it as exactly zero skill, and
`conf = 0.25 + edge * 0.9` then prints the same 0.25 for every ML member on every
symbol, under a detail line that said "walk-forward hit-rate=50.0%" as though a
measurement had been taken. Measured against the module this replaces: every test
window it produced was five bars long, and of the hit rates it printed across the
cached instruments — 40 of them — every one was exactly 0.500.

The third line is what made this worth more than a wrong constant. With real
windows being scored, `abs()` hands a model that is consistently *wrong* the same
confidence as one that is consistently right — and the vote that confidence
rides on still points wherever the model says it does.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import kairos.models_ml as ML

_RATE = re.compile(r"walk-forward hit-rate=([\d.]+)% on (\d+) folds")


def _frame(seed: int = 0, n: int = 340) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0035, 0.006, n))
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.004,
        "low": close * 0.996,
        "close": close,
        "volume": np.full(n, 1e6),
    })


def _window_len(split) -> int:
    """How many bars the fold hands the scorer."""
    return split[3] - split[2]


def _gap_len(split) -> int:
    """How many bars sit between the end of training and the start of testing."""
    return split[2] - split[1]


# ---------------------------------------------------------------------------
# 1. The window is the thing the function says it is.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [300, 500, 725, 1200])
def test_the_test_window_is_the_size_the_function_computes(n: int):
    """`test_size` is not a decoration; it is the length of the block scored."""
    expected = max(15, n // 6)
    splits = list(ML._walk_forward_splits(n))
    assert splits, f"a {n}-bar history produced no folds"
    for split in splits:
        assert _window_len(split) == expected, (
            f"n={n}: the split yielded a {_window_len(split)}-bar test window, "
            f"but the function computed test_size={expected}")


@pytest.mark.parametrize("n", [300, 500, 1200])
def test_the_gap_between_train_and_test_is_the_embargo(n: int):
    """The embargo is a dead zone, and it was the window's turn to be one."""
    splits = list(ML._walk_forward_splits(n, embargo=5))
    assert splits
    for split in splits:
        assert _gap_len(split) == 5, (
            f"n={n}: the train/test gap came out {_gap_len(split)}, not the embargo")


@pytest.mark.parametrize("n", [40, 60, 90])
def test_short_history_still_yields_only_usable_folds(n: int):
    """The 30-bar floor still refuses a fold that cannot hold both a train set
    and a test block of the size it asked for."""
    for split in ML._walk_forward_splits(n):
        assert split[1] >= 30, f"n={n}: trained on {split[1]} bars, under the floor"
        assert _window_len(split) == max(15, n // 6)


# ---------------------------------------------------------------------------
# 2. The embargo is wired to the horizon whose labels it has to exclude.
# ---------------------------------------------------------------------------
def test_the_gap_is_widened_to_the_horizon_being_predicted():
    """A horizon-10 label reaches ten bars past the training end.

    The dead zone has to be at least as long as the horizon or the fold scores
    the model on outcomes it trained on. `embargo` defaulted to 5 and the call
    site never passed the horizon it was forecasting, so the two only agreed by
    the horizon also happening to be 5.
    """
    seen: list[int] = []
    real = ML._walk_forward_splits

    def spy(n: int, n_splits: int = 4, embargo: int = 5):
        seen.append(embargo)
        yield from real(n, n_splits, embargo)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ML, "_walk_forward_splits", spy)
    try:
        ML.tree_stack_forecast(_frame(), horizon=10)
    finally:
        monkey.undo()
    assert seen, "the battery never asked for a split"
    assert all(e == 10 for e in seen), f"the gap ignored the horizon: {seen}"


# ---------------------------------------------------------------------------
# 3. A window too short to score yields nothing, not a half.
# ---------------------------------------------------------------------------
def test_an_unmeasurable_window_is_not_reported_as_chance():
    assert ML._hit_rate(np.array([0.01, -0.02, 0.03, 0.01, -0.01]),
                        np.array([1.0, -1.0, 1.0, 1.0, 1.0])) is None


def test_a_measurable_window_still_yields_a_rate():
    y = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, -0.01])
    p = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert ML._hit_rate(y, p) == pytest.approx(float(np.mean(np.sign(y) == p)))


def test_a_fold_that_cannot_be_scoreed_produces_no_vote():
    """Shrink the window back to the five bars the defect handed the scorer and
    the battery must go silent, not paint 0.25 confidence on every member."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ML, "_walk_forward_splits",
                   lambda n, n_splits=4, embargo=5: [(0, 200, 200, 205)])
    try:
        out = ML.tree_stack_forecast(_frame(), horizon=5)
    finally:
        monkey.undo()
    assert out == [], f"unscoreable folds still produced votes: {[f.model for f in out]}"


def test_the_scored_fold_count_is_printed_with_the_rate():
    """The row has to say how many windows the rate came from, the way the ARIMA
    row says how much of its grid finished."""
    out = ML.tree_stack_forecast(_frame(), horizon=5)
    assert out
    for f in out:
        m = _RATE.search(f.detail)
        assert m, f"the ML row no longer prints the rate it measured: {f.detail!r}"
        assert int(m.group(2)) == 4, f"only {m.group(2)} folds reached the rate"


# ---------------------------------------------------------------------------
# 4. Below chance earns nothing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("rate", "pct", "want_conf"),
                         [(0.0, "0.0", 0.25), (0.413, "41.3", 0.25), (1.0, "100.0", 0.90)])
def test_the_credit_rule_only_points_one_way(rate: float, pct: str, want_conf: float):
    """`abs(hit - 0.5)` paid a perfect contrarian as much as a perfect forecaster.

    The fold is scored on the sign the model actually voted, so a member that is
    consistently wrong is not a member that is consistently right about the
    opposite — its vote still points where it says it points. With real windows
    being scored this is reachable on live history: measured across the cached
    instruments, the 40 member-means spanned 41.3% to 58.1% and 17 of them sat
    below chance. The 41.3% one — a random forest right far less often than a
    coin flip — used to buy `abs(0.413 - 0.5) * 2 = 0.174` of edge and therefore
    confidence 0.407, on a vote pointing the wrong way. The rate is injected here
    because this test is about the credit rule, not the scorer: `_hit_rate`'s own
    contract is the two tests above, and the battery's real members are the ones
    being scored.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ML, "_hit_rate", lambda y_true, y_pred: rate)
    try:
        out = ML.tree_stack_forecast(_frame(), horizon=5)
    finally:
        monkey.undo()
    assert len(out) == 4, [f.model for f in out]
    for f in out:
        m = _RATE.search(f.detail)
        assert m and m.group(1) == pct, f.detail
        assert f.confidence == pytest.approx(want_conf), (
            f"{f.model}: a member that was right {pct}% of the windows it was "
            f"scored on carries confidence {f.confidence}")
