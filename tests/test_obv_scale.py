"""The volume-balance feature divided by a number that passes through zero.

`obv_sl` is OBV's five-bar change over `|OBV|`. OBV is a cumulative sum of signed
volume, so its level is arbitrary and it crosses zero whenever the signed volume
balances out — and on those bars the denominator is `1e-9` while the numerator is
a whole day's volume. Measured on the cached universe: NIKKEI reached **4.95e16**
against a panel whose next-largest column is **1e2**, so one column of the design
matrix stood fourteen orders of magnitude above the rest. `RidgeClassifier` did
solve it, and said what it cost every time — five `LinAlgWarning`s per battery
run, `rcond` between 2.3e-34 and 3.9e-34, on the member whose probability feeds
the ensemble weight and the size channel.

The same shape sat in `compute_features`, which the live strategies read, where
`s90` turned it into `min(0.6, abs(o) * 4)` — a confidence pinned at its 0.6
ceiling by a number that meant nothing.

What the change is measured against is the volume traded over the same five
bars, so the feature is net signed volume as a share of everything that traded:
bounded in [-1, 1], and the same sign it always had. Measured across 87 cached
series and 47,247 bars, the sign matched the old definition on every one, and
the live row and the ML panel agree on the last bar of all 87.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import LinAlgWarning

import kairos.models_ml as ML
from kairos.features import compute_features, feature_panel


def _wrap(close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.004,
        "low": close * 0.996,
        "close": close,
        "volume": np.full(len(close), 1e6),
    })


def _frame(n: int = 400, seed: int = 0, amp: float = 0.01) -> pd.DataFrame:
    """Alternating up and down bars: the signed-volume balance keeps returning
    to zero, which is the bar the old denominator divided by 1e-9."""
    rng = np.random.default_rng(seed)
    step = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return _wrap(100.0 * np.cumprod(1.0 + step * amp + rng.normal(0.0, 0.002, n)))


def _old_panel(df: pd.DataFrame) -> pd.Series:
    """The definition this module shipped: change over |OBV|."""
    c, v = df["close"], df["volume"].fillna(0)
    obv = (np.sign(c.pct_change().fillna(0.0)) * v).cumsum()
    return obv.diff(5) / (obv.abs() + 1e-9)


# ---------------------------------------------------------------------------
# 1. The fixture is the trap, and the feature no longer falls into it.
# ---------------------------------------------------------------------------
def test_the_fixture_reproduces_the_blow_up():
    """A test that cannot fail against the old code is not a description of it.

    The old expression on this frame must reach the magnitudes that were
    measured on NIKKEI, or the bound tested below is being tested on something
    that was never broken.
    """
    blown = _old_panel(_frame()).abs().max()
    assert blown > 1e6, (
        f"the old denominator only reached {blown:.2e} on this frame, so the "
        f"fixture does not exercise the defect")


def test_the_feature_stays_inside_its_bound():
    """Net signed volume as a share of traded volume cannot leave [-1, 1]."""
    for seed in (0, 1, 2):
        got = feature_panel(_frame(seed=seed))["obv_sl"]
        assert np.isfinite(got).all()
        assert got.abs().max() <= 1.0 + 1e-12, (
            f"seed {seed}: obv_sl reached {got.abs().max():.3e}")


def test_the_sign_is_the_one_the_old_definition_had():
    """Every strategy reads this feature for its direction, so the fix has to
    leave the direction alone — only its scale was wrong."""
    df = _frame()
    old, new = _old_panel(df), feature_panel(df)["obv_sl"]
    both = np.isfinite(old) & np.isfinite(new)
    assert both.any()
    assert (np.sign(old[both]) == np.sign(new[both])).all()


# ---------------------------------------------------------------------------
# 2. What the number means now.
# ---------------------------------------------------------------------------
def _with_tail(step: float, n: int = 120) -> pd.DataFrame:
    df = _frame(n=n)
    close = df["close"].to_numpy().copy()
    close[-6:] = close[-7] * np.cumprod(np.full(6, step))
    return _wrap(close)


def test_five_up_bars_is_the_whole_of_the_volume():
    assert feature_panel(_with_tail(1.01))["obv_sl"].iloc[-1] == pytest.approx(1.0)


def test_five_down_bars_is_minus_the_whole_of_the_volume():
    assert feature_panel(_with_tail(0.99))["obv_sl"].iloc[-1] == pytest.approx(-1.0)


def test_a_balanced_window_is_quiet():
    """Three up bars and two down is not a volume surge in either direction."""
    got = feature_panel(_frame())["obv_sl"]
    assert got.abs().max() < 0.5


def test_a_drought_in_the_window_is_no_signal_not_a_huge_one():
    """Volume fell to zero in the last five bars after trading before it. That
    is no evidence of accumulation, and it must not become a number of order
    1e15 — which is what the old guard did here."""
    df = _frame()
    df.loc[df.index[-5:], "volume"] = 0.0
    assert compute_features(df)["obv_slope"] == 0.0
    assert feature_panel(df)["obv_sl"].iloc[-1] == 0.0


def test_the_live_row_and_the_ml_panel_report_the_same_number():
    """One definition, two code paths. They must not disagree at the last bar."""
    df = _frame()
    row = compute_features(df)["obv_slope"]
    panel = float(feature_panel(df)["obv_sl"].iloc[-1])
    assert row == pytest.approx(panel)


# ---------------------------------------------------------------------------
# 3. The fit that was paying for it.
# ---------------------------------------------------------------------------
def test_the_ridge_solve_is_no_longer_ill_conditioned():
    """The members are fitted on this matrix, and this is the warning they were
    fitted under: five per run on NIKKEI, rcond down at 2.3e-34."""
    df = _frame()
    X = feature_panel(df)
    y_fwd = df["close"].shift(-5) / df["close"] - 1.0
    valid = y_fwd.notna() & X.notna().all(axis=1)
    from sklearn.linear_model import RidgeClassifier
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        RidgeClassifier(alpha=1.0).fit(
            X[valid].to_numpy(), np.where(y_fwd[valid].to_numpy() > 0, 1.0, -1.0))


def test_the_battery_still_votes_and_emits_no_conditioning_warning():
    df = _frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ML.tree_stack_forecast(df, horizon=5)
    conditioning = [w for w in caught if issubclass(w.category, LinAlgWarning)]
    assert conditioning == [], (
        f"the battery still solved an ill-conditioned matrix: "
        f"{[str(w.message)[:60] for w in conditioning]}")
    assert len(out) == 4, [f.model for f in out]
