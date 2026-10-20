"""Alignment and invariants in the numeric paths that feed live sizing.

These are characterization tests: each guards a refactor against silently
moving a number. A weight paired with the wrong model, or a Hurst exponent fit
against the wrong lag, does not raise — it changes the conviction that the
entry threshold is compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos import ensemble as ENS
from kairos.features import _hurst
from kairos.models_stat import ModelForecast
from kairos.quantum import optimize_portfolio


# --------------------------------------------------------------------- hurst
def _walk(n: int = 400, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))))


def test_hurst_values_are_pinned_through_the_refactor():
    """Characterization: the lag filter used to recompute every dispersion a
    second time so tau and its lag dropped as pairs. Halving that work must not
    move one exponent."""
    assert _hurst(_walk(400, seed=0)) == pytest.approx(0.4552, abs=1e-3)
    assert _hurst(_walk(400, seed=3)) == pytest.approx(0.5090, abs=1e-3)
    assert _hurst(_walk(400, seed=4)) == pytest.approx(0.5543, abs=1e-3)
    # a smooth price trend saturates the top clip, a bounded oscillation the bottom
    assert _hurst(pd.Series(np.linspace(100, 200, 400))) == pytest.approx(0.95)
    assert _hurst(pd.Series(np.tile(np.linspace(100, 120, 24), 20)
                            + np.linspace(0, 1, 480))) == pytest.approx(0.1791, abs=1e-3)


def test_degenerate_series_return_the_neutral_half():
    assert _hurst(pd.Series([100.0] * 80)) == 0.5        # zero dispersion
    assert _hurst(pd.Series([100.0, 101.0, 102.0])) == 0.5      # too short


def test_periodic_series_matches_a_reference_paired_by_lag():
    """Every lag that is an exact multiple of the period has zero dispersion
    and drops out. tau and its lag must be dropped as a pair."""
    s = pd.Series(np.tile([100.0, 103.0, 101.5], 60))
    prices = np.log(s.values.astype(float))
    lags = np.arange(2, min(21, len(prices) // 4))
    tau = np.array([np.std(prices[lag:] - prices[:-lag]) for lag in lags])
    keep = tau > 0
    assert keep.sum() >= 4 and (~keep).sum() >= 1        # the edge really occurs
    ref = float(np.polyfit(np.log(lags[keep]), np.log(tau[keep]), 1)[0])
    assert _hurst(s) == pytest.approx(float(np.clip(ref, 0.05, 0.95)), abs=1e-9)


def test_hurst_is_deterministic_and_bounded():
    s = _walk(600, seed=3)
    vals = [_hurst(s) for _ in range(3)]
    assert len(set(vals)) == 1
    assert all(0.05 <= v <= 0.95 for v in vals)


# ------------------------------------------------------------------- ensemble
def _fc(model: str, exp: float, conf: float) -> ModelForecast:
    return ModelForecast(model=model, direction=1 if exp > 0 else -1,
                         expected_return=exp, horizon=5, confidence=conf)


def test_blend_pairs_each_return_with_its_own_weight():
    """An unknown regime means a multiplier of 1.0, so the blend is exactly
    confidence-weighted. Shifting the pairing by one forecast would flip it."""
    fcs = [_fc("ARIMA", 0.05, 0.9), _fc("ARIMA", -0.01, 0.1)]
    sig = ENS.fuse("OGDC", fcs, [], "no-such-regime", 5)
    raw = sig.expected_return / sig.details["shrinkage"]
    assert raw == pytest.approx(0.9 * 0.05 + 0.1 * -0.01, abs=2e-4)
    assert raw > 0                                    # follows the confident one


def test_a_zero_confidence_model_cannot_dilute_a_confident_one():
    fcs = [_fc("ARIMA", 0.05, 1.0), _fc("OU-MeanRev", -0.05, 0.0)]
    sig = ENS.fuse("OGDC", fcs, [], "no-such-regime", 5)
    assert sig.expected_return == pytest.approx(
        0.05 * sig.details["shrinkage"], abs=1e-5)
    assert sig.expected_return > 0


def test_no_inputs_is_an_explicit_hold():
    sig = ENS.fuse("OGDC", [], [], "trend", 5)
    assert (sig.score, sig.direction) == (0.0, "HOLD")


# ------------------------------------------------------------------- portfolio
def _rets(n_syms: int = 6, days: int = 180, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"S{i}": rng.normal(0.0005 * i, 0.01, days) for i in range(n_syms)}
    return pd.DataFrame(cols)


def test_selected_symbols_are_all_named_in_the_book():
    rets = _rets()
    sol = optimize_portfolio(rets, k=3)
    assert sol is not None
    # a short selection vector used to drop the tail columns while the reported
    # cardinality still counted them: every weight must name a real instrument
    assert set(sol.weights) <= set(rets.columns)
    assert 0 < len(sol.weights) <= sol.cardinality
    assert sum(sol.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_too_thin_a_history_returns_none_instead_of_garbage():
    assert optimize_portfolio(_rets(n_syms=1)) is None
    assert optimize_portfolio(_rets(days=10)) is None
    assert optimize_portfolio(None) is None
