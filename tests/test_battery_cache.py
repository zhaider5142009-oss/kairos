"""Offline test: the model battery must be refit only when its input bars change.

This is the single biggest cost on a small machine — one refit is ~45s — so the
autopilot cycle is only survivable if a warm bar set is never recomputed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.engine import ENGINE
from kairos.universe import resolve


def _frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes, "date": idx.strftime("%Y-%m-%d")}, index=idx)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the three batteries with counters; no data, no fitting."""
    counts = {"stat": 0, "ml": 0, "neural": 0}
    monkeypatch.setattr("kairos.engine.run_statistical_battery",
                        lambda close, h: counts.__setitem__("stat", counts["stat"] + 1) or [])
    monkeypatch.setattr("kairos.engine.tree_stack_forecast",
                        lambda df, h: counts.__setitem__("ml", counts["ml"] + 1) or [])
    monkeypatch.setattr("kairos.engine.neural_battery",
                        lambda df, h: counts.__setitem__("neural", counts["neural"] + 1) or [])
    ENGINE._battery_cache.clear()
    yield counts
    ENGINE._battery_cache.clear()


def test_same_bars_refit_once(stubbed):
    inst, df = resolve("GOLD"), _frame([100.0, 101.0, 102.0])
    first = ENGINE._battery(inst, df, 5)
    second = ENGINE._battery(inst, df, 5)
    assert stubbed == {"stat": 1, "ml": 1, "neural": 1}
    assert first == second


def test_a_new_bar_refits(stubbed):
    inst = resolve("GOLD")
    ENGINE._battery(inst, _frame([100.0, 101.0, 102.0]), 5)
    ENGINE._battery(inst, _frame([100.0, 101.0, 102.0, 103.0]), 5)
    assert stubbed["stat"] == 2


def test_a_moved_last_close_refits(stubbed):
    """The forecasts are anchored on the last close, so a moved close must never
    reuse the old levels — that would price a trade off a stale anchor."""
    inst = resolve("GOLD")
    ENGINE._battery(inst, _frame([100.0, 101.0, 102.0]), 5)
    ENGINE._battery(inst, _frame([100.0, 101.0, 109.0]), 5)
    assert stubbed["stat"] == 2


def test_horizon_and_symbol_are_separate_entries(stubbed):
    gold, wti = resolve("GOLD"), resolve("WTI")
    df = _frame([100.0, 101.0, 102.0])
    ENGINE._battery(gold, df, 5)
    ENGINE._battery(gold, df, 21)
    ENGINE._battery(wti, df, 5)
    assert len(ENGINE._battery_cache) == 3


def test_cache_is_bounded(stubbed):
    inst = resolve("GOLD")
    for i in range(80):
        ENGINE._battery(inst, _frame([float(i), float(i) + 1, float(i) + 2]), 5)
    assert len(ENGINE._battery_cache) <= 64
