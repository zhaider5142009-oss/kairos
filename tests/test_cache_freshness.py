"""The two panels shared by a whole cycle still have to expire.

`intermarket()` (ten sockets) and the cross-sectional factor row (one scan plus
pandas) are deliberately not refetched per prediction: every symbol in a cycle
should be scored against the *same* macro state. The caching was right; the
expiry was missing. Nothing in the autopilot ever asked for a refresh, so a
daemon that stayed up for a day sized every position from the panel and factor
rows it booted with, while the desk showed an `as_of` date that looked current.

These pin the clock, and pin the failure shape: a refresh that raises keeps the
last good panel for the next attempt rather than quietly serving it as new.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos import engine as E


def _frame(n: int, last: float) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": days, "open": last, "high": last, "low": last,
                         "close": [last] * n, "volume": [1.0] * n})


class _Md:
    """Counts the two lanes separately: sockets vs the exchange snapshot."""

    def __init__(self) -> None:
        self.history_calls = 0
        self.snapshot_calls = 0
        self.fail = False

    def history(self, instrument):
        self.history_calls += 1
        if self.fail:
            raise OSError(f"{instrument.key} is down")
        return _frame(60, 100.0)

    def psx_snapshot(self):
        self.snapshot_calls += 1
        return pd.DataFrame({"name": ["OGDC"], "close": [100.0]})


# ------------------------------------------------------------------ macro panel
def test_the_macro_panel_is_fetched_once_per_cycle_and_expires_on_its_clock():
    md = _Md()
    eng = E.PredictionEngine(md=md)
    eng.intermarket()
    per_fetch = md.history_calls
    assert per_fetch > 0
    eng.intermarket()
    assert md.history_calls == per_fetch, "twelve symbols paid for twelve panels"
    eng.intermarket(refresh_after_s=0.0)
    assert md.history_calls == per_fetch * 2, "the clock never expired the panel"


def test_a_failed_refresh_keeps_the_last_good_panel_and_raises():
    md = _Md()
    eng = E.PredictionEngine(md=md)
    good = eng.intermarket()
    md.fail = True
    with pytest.raises(OSError):
        eng.intermarket(refresh_after_s=0.0)
    # the failure is visible (the symbol's prediction fails) and the panel is
    # kept for the next attempt — it is never reported as a fresh one
    assert eng._intermarket is good
    md.fail = False
    assert set(eng.intermarket(refresh_after_s=0.0)) == set(good)


# ---------------------------------------------------------- cross-section row
def _fake_factors(monkeypatch):
    calls: list[pd.DataFrame] = []

    def fake(snap):
        calls.append(snap)
        return pd.DataFrame({"name": ["OGDC"], "composite_score": [0.5]})

    monkeypatch.setattr(E, "cross_sectional_factors", fake)
    return calls


def test_the_factor_row_expires_on_the_snapshot_clock(monkeypatch):
    calls = _fake_factors(monkeypatch)
    md = _Md()
    eng = E.PredictionEngine(md=md)
    inst = type("I", (), {"market": "PSX", "key": "OGDC"})()

    row = eng.factor_row(inst)
    assert row is not None and float(row["composite_score"]) == 0.5
    assert md.snapshot_calls == 1 and len(calls) == 1

    eng.factor_row(inst)
    assert md.snapshot_calls == 1, "the row was recomputed within its own clock"

    eng.factor_row(inst, refresh_after_s=0.0)
    assert md.snapshot_calls == 2, "the clock never expired the factor row"


def test_a_non_psx_instrument_never_asks_for_the_cross_section(monkeypatch):
    calls = _fake_factors(monkeypatch)
    md = _Md()
    eng = E.PredictionEngine(md=md)
    inst = type("I", (), {"market": "PMEX", "key": "GOLD"})()
    assert eng.factor_row(inst) is None
    assert md.snapshot_calls == 0 and calls == []


# ------------------------------------------------------------- bridge cache
def test_the_bridge_cache_is_content_addressed_and_bounded(monkeypatch):
    """Same series → one subprocess; a new close every cycle → a new key every
    cycle. Unbounded, that dict grows for the life of the daemon."""
    from kairos.wsl_bridge import WslBridge

    bridge = WslBridge()
    calls: list[dict] = []

    def fake_call(payload):
        calls.append(payload)
        return {"ok": True, "point": 100.0}

    monkeypatch.setattr(bridge, "_call", fake_call)
    bridge.forecast("chronos", [100.0, 101.0], 5)
    bridge.forecast("chronos", [100.0, 101.0], 5)
    assert len(calls) == 1, "the same bars were fitted twice"

    for i in range(WslBridge.CACHE_MAX + 20):
        bridge.forecast("chronos", [100.0 + i, 101.0 + i], 5)
    assert len(bridge._cache) <= WslBridge.CACHE_MAX


def test_a_scan_and_a_prediction_share_one_snapshot(monkeypatch):
    """`scan()` writes the same row `factor_row` reads; it must stamp the same
    clock, or the next prediction immediately scans the exchange again."""
    calls = _fake_factors(monkeypatch)
    md = _Md()
    eng = E.PredictionEngine(md=md)
    out = eng.scan()
    assert out["ok"] is True and len(calls) == 1
    inst = type("I", (), {"market": "PSX", "key": "OGDC"})()
    assert eng.factor_row(inst) is not None
    assert md.snapshot_calls == 1 and len(calls) == 1
