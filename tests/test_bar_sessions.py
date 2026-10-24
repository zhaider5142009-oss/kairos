"""A bar is a session, and a session's range contains its own open and close.

The PSX accumulators fill the bar store from a snapshot that carries a close and
a percentage change and nothing else. Two things went wrong with that: the range
was synthesised as if every day closed up (on a down day the file held a high
below its own low), and the bar was stamped with the calendar day rather than
the session, so every weekend wrote Friday's print a second and third time.
"""

from __future__ import annotations

import sys

import pandas as pd

from kairos import data as D
from kairos.config import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import repair_bar_store as R  # noqa: E402


def _row(name: str, close: float, change: float, volume: float) -> pd.DataFrame:
    return pd.DataFrame([{"name": name, "close": close, "change": change,
                          "volume": volume}])


def test_a_down_day_from_the_board_is_not_written_upside_down(tmp_path):
    st = D.Store(tmp_path)
    D._accumulate_bars(st, _row("OGDC", 95.0, -5.0, 1000.0), "2026-10-22")
    bar = st.load("OGDC").iloc[-1]
    assert bar["high"] >= max(bar["open"], bar["close"])
    assert bar["low"] <= min(bar["open"], bar["close"])
    # the previous close and today's close are the only prices in hand
    assert float(bar["high"]) == 100.0 and float(bar["low"]) == 95.0


def test_a_weekend_sweep_writes_no_bar_at_all(tmp_path):
    st = D.Store(tmp_path)
    D._accumulate_bars(st, _row("OGDC", 100.0, 0.0, 500.0), "2026-10-24")  # Saturday
    assert st.load("OGDC") is None


def test_a_repeated_print_does_not_start_a_new_session(tmp_path):
    """A holiday, or a pre-open poll, re-serves the last session on a weekday.
    The same close and the same volume are that session, not a new one."""
    st = D.Store(tmp_path)
    D._accumulate_bars(st, _row("OGDC", 100.0, 0.0, 500.0), "2026-10-23")
    D._accumulate_bars(st, _row("OGDC", 100.0, 0.0, 500.0), "2026-10-26")
    df = st.load("OGDC")
    assert len(df) == 1
    assert str(df["date"].iloc[-1].date()) == "2026-10-23"


def test_a_real_flat_session_with_its_own_volume_is_still_written(tmp_path):
    st = D.Store(tmp_path)
    D._accumulate_bars(st, _row("OGDC", 100.0, 0.0, 500.0), "2026-10-23")
    D._accumulate_bars(st, _row("OGDC", 100.0, 0.0, 900.0), "2026-10-26")
    df = st.load("OGDC")
    assert len(df) == 2
    assert str(df["date"].iloc[-1].date()) == "2026-10-26"


def test_the_store_never_files_a_bar_below_its_own_close(tmp_path):
    """The provider's rows are persisted as printed, so a vendor bar whose close
    sits under its own low came back on the next merge. The store widens on the
    way in, which is the only place every writer passes through."""
    st = D.Store(tmp_path)
    bad = pd.DataFrame({"date": pd.to_datetime(["2026-10-23"]), "open": [13.02],
                        "high": [13.02], "low": [12.95], "close": [12.57],
                        "volume": [150.0]})
    st.save("PIM", bad)
    back = st.load("PIM")
    assert D.out_of_range_bars(back) == 0
    assert float(back["low"].iloc[0]) == 12.57


def test_the_repair_tool_drops_fake_sessions_and_widens_inverted_bars():
    psx = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-23", "2026-10-24", "2026-10-25",
                                "2026-10-26"]),
        "open": [394.0, 394.0, 394.0, 400.0],
        "high": [394.0, 386.16, 386.16, 395.0],
        "low": [383.0, 394.0, 394.0, 400.0],
        "close": [386.16, 386.16, 386.16, 390.0],
        "volume": [607031.0, 607031.0, 607031.0, 510000.0]})
    fixed, dropped, widened = R.repair_frame(psx, psx=True)
    assert dropped == 2 and len(fixed) == 2
    assert widened == 1
    assert float(fixed["high"].iloc[-1]) == 400.0
    assert float(fixed["low"].iloc[-1]) == 390.0


def test_the_repair_tool_leaves_a_weekend_the_venue_actually_traded():
    btc = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-23", "2026-10-24", "2026-10-25"]),
        "open": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0], "close": [101.0, 102.0, 102.0],
        "volume": [10.0, 10.0, 10.0]})
    fixed, dropped, _widened = R.repair_frame(btc, psx=False)
    assert dropped == 0
    assert len(fixed) == 3
