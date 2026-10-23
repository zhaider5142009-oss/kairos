"""The PSX history merge has to disagree with a vendor by day, not by timestamp.

`MarketData.history` answers a PSX read with the union of the vendor's frame and
the local bar store, which is how the store grew real history out of quote-only
days. That one line of `pd.concat` was the only part of the accumulation path
with no test on it, and it turns out the union was wrong for any source that
stamps its bars with a time of day: the store is day-grained by construction, so
`drop_duplicates(subset="date")` compared a `15:30` print against the `00:00` row
already saved and concluded they were two different days.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos import data as D


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    st = D.Store(tmp_path)
    monkeypatch.setattr(D, "STORE", st)
    return st


def bar(date, close: float) -> dict:
    return {"date": pd.Timestamp(date), "open": close, "high": close,
            "low": close, "close": close, "volume": 1000.0}


def frame(*bars: dict) -> pd.DataFrame:
    return pd.DataFrame(list(bars))


def days(df: pd.DataFrame) -> list[str]:
    return [str(d.date()) for d in df["date"]]


# ------------------------------------------------------------------ the grain
def test_one_trading_day_never_becomes_two_bars(tmp_store):
    """An intraday print restates a stored day; it must not add a bar.

    Rolling windows count rows, not dates, so a duplicated day would move every
    indicator read on that instrument by one bar — silently, because the row count
    simply looked like a history that had grown.
    """
    tmp_store.save("SYS", frame(bar("2026-10-20", 150.0), bar("2026-10-21", 160.0)))
    live = frame(bar("2026-10-21 15:30:00", 170.0), bar("2026-10-22", 180.0))

    merged = D.MarketData._merge_with_store("SYS", live)

    assert days(merged) == ["2026-10-20", "2026-10-21", "2026-10-22"]
    assert float(merged[merged["date"].dt.date == pd.Timestamp("2026-10-21").date()]
                 ["close"].iloc[0]) == 170.0


def test_the_saved_file_and_the_reloaded_frame_agree_with_the_merge(tmp_store):
    """What the reader got and what the store kept must be the same history.

    `Store.load` dedupes to the first row of a day, so a merge that left two rows
    for 2026-10-21 in the CSV would hand the *next* reader the older close: the
    frame under a strategy and the frame on the disk would disagree about the same
    day, and nothing on the read path can tell.
    """
    tmp_store.save("SYS", frame(bar("2026-10-20", 150.0), bar("2026-10-21", 160.0)))
    merged = D.MarketData._merge_with_store(
        "SYS", frame(bar("2026-10-21 15:30:00", 170.0), bar("2026-10-22", 180.0)))

    text = (tmp_store.dir / "SYS.CSV").read_text(encoding="utf-8")
    assert text.count("2026-10-21,") == 1, text
    back = tmp_store.load("SYS")
    assert days(back) == days(merged)
    assert list(back["close"]) == list(merged["close"])


def test_the_callers_frame_is_not_restamped_by_the_merge(tmp_store):
    """The merge copies before normalizing — the caller's frame is shared.

    `history` parks the vendor frame in `_hist_cache` and hands the same object to
    the store merge. Restamping that in place would rewrite what every later
    reader of the cache sees, including a non-PSX instrument that shares a Yahoo
    symbol with this one.
    """
    tmp_store.save("SYS", frame(bar("2026-10-20", 150.0)))
    live = frame(bar("2026-10-21 15:30:00", 170.0))
    before = list(live["date"])

    D.MarketData._merge_with_store("SYS", live)

    assert list(live["date"]) == before


# ------------------------------------------------------------------- the rest
def test_the_live_print_wins_the_day_the_store_already_held(tmp_store):
    """The documented rule, at the grain the store actually keeps.

    Both sides arrive at midnight here, which is the shape Yahoo's daily chart
    has, and which is why this half worked while the other half did not.
    """
    tmp_store.save("SYS", frame(bar("2026-10-21", 160.0)))
    merged = D.MarketData._merge_with_store("SYS", frame(bar("2026-10-21", 172.5)))

    assert days(merged) == ["2026-10-21"]
    assert float(merged["close"].iloc[0]) == 172.5


def test_a_store_with_no_history_returns_the_fetch_and_remembers_it(tmp_store):
    live = frame(bar("2026-10-21", 170.0), bar("2026-10-22", 180.0))
    merged = D.MarketData._merge_with_store("NEW", live)

    assert days(merged) == ["2026-10-21", "2026-10-22"]
    assert tmp_store.load("NEW") is not None


def test_a_fetch_that_adds_no_new_day_grows_nothing(tmp_store):
    """A restated day must not lengthen the series, however often it is read.

    The autopilot re-reads every PSX name each cycle; a merge that appended on
    every read would turn a thirty-day history into a different instrument inside
    a week.
    """
    stored = frame(bar("2026-10-20", 150.0), bar("2026-10-21", 160.0))
    tmp_store.save("SYS", stored)

    for _ in range(3):
        merged = D.MarketData._merge_with_store("SYS", stored.copy())
        assert days(merged) == ["2026-10-20", "2026-10-21"]

    assert len(tmp_store.load("SYS")) == 2
