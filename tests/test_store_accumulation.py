"""The whole-exchange bar sweep must not be on the read path.

`psx_snapshot` is what the factor row, the desk and every PSX prediction read.
It used to spend most of its time restating 484 CSVs that already said exactly
what the snapshot says, and each of those reads came back as "the desk took six
seconds to answer".

Now: unchanged bars are not rewritten, and the sweep that does write runs on a
worker. Both halves are checked here without touching the network.
"""

from __future__ import annotations

import os
import time
from datetime import datetime as _real_datetime

import pandas as pd
import pytest

from kairos import data as D


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    st = D.Store(tmp_path)
    monkeypatch.setattr(D, "STORE", st)
    return st


@pytest.fixture
def weekday_clock(monkeypatch):
    """The sweep refuses a weekend day, so a test that expects a written bar must
    not inherit whichever weekday the suite happens to run on — the suite is green
    Monday to Friday and, before this, silently wrote nothing on a Saturday."""
    class Clock:
        instant = "2026-10-21 09:30:00"

        @classmethod
        def now(cls, *a, **k):
            return _real_datetime.strptime(cls.instant, "%Y-%m-%d %H:%M:%S")

    monkeypatch.setattr(D, "datetime", Clock)
    return Clock


# ------------------------------------------------------------------ upsert_row
def test_an_unchanged_bar_does_not_rewrite_the_file(tmp_store):
    tmp_store.upsert_row("LUCK", "2026-10-21", 100.0, 104.0, 100.0, 104.0, 9000.0)
    p = tmp_store._path("LUCK")
    before = (p.stat().st_mtime_ns, p.read_text(encoding="utf-8"))
    again = tmp_store.upsert_row("LUCK", "2026-10-21", 100.0, 104.0, 100.0, 104.0, 9000.0)
    assert (p.stat().st_mtime_ns, p.read_text(encoding="utf-8")) == before
    assert len(again) == 1
    assert float(again["close"].iloc[-1]) == 104.0


def test_a_moved_bar_is_rewritten_not_skipped(tmp_store):
    tmp_store.upsert_row("LUCK", "2026-10-21", 100.0, 104.0, 100.0, 104.0, 9000.0)
    out = tmp_store.upsert_row("LUCK", "2026-10-21", 100.0, 105.0, 100.0, 105.0, 9500.0)
    assert float(out["close"].iloc[-1]) == 105.0
    assert len(out) == 1                       # restated, not duplicated


def test_a_newer_bar_still_appends(tmp_store):
    tmp_store.upsert_row("LUCK", "2026-10-20", 99.0, 100.0, 99.0, 100.0, 8000.0)
    out = tmp_store.upsert_row("LUCK", "2026-10-21", 100.0, 104.0, 100.0, 104.0, 9000.0)
    assert [str(d.date()) for d in out["date"]] == ["2026-10-20", "2026-10-21"]


# ---------------------------------------------------------------- psx_snapshot
def _scan(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"S{i}" for i in range(rows)],
                         "close": [100.0 + i for i in range(rows)],
                         "change": [1.0] * rows,
                         "volume": [1000.0 + i for i in range(rows)]})


class _Md(D.MarketData):
    """A MarketData whose TradingView scan is already in hand."""

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__()
        self._frame = frame
        self.scans = 0

    def _scan_psx(self, *a, **k) -> pd.DataFrame:
        self.scans += 1
        return self._frame


def _drain_sweep(timeout: float = 20.0) -> None:
    # the sweep is guarded by a semaphore; taking it means the previous one
    # finished, which is the only honest way to wait for it from a test.
    assert D._bar_sweep.acquire(timeout=timeout), "the bar sweep never finished"
    D._bar_sweep.release()


def test_psx_snapshot_answers_before_the_sweep_is_done(tmp_store, monkeypatch,
                                                      weekday_clock):
    md = _Md(_scan(300))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    t0 = time.perf_counter()
    snap = md.psx_snapshot()
    waited = time.perf_counter() - t0
    assert len(snap) == 300
    assert waited < 1.0, f"snapshot blocked for {waited:.1f}s on 300 CSV writes"
    _drain_sweep()
    assert tmp_store._path("S7").exists()
    bar = tmp_store.load("S7")
    assert float(bar["close"].iloc[-1]) == 107.0
    assert str(bar["date"].iloc[-1].date()) == "2026-10-21"


def test_a_refresh_during_a_sweep_skips_instead_of_queuing(tmp_store, monkeypatch,
                                                           weekday_clock):
    md = _Md(_scan(50))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    _drain_sweep()
    monkeypatch.setattr(md, "_tv_fetched_at", 0.0)   # force a refresh
    assert len(tmp_store.load("S9")) == 1

    # hold the sweep slot the way a real in-flight sweep holds it: a second
    # refresh must not block the reader and must not double-write either
    assert D._bar_sweep.acquire(blocking=False)
    try:
        monkeypatch.setattr(md, "_tv_fetched_at", 0.0)
        t0 = time.perf_counter()
        md.psx_snapshot()
        md.psx_snapshot()
        assert time.perf_counter() - t0 < 0.5, "the reader waited on the sweep"
    finally:
        D._bar_sweep.release()


def test_a_bad_row_in_the_scan_does_not_stop_the_sweep(tmp_store, monkeypatch,
                                                      weekday_clock):
    frame = _scan(5)
    frame.loc[2, "close"] = None
    frame.loc[3, "name"] = ""
    md = _Md(frame)
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    _drain_sweep()
    # sorted: glob order is the filesystem's, and ext4 gave these back in a
    # different order than NTFS did — the claim is which files landed, not when
    assert sorted(p.name for p in tmp_store.dir.glob("*.csv")) == ["S0.csv", "S1.csv", "S4.csv"]


def test_the_sweep_does_not_raise_out_of_a_worker(tmp_store, monkeypatch):
    # a store directory that cannot be written must cost the reader nothing
    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(D.STORE, "upsert_row", explode)
    md = _Md(_scan(8))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    _drain_sweep()


def test_a_fresh_instance_reuses_the_cached_snapshot(tmp_store, monkeypatch):
    md = _Md(_scan(3))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    md.psx_snapshot()
    assert md.scans == 1
    _drain_sweep()


def test_store_dir_files_are_not_left_half_written_by_the_sweep(tmp_store, monkeypatch,
                                                               weekday_clock):
    md = _Md(_scan(40))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    _drain_sweep()
    leftovers = [f for f in os.listdir(tmp_store.dir) if f.endswith(".tmp")]
    assert leftovers == []


def test_a_sweep_that_runs_past_midnight_stamps_the_day_it_read(tmp_store, monkeypatch,
                                                                weekday_clock):
    md = _Md(_scan(4))
    monkeypatch.setattr(md.tv, "scan_psx", md._scan_psx)
    md.psx_snapshot()
    weekday_clock.instant = "2026-10-22 00:01:00"   # the worker is slow, not the read
    _drain_sweep()
    assert str(tmp_store.load("S1")["date"].iloc[-1].date()) == "2026-10-21"
