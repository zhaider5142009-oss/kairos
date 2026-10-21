"""Which run the ledger is allowed to describe.

The track record's strategy sweep section is parsed out of the logs in
`reports/`, so the generator has to pick the right one. It did not: the widest
completed run won, and equal widths fell to whichever log `sorted()` happened to
return first — alphabetically. A refreshed sweep published nothing. The old run
of the same width stayed in the ledger under a rebuild stamp for the day the new
one finished, which is a stale table wearing a fresh date, and that is the one
failure mode this file is supposed to make impossible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import make_track_record as gen  # noqa: E402  (the ledger generator is the thing under test)

ROW = ("{sym}: PROFITABLE ret=+10.0% bench=+2.0% sharpe=1.5 hit=0.600 trades=9 "
       "dd=-5.0%\n")
LOSE = ("{sym}: UNPROFITABLE ret=-10.0% bench=+2.0% sharpe=-1.5 hit=0.400 "
        "trades=9 dd=-5.0%\n")
SKIP = "{sym}: insufficient history (90 bars)\n"

# The older run is named so that it sorts FIRST. That is the case that matters:
# `spectrum_full2.log` (the run that was published) sorts before
# `spectrum_v261.log` (the refresh), which is how the stale table survived being
# re-run. A fixture named the other way round passes under both rules, and a
# guard that passes under the bug it is guarding against is not a guard.
EARLIER, LATER = "spectrum_a_earlier.log", "spectrum_z_later.log"


def _log(dirn: Path, name: str, syms: list[str], mtime: float,
         losing: list[str] = (), skipped: list[str] = ()) -> None:
    body = "".join(ROW.format(sym=s) for s in syms)
    body += "".join(LOSE.format(sym=s) for s in losing)
    body += "".join(SKIP.format(sym=s) for s in skipped)
    # the sweep script's own summary counts what it was ASKED to run, so a skipped
    # instrument lands in its denominator and its loss count; a fixture that
    # printed len(rows) here could not tell a honest headline from that one
    body += "summary: {'instruments': %d, 'profitable': %d, 'marginal': 0, " \
            "'unprofitable': %d}\n" % (len(syms) + len(losing) + len(skipped),
                                       len(syms), len(losing) + len(skipped))
    path = dirn / name
    path.write_text(body, encoding="utf-8")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def reports(tmp_path, monkeypatch):
    d = tmp_path / "reports"
    d.mkdir()
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    return d


def test_a_refreshed_sweep_of_the_same_width_replaces_the_older_one(reports):
    """Newest wins on a tie, whatever the filenames sort to."""
    _log(reports, EARLIER, ["GOLD", "SILVER"], 1_600_000_000)
    _log(reports, LATER, ["GOLD", "WTI"], 1_700_000_000)
    rows, _summary, _run = gen._spectrum()
    assert [r["symbol"] for r in rows] == ["GOLD", "WTI"], rows


def test_a_wider_sweep_still_beats_a_newer_narrower_one(reports):
    """Coverage outranks recency: a partial re-run is not the published table."""
    _log(reports, EARLIER, ["GOLD", "SILVER", "WTI"], 1_600_000_000)
    _log(reports, LATER, ["GOLD"], 1_700_000_000)
    rows, _summary, _run = gen._spectrum()
    assert len(rows) == 3


def test_the_skipped_instruments_are_read_out_of_the_log(reports):
    """A run that gave up on an instrument prints it, and the ledger has to know.

    The generator used to have nothing but the row lines and the summary line to
    work with, and the summary line is the wrong one to trust for a denominator.
    """
    _log(reports, LATER, ["GOLD"], 1_700_000_000, losing=["SILVER"], skipped=["ENGRO"])
    rows, summary, _run = gen._spectrum()
    assert len(rows) == 2, rows
    assert summary["skipped"] == ["ENGRO"], summary


def test_the_run_summary_counts_the_table_not_the_sweeps_own_denominator(reports):
    """The headline an investor reads cannot argue with the rows above it.

    The published ledger printed "5/15 instruments profitable, 10 not" over a
    fourteen-row table, because the sweep script counts the instrument it skipped
    for lack of history in both numbers. A data gap read as a loss, and a reader
    with a calculator could see it.
    """
    _log(reports, LATER, ["GOLD", "WTI"], 1_700_000_000,
         losing=["SILVER", "COPPER"], skipped=["ENGRO"])
    rows, summary, _run = gen._spectrum()
    assert summary["instruments"] == 5 and len(rows) == 4
    line = gen._run_summary(rows, summary)
    assert "2/4 instruments profitable" in line, line
    assert "the other 2 unprofitable" in line, line
    assert "/5" not in line.replace("2/4", ""), line
    assert "ENGRO" in line and "not in the table" in line, line


def test_the_summary_comes_from_the_log_that_was_chosen(reports):
    """The counts printed under the table belong to the run that produced it."""
    _log(reports, EARLIER, ["GOLD", "SILVER"], 1_600_000_000)
    _log(reports, LATER, ["GOLD", "SILVER", "WTI"], 1_700_000_000)
    rows, summary, _run = gen._spectrum()
    assert summary["instruments"] == len(rows) == 3


def test_a_log_the_sweep_did_not_finish_is_not_widest(reports):
    """A crashed run has fewer parseable rows, and must not displace a complete one."""
    _log(reports, EARLIER, ["GOLD", "SILVER", "WTI"], 1_600_000_000)
    (reports / LATER).write_text(ROW.format(sym="GOLD"), encoding="utf-8")
    os.utime(reports / LATER, (1_700_000_000,) * 2)
    rows, _summary, _run = gen._spectrum()
    assert len(rows) == 3
