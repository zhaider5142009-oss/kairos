"""A value that goes to disk must come back as the same *type* it went in as.

`json.dumps(..., default=str)` sat on every write path in this system — the
autopilot state, the audit journal, the news desk cache and the CLI reports —
and it will convert *anything*. A `numpy.bool_(False)` became the string
`"False"`, which is truthy, so a quiet market was stored as a loud one (the
incident recorded at `features.py:462` and `macro.py:82`); a `numpy.float64`
equity became `"1000000"`, which the desk's coercing `_num` read back as a
number while anything that forgot to coerce first did string arithmetic on it.
The serializer now converts the types this code actually produces and refuses
the rest, so a wrong type is an error the caller reports instead of a wrong
fact that lands on disk. Two of these files are also the loop's memory of its
own risk state, so the other half of this contract is that a state file which
cannot be read no longer has the same shape as a first run.
"""

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pytest

import kairos.execution as E
from kairos.autopilot import Autopilot
from kairos.config import CFG
from kairos.store import dumps_strict


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Keep every write, read and kill-switch probe inside this test's directory."""
    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    monkeypatch.setattr(E.KILL, "path", tmp_path / "KILL_SWITCH")
    monkeypatch.setattr(E.JOURNAL, "path", tmp_path / "audit_journal.jsonl")
    return tmp_path


def _journal_events(monkeypatch) -> list:
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(E.JOURNAL, "log",
                         lambda event, payload=None: seen.append((event, payload or {})))
    return seen


# --------------------------------------------------------------- the serializer
def test_a_numpy_scalar_is_still_the_same_type_when_it_comes_back():
    payload = {"high_volatility": np.bool_(False), "equity": np.float64(1_000_000.0),
               "qty": np.int64(3), "bar": np.array([1.0, 2.0])}
    got = json.loads(dumps_strict(payload))
    assert got["high_volatility"] is False
    assert isinstance(got["equity"], float) and got["equity"] == 1_000_000.0
    assert isinstance(got["qty"], int)
    assert got["bar"] == [1.0, 2.0]


def test_the_string_false_is_the_exact_thing_default_str_used_to_write():
    """Not a hypothetical: this is the record the old serializer produced, and a
    reader that trusted it called a quiet market loud."""
    old = json.loads(json.dumps({"high_volatility": np.bool_(False)}, default=str))
    assert old["high_volatility"] == "False"
    assert bool(old["high_volatility"]) is True


def test_a_type_with_no_conversion_names_itself_instead_of_becoming_a_string():
    with pytest.raises(TypeError) as exc:
        dumps_strict({"peers": {6, 7}})
    assert "set" in str(exc.value)


def test_the_types_a_market_record_actually_holds_survive_the_trip():
    payload = {"ts": dt.datetime(2026, 10, 23, 12, 30), "day": dt.date(2026, 10, 23),
               "span": dt.timedelta(seconds=90), "path": Path("data/book.json")}
    got = json.loads(dumps_strict(payload))
    assert got["ts"] == "2026-10-23T12:30:00"
    assert got["day"] == "2026-10-23"
    assert got["span"] == 90.0
    assert Path(got["path"]).name == "book.json"


# ----------------------------------------------------------------- audit journal
def test_a_journal_row_does_not_re_type_what_it_records(data):
    assert E.JOURNAL.log("TEST_ROW", {"realized_pnl": np.float64(-13.05),
                                      "flat": np.bool_(False)})
    row = json.loads((data / "audit_journal.jsonl").read_text(encoding="utf-8"))
    assert row["realized_pnl"] == -13.05
    assert row["flat"] is False


def test_a_row_that_cannot_be_written_is_reported_and_leaves_no_half_record(data):
    assert not E.JOURNAL.log("TEST_ROW", {"blob": {6, 7}})
    assert "not serializable" in (E.JOURNAL.last_error or "")
    assert not (data / "audit_journal.jsonl").exists()


# -------------------------------------------------------------- autopilot state
def test_a_saved_state_stamps_the_write_and_clears_the_fault(tmp_path):
    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a.st["cycles"] = 42
    a.st["state_error"] = "an earlier fault"
    a._save()
    assert a.st["state_error"] is None
    assert isinstance(a.st["state_saved"], float)
    on_disk = json.loads(a.state_path.read_text(encoding="utf-8"))
    assert on_disk["cycles"] == 42 and on_disk["state_error"] is None


def test_a_state_that_cannot_be_serialized_says_why_in_status(data, monkeypatch):
    """`except Exception: pass` used to sit in `_save`, so a state file the loop
    could no longer write left the desk reporting hours-old cycles, peak and
    losing streak as live fact."""
    a = Autopilot()
    a.state_path = data / "autopilot.json"
    events = _journal_events(monkeypatch)
    a.st["macro"] = {6, 7}
    a._save()
    assert "serialize" in a.status()["state_error"]
    assert not (data / "autopilot.json").exists(), "a refused write leaves no half state"
    assert [e[0] for e in events] == ["AUTOPILOT_STATE_SERIALIZE_FAILED"]


def test_one_failing_state_write_journals_one_row_not_one_per_cycle(data, monkeypatch):
    a = Autopilot()
    a.state_path = data / "autopilot.json"
    a.st["macro"] = {6, 7}
    events = _journal_events(monkeypatch)
    for _ in range(5):
        a._save()
    assert len(events) == 1, "a full disk fails every cycle; 288 rows a day bury the event"


def test_a_destination_that_refuses_the_write_is_reported_as_a_write_failure(tmp_path):
    a = Autopilot()
    a.state_path = tmp_path          # a directory: the replace can never land
    a._save()
    assert "write" in a.st["state_error"]


def test_a_torn_state_file_announces_instead_of_looking_like_a_first_run(tmp_path):
    p = tmp_path / "autopilot.json"
    p.write_text('{"cycles": 379, "equity_peak": 1000000', encoding="utf-8")
    a = Autopilot()
    a.state_path = p
    st = a._load()
    assert st["cycles"] == 0, "it does run on defaults — that is the honest part"
    assert st["state_error"] and "unreadable" in st["state_error"]


def test_a_state_file_holding_the_wrong_shape_is_a_fault_too(tmp_path):
    p = tmp_path / "autopilot.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    a = Autopilot()
    a.state_path = p
    assert "unreadable" in a._load()["state_error"]


def test_a_missing_state_file_is_a_first_run_and_not_a_fault(tmp_path):
    a = Autopilot()
    a.state_path = tmp_path / "nothing_here.json"
    assert a._load()["state_error"] is None


# ---------------------------------------------------------------- news desk cache
def test_a_restored_reading_keeps_its_score_as_a_number(tmp_path):
    """`predict` reads the desk off this cache, so the file is the desk's answer
    for every process that did not do the fetching."""
    from kairos.news import NewsDesk, ToneReading

    p = tmp_path / "news_desk.json"
    desk = NewsDesk(p)
    desk._cache["HBL"] = (time.time(),
                          ToneReading(key="HBL", score=np.float64(0.42), matching=3))
    desk._save()
    assert desk.save_error is None
    reading = NewsDesk(p)._cache["HBL"][1]
    assert isinstance(reading.score, float)
    assert reading.available is True


# --------------------------------------------------------------------- the CLI
def test_a_report_the_serializer_refuses_fails_the_command(tmp_path):
    from kairos.cli import _write_json

    printed: list[str] = []

    class _Console:
        def print(self, *args, **kwargs):
            printed.append(str(args[0] if args else ""))

    good = tmp_path / "reports" / "predict_X.json"
    assert _write_json(_Console(), good, {"up": np.float64(1.5)})
    assert json.loads(good.read_text(encoding="utf-8"))["up"] == 1.5

    assert not _write_json(_Console(), tmp_path / "macro.json", {"peers": {6, 7}})
    assert any("set" in line for line in printed), printed
    assert not (tmp_path / "macro.json").exists()
