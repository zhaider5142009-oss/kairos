"""A non-finite number must not reach the files that *are* the record.

`json.dumps` writes `NaN` and `Infinity` by default, and neither token is JSON.
Python's own reader accepts them back, so a book or journal holding one keeps
working on this machine — the equity arithmetic quietly turns into NaN, every
comparison against it answers no, and a standard parser anywhere else stops at
that line. The serializer in `store.py` now refuses one and names the field, and
these tests hold both halves: the refusal, and the promise that a healthy file
is written exactly as it was before.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from kairos.store import dumps_strict


def _strict_loads(text: str):
    """Parse as a real JSON reader would: NaN and Infinity are a syntax error."""
    def _reject(token: str):
        raise AssertionError(f"the file contains `{token}`, which is not JSON")
    return json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# the hole, and the refusal
# ---------------------------------------------------------------------------
def test_the_old_line_put_a_token_json_does_not_define_into_the_file():
    """Why this row exists: the default serializer produced an invalid file.

    Python reads it back, so nothing looked broken — `jq`, a spreadsheet
    import, a standards-compliant parser and the track-record publisher do not.
    """
    text = json.dumps({"equity": float("nan"), "leverage": float("inf")})
    assert "NaN" in text and "Infinity" in text
    assert json.loads(text)["equity"] != json.loads(text)["equity"]  # round-trips as NaN
    with pytest.raises(AssertionError):
        _strict_loads(text)


def test_dumps_strict_refuses_a_nan_and_names_where_it_is():
    state = {"capital": 1e6, "positions": {"MARI": {"qty": 100, "last": float("nan")}},
             "fills": [{"pnl": 1.0}, {"pnl": 2.0}, {"pnl": float("nan")}]}
    with pytest.raises(ValueError) as exc:
        dumps_strict(state)
    message = str(exc.value)
    assert "$.positions.MARI.last" in message
    assert "not a finite number" in message


def test_dumps_strict_refuses_an_infinity_in_the_pnl_field():
    with pytest.raises(ValueError, match=r"\$\.realized_pnl"):
        dumps_strict({"realized_pnl": math.inf})


@pytest.mark.parametrize("hole", [np.float64("nan"), np.float32("-inf")])
def test_a_numpy_hole_is_refused_not_converted_by_the_default_hook(hole):
    # `_json_default` turns a numpy scalar into a Python one; that happens
    # *after* this check, so a numpy NaN cannot slip through as a "supported type"
    with pytest.raises(ValueError):
        dumps_strict({"score": hole})


def test_a_hole_inside_an_array_is_refused():
    with pytest.raises(ValueError, match=r"\$\.closes"):
        dumps_strict({"closes": np.array([1.0, float("nan"), 3.0])})


def test_finite_state_is_written_exactly_as_before():
    """The guard adds a refusal, not a new format — the healthy path is unchanged."""
    obj = {"equity": 1_000_000.0, "qty": np.int64(4000), "flag": np.bool_(True),
           "ratio": np.float64(0.25), "when": datetime(2026, 10, 24, 5, 54, 28),
           "path": Path("cache"), "nested": [{"x": 1.5}, {"y": 0.0}],
           "note": None, "name": "ENGRO"}
    got = dumps_strict(obj, indent=1)
    assert got == json.dumps({"equity": 1_000_000.0, "qty": 4000, "flag": True,
                              "ratio": 0.25, "when": "2026-10-24T05:54:28",
                              "path": "cache", "nested": [{"x": 1.5}, {"y": 0.0}],
                              "note": None, "name": "ENGRO"}, indent=1)
    back = _strict_loads(got)
    assert back["flag"] is True and back["nested"][0]["x"] == 1.5


def test_the_word_nan_is_not_the_number_nan():
    """The refusal is about a value arithmetic cannot use, not about a string."""
    text = dumps_strict({"why": "nan", "veto": None, "regime": "Infinity-ish"})
    assert _strict_loads(text)["why"] == "nan"


# ---------------------------------------------------------------------------
# the files themselves
# ---------------------------------------------------------------------------
def test_the_paper_book_keeps_the_last_good_file_when_a_number_loses_its_sense():
    from kairos.execution import PaperBroker

    book = PaperBroker(capital=1_000_000.0)
    book.realized_pnl = 1234.0
    book._save()
    good = book.state_path.read_text(encoding="utf-8")
    assert _strict_loads(good)["realized_pnl"] == 1234.0

    book.realized_pnl = float("nan")
    with pytest.raises(ValueError, match=r"\$\.realized_pnl"):
        book._save()
    # the file on disk is still the previous, parseable book — not a half-written
    # one and not a NaN the next cycle would read back as a number
    assert book.state_path.read_text(encoding="utf-8") == good
    assert _strict_loads(good)


def test_the_prediction_journal_appends_nothing_for_a_hollow_prediction(tmp_path, monkeypatch):
    from kairos import feedback as FB

    journal = tmp_path / "prediction_journal.jsonl"
    monkeypatch.setattr(FB, "JOURNAL", journal)

    def res(expected_return):
        return {"ok": True, "symbol": "ENGRO", "price": 4321.0,
                "regime": {"regime": "trend"},
                "top_strategies": [],
                "signal": {"score": 0.4, "confidence": 0.3, "horizon_days": 5,
                           "expected_return": expected_return, "direction": 1}}

    FB.record(res(0.02))
    assert len(journal.read_text(encoding="utf-8").strip().splitlines()) == 1

    with pytest.raises(ValueError, match=r"\$\.expected_return"):
        FB.record(res(float("nan")))

    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert _strict_loads(lines[0])["expected_return"] == 0.02


def test_the_order_queue_refusal_is_a_refusal_not_a_crash():
    from kairos import jsgtrade as JG

    rows = [{"id": "99999", "qty": 100, "price": 1234.0}]
    JG._save(rows)
    good = JG.PENDING_PATH.read_text(encoding="utf-8")
    assert _strict_loads(good) == rows

    with pytest.raises(JG.JsGlobalError, match="could not be written"):
        JG._save([{"id": "99999", "qty": 100, "price": float("inf")}])
    # the queue a human's confirmed intent is sitting in is untouched
    assert JG.PENDING_PATH.read_text(encoding="utf-8") == good


def test_the_doctrine_file_is_not_rewritten_with_an_infinite_limit():
    from kairos import rules as R

    R.save_overrides({"max_risk_per_trade_pct": 2.0})
    good = R._runtime_file().read_text(encoding="utf-8")
    assert _strict_loads(good)["max_risk_per_trade_pct"] == 2.0

    with pytest.raises(ValueError, match=r"\$\.max_risk_per_trade_pct"):
        R.save_overrides({"max_risk_per_trade_pct": float("nan")})
    assert R._runtime_file().read_text(encoding="utf-8") == good
