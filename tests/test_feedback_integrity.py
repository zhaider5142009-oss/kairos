"""Prediction-journal integrity: the track record an investor reads must be a
record of calls, not of non-trades.

Two defects these guard:

1. `correct` was true whenever the score sat inside the no-trade band, so every
   HOLD resolved as a win. The journal read 140/140 = 100% while containing no
   directional prediction at all — and confidence_adjustment() turned that into
   a 1.05x confidence bonus in the live engine.
2. A 5-day horizon was scored on the last available bar when the future bars
   had not printed, so realized_return came out 0.00000 and "predicted -0.04%,
   realized 0.00%, correct".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from kairos import feedback as F


def _bars(days: int, start_price: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    d0 = datetime(2026, 1, 5)
    rows = []
    for i in range(days):
        rows.append({"date": pd.Timestamp(d0 + timedelta(days=i)),
                     "close": start_price * (1 + drift) ** i})
    return pd.DataFrame(rows)


class _Md:
    """Minimal market-data stub: history() returns a fixed frame."""

    def __init__(self, df):
        self.df = df

    def history(self, inst, **kw):
        return self.df


def _row(symbol="OGDC", score=0.4, ts="2026-01-05T00:00:00", horizon=5,
         exp=0.03, conf=0.6, **extra) -> dict:
    r = {"ts": ts, "symbol": symbol, "horizon": horizon, "score": score,
         "confidence": conf, "expected_return": exp, "price": 100.0,
         "direction": "BUY" if score > 0.18 else "SELL" if score < -0.18 else "HOLD",
         "resolved": False}
    r.update(extra)
    return r


@pytest.fixture(autouse=True)
def any_symbol(monkeypatch):
    """resolve_outcomes() drops unknown tickers; the stub bars work for any."""
    monkeypatch.setattr("kairos.universe.resolve",
                        lambda s: type("Inst", (), {"key": s, "name": s})(),
                        raising=True)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "prediction_journal.jsonl"
    monkeypatch.setattr(F, "JOURNAL", path)
    return path


def _write(journal, rows):
    journal.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _resolved(journal):
    return [json.loads(l) for l in
            journal.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_hold_is_not_a_win(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(score=0.02, exp=0.0)])
    n = F.resolve_outcomes(_Md(_bars(30, drift=0.004)))
    assert n == 1
    e = _resolved(journal)[0]
    assert e["directional"] is False
    assert e["correct"] is False
    assert e["realized_return"] != 0.0          # the horizon really elapsed


def test_unmatured_horizon_stays_unresolved(journal, monkeypatch):
    """Data that ends inside the horizon must not be clamped to the last bar."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(score=0.4)])
    assert F.resolve_outcomes(_Md(_bars(7))) == 0
    assert _resolved(journal)[0]["resolved"] is False


def test_directional_call_that_was_wrong_is_recorded_wrong(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(score=0.4, exp=0.03)])
    F.resolve_outcomes(_Md(_bars(30, drift=-0.004)))
    e = _resolved(journal)[0]
    assert e["directional"] is True and e["correct"] is False


def test_hit_rate_counts_only_calls(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    rows = [_row(symbol="OGDC", score=0.4), _row(symbol="HUBC", score=0.02)]
    _write(journal, rows)
    F.resolve_outcomes(_Md(_bars(30, drift=0.004)))
    tr = F.track_record()
    assert tr["resolved"] == 2
    assert tr["directional_resolved"] == 1
    assert tr["non_directional_resolved"] == 1
    assert tr["hit_rate"] == 1.0                # one call, correct


def test_all_hold_journal_has_no_hit_rate(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(score=0.02, exp=0.0), _row(score=-0.01, exp=0.0)])
    F.resolve_outcomes(_Md(_bars(30, drift=0.001)))
    tr = F.track_record()
    assert tr["hit_rate"] is None
    assert "HOLD" in tr["reason"]
    # the desk's per-row ✓ column must not be fed non-trades
    assert tr["recent"] == []


def test_legacy_rows_without_the_flag_are_rederived(journal, monkeypatch):
    """Rows written before the distinction existed must not enter the hit rate
    just because they were never re-scored."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    legacy = [{"ts": "2026-01-05T00:00:00", "symbol": "GOLD", "horizon": 5,
               "score": 0.01, "confidence": 0.3, "expected_return": 0.0,
               "price": 4181.1, "direction": "HOLD", "resolved": True,
               "realized_return": 0.0, "correct": True, "abs_err": 0.0}]
    _write(journal, legacy)
    tr = F.track_record()
    assert tr["directional_resolved"] == 0
    assert tr["hit_rate"] is None


def test_no_evidence_does_not_inflate_confidence(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(score=0.02, exp=0.0)] * 20)
    F.resolve_outcomes(_Md(_bars(30, drift=0.001)))
    assert F.confidence_adjustment() == 1.0


def test_genuine_strength_still_earns_the_bonus(journal, monkeypatch):
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(symbol=f"S{i}", score=0.4, exp=0.01) for i in range(12)])
    F.resolve_outcomes(_Md(_bars(30, drift=0.004)))
    assert F.track_record()["hit_rate"] == 1.0
    assert F.confidence_adjustment() == 1.05
