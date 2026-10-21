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

And the third, found while testing the fix for the first two: the repair
re-derived "was this a call?" from each row's score against the entry threshold
*currently* configured, and left the stale `correct` column in the arithmetic.
The threshold is a knob the operator turns, so turning it down would have
promoted last month's 140 HOLDs into 140 wins and republished a 100% hit rate
from a journal with no calls in it; turning it up would have deleted real calls
that were placed and won. Whether a row was a claim is now read from the label
written that day, and a legacy row's verdict is recomputed from the sign of the
return against the direction claimed. Neither depends on today's config. The first cut of
that repair had a hole of its own, and it is the reason the label is matched on a substring:
it matched `BUY` exactly, while the ensemble's vocabulary is `STRONG BUY / BUY / HOLD / SELL /
STRONG SELL`, so the strongest band — exactly the one the strong-signal rate is measured on —
was filed as a loss it had actually won. And the write path had the same coupling all along:
`directional` and `correct` are stamped at resolution time and read back for the life of the
record, so they now come from the row's own claim too, with the band only a fallback.
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
    """The stub bars answer for any symbol, so resolution depends only on the
    horizon having printed."""
    monkeypatch.setattr("kairos.universe.resolve",
                        lambda s: type("Inst", (), {"key": s, "name": s})(),
                        raising=True)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "prediction_journal.jsonl"
    monkeypatch.setattr(F, "JOURNAL", path)
    return path


def _write(journal, rows):
    # terminated, the way both production writers leave the file
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                       encoding="utf-8")


def _resolved(journal):
    return [json.loads(ln) for ln in
            journal.read_text(encoding="utf-8").splitlines() if ln.strip()]


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


def test_lowering_the_threshold_cannot_promote_yesterdays_holds(journal,
                                                                 monkeypatch):
    """The entry threshold is operator-tuned; the record is not.

    A legacy HOLD whose score sat under the old band still carries the pre-fix
    correct=True. If direction is re-derived from whatever threshold is set
    today, turning the knob down reaches backwards and edits history: these
    rows become directional wins and the ledger publishes a perfect hit rate
    built entirely from non-trades.
    """
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.05)
    _write(journal, [
        {"ts": "2026-01-05T00:00:00", "symbol": s, "horizon": 5, "score": sc,
         "confidence": 0.3, "expected_return": 0.0, "price": 100.0,
         "direction": "HOLD", "resolved": True, "realized_return": rr,
         "correct": True, "abs_err": 0.001}
        for s, sc, rr in (("GOLD", 0.152, 0.011), ("WTI", -0.14, -0.007),
                          ("SPX", 0.09, 0.004))])
    tr = F.track_record()
    assert tr["directional_resolved"] == 0
    assert tr["hit_rate"] is None
    assert tr["recent"] == []


def test_a_legacy_call_is_judged_on_the_sign_not_the_stale_column(journal,
                                                                  monkeypatch):
    """The old scorer's `correct` meant "landed inside the no-trade band", so a
    wrong BUY could be filed as right. Recompute from the direction claimed
    against the return that happened."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [
        {"ts": "2026-01-05T00:00:00", "symbol": "OGDC", "horizon": 5,
         "score": 0.44, "confidence": 0.6, "expected_return": 0.03,
         "price": 100.0, "direction": "BUY", "resolved": True,
         "realized_return": -0.021, "correct": True, "abs_err": 0.051},
        {"ts": "2026-01-05T00:00:00", "symbol": "HUBC", "horizon": 5,
         "score": -0.51, "confidence": 0.5, "expected_return": -0.02,
         "price": 100.0, "direction": "SELL", "resolved": True,
         "realized_return": -0.03, "correct": False, "abs_err": 0.01}])
    tr = F.track_record()
    assert tr["directional_resolved"] == 2
    # the BUY lost (wrong, despite its stale ✓); the SELL won
    assert tr["hit_rate"] == 0.5
    recent = {e["symbol"]: e["correct"] for e in tr["recent"]}
    assert recent == {"OGDC": False, "HUBC": True}


def test_the_claim_is_what_was_recorded_even_below_todays_band(journal,
                                                               monkeypatch):
    """A raised threshold must not retroactively un-make a call that was
    actually placed as one."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.60)
    _write(journal, [
        {"ts": "2026-01-05T00:00:00", "symbol": "LUCK", "horizon": 5,
         "score": 0.22, "confidence": 0.4, "expected_return": 0.01,
         "price": 100.0, "direction": "BUY", "resolved": True,
         "realized_return": 0.02, "correct": True, "abs_err": 0.01}])
    tr = F.track_record()
    assert tr["directional_resolved"] == 1
    assert tr["hit_rate"] == 1.0


def test_strong_labels_are_the_same_claim(journal, monkeypatch):
    """The ensemble says STRONG BUY / BUY / HOLD / SELL / STRONG SELL.

    Matching the label exactly would file the strongest band — the one the
    published strong-signal rate is measured on — as a non-trade or, worse, as
    a loss it actually won.
    """
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [
        {"ts": "2026-01-05T00:00:00", "symbol": "BTC", "horizon": 5,
         "score": 0.71, "confidence": 0.6, "expected_return": 0.04,
         "price": 100.0, "direction": "STRONG BUY", "resolved": True,
         "realized_return": 0.061, "correct": False, "abs_err": 0.021},
        {"ts": "2026-01-05T00:00:00", "symbol": "OGDC", "horizon": 5,
         "score": -0.66, "confidence": 0.6, "expected_return": -0.03,
         "price": 100.0, "direction": "STRONG SELL", "resolved": True,
         "realized_return": 0.012, "correct": True, "abs_err": 0.042}])
    tr = F.track_record()
    assert tr["directional_resolved"] == 2
    assert tr["strong_signal_hit_rate"] == 0.5     # the BUY won, the SELL lost
    assert tr["hit_rate"] == 0.5


def test_a_row_with_no_label_at_all_falls_back_to_the_score(journal,
                                                            monkeypatch):
    """The score path is only for rows too old to say what they claimed."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [
        {"ts": "2026-01-05T00:00:00", "symbol": "SILVER", "horizon": 5,
         "score": 0.62, "confidence": 0.5, "expected_return": 0.02,
         "price": 100.0, "resolved": True, "realized_return": 0.03,
         "correct": True, "abs_err": 0.01}])
    tr = F.track_record()
    assert tr["directional_resolved"] == 1
    assert tr["hit_rate"] == 1.0          # up claimed, up delivered


def test_resolution_writes_the_claim_not_todays_band(journal, monkeypatch):
    """Same rule at write time as at read time.

    `directional` and `correct` are stamped once and read back forever, so if
    the stamp came from whatever threshold is configured that day, tuning the
    gate tomorrow would choose which of today's snapshots ever counted — the
    landmine one step earlier in the pipeline.
    """
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.60)
    _write(journal, [_row(symbol="OGDC", score=0.4, exp=0.03),
                     _row(symbol="HUBC", score=0.02, exp=0.0)])
    F.resolve_outcomes(_Md(_bars(30, drift=0.004)))
    out = _resolved(journal)
    buy, hold = [r for r in out if r["symbol"] == "OGDC"][0], \
                [r for r in out if r["symbol"] == "HUBC"][0]
    assert buy["directional"] is True        # labelled BUY: the claim stands
    assert buy["correct"] is True            # ...and it came in
    assert hold["directional"] is False      # labelled HOLD: never a call


def test_a_tightened_gate_cannot_un_make_a_call(journal, monkeypatch):
    """The desk gate is tunable; a claim already recorded is not retracted by it."""
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.90)
    _write(journal, [_row(symbol="LUCK", score=0.42, exp=0.03)])
    F.resolve_outcomes(_Md(_bars(30, drift=0.004)))
    tr = F.track_record()
    assert tr["directional_resolved"] == 1
    assert tr["hit_rate"] == 1.0


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


def test_a_symbol_the_registry_cannot_answer_leaves_the_call_pending(journal,
                                                                     monkeypatch):
    """An unanswerable lookup is a reason to retry, not a verdict.

    Rows for a symbol `universe.resolve()` could not name were resolved as
    "no-instrument" — permanently, and with `directional` stamped from the
    entry threshold configured that day. A watchlist ticker whose dynamic
    registry had not been rebuilt, or a journal copied to a fresh deploy
    without it, therefore deleted every pending call on that symbol from the
    forward record in one cycle, and nothing raised an error to say so.
    """
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    monkeypatch.setattr("kairos.universe.resolve", lambda s: None, raising=True)
    _write(journal, [_row(score=0.40), _row(score=0.90)])
    assert F.resolve_outcomes(_Md(_bars(30, drift=0.004))) == 0
    out = _resolved(journal)
    assert [r["resolved"] for r in out] == [False, False], out
    assert not any("outcome" in r for r in out), out

    # The same two calls are still scorable once the symbol answers again.
    monkeypatch.setattr("kairos.universe.resolve",
                        lambda s: type("Inst", (), {"key": s})(), raising=True)
    assert F.resolve_outcomes(_Md(_bars(30, drift=0.004))) == 2
    assert [r["directional"] for r in _resolved(journal)] == [True, True]


def test_a_snapshot_recorded_mid_resolve_survives_the_rewrite(journal, monkeypatch):
    """The journal is rewritten whole, so a row appended during the rewrite dies.

    `resolve_outcomes` reads the file, walks its in-memory copy for as long as
    the bars take, then overwrites the journal from that copy. Every other
    thread that finishes a fresh prediction appends to the same file — and a
    dashboard visitor, the desk's own auto-refresh or a `kairos predict` in a
    second terminal is such a thread. A snapshot that landed inside the window
    was not marked unresolved or mis-scored: it was never in the file again, so
    the forward record quietly stopped counting the call the investor was
    promised. Resolution had written nothing until a horizon actually matured,
    so this armed itself on the first publishable day rather than showing up
    in the days before it.
    """
    monkeypatch.setattr(F, "_direction_threshold", lambda: 0.18)
    _write(journal, [_row(symbol="OGDC", score=0.40), _row(symbol="SCB", score=0.30)])
    late = _row(symbol="HUBC", score=0.55, ts="2026-10-21T12:00:00")

    class _AppendMidResolve(_Md):
        """Marks the point where another writer can slip a row in: after the
        journal has been read, before it is written back."""

        def __init__(self, df, row):
            super().__init__(df)
            self.row, self.fired = row, False

        def history(self, inst, **kw):
            if not self.fired:
                self.fired = True
                with journal.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(self.row) + "\n")
            return self.df

    assert F.resolve_outcomes(_AppendMidResolve(_bars(30, drift=0.004), late)) == 2
    out = _resolved(journal)
    assert [r["symbol"] for r in out] == ["OGDC", "SCB", "HUBC"], out
    assert [r["resolved"] for r in out] == [True, True, False], out
    # the two matured calls keep their verdicts; the newcomer keeps its turn
    assert [r["correct"] for r in out[:2]] == [True, True], out
    assert F.track_record()["directional_resolved"] == 2


def test_recording_onto_a_copied_journal_does_not_fuse_two_rows(tmp_path, monkeypatch):
    """A journal that arrives unterminated must not swallow the next snapshot.

    Restoring or copying the record onto a fresh deploy leaves the last line
    without its newline, and a plain append then welds it to the new row: one
    line no parser reads, which raises for the resolver and the desk alike and
    stops the forward record while the file still looks populated.
    """
    path = tmp_path / "prediction_journal.jsonl"
    monkeypatch.setattr(F, "JOURNAL", path)
    path.write_text(json.dumps(_row(symbol="OGDC")), encoding="utf-8")
    F.record({"ok": True, "symbol": "HUBC", "price": 100.0,
              "signal": {"horizon_days": 5, "score": 0.5, "confidence": 0.6,
                         "expected_return": 0.02, "direction": "BUY"}})
    out = _resolved(path)
    assert [r["symbol"] for r in out] == ["OGDC", "HUBC"], out
