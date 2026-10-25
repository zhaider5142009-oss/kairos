"""Attribution: the parts of the library must be paid for what they got right.

Five defects this guards, all of them the same shape — a component that was
never asked whether it worked:

1. `ensemble.fuse` polled 100 strategies by their own stated strength and no
   outcome in the journal was ever credited to the strategy that produced it,
   so a component that is right only in a trending tape carried the same weight
   in a range, and `feedback` could only move one global confidence number.
2. A per-strategy hit rate measured in absolute terms is not skill, it is
   drift: in a bull sample every long-biased strategy reads 60-75% correct.
   Weights are therefore a *difference* from the rate at which that horizon
   rose in that regime, and where the artifact cannot say how often the tape
   rose, no weight is applied and the reason is reported.
3. `run_strategy_library` dropped every strategy that raised, which made a
   broken component indistinguishable from a careful one that abstained — and
   an attribution run would have quietly recorded the broken one as having no
   edge.
4. `load` had to refuse, not guess. An unreadable, stale, short or
   differently-sized artifact must leave the poll flat and say why; returning
   `{}` would let "nothing has demonstrated skill" be read off a file that was
   never readable in the first place.
5. Thin buckets: a 75% hit rate on twelve votes is a coin flip that landed
   well. Shrinkage by count is the only thing standing between a noisy bucket
   and an order.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from kairos import attribution as A
from kairos.ensemble import FusedSignal, StrategyVote, fuse
from kairos.feedback import Snapshot, record
from kairos.strategies import run_strategy_library


TODAY = date(2026, 10, 20)
GEN = "2026-10-18"                      # two days before TODAY: fresh enough
SIZE = 100                              # library size the artifact was measured at


def _counts(*, graded: int = A.MIN_VOTES) -> dict:
    counts = A.new_counts(SIZE, A.HORIZONS)
    counts["generated"] = GEN
    counts["graded"] = graded
    return counts


def _strategy(counts, name, *, category="Trend", horizon=5, pooled=None,
              in_regime=None, errors=0, slices=100, tape_up=None,
              regime_up=None) -> dict:
    """Fill one strategy row from stated buckets.

    `pooled` is `(votes, hits, longs)` for the whole horizon; `in_regime` adds
    `(regime, votes, hits, longs)`. Shorts are the remainder because in these
    buckets every vote is directional — which is exactly the split the weight
    needs, since a long in a bull tape and a short in a bear one both look
    right before the tape is subtracted.

    `tape_up` / `regime_up` are how often the tape rose on the slices *this
    strategy voted on* — the paired denominator its hit rate is measured
    against. They are separate from `_base`, which is the whole-condition
    frequency the desk prints. A strategy that abstains in half the slices has
    two different references, and confusing them is the defect this pins.
    """
    st = A.ensure_strategy(counts, name, category, tuple(counts["horizons"]))
    st["errors"] = errors
    st["slices_seen"] = slices
    if pooled is not None:
        n, hits, longs = pooled
        st["by_horizon"][str(horizon)] = [n, hits, longs, n - longs]
        if tape_up is not None:
            st["base_by_horizon"][str(horizon)] = [n, int(round(tape_up * n))]
    if in_regime is not None:
        regime, n, hits, longs = in_regime
        st["by_bucket"][f"{horizon}|{regime}"] = [n, hits, longs, n - longs]
        if regime_up is not None:
            st["base_by_bucket"][f"{horizon}|{regime}"] = [n, int(round(regime_up * n))]
    return st


def _base(counts, horizon, p_up, *, regime=None, slices=400) -> None:
    """How often the tape actually rose — the reference every claim is measured
    against. `_all` is the pooled bucket; `regime` names a conditional one."""
    hb = counts["base"].setdefault(str(horizon), {})
    up = int(round(p_up * slices))
    if regime is None:
        hb["_all"] = [slices, up]
    else:
        hb[regime] = [slices, up]


def _write(tmp_path, counts) -> str:
    path = tmp_path / "strategy_attribution.json"
    path.write_text(json.dumps(counts), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- one weight
def test_skill_is_a_difference_from_the_tape_not_a_ratio():
    """Long-only and right 70% in a tape that rose 70% of the time has
    demonstrated nothing, and must not be promoted."""
    counts = _counts()
    _strategy(counts, "Drift Rider", pooled=(400, 280, 400), tape_up=0.70)
    _base(counts, 5, 0.70)
    attr = A.Attribution(counts)
    w, tag = attr.weight_tag("Drift Rider", 5, "Range")
    assert tag == "pooled"
    assert w == pytest.approx(1.0, abs=0.001), f"drift was read as skill: {w}"


def test_skill_is_measured_on_the_slices_the_strategy_spoke_on():
    """A hit rate must be compared with the tape over the same slices.

    `_derive` used the condition-wide up-frequency — every slice of every
    instrument in that condition — while the hit rate came only from the slices
    the strategy actually voted on. A member that abstains in down tapes then
    reads as skilled for having skipped them: below, 360 hits on 400 long votes
    (90%) against a pooled tape that rose 50% of the time is the maximum
    promotion, while the 400 slices it sat on rose 95% of the time, on which
    the same record is ten points *behind* always-long. Paid either way, the
    weight is a payment for when the member chose to speak, not for what it
    said.
    """
    counts = _counts()
    _strategy(counts, "Fair Weather", pooled=(400, 360, 400), tape_up=0.95)
    _base(counts, 5, 0.50, slices=800)
    w, tag = A.Attribution(counts).weight_tag("Fair Weather", 5, "Range")
    assert tag == "pooled"
    assert w < 0.95, f"the call was paid for the tape it only spoke on: {w}"


def test_a_real_edge_moves_the_weight_both_ways():
    counts = _counts()
    # 60% of 400 long votes in a tape that rose 45% of the time: genuinely ahead.
    _strategy(counts, "Good", pooled=(400, 240, 400), tape_up=0.45)
    # 30% of 400 in the same tape: genuinely behind.
    _strategy(counts, "Bad", pooled=(400, 120, 400), tape_up=0.45)
    _base(counts, 5, 0.45)
    attr = A.Attribution(counts)
    good = attr.weight("Good", 5, "Range")
    bad = attr.weight("Bad", 5, "Range")
    assert good > 1.05 and bad < 0.95
    assert A.MIN_WEIGHT <= bad and good <= A.MAX_WEIGHT
    # a short seller is judged against the same tape from the other side: 60%
    # of 400 in a tape that fell 55% of the time is only just ahead
    counts2 = _counts()
    _strategy(counts2, "Shorty", pooled=(400, 240, 0), tape_up=0.45)
    _base(counts2, 5, 0.45)
    assert A.Attribution(counts2).weight("Shorty", 5, "Range") > 1.05


def test_a_thin_bucket_cannot_move_a_weight_at_all():
    """MIN_CALLS is the gate, not a soft nudge: below it a bucket has not
    separated skill from a coin flip, so it must not touch an order."""
    counts = _counts()
    _base(counts, 5, 0.5)
    _strategy(counts, "Twelve", pooled=(12, 9, 12), tape_up=0.5)          # 75% on a dozen
    _strategy(counts, "FourHundred", pooled=(400, 300, 400), tape_up=0.5)  # 75% on four hundred
    attr = A.Attribution(counts)
    thin, thin_tag = attr.weight_tag("Twelve", 5, "Range")
    thick, thick_tag = attr.weight_tag("FourHundred", 5, "Range")
    assert thin == 1.0 and thin_tag == "no coverage", \
        f"a dozen votes changed a weight: {thin}"
    assert thick_tag == "pooled" and thick > 1.4


def test_regime_evidence_is_used_when_there_is_enough_of_it():
    """A strategy good in a range and mediocre overall must not be averaged into
    one number — that averaging is the thing the whole layer exists to undo."""
    counts = _counts()
    _base(counts, 5, 0.55)                                  # pooled: tape up 55%
    _base(counts, 5, 0.60, regime="Range")
    _base(counts, 5, 0.35, regime="Bull-Volatile")
    _strategy(counts, "Split", pooled=(400, 200, 400), tape_up=0.55,
              in_regime=("Range", 120, 100, 120), regime_up=0.60)
    attr = A.Attribution(counts)
    # in Range: 100/120 = 83% long against a tape up 60% → ahead
    # pooled:   200/400 = 50% long against a tape up 55% → behind
    w_regime, tag = attr.weight_tag("Split", 5, "Range")
    assert tag == "regime"
    assert w_regime > 1.2, f"the regime bucket was not used: {w_regime}"
    pooled, pooled_tag = attr.weight_tag("Split", 5, "MeanRev-Quiet")
    assert pooled_tag == "pooled"
    assert pooled < 1.0, "the regime-only evidence leaked into an unknown regime"
    # and one regime with a base rate of its own must not borrow another's
    assert attr.weight("Split", 5, "Bull-Volatile") == pytest.approx(pooled, abs=1e-9)


def test_a_regime_where_nothing_rose_still_has_a_base_rate():
    """A base rate of exactly 0.0 is a measurement, not an empty cell.

    `_derive` chained the regime's rate onto the pooled one with `or`, and 0.0 is
    falsy — so the one regime where subtracting the tape matters most, a tape in
    which *nothing* rose, silently scored against the pooled rate instead. That
    cuts both ways and only one way is safe: measured against a 70%-up pooled
    reference, 40 short votes that were right on all 40 slices look like huge skill
    because the shorts were never credited to the drift that made them free, while
    longs that were right *against* a tape that never rose are scored as barely
    ahead of their own bias. The first buys leverage with nothing; the second is
    the evidence the layer exists to find. On the fixture below the two weights
    were inverted: 1.500 for the drift and 1.098 for the skill before the `or`,
    1.000 and 1.500 after.
    """
    counts = _counts()
    _base(counts, 5, 0.70)                                  # pooled: tape up 70%
    _base(counts, 5, 0.00, regime="Crash")                  # in Crash: never rose
    _strategy(counts, "Down Rider", in_regime=("Crash", 40, 40, 0), regime_up=0.0)       # 40 shorts, all hit
    _strategy(counts, "Against The Tape", in_regime=("Crash", 40, 32, 40), regime_up=0.0)  # 40 longs, 32 hit
    attr = A.Attribution(counts)
    assert A._p_up([400, 0]) == 0.0, "the fixture no longer is the case under test"
    assert attr.weight("Down Rider", 5, "Crash") == pytest.approx(1.0, abs=1e-9), \
        "shorts in a tape that never rose were paid as skill"
    assert attr.weight("Against The Tape", 5, "Crash") > 1.2, \
        "counter-trend longs were scored against the pooled rate, not their own regime"


def test_no_coverage_and_broken_and_no_base_are_all_said_loudly():
    counts = _counts()
    _strategy(counts, "Silent")                             # never voted
    _strategy(counts, "Thrower", pooled=(400, 100, 400),   # raised on half
              errors=50, slices=100, tape_up=0.5)
    _strategy(counts, "Orphan", pooled=(400, 300, 400))     # but no base rate
    attr = A.Attribution(counts)
    assert attr.weight("Silent", 5, "Range") == 1.0
    assert attr.weight("Thrower", 5, "Range") == 1.0, "a broken module was demoted"
    assert attr.weight("Orphan", 5, "Range") == 1.0, "no base rate but a weight anyway"
    st = attr.status()
    assert st["broken"] == ["Thrower"]
    assert st["no_coverage"] == 1 and st["no_base_rate"] == 1
    assert st["demoted"] == 0 and st["promoted"] == 0
    assert st["weighted"] == 3


# ----------------------------------------------------------------- refusals
def test_load_accepts_a_good_artifact_and_refuses_each_bad_one(tmp_path):
    good = _counts()
    _strategy(good, "Ok", pooled=(400, 300, 400))
    _base(good, 5, 0.5)
    attr, reason = A.load(_write(tmp_path, good), library_size=SIZE, today=TODAY)
    assert attr is not None and reason == ""

    cases = {
        "absent": None,
        "truncated_json": '{"strategies": {"a": ',
        "not_a_table": '{"strategies": 7, "schema": 1}',
        "wrong_schema": json.dumps(dict(good, schema=99)),
        "wrong_library": json.dumps(dict(good, library_size=97)),
        "under_voted": json.dumps(dict(good, graded=17)),
        "undated": json.dumps(dict(good, generated=None)),
        "stale": json.dumps(dict(good, generated="2020-01-01")),
        "future_dated": json.dumps(dict(good, generated="2030-01-01")),
    }
    needles = {
        "absent": "no attribution artifact",
        "truncated_json": "not valid JSON",
        "not_a_table": "does not hold",
        "wrong_schema": f"schema {99}",
        "wrong_library": f"the library now holds {SIZE}",
        "under_voted": "only 17 directional votes",
        "undated": "no readable generation date",
        "stale": "days old",
        "future_dated": "days from now",
    }
    for label, payload in cases.items():
        path = tmp_path / f"{label}.json"
        if payload is not None:
            path.write_text(payload, encoding="utf-8")
        attr, reason = A.load(str(path), library_size=SIZE, today=TODAY)
        assert attr is None, f"{label}: a refused artifact loaded anyway"
        assert needles[label] in reason, f"{label}: reason was {reason!r}"


def test_a_damaged_file_is_not_read_as_no_edge(tmp_path):
    """The reading this refuses: an empty strategy table means "nobody has an
    edge", but it also means "the file was truncated" — and only one of those
    is a statement about the market."""
    p = tmp_path / "strategy_attribution.json"
    p.write_text(json.dumps(dict(_counts(), strategies={}, graded=A.MIN_VOTES)),
                 encoding="utf-8")
    attr, reason = A.load(str(p), library_size=SIZE, today=TODAY)
    assert attr is not None and reason == ""   # the file itself is intact
    assert attr.status()["graded_votes"] == A.MIN_VOTES
    v = StrategyVote("Anything", "Trend", 1, 0.5)
    sig = fuse("X", [], [v], "Range", 5, attribution=attr.view())
    assert sig.details["attribution"]["weighted"] == 1
    assert sig.details["attribution"]["no_coverage"] == 1
    assert sig.details["strategies"]["Anything"]["dir"] == 1


# --------------------------------------------------------------------- fuse
def _flat_signal(votes, regime="Range", horizon=5):
    return fuse("X", [], votes, regime, horizon)


def test_fuse_without_attribution_is_the_old_flat_poll():
    votes = [StrategyVote("A", "Trend", 1, 0.8), StrategyVote("B", "Trend", -1, 0.8)]
    before = _flat_signal(votes)
    assert isinstance(before, FusedSignal)
    assert before.strategy_consensus == 0.0
    assert "attribution" not in before.details
    assert "w" not in before.details["strategies"]["A"]


def test_a_demoted_strategy_still_speaks_at_half_voice():
    counts = _counts()
    _base(counts, 5, 0.5)
    _strategy(counts, "Bull", pooled=(400, 40, 400), tape_up=0.5)      # badly wrong, long
    _strategy(counts, "Bear", pooled=(400, 360, 0), tape_up=0.5)       # badly right, short
    attr = A.Attribution(counts).view()
    votes = [StrategyVote("Bull", "Trend", 1, 0.9),
             StrategyVote("Bear", "Reversion", -1, 0.9)]
    flat = _flat_signal(votes)
    weighted = fuse("X", [], votes, "Range", 5, attribution=attr)
    assert flat.strategy_consensus == 0.0, "two equal votes were not flat before"
    assert weighted.strategy_consensus < 0.0, "attribution did not reach the poll"
    # the point of the whole layer: a re-weighted poll can cross the band the
    # desk trades on, so the book changes without a new vote being added
    assert flat.direction == "HOLD" and weighted.direction == "SELL"
    d = weighted.details
    assert d["attribution"]["applied"] is True
    assert d["attribution"]["demoted"] == 1 and d["attribution"]["promoted"] == 1
    assert d["strategies"]["Bull"]["w"] == A.MIN_WEIGHT
    assert d["strategies"]["Bear"]["w"] == A.MAX_WEIGHT
    assert d["strategy_bear"] > d["strategy_bull"]


def test_attribution_status_does_not_accumulate_across_a_watchlist():
    """`view()` exists because the cached artifact is one object and the desk
    shows per-prediction counters; a status that grew with every symbol would
    report the process's history inside tonight's signal."""
    counts = _counts()
    _base(counts, 5, 0.5)
    _strategy(counts, "Ok", pooled=(400, 300, 400), tape_up=0.5)
    attr = A.Attribution(counts)
    votes = [StrategyVote("Ok", "Trend", 1, 0.5)]
    first = fuse("A", [], votes, "Range", 5, attribution=attr.view()).details
    second = fuse("B", [], votes, "Range", 5, attribution=attr.view()).details
    assert first["attribution"]["weighted"] == 1
    assert second["attribution"]["weighted"] == 1, \
        "the second symbol reported the process's history, not its own signal"


# ------------------------------------------------------- library reporting
def test_a_strategy_that_raises_is_counted_as_broken_not_as_abstaining():
    import kairos.strategies as S
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                       "open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
                       "close": [1.0] * 5, "volume": [10.0] * 5})
    before = len(S.STRATEGIES)

    @S.strategy("Test Thrower", "Trend")
    def _boom(df_, f_, ctx_):
        raise KeyError("simulated broken component")

    @S.strategy("Test Abstainer", "Trend")
    def _quiet(df_, f_, ctx_):
        return None

    try:
        report: dict = {}
        votes = run_strategy_library(df, {}, None, report=report)
        assert all(v.name != "Test Thrower" for v in votes)
        assert "simulated broken component" in report["errors"]["Test Thrower"]
        assert "Test Abstainer" not in report["errors"]
        assert {"Test Thrower", "Test Abstainer"} <= set(report["seen"])
        assert len(report["seen"]) == len(S.STRATEGIES)
    finally:
        del S.STRATEGIES[before:]


def test_slice_counts_and_base_rates_reach_strategies_that_never_vote():
    counts = A.new_counts(2, (5,))
    A.note_slice(counts, ["Voter", "Quiet"], {"Voter": "ValueError: boom"})
    A.note_outcome(counts, "Range", {5: 0.02}, (5,))
    A.note_outcome(counts, "Range", {5: -0.01}, (5,))
    A.grade(counts, "Voter", "Trend", 1, 0.5, "Range", {5: 0.02})
    assert counts["strategies"]["Quiet"]["slices_seen"] == 1
    assert counts["strategies"]["Quiet"]["votes"] == 0
    assert counts["strategies"]["Voter"]["errors"] == 1
    # the pooled base counts every slice, including the ones nobody voted on
    assert counts["base"]["5"]["_all"] == [2, 1]
    assert counts["base"]["5"]["Range"] == [2, 1]
    # the denominator a weight is measured against counts only the slices that
    # strategy voted on — here one, on which the tape rose
    assert counts["strategies"]["Voter"]["base_by_horizon"]["5"] == [1, 1]
    assert counts["strategies"]["Voter"]["base_by_bucket"]["5|Range"] == [1, 1]
    assert counts["strategies"]["Quiet"]["base_by_horizon"] == {}
    assert counts["graded"] == 1
    # a vote that is not a claim is not graded
    A.grade(counts, "Voter", "Trend", 0, 0.5, "Range", {5: 0.02})
    A.grade(counts, "Voter", "Trend", 1, 0.0, "Range", {5: 0.02})
    assert counts["graded"] == 1


def test_a_graded_vote_lands_in_its_own_column_and_nowhere_else():
    """`grade` writes a four-column bucket, and the first full walk-forward run
    produced more hits than votes: the long count was being added to the hit
    column. Every weight is built from these four numbers, so the columns are
    asserted by name, not inferred from a ratio that happens to look plausible."""
    counts = A.new_counts(2, (5,))
    A.grade(counts, "Long", "Trend", 1, 0.5, "Range", {5: 0.01})    # right
    A.grade(counts, "Long", "Trend", 1, 0.5, "Range", {5: -0.01})   # wrong
    A.grade(counts, "Short", "Trend", -1, 0.5, "Range", {5: -0.02})  # right
    assert counts["strategies"]["Long"]["by_horizon"]["5"] == [2, 1, 2, 0]
    assert counts["strategies"]["Short"]["by_horizon"]["5"] == [1, 1, 0, 1]
    assert counts["strategies"]["Long"]["by_bucket"]["5|Range"] == [2, 1, 2, 0]
    # one paired base observation per graded vote, and it records the tape, not
    # the call: Long saw one up and one down slice; Short saw one down slice
    assert counts["strategies"]["Long"]["base_by_horizon"]["5"] == [2, 1]
    assert counts["strategies"]["Short"]["base_by_horizon"]["5"] == [1, 0]
    assert counts["strategies"]["Long"]["base_by_bucket"]["5|Range"] == [2, 1]
    for st in counts["strategies"].values():
        for bucket in list(st["by_horizon"].values()) + list(st["by_bucket"].values()):
            assert bucket[1] <= bucket[0], f"hits exceed votes: {bucket}"
            assert bucket[2] + bucket[3] == bucket[0], f"votes lost in the split: {bucket}"


def test_a_bucket_without_its_paired_denominator_is_not_published():
    """The sweep's own gate on the column `_derive` now reads.

    A run where `grade` forgot the paired base would still produce a
    well-shaped artifact; every strategy would load as *no base rate* and the
    poll would stay flat, which is indistinguishable at the desk from "nothing
    has an edge". The tool therefore refuses to publish it. An abstaining
    member — a row whose pre-created horizon bucket is still all zeros — is not
    a missing denominator and must not block the run.
    """
    import sys
    from pathlib import Path as _Path
    tools = _Path(__file__).resolve().parent.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from strategy_attribution import _bad_buckets

    counts = _counts()
    _strategy(counts, "Voter", pooled=(40, 30, 40), tape_up=0.5)
    _strategy(counts, "Quiet")
    assert _bad_buckets(counts) == []

    del counts["strategies"]["Voter"]["base_by_horizon"]["5"]
    bad = _bad_buckets(counts)
    assert any("Voter" in b and "no paired base rate" in b for b in bad), bad

    counts["strategies"]["Voter"]["base_by_horizon"]["5"] = [40, 41]
    assert any("41 up-slices on 40 slices" in b for b in _bad_buckets(counts))


# ------------------------------------------------------------------ journal
def test_the_journal_records_who_spoke_so_live_outcomes_can_be_attributed(tmp_path):
    """Backtest weights are a hypothesis. These two columns are the only way
    tonight's book can ever confirm or refute it, so they have to start being
    written now, while there is nothing to attribute, rather than after the
    evidence that would settle it has been thrown away."""
    rows: list[dict] = []
    import kairos.feedback as FB

    class _Handle:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, line):
            rows.append(json.loads(line.strip()))

    class _J:
        parent = tmp_path       # record() creates the journal's directory first

        def open(self, *a, **kw):
            return _Handle()

    res = {"ok": True, "symbol": "TEST", "price": 10.0,
           "signal": {"horizon_days": 5, "score": 0.4, "confidence": 0.6,
                      "expected_return": 0.02, "direction": "BUY"},
           "regime": {"regime": "Bull-Volatile"},
           "top_strategies": [{"name": "MACD Signal", "direction": 1},
                              {"name": "Quiet One", "direction": 0},
                              {"name": "", "direction": -1},
                              {"name": "Thrower", "direction": None}]}
    keep = (FB.JOURNAL, FB._suffix_if_unterminated)
    FB.JOURNAL, FB._suffix_if_unterminated = _J(), (lambda: "")
    try:
        record(res)
    finally:
        FB.JOURNAL, FB._suffix_if_unterminated = keep
    assert len(rows) == 1
    assert rows[0]["regime"] == "Bull-Volatile"
    assert rows[0]["strategies"] == {"MACD Signal": 1}
    assert rows[0]["resolved"] is False


def test_a_legacy_journal_row_still_reads_without_the_new_columns():
    """The resolver and the desk both read rows written before these fields
    existed; a snapshot reconstructed from such a row must not invent a regime
    or a strategy that was never recorded."""
    legacy = {"ts": "2026-01-01T00:00:00", "symbol": "TEST", "horizon": 5,
              "score": 0.4, "confidence": 0.6, "expected_return": 0.02,
              "price": 10.0, "direction": "BUY"}
    snap = Snapshot(**legacy)
    out = json.loads(json.dumps(snap.to_dict()))
    assert out["regime"] is None and out["strategies"] is None
    assert out["resolved"] is False
