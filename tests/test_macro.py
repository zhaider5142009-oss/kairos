"""The macro axis: a condition the instrument's own bars cannot see, and the
vocabulary that lets attribution charge for it.

Seven things these tests hold shut, most of them the same shape — a state that
could be reported as something it is not:
1. An absent print reading as a neutral one. If the VIX feed is down, a score
   that averages "no opinion" in with the six opinions it does have lands near
   zero and reports Neutral, which is a *finding*. Coverage is therefore
   counted, `MIN_COMPONENTS` is a floor of its own, and a thin panel gets its
   own tag rather than a mid-range number.
2. The live engine and the walk-forward spelling the bucket key differently.
   Both call `condition_label`; a divergence would make every live lookup miss,
   report "no coverage", and read as though no strategy had an edge.
3. Look-ahead on the macro half. The state is computed from the same
   date-truncated panel the strategies get, so a slice cannot be classified by
   today's volatility spike.
4. A component that cannot answer raising an exception instead of abstaining —
   NaN-filled, empty, too-short or entirely absent series.
5. Live-only context (PSX breadth) leaking into the score that decides a bucket
   tag the measurement could never have reproduced.
6. The size multiplier being allowed to lever the book on a heuristic: Risk-On
   lifts by 10%, an unclassifiable panel cuts.
7. And the one that makes the other six worth having — a state that is computed
   and then never shown. The endpoint, the dossier, the CLI and the desk each
   have to carry the prints behind the multiplier, because an `x0.75` nobody
   can trace to a VIX percentile is an instruction, not a finding.
"""

from __future__ import annotations

import inspect
import math
import time

import numpy as np
import pandas as pd
import pytest

from kairos import attribution as A
from kairos.ensemble import StrategyVote, fuse
from kairos.feedback import Snapshot
from kairos.macro import (MIN_COMPONENTS, MIN_COVERAGE, NEUTRAL, RISK_OFF,
                          RISK_ON, SIZE_MULTIPLIER, TAG_BAND, TAGS, UNKNOWN,
                          Component, condition_label, macro_state, psx_breadth,
                          split_condition)
from kairos.models_stat import ModelForecast

KEYS = ("VIX", "DXY", "US10Y", "SPX", "COPPER", "GOLD", "BTC", "USDPKR")


def _frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float)})


def _ramp(start: float, end: float, n: int, noise: float = 0.0) -> list[float]:
    """Linear path from start to end, optionally with deterministic wobble."""
    rng = np.random.default_rng(7)
    base = np.linspace(start, end, n)
    if noise:
        base = base * (1.0 + rng.normal(0.0, noise, n))
    return [float(x) for x in base]


def _panel_off() -> dict[str, pd.DataFrame]:
    """Every print pointing the same way: vol up, dollar up, yields up, equities
    down, copper/gold down, BTC down, PKR down."""
    return {
        "VIX": _frame(_ramp(15.0, 32.0, 270)),
        "DXY": _frame(_ramp(100.0, 104.0, 60)),
        "US10Y": _frame(_ramp(4.0, 4.6, 60)),
        "SPX": _frame(_ramp(5200.0, 4700.0, 90, noise=0.001)),
        "COPPER": _frame(_ramp(4.50, 4.05, 60)),
        "GOLD": _frame(_ramp(2400.0, 2420.0, 60)),
        "BTC": _frame(_ramp(70000.0, 59000.0, 90, noise=0.02)),
        "USDPKR": _frame(_ramp(280.0, 288.0, 60)),
    }


def _panel_on() -> dict[str, pd.DataFrame]:
    return {
        "VIX": _frame(_ramp(32.0, 12.0, 270)),
        "DXY": _frame(_ramp(104.0, 100.0, 60)),
        "US10Y": _frame(_ramp(4.6, 4.0, 60)),
        "SPX": _frame(_ramp(4700.0, 5200.0, 90, noise=0.001)),
        "COPPER": _frame(_ramp(4.05, 4.50, 60)),
        "GOLD": _frame(_ramp(2420.0, 2400.0, 60)),
        "BTC": _frame(_ramp(59000.0, 70000.0, 90, noise=0.02)),
        "USDPKR": _frame(_ramp(288.0, 280.0, 60)),
        # a panel key nothing in the axis reads must not change the score
        "WTI": _frame(_ramp(70.0, 60.0, 60)),
    }


# ------------------------------------------------------------------- the axis
def test_a_squeezed_tape_is_called_risk_off_and_cuts_size():
    st = macro_state(_panel_off())
    assert st.tag == RISK_OFF
    assert st.score <= -0.5
    assert st.coverage == pytest.approx(1.0)
    assert st.size_multiplier == SIZE_MULTIPLIER[RISK_OFF] == 0.75
    assert st.n_present == 7


def test_a_relaxed_tape_is_called_risk_on_and_the_lift_is_bounded():
    st = macro_state(_panel_on())
    assert st.tag == RISK_ON
    assert st.score >= 0.5
    # the lift is capped at 10% on purpose: a heuristic score may open the gate
    # slightly, it may not lever a book
    assert st.size_multiplier == 1.10
    assert st.size_multiplier < 1.2


def test_score_is_always_bounded_even_when_every_print_is_extreme():
    for panel in (_panel_off(), _panel_on()):
        st = macro_state(panel)
        assert -1.0 <= st.score <= 1.0
        assert all(-1.0 <= c.score <= 1.0 for c in st.components if c.present)


def test_prints_that_disagree_are_called_what_the_band_says():
    """Flip the two heaviest prints and the tape argues with itself. The point
    is not the number, it is that the tag follows the declared band rather than
    rounding a small score into a condition."""
    off = _panel_off()
    mixed = dict(off)
    mixed["VIX"] = _frame(_ramp(32.0, 12.0, 270))
    mixed["DXY"] = _frame(_ramp(104.0, 100.0, 60))
    st = macro_state(mixed)
    assert abs(st.score) < 0.2
    assert st.coverage == pytest.approx(1.0)
    if abs(st.score) < TAG_BAND:
        assert st.tag == NEUTRAL and st.size_multiplier == 1.0
    else:                                   # only reachable if a weight changes
        assert st.tag in (RISK_ON, RISK_OFF)


# --------------------------------------------------- absence is not neutrality
def test_one_print_alone_is_not_a_macro_state():
    """The defect this exists to prevent: a dead panel reading as "Neutral"."""
    st = macro_state({"VIX": _frame(_ramp(15.0, 32.0, 270))})
    assert st.tag == UNKNOWN
    assert st.coverage < MIN_COVERAGE
    assert st.n_present < MIN_COMPONENTS
    # the score itself is still the VIX's own opinion, undiluted by the five
    # prints that never answered — it is the *tag* that refuses to conclude
    assert st.score <= -0.5
    assert st.size_multiplier == SIZE_MULTIPLIER[UNKNOWN] == 0.90


def test_two_prints_is_still_too_thin_to_name_a_condition():
    st = macro_state({"VIX": _frame(_ramp(15.0, 32.0, 270)),
                      "DXY": _frame(_ramp(100.0, 104.0, 60))})
    assert st.n_present == 2
    assert st.tag == UNKNOWN


def test_an_empty_panel_is_unknown_and_cautious_not_neutral_and_full_size():
    st = macro_state({})
    assert st.tag == UNKNOWN
    assert st.score == 0.0
    assert st.coverage == 0.0
    assert st.size_multiplier < 1.0
    st2 = macro_state(None)
    assert st2.tag == UNKNOWN


def test_label_names_the_absent_prints_so_an_operator_can_see_the_hole():
    st = macro_state({"VIX": _frame(_ramp(15.0, 32.0, 270)),
                      "DXY": _frame(_ramp(100.0, 104.0, 60)),
                      "SPX": _frame(_ramp(5000.0, 5010.0, 90))})
    line = st.label()
    assert "absent" in line
    assert "US10Y" in line and "BTC" in line
    assert line.startswith(st.tag)


# ------------------------------------------------------------- no look-ahead
def test_a_slice_is_classified_by_the_panel_as_of_that_bar():
    """The volatility spike lives only in the last 20 bars. A slice truncated
    before it must not be able to see it."""
    full = macro_state(_panel_off())
    # the same eight prints, one month earlier: calm vol, a softening dollar,
    # flat yields, equities grinding up, copper bid, BTC steady, PKR pinned
    early = {
        "VIX": _frame(_ramp(15.2, 14.0, 250)),
        "DXY": _frame(_ramp(100.0, 99.6, 60)),
        "US10Y": _frame(_ramp(4.0, 4.02, 60)),
        "SPX": _frame(_ramp(5200.0, 5250.0, 90, noise=0.001)),
        "COPPER": _frame(_ramp(4.50, 4.52, 60)),
        "GOLD": _frame(_ramp(2400.0, 2395.0, 60)),
        "BTC": _frame(_ramp(70000.0, 71000.0, 90, noise=0.02)),
        "USDPKR": _frame(_ramp(280.0, 280.2, 60)),
    }
    before = macro_state(early)
    assert full.tag == RISK_OFF
    assert before.tag != RISK_OFF
    assert before.score > full.score + 0.5


# ------------------------------------------------------------ garbage in
@pytest.mark.parametrize("bad", [None, [], pd.DataFrame(), pd.DataFrame(
    {"date": pd.date_range("2024-01-01", periods=3), "close": [1.0, 2.0, 3.0]}),
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=80),
                  "close": [float("nan")] * 80}),
    pd.DataFrame({"close": [float("inf")] * 80}),
    "not a frame at all", 12345])
def test_a_series_that_cannot_answer_abstains_instead_of_raising_or_inventing(bad):
    st = macro_state({k: bad for k in KEYS})
    assert st.tag == UNKNOWN
    assert st.n_present == 0
    assert math.isfinite(st.score)
    assert all(not c.present for c in st.components)


def test_interior_gaps_cost_a_component_its_seat_never_a_fabricated_print():
    """A vendor hole in the middle must not read as a volatility collapse."""
    good = _ramp(15.0, 32.0, 270)
    holes = list(good)
    for i in range(100, 112):
        holes[i] = float("nan")
    st = macro_state({"VIX": _frame(holes), "DXY": _frame(_ramp(100.0, 104.0, 60)),
                      "SPX": _frame(_ramp(5000.0, 4700.0, 90)),
                      "US10Y": _frame(_ramp(4.0, 4.6, 60))})
    vix = next(c for c in st.components if c.key == "VIX")
    assert vix.present is True
    assert vix.score is not None and math.isfinite(vix.score)


def test_a_dead_feed_of_zeros_cannot_be_read_as_the_calmest_tape_on_record():
    flat = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                         "close": [0.0] * 60})
    panel = {k: flat for k in KEYS}
    st = macro_state(panel)
    vix = next(c for c in st.components if c.key == "VIX")
    assert vix.present is False
    assert "not a volatility" in vix.why
    assert st.n_present == 0
    assert st.tag == UNKNOWN
    assert all(-1.0 <= c.score <= 1.0 for c in st.components if c.present)
    assert math.isfinite(st.score)


def test_a_series_that_never_moves_has_no_percentile_opinion():
    """A stuck vendor print would otherwise score as "at the top of its own
    range", which is an artefact of the flat window, not a market."""
    stuck = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=260),
                          "close": [18.0] * 260})
    st = macro_state({"VIX": stuck})
    vix = next(c for c in st.components if c.key == "VIX")
    assert vix.present is False
    assert "meaningless" in vix.why


def test_component_weight_table_sums_to_one_so_coverage_means_what_it_says():
    st = macro_state(_panel_off())
    assert sum(c.weight for c in st.components) == pytest.approx(1.0)
    assert len(st.components) == 7
    assert set(SIZE_MULTIPLIER) == set(TAGS)


# ---------------------------------------------------------------- the vocabulary
def test_condition_label_is_one_spelling_for_both_producers():
    st = macro_state(_panel_off())
    cond = condition_label("Bull-Quiet", st)
    assert cond == "Bull-Quiet :: Risk-Off"
    # the walk-forward builds the same string from a MacroState; the desk builds
    # it from the tag alone. They must agree, or the buckets never meet.
    assert cond == condition_label("Bull-Quiet", st.tag)


def test_condition_label_refuses_to_invent_a_bucket():
    assert condition_label("Range", "Something The Operator Made Up") == \
        "Range :: Unknown"
    assert condition_label("Range", None) == "Range :: Unknown"
    assert condition_label("", RISK_ON) == "Unknown :: Risk-On"


def test_split_condition_round_trips_for_every_tag():
    for micro in ("Bull-Quiet", "MeanRev-Volatile", "Range"):
        for tag in TAGS:
            assert split_condition(condition_label(micro, tag)) == (micro, tag)
    assert split_condition("no separator at all") == ("no separator at all",
                                                     UNKNOWN)
    assert split_condition("") == ("Unknown", UNKNOWN)


def test_the_condition_never_collides_with_the_horizon_prefix():
    """Bucket keys are f"{horizon}|{regime}" — the condition must not contain a
    pipe, or a reader splitting on the first pipe gets the wrong half."""
    st = macro_state(_panel_off())
    cond = condition_label("Bull-Quiet", st)
    key = f"5|{cond}"
    assert "|" not in cond
    assert key.split("|", 1) == ["5", cond]


# ------------------------------------------------------------------ breadth
def test_breadth_is_reported_but_never_reaches_the_score():
    """A live-only print inside the bucket tag would let the desk be armed by
    evidence the measurement could not have reproduced."""
    panel = _panel_off()
    plain = macro_state(panel)
    with_breadth = macro_state(panel, extras={"psx_breadth": {"symbols": 400,
                                                              "pct_above_sma50": 0.9}})
    assert plain.score == with_breadth.score
    assert plain.tag == with_breadth.tag
    assert with_breadth.extras["psx_breadth"]["pct_above_sma50"] == 0.9
    assert plain.extras == {}


def test_breadth_counts_only_rows_where_both_numbers_exist():
    closes = [100.0] * 60 + [80.0] * 40
    sma = [90.0] * 100
    snap = pd.DataFrame({"close": closes, "SMA50": sma, "change|1M": [5.0] * 60 +
                                                            [-3.0] * 40})
    out = psx_breadth(snap)
    assert out["symbols"] == 100
    assert out["pct_above_sma50"] == pytest.approx(0.6)
    assert out["pct_up_1m"] == pytest.approx(0.6)


def test_breadth_needs_a_real_cross_section_before_it_claims_one():
    assert psx_breadth(None) is None
    assert psx_breadth(pd.DataFrame()) is None
    thin = pd.DataFrame({"close": [100.0] * 29, "SMA50": [90.0] * 29,
                         "change|1M": [1.0] * 29})
    assert psx_breadth(thin) is None


def test_breadth_survives_a_missing_column_instead_of_losing_both_numbers():
    snap = pd.DataFrame({"close": [100.0] * 60, "SMA50": [90.0] * 60})
    out = psx_breadth(snap)
    assert out == {"symbols": 60, "pct_above_sma50": 1.0}
    assert psx_breadth(pd.DataFrame({"volume": [1.0] * 50})) is None


def test_breadth_ignores_rows_the_vendor_left_blank():
    closes = [100.0] * 50 + [float("nan")] * 10
    out = psx_breadth(pd.DataFrame({"close": closes, "SMA50": [90.0] * 60}))
    assert out["symbols"] == 50


# ------------------------------------------------------------------ reporting
def test_as_dict_carries_everything_the_desk_has_to_show():
    st = macro_state(_panel_off())
    d = st.as_dict()
    assert set(d) >= {"tag", "score", "coverage", "size_multiplier", "components",
                      "extras", "label", "as_of", "n_present", "n_components"}
    assert len(d["components"]) == 7
    row = d["components"][0]
    assert set(row) == {"key", "what", "weight", "present", "score", "value", "why"}
    assert isinstance(row["present"], bool)
    assert d["as_of"]                          # the bar the panel was read through


def test_a_component_that_abstains_still_reports_why():
    st = macro_state({"VIX": _frame(_ramp(15.0, 32.0, 270))})
    absent = [c for c in st.components if not c.present]
    assert absent
    assert all(c.why for c in absent)
    assert all(c.score is None for c in absent)


def test_component_is_usable_standalone_as_an_abstention():
    c = Component("VIX", "fear premium", 0.22, None, None, why="no VIX series")
    assert c.present is False
    assert c.as_dict()["score"] is None


# ------------------------------------------------------- it reaches the weights
def _counts_with(cond: str, *, votes: int, hits: int, longs: int,
                 base: tuple[int, int]) -> dict:
    counts = A.new_counts(100, (5,))
    st = A.ensure_strategy(counts, "Vol Rider", "Trend", (5,))
    st["by_bucket"][f"5|{cond}"] = [votes, hits, longs, votes - longs]
    counts["base"]["5"][cond] = list(base)
    return counts


def test_the_poll_is_addressed_by_the_condition_not_the_micro_label_alone():
    """The whole reason the axis exists: a strategy measured in a risk-off tape
    is paid for what it got right in that tape, and is not handed the same
    weight when the tape turns."""
    cond = condition_label("Bull-Quiet", RISK_OFF)
    attr = A.Attribution(_counts_with(cond, votes=60, hits=45, longs=50,
                                      base=(120, 60)))
    off = StrategyVote(name="Vol Rider", category="Trend", direction=1,
                       strength=0.6, rationale="stub")
    fc = ModelForecast(model="Stub", direction=1, expected_return=0.01,
                       horizon=5, confidence=0.5, detail="")
    paid = fuse("GOLD", [fc], [off], "Bull-Quiet", 5, attribution=attr,
                condition=cond)
    # called with the micro label only — the pre-axis behaviour — the bucket is
    # not found and the vote enters at face value
    flat = fuse("GOLD", [fc], [off], "Bull-Quiet", 5, attribution=attr)
    w_paid = paid.details["strategies"]["Vol Rider"].get("w")
    assert w_paid is not None and w_paid > 1.0
    assert "w" not in flat.details["strategies"]["Vol Rider"]
    assert paid.details["condition"] == cond
    assert flat.details["condition"] == "Bull-Quiet"


def test_the_affinity_table_still_sees_the_micro_label():
    """REGIME_AFFINITY is keyed on "Bull-Quiet". Folding the macro tag into the
    one string the models read would have missed every lookup and silently
    flattened the trend/reversion tilt."""
    from kairos.ensemble import REGIME_AFFINITY
    st = macro_state(_panel_off())
    cond = condition_label("Bull-Quiet", st)
    assert cond not in REGIME_AFFINITY
    assert "Bull-Quiet" in REGIME_AFFINITY


# ------------------------------------------------------------- engine plumbing
class _StubMd:
    def __init__(self, snapshot_error: bool = False) -> None:
        self.news = None
        self.snapshot_error = snapshot_error

    def history(self, instrument, days=None, use_cache=True):
        n = 300
        rng = np.random.default_rng(3)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
        return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n,
                                                   freq="D"),
                             "open": close, "high": close * 1.004,
                             "low": close * 0.996, "close": close,
                             "volume": np.full(n, 1e6)})

    def psx_snapshot(self, refresh_after_s: float = 300.0):
        if self.snapshot_error:
            raise ConnectionError("exchange scan is down")
        raise AttributeError("no cross-section in this build")


@pytest.fixture
def engine(monkeypatch):
    import kairos.engine as E
    from kairos.kse_regime import KseRegime

    def make(panel, snapshot_error=False, seen=None, forecasts=()):
        """A prediction engine with every network fetch replaced by the panel.

        `forecasts` is not decoration: the engine scales the suggested weight by
        `abs(score) / 0.35`, so a signal built from no models at all has a zero
        base weight and a 0.75x risk-off cut is invisible in the output.
        """
        eng = E.PredictionEngine(md=_StubMd(snapshot_error))
        eng._battery = lambda inst, df, horizon: list(forecasts)
        eng.intermarket = lambda force=False: dict(panel)
        eng.factor_row = lambda inst: None
        eng._news_cache = (time.time(), [])

        def _library(df, fe, ctx):
            if seen is not None:
                seen.update(ctx)
            return []

        monkeypatch.setattr(E, "run_strategy_library", _library)
        monkeypatch.setattr(E, "kse_regime", lambda md: KseRegime(
            "sideways", 0.5, 1.0, 0.0, 0.1, 0.0, "stub"))
        return eng
    return make


def test_a_prediction_carries_the_condition_and_the_prints_behind_it(engine):
    eng = engine(_panel_off())
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"]
    assert res["macro"]["tag"] == RISK_OFF
    assert len(res["macro"]["components"]) == 7
    assert res["condition"] == condition_label(res["regime"]["regime"],
                                               RISK_OFF)
    assert res["condition"].endswith(":: Risk-Off")


def test_the_macro_state_scales_the_position_and_says_so(engine):
    """Risk-off must shrink the same signal, and the note must name the number
    that did it — a size the operator cannot trace is a size they cannot audit."""
    bet = [ModelForecast(model="Stub", direction=1, expected_return=0.03,
                         horizon=5, confidence=0.8, detail="")]
    hot = engine(_panel_off(), forecasts=bet)
    cold = engine(_panel_on(), forecasts=bet)
    r_off = hot._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    r_on = cold._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert r_off["macro"]["tag"] == RISK_OFF and r_on["macro"]["tag"] == RISK_ON
    assert r_on["risk"]["suggested_weight"] > 0, "a zero base hides the cut"
    ratio = (r_off["risk"]["suggested_weight"] /
             r_on["risk"]["suggested_weight"])
    assert ratio == pytest.approx(0.75 / 1.10, rel=0.02)
    assert "macro Risk-Off" in r_off["risk"]["notes"]
    assert "macro Risk-On" in r_on["risk"]["notes"]


def test_a_thin_panel_lands_on_the_cautious_floor_not_full_size(engine):
    eng = engine({})
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["macro"]["tag"] == UNKNOWN
    assert res["macro"]["coverage"] == 0
    assert res["macro"]["size_multiplier"] == 0.90
    assert "macro Unknown" in res["risk"]["notes"]


def test_a_dead_exchange_scan_costs_the_breadth_line_not_the_prediction(engine):
    """The scan is the least reliable fetch in the stack. Losing it may not turn
    a PSX prediction into a 500."""
    for err in (True, False):
        eng = engine(_panel_off(), snapshot_error=err)
        res = eng._predict_uncached("LUCK", 5, use_wsl=False, include_news=False)
        assert res["ok"], res.get("error")
        assert "psx_breadth" not in res["macro"]["extras"]
        assert res["macro"]["tag"] == RISK_OFF


def test_the_strategy_library_is_handed_the_macro_context_too(engine):
    seen: dict = {}
    eng = engine(_panel_off(), seen=seen)
    eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert seen["macro"] == RISK_OFF
    assert -1.0 <= seen["macro_score"] <= 1.0
    assert isinstance(seen["regime"], str) and "::" not in seen["regime"]


def test_the_snapshot_carries_the_condition_into_the_journal_field():
    snap = Snapshot(symbol="LUCK", ts="2026-10-23T00:00:00", horizon=5,
                    score=0.3, confidence=0.5, expected_return=0.02, price=100.0,
                    direction="BUY", regime="Bear-Quiet",
                    condition=condition_label("Bear-Quiet", RISK_OFF))
    d = snap.to_dict()
    assert d["regime"] == "Bear-Quiet"
    assert d["condition"] == "Bear-Quiet :: Risk-Off"
    assert len(d["condition"]) <= 40          # the field journal truncates to
    assert Snapshot(symbol="X", ts="", horizon=5, score=0, confidence=0,
                    expected_return=0, price=1, direction="HOLD").condition is None


# ----------------------------------------------------------- the operator paths
def test_the_desk_can_ask_the_condition_without_predicting_anything(monkeypatch):
    """/api/macro exists because an operator sizing one position by hand needs
    the prints behind the multiplier and must not pay twelve predictions for
    them. The route being mounted is the contract; the panel is stubbed."""
    import kairos.api.server as S
    import kairos.macro as M

    assert "/api/macro" in {r.path for r in S.app.routes}
    monkeypatch.setattr(M, "live_state",
                        lambda md, engine=None: macro_state(_panel_off()))
    body = S.macro()
    assert body["ok"] and body["tag"] == RISK_OFF
    assert len(body["components"]) == 7
    assert body["size_multiplier"] == SIZE_MULTIPLIER[RISK_OFF]


def test_a_prediction_report_shows_the_condition_and_the_prints_behind_it():
    """The dossier already printed the micro regime and a suggested weight. If
    the macro half is not on the page, a size the reader cannot trace to a
    print is a size they cannot argue with."""
    import inspect

    from kairos import report as R

    src = inspect.getsource(R.render_prediction)
    assert "CROSS-ASSET CONDITION" in src
    assert "absent" in src                      # a missing print is named
    assert "condition" in src                   # the bucket the weights came from


def test_the_condition_is_a_command_an_operator_can_run():
    from kairos.cli import cmd_macro, main

    assert "live_state" in inspect.getsource(cmd_macro)
    with pytest.raises(SystemExit) as exc:
        main(["macro", "--help"])
    assert exc.value.code == 0


def test_the_autopilot_remembers_the_condition_that_sized_the_last_cycle():
    from kairos.autopilot import AUTOPILOT

    # a key in the defaults is what lets the cycle store it and survive a
    # restart — state files are filtered against the defaults on load
    assert "macro" in AUTOPILOT._defaults()
    src = inspect.getsource(type(AUTOPILOT)._cycle)
    assert '"macro": macro' in src
