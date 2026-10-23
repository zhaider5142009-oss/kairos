"""The doctrine that binds: every declared rule is read by something, or admits
it is not.

Seven defects these guard:

1. `describe()` printed the whole declared doctrine as though each line governed
   the book, and the CLI panel, the command plane and the LLM council prompt all
   quote it. Nine keys — the ATR stop/target/trail multipliers, `volume_min_ratio`,
   `min_market_cap_pkr`, `max_correlation`, `earnings_blackout_days`, `fee_pct`,
   `tax_pct` — were read by nothing. The worst was the stop claim: the doctrine
   said "stop = entry - 2*ATR" while the autopilot took its stop from the
   prediction's 10th/90th percentile band, so the operator and the model were
   both told a *method* that did not exist. `ENFORCED` / `NOT_ENFORCED` now
   classify every key, `audit()` reports the split, and `describe()` says which
   leg comes from where.
2. `require_above_sma50`, `trend_sma_period`, `rsi_overbought` and `rsi_oversold`
   were declared for eighteen months and read by nothing: rules 3 and 4 of the
   file header ("trend is your friend", "buy on fear, sell on greed") were
   comments. They now gate entries in `check_plan`.
3. `max_open_positions` and `max_hold_days` were the same shape — a cap with no
   reader. The name cap is enforced at the entry gate and an unreadable book is
   refused, not assumed to fit; the hold cap is a second maturity brake, and the
   tighter of it and the autopilot's own horizon governs.
4. `min_avg_volume` and `psx_min_adv_shares` were one liquidity floor with two
   names and one reader, so tightening the other knob changed nothing.
   `adv_share_floor` resolves the pair the way `daily_loss_fraction` does.
5. An opening order whose stop was missing passed the plan check silently. The
   band percentile was treated as decoration, so a prediction with no 10th
   percentile reached the venue with `stop_loss=None`, `_manage_exit` had no hard
   stop to test and no `stop0` for the ratchet to seed from, and the only ways
   out were a signal flip or the maturity date. `check_plan` now refuses it.
6. Two of those fixes were defects on arrival. `fugu.py` read `plan["failures"]`
   from a check_plan that had no such key — a `KeyError` on every doctrine
   refusal — and once the cap and trend checks existed the advisor callers could
   not answer the new questions, so a fail-closed control muted the conductor
   instead of guarding it. Hence the three-way book state: a count, `None` (a
   book that failed to read → refuse), and `NO_BOOK` (no book view at all →
   report the cap as not evaluated).
7. Wiring the filters was not the same as applying them: the trend and heat rows
   were appended to `checks` *after* the verdict had been accumulated, so each
   one printed its refusal inside a plan that still returned `passed: True`. The
   first run of these gates failed that test — a control that reports a failure
   without acting on it is the most dangerous shape here, because it looks like
   a control and reads in the log like one too.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pandas as pd
import pytest

from kairos import fugu as FU
from kairos import rules as R
from kairos.autopilot import Autopilot
from kairos.execution import PaperBroker
from kairos.features import doctrine_snapshot
from kairos.universe import Instrument


# ------------------------------------------------------------------ the guard
# The two classification tables are claims about other code, so they are not
# evidence of it. Everything below reads the module's parsed literals with those
# tables removed — otherwise `ENFORCED` would vouch for itself.
_CLAIM_TABLES = {"ENFORCED", "NOT_ENFORCED"}
_PKG = pathlib.Path(R.__file__).parent


def _module_source(name: str) -> str:
    return (_PKG / name).read_text(encoding="utf-8")


def _string_literals(name: str) -> list[str]:
    """Every string literal in a package module, minus the claim tables."""
    tree = ast.parse(_module_source(name), filename=name)
    out: list[str] = []
    for stmt in tree.body:
        targets = (stmt.targets if isinstance(stmt, ast.Assign)
                   else [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id in _CLAIM_TABLES for t in targets):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.append(node.value)
    return out


def _read_sites(name: str, key: str) -> int:
    return sum(1 for s in _string_literals(name) if s == key)


def _package_modules() -> list[str]:
    return sorted(p.name for p in _PKG.glob("*.py"))


@pytest.fixture
def doctrine(monkeypatch):
    """Install an effective doctrine built from the shipped defaults."""
    def _install(**over):
        merged = dict(R.DEFAULT_RULES, **over)
        monkeypatch.setattr(R, "rules", lambda: merged)
        return merged
    return _install


# ------------------------------------------------------- 1. classification
def test_every_declared_key_is_classified():
    """A key may be unread, but it may not be unclassified."""
    audit = R.audit()
    assert audit["unclassified"] == [], (
        f"{audit['unclassified']} declared in neither ENFORCED nor NOT_ENFORCED")
    assert audit["counts"]["declared"] >= len(R.DEFAULT_RULES)
    assert (audit["counts"]["enforced"] + audit["counts"]["inert"]
            == audit["counts"]["declared"])


def test_the_two_tables_do_not_claim_opposite_things():
    assert not set(R.ENFORCED) & set(R.NOT_ENFORCED)


@pytest.mark.parametrize("key,modules", sorted(R.ENFORCED.items()))
def test_an_enforcement_claim_points_at_a_real_read(key, modules):
    """`ENFORCED` says which file reads the key; the file is asked."""
    assert modules, f"{key} claims enforcement with nowhere to look"
    for name in modules:
        sites = _read_sites(name, key)
        if name == "rules.py" and key in R.DEFAULT_RULES:
            # one hit is the declaration itself — enforcement needs a reader too
            assert sites >= 2, f"{name} declares {key} but never reads it"
        else:
            assert sites >= 1, f"{key} is not read in {name}"


@pytest.mark.parametrize("key", sorted(R.NOT_ENFORCED))
def test_a_not_enforced_admission_is_still_true(key):
    """The other direction: once something reads it, the admission is a lie."""
    in_rules = _read_sites("rules.py", key)
    assert in_rules == (1 if key in R.DEFAULT_RULES else 0), (
        f"{key} is now read in rules.py but still declared not enforced")
    elsewhere = {m: _read_sites(m, key) for m in _package_modules()
                 if m != "rules.py" and _read_sites(m, key)}
    assert not elsewhere, (
        f"{key} is read in {sorted(elsewhere)} — move it to ENFORCED with the "
        "file that reads it")


def test_audit_names_a_real_reason_for_every_inert_key():
    for row in R.audit()["declared_not_enforced"]:
        assert len(row["why"].strip()) > 20, f"{row['key']} has no real reason"


# ---------------------------------------------------- 2. one floor, two names
def test_the_tighter_share_floor_governs_and_says_which():
    floor, src = R.adv_share_floor({"psx_min_adv_shares": 20_000,
                                    "min_avg_volume": 50_000})
    assert (floor, src) == (20_000.0, "psx_min_adv_shares")
    floor, src = R.adv_share_floor({"psx_min_adv_shares": 80_000,
                                    "min_avg_volume": 50_000})
    assert (floor, src) == (50_000.0, "min_avg_volume")


def test_an_unreadable_share_floor_screens_everything_out():
    """Zero would let every name through; the floor cannot be assumed away."""
    for bad in ("many", float("nan"), -1):
        floor, src = R.adv_share_floor({"psx_min_adv_shares": bad,
                                        "min_avg_volume": 50_000})
        assert math.isinf(floor) and "not a" in src, f"{bad!r} read as no floor"


def test_no_configured_floor_is_not_a_free_pass():
    floor, src = R.adv_share_floor({"unrelated": 1})
    assert math.isinf(floor) and src == "no share floor configured"


def test_tightening_either_key_changes_the_answer(doctrine):
    """Both knobs bind, so both are worth turning — the point of resolving them."""
    doctrine(psx_min_adv_shares=100_000, min_avg_volume=50_000)
    assert R.adv_share_floor()[0] == 50_000.0
    doctrine(min_avg_volume=25_000)
    assert R.adv_share_floor()[0] == 25_000.0


# ----------------------------------------------------------- 3. the name cap
def test_the_name_cap_binds_at_the_boundary():
    assert R.open_slot(7, {"max_open_positions": 8})["allowed"] is True
    slot = R.open_slot(8, {"max_open_positions": 8})
    assert slot["allowed"] is False and "full" in slot["why"]
    assert R.open_slot(9, {"max_open_positions": 8})["allowed"] is False


def test_an_unreadable_book_is_refused_not_assumed_empty():
    """Holding nothing and failing to count are different answers."""
    slot = R.open_slot(None, {"max_open_positions": 8})
    assert slot["allowed"] is False and "refused" in slot["why"]


@pytest.mark.parametrize("bad", ["eight", float("nan"), 0, -1, None])
def test_an_unusable_cap_is_a_refusal_not_an_open_door(bad):
    assert R.open_slot(0, {"max_open_positions": bad})["allowed"] is False


# ---------------------------------------------------------- 4. hold maturity
def test_the_tighter_exit_timer_governs():
    assert R.maturity_days(5.0, {"max_hold_days": 60}) == (
        5.0, "autopilot.horizon_days")
    assert R.maturity_days(90.0, {"max_hold_days": 60}) == (
        60.0, "rules.max_hold_days")


def test_an_unreadable_cap_does_not_become_an_infinite_hold():
    """The horizon is the exit that works, so it keeps binding."""
    due, src = R.maturity_days(7.0, {"max_hold_days": "sixty"})
    assert due == 7.0 and "horizon governs" in src
    assert R.maturity_days(7.0, {"max_hold_days": -5})[0] == 7.0
    assert R.maturity_days("n/a", {"max_hold_days": 60})[0] == 5.0


# ------------------------------------------------------- 5. stops and geometry
def _plan(**over):
    """A sound long at Rs 100, stop 95, target 115: 1:3, 0.05% of equity."""
    kwargs = dict(entry=100.0, stop=95.0, target=115.0, equity=1_000_000.0,
                  confidence=0.6, signal_dir=1, qty=100.0,
                  n_held=0, sma=90.0, rsi14=55.0)
    kwargs.update(over)
    return R.check_plan(**kwargs)


def test_a_silent_plan_check_is_not_a_passing_one():
    """The regression this whole file exists for: no rows means nothing ran."""
    plan = _plan()
    assert plan["passed"] is True
    rows = {c["rule"] for c in plan["checks"]}
    assert {"Stop defined", "Risk:Reward", "Confidence",
            "Trend", "Heat", "Name cap"} <= rows


def test_an_opening_order_without_a_stop_is_refused():
    plan = _plan(stop=None)
    assert plan["passed"] is False
    assert "Stop defined" in plan["failures"]


def test_the_stop_rule_applies_to_opens_not_exits():
    """A close is never refused for lacking a stop it already exercised."""
    plan = _plan(stop=None, signal_dir=0)
    assert "Stop defined" not in [c["rule"] for c in plan["checks"]]


def test_a_missing_reward_leg_is_reported_not_assumed():
    plan = _plan(target=None)
    assert "Risk:Reward" in plan["blind"]
    assert plan["passed"] is True


def test_failures_is_a_real_key_on_every_refusal():
    """fugu.py reads plan["failures"]; it used to exist only in the docstring."""
    for kwargs in ({"stop": None}, {"sma": None}, {"rsi14": None},
                   {"n_held": 99}, {"confidence": 0.1},
                   {"rsi14": 80.0, "sma": 120.0}):
        plan = _plan(**kwargs)
        assert not plan["passed"], kwargs
        assert isinstance(plan["failures"], str) and plan["failures"], kwargs


# ------------------------------------------------------- 6. trend and heat
def test_a_filter_that_fails_changes_the_verdict_not_only_the_print():
    """The rows used to be appended after `passed` was accumulated, so a printed
    refusal still cleared the order — a fail-open that looked like a filter."""
    plan = _plan(sma=105.0)
    assert plan["passed"] is False
    assert [c["rule"] for c in plan["checks"] if not c["passed"]] == ["Trend"]


def test_a_long_below_the_trend_mean_is_refused():
    assert _plan(sma=105.0)["passed"] is False
    assert _plan(sma=100.0)["passed"] is True       # "at or above"
    assert "Trend" in _plan(sma=105.0)["failures"]


def test_a_long_with_no_readable_mean_is_refused_not_waved_through():
    """A screen that cannot see answers reject, like `_psx_liquid`."""
    for missing in (None, float("nan")):
        plan = _plan(sma=missing)
        assert plan["passed"] is False, missing
    assert "Trend" in _plan(sma=None)["blind"]


def test_the_trend_filter_is_declared_for_longs_only():
    """`describe()` says so; the code must not imply a general trend rule."""
    assert _plan(signal_dir=-1, stop=105.0, target=85.0,
                 sma=110.0)["passed"] is True


def test_the_heat_gate_is_symmetric():
    assert _plan(rsi14=75.0)["passed"] is False      # long into overbought
    assert _plan(rsi14=65.0)["passed"] is True
    short_hot = _plan(signal_dir=-1, stop=105.0, target=85.0, rsi14=25.0)
    assert short_hot["passed"] is False             # short into oversold
    assert _plan(signal_dir=-1, stop=105.0, target=85.0,
                 rsi14=35.0)["passed"] is True


def test_an_unreadable_rsi_is_a_refusal_named_in_the_blind():
    plan = _plan(rsi14=None)
    assert plan["passed"] is False and "Heat" in plan["blind"]


def test_turning_the_trend_filter_off_turns_off_the_check(doctrine):
    doctrine(require_above_sma50=False)
    assert _plan(sma=110.0)["passed"] is True
    assert "Trend" not in [c["rule"] for c in _plan(sma=110.0)["checks"]]


def test_a_heat_gate_disabled_in_both_directions_stops_asking_for_an_rsi(
        doctrine):
    """over=100 and under=0 mean "no heat rule", so no number is needed."""
    doctrine(rsi_overbought=100, rsi_oversold=0)
    plan = _plan(rsi14=None)
    assert plan["passed"] is True and "Heat" not in plan["blind"]


# ------------------------------------------------------------- 7. cap in gate
def test_the_name_cap_is_enforced_inside_the_plan_check(doctrine):
    doctrine(max_open_positions=8)
    assert _plan(n_held=7)["passed"] is True
    plan = _plan(n_held=8)
    assert plan["passed"] is False and "Name cap" in plan["failures"]


def test_an_unreadable_book_refuses_the_entry_inside_the_plan_check(doctrine):
    doctrine(max_open_positions=8)
    plan = _plan(n_held=None)
    assert plan["passed"] is False
    assert "Name cap" in plan["failures"] and "?" in plan["failures"]


def test_a_caller_with_no_book_view_abstains_instead_of_blindly_refusing(
        doctrine):
    """The conductor prices one name and never holds the position list. A cap
    that only ever passes callers which assume zero is theatre, and one that
    always refuses mutes the advisor. So it reports the gap instead."""
    doctrine(max_open_positions=8)
    plan = _plan(n_held=R.NO_BOOK)
    assert plan["passed"] is True
    assert "Name cap" in plan["blind"]
    assert "Name cap" not in [c["rule"] for c in plan["checks"]]


# --------------------------------------------------- 8. the snapshot itself
def _frame(closes):
    return pd.DataFrame({"close": [float(c) for c in closes],
                         "volume": [1_000.0] * len(closes)})


def test_the_snapshot_returns_the_numbers_the_doctrine_gates_on():
    doc = doctrine_snapshot(_frame(range(60, 120)))
    assert doc["close"] == pytest.approx(119.0)
    assert doc["sma"] == pytest.approx(sum(range(70, 120)) / 50.0)
    assert doc["rsi14"] is not None and 0.0 <= doc["rsi14"] <= 100.0


def test_two_bars_of_history_is_not_reported_as_a_neutral_reading():
    """`_rsi` fills 50.0 for a series too short for its own min_periods. A gate
    that trusts it buys a name the trend filter never looked at."""
    doc = doctrine_snapshot(_frame([100.0, 101.0]))
    assert doc["close"] == 101.0
    assert doc["sma"] is None and doc["rsi14"] is None


def test_an_unreadable_period_leaves_the_mean_unset():
    doc = doctrine_snapshot(_frame(range(60, 120)), trend_period="fifty")
    assert doc["sma"] is None
    assert doc["rsi14"] is not None     # the heat filter still has its number


def test_a_zero_or_negative_print_gives_the_gate_nothing_to_trust():
    blank = {"close": None, "sma": None, "rsi14": None}
    for closes in ([100.0, 0.0], [-5.0, -3.0]):
        assert doctrine_snapshot(_frame(closes)) == blank


def test_a_frame_the_gate_cannot_read_is_answered_not_raised():
    blank = {"close": None, "sma": None, "rsi14": None}
    assert doctrine_snapshot(pd.DataFrame({"last": [1.0, 2.0]})) == blank
    assert doctrine_snapshot(None) == blank
    assert doctrine_snapshot(_frame([])) == blank


# ------------------------------------------------------ 9. what we tell people
def test_the_doctrine_paragraph_claims_no_method_that_does_not_exist():
    d = R.describe()
    assert "ATR" not in d, "the stop method is a band percentile, not ATR"
    assert "percentile" in d


def test_the_doctrine_paragraph_admits_what_it_does_not_apply():
    d = R.describe()
    assert "NOT enforced" in d
    for key in R.NOT_ENFORCED:
        assert key in d, f"{key} is not mentioned in the doctrine paragraph"


def test_the_doctrine_paragraph_states_the_cap_and_the_filters():
    d = R.describe()
    assert "8 simultaneous names" in d
    assert "50-bar mean" in d and "RSI" in d
    assert "20-day average shares" in d


def test_an_unreadable_share_floor_is_described_as_unreadable(doctrine):
    doctrine(psx_min_adv_shares="many")
    assert "unreadable share floor" in R.describe()


def test_the_audit_splits_the_book_of_rules():
    audit = R.audit()
    assert "max_hold_days" in {row["key"] for row in audit["enforced"]}
    inert = {row["key"] for row in audit["declared_not_enforced"]}
    assert "min_market_cap_pkr" in inert
    for row in audit["enforced"]:
        assert row["read_in"], row["key"]


# --------------------------------------------- 10. the callers really ask
def _res(**over):
    res = {"price": 100.0,
           "signal": {"direction": "BUY", "score": 0.5, "confidence": 0.8,
                      "horizon_days": 5},
           "models": [{"direction": 1}, {"direction": 1}],
           "regime": {"regime": "trending"},
           "confidence_band": {"p10": 95.0, "p90": 115.0},
           "targets": {"5d": 115.0},
           "doctrine": {"close": 100.0, "sma": 90.0, "rsi14": 55.0}}
    res.update(over)
    return res


def test_the_conductor_is_gated_by_the_same_trend_filter():
    """The advisor that never looked at the mean approved what the gate refuses."""
    ok = FU.conduct("GOLD", res=_res())
    assert ok["action"] == "EXECUTE", ok["reasons"]
    cold = FU.conduct("GOLD", res=_res(doctrine={"close": 100.0, "sma": 110.0,
                                                 "rsi14": 55.0}))
    assert cold["action"] == "PASS"
    assert any("Trend" in r for r in cold["reasons"])


def test_the_conductor_is_gated_by_the_same_heat_filter():
    hot = FU.conduct("GOLD", res=_res(doctrine={"close": 100.0, "sma": 90.0,
                                                "rsi14": 78.0}))
    assert hot["action"] == "PASS"
    assert any("Heat" in r for r in hot["reasons"])


def test_the_conductor_reports_the_controls_it_could_not_run():
    """No posture means no book to count — say so, do not assume zero."""
    assert "Name cap" in FU.conduct("GOLD", res=_res())["plan_blind"]
    held = FU.conduct("GOLD", res=_res(),
                      posture={"equity": 1_000_000.0, "n_positions": 2})
    assert "Name cap" not in held["plan_blind"]
    assert held["action"] == "EXECUTE", held["reasons"]


def test_a_book_that_cannot_be_read_stops_the_conductor():
    verdict = FU.conduct("GOLD", res=_res(),
                         posture={"equity": 1_000_000.0, "n_positions": None})
    assert verdict["action"] == "PASS"
    assert any("Name cap" in r for r in verdict["reasons"])


def test_an_advisor_refusal_never_raises_on_a_missing_key():
    for over in ({"doctrine": {"close": 100.0}},
                 {"doctrine": {"close": 100.0, "sma": 90.0}}):
        verdict = FU.conduct("GOLD", res=_res(**over))
        assert verdict["action"] == "PASS", over
        assert verdict["reasons"], over


def test_the_engine_snapshot_reaches_the_plan_check_through_the_autopilot(
        tmp_path):
    """The autopilot must read the numbers off the prediction, not invent them."""
    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a._save()
    broker = PaperBroker(capital=1_000_000.0)
    inst = Instrument(key="GOLD", name="Gold", asset_class="commodity",
                      sector="materials", currency="USD", market="GLOBAL")
    act, why = a._execute(inst, "BUY", 100.0, 100.0,
                          dict(_res(), doctrine={}), broker,
                          live=False, close=False)
    assert act == "PLANCHECK"
    assert "Trend" in why and "not evaluated" in why
    good, _ = a._execute(inst, "BUY", 100.0, 100.0, _res(), broker,
                         live=False, close=False)
    assert good != "PLANCHECK"


def test_the_predict_payload_carries_the_doctrine_numbers():
    """`predict` is the only place in the cycle holding the price frame."""
    from kairos import engine as EN
    src = _module_source("engine.py")
    assert '"doctrine"' in src and "doctrine_snapshot" in src
    assert callable(EN._trend_period)
