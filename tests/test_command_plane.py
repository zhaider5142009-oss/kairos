"""The command plane may plan, but it may never move money or invent a number.

`kairos/command.py` turns plain language into a multi-step plan over the quant
tools and runs it against the live engine. Three properties have to survive
future edits, and none of them is obvious from reading the module:

* the registry contains no mutating tool, so the *worst* a confused planner can
  do is run an analysis — including the model-planned path, whose choices are
  re-validated against the registry rather than trusted;
* a request that asks for movement is refused before any planning happens, so
  the refusal does not depend on the planner understanding it;
* every figure in an answer is rendered by the tool that produced it, and the
  plane answers correctly with no provider enabled at all.

One of these caught a real problem while the module was being written: the
obvious implementation of the backtest tool was `run_paper_session()`, which
also submits a demo order through the paper broker. A question in the desk chat
would then have left a fill in the book the same chat reports on. The tool calls
`walk_forward()` directly for that reason, and a test here keeps it that way.
"""

from __future__ import annotations

import inspect
import time

import pytest

from kairos import command as C


def _fake(name: str, run, line=None, args=(), seconds: float = 0.1) -> C.Tool:
    return C.Tool(name=name, what=f"test {name}", run=run,
                  line=line or (lambda d: f"{name} ran"), args=args,
                  limits={}, seconds=seconds)


# ------------------------------------------------------------------ the plan
def test_a_plain_question_compiles_a_two_expert_plan():
    p = C.plan("is OGDC a buy this week")
    assert [s["tool"] for s in p["steps"]] == ["predict", "conduct"]
    assert p["steps"][0]["args"]["symbol"] == "OGDC"
    assert p["steps"][0]["args"]["horizon"] == 5
    assert not p["movement"]


def test_a_screen_feeds_the_conductor_from_its_own_output():
    """The point of a plan: step two's symbol is step one's result."""
    p = C.plan("screen the PSX and tell me which name is worth trading")
    tools = [s["tool"] for s in p["steps"]]
    assert tools == ["scan", "conduct", "posture"]
    assert p["steps"][1]["args"]["symbol"].startswith("$0.")


def test_an_unfamiliar_ask_plans_nothing_instead_of_guessing():
    assert C.plan("hello there")["steps"] == []


# ------------------------------------------------------------------ the guard
# A tool is only dangerous through what its code calls. Checking the prose
# descriptions instead would fail on words the registry earns honestly: the
# read-only `livebook`, `posture` reporting the kill-switch state, `backtest`
# promising "no orders placed". So this scans the callables, not the marketing.
MONEY_CALLS = ("place_order", "close_position", "modify_sl", ".submit(",
               "engage(", "release(", "make_sell_ticket", "livebook.add",
               "livebook.remove", "livebook.set_stop", "arm_live", "set_armed",
               ".connect(", "shutdown(", "build_ticket")
MUTATING_NAMES = ("place", "submit", "execute", "cancel", "arm", "withdraw",
                  "transfer", "buy", "sell", "open_position", "close")


def test_the_registry_has_no_way_to_move_money():
    for name, tool in C.TOOLS.items():
        assert not any(w in name.lower() for w in MUTATING_NAMES), name
        source = inspect.getsource(tool.run)
        for call in MONEY_CALLS:
            assert call not in source, f"{name} can call {call}"


def test_the_plane_never_imports_the_execution_venue():
    """The tools reach the engine, not the broker. An import of the MT5 bridge or
    the order-ticket builder here would mean the desk can touch a venue."""
    source = inspect.getsource(C)
    for forbidden in ("place_order", "build_ticket", "close_position",
                      "from .mt5", "import mt5"):
        assert forbidden not in source


@pytest.mark.parametrize("ask", [
    "place an order on OGDC", "buy HBL now", "arm live mode",
    "cancel my order", "switch to live", "withdraw my money",
])
def test_movement_asks_are_recognised(ask):
    assert C._asks_for_movement(ask), ask


@pytest.mark.parametrize("ask", [
    "should I buy HBL now?", "is it a good time to sell OGDC",
    "what do you think — buy or pass", "can gold run this month",
])
def test_questions_about_movement_are_answered_not_refused(ask):
    """The desk exists to answer exactly these. Refusing them would turn the
    guard into a refusal to advise, which is worse than no guard at all."""
    assert not C._asks_for_movement(ask), ask


def test_a_movement_ask_is_refused_before_any_step_runs(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a refused request must not reach a tool")

    monkeypatch.setattr(C, "execute", boom)
    monkeypatch.setattr(C, "plan", boom)
    out = C.run("place an order on OGDC")
    assert out["intent"] == "refused"
    assert out["steps"] == []
    assert "no order" in out["answer"] or "cannot move money" in out["answer"].lower()


def test_buy_and_sell_asis_are_not_movement():
    """The verbs that make a market question must not trip the guard."""
    assert not C._asks_for_movement("is OGDC a buy this week")
    assert not C._asks_for_movement("should I sell HBL into the rally")


# ------------------------------------------------------------------ execution
def test_a_symbol_the_universe_does_not_have_is_rejected_not_run(monkeypatch):
    called = []
    monkeypatch.setitem(C.TOOLS, "conduct",
                        _fake("conduct", lambda **kw: called.append(kw) or {"ok": True},
                              args=("symbol",)))
    out = C.execute([{"tool": "conduct", "args": {"symbol": "NOTAREALTICKER"}}])
    assert out[0]["ok"] is False
    assert "known instrument" in out[0]["error"]
    assert called == []


def test_a_reference_resolves_into_the_previous_steps_output(monkeypatch):
    seen = {}

    def scan(**kw):
        return {"ok": True, "top": [{"name": "OGDC"}, {"name": "HBL"}]}

    def conduct(**kw):
        seen.update(kw)
        return {"ok": True, "action": "PASS"}

    monkeypatch.setitem(C.TOOLS, "scan", _fake("scan", scan, args=("top",)))
    monkeypatch.setitem(C.TOOLS, "conduct", _fake("conduct", conduct, args=("symbol",)))
    out = C.execute([{"tool": "scan", "args": {"top": 2}},
                     {"tool": "conduct", "args": {"symbol": "$0.top.0.name"}}])
    assert out[1]["ok"] is True
    assert seen == {"symbol": "OGDC"}


def test_an_argument_a_tool_does_not_take_fails_that_step_only(monkeypatch):
    monkeypatch.setitem(C.TOOLS, "record", _fake("record", lambda **kw: {"ok": True}))
    monkeypatch.setitem(C.TOOLS, "posture", _fake("posture", lambda **kw: {"ok": True}))
    out = C.execute([{"tool": "record", "args": {"capital": 5}},
                     {"tool": "posture", "args": {}}])
    assert out[0]["ok"] is False and "takes no argument" in out[0]["error"]
    assert out[1]["ok"] is True


def test_the_budget_stops_the_plan_rather_than_the_desk(monkeypatch):
    def slow(**kw):
        time.sleep(0.25)
        return {"ok": True}

    monkeypatch.setitem(C.TOOLS, "slow", _fake("slow", slow))
    monkeypatch.setitem(C.TOOLS, "quick", _fake("quick", lambda **kw: {"ok": True}))
    out = C.execute([{"tool": "slow", "args": {}}, {"tool": "quick", "args": {}}],
                    budget_s=0.05)
    assert out[0]["ok"] is True            # the step already running finishes
    assert out[1]["ok"] is False and "budget" in out[1]["error"]


def test_a_tool_that_raises_is_reported_not_propagated(monkeypatch):
    def explode(**kw):
        raise RuntimeError("venue down")

    monkeypatch.setitem(C.TOOLS, "boom", _fake("boom", explode))
    out = C.execute([{"tool": "boom", "args": {}}])
    assert out[0]["ok"] is False
    assert "venue down" in out[0]["error"]


# ------------------------------------------------------------------ the answer
def test_the_answer_carries_the_numbers_the_tools_produced(monkeypatch):
    monkeypatch.setitem(C.TOOLS, "record", _fake(
        "record", lambda **kw: {"ok": True, "hit_rate": 0.61},
        line=lambda d: f"hit rate {d['hit_rate'] * 100:.0f}%"))
    out = C.run("how is your track record", use_model=False)
    assert "hit rate 61%" in out["answer"]


def test_no_provider_still_answers(monkeypatch):
    """A suspended account costs polish, never truth."""
    import kairos.llm as L

    def never(*_a, **_k):
        raise AssertionError("with no provider enabled the plane must not call a model")

    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: False)
    monkeypatch.setattr(L.ROUTER, "chat", never)
    monkeypatch.setitem(C.TOOLS, "posture", _fake(
        "posture", lambda **kw: {"ok": True, "mode": "paper", "cycles": 3},
        line=lambda d: f"running in {d['mode']} mode through {d['cycles']} cycles, "
                       f"entries are open"))
    monkeypatch.setattr(C, "plan", lambda text: {
        "intent": "posture", "steps": [{"tool": "posture", "args": {}, "why": "x"}],
        "symbols": [], "movement": False})
    out = C.run("what is your posture", use_model=True)
    assert out["answer"].startswith("running in paper mode through 3 cycles")


# ------------------------------------------------------------ the model path
class _Resp:
    def __init__(self, text, ok=True):
        self.ok, self.text, self.provider, self.model = ok, text, "stub", "stub"


def test_the_model_cannot_choose_a_tool_that_is_not_in_the_registry(monkeypatch):
    import kairos.llm as L

    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: True)

    def chat(prompt, **kw):
        return _Resp('{"steps": [{"tool": "place_order", "args": {}},'
                     '{"tool": "record", "args": {}, "why": "grade it"},'
                     '{"tool": "predict", "args": {"nonsense": 1}}]}')

    monkeypatch.setattr(L.ROUTER, "chat", chat)
    steps, why = C._llm_plan("anything")
    assert [s["tool"] for s in steps] == ["record"]
    assert why == ""


def test_a_model_plan_with_nothing_valid_leaves_the_rule_plan_in_charge(monkeypatch):
    import kairos.llm as L

    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: True)
    monkeypatch.setattr(L.ROUTER, "chat", lambda prompt, **kw: _Resp('{"steps": []}'))
    steps, why = C._llm_plan("anything")
    assert steps == []
    assert "chose nothing" in why


def test_the_model_is_not_asked_when_no_provider_is_enabled(monkeypatch):
    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: False)
    monkeypatch.setattr(C, "TOOLS", {})
    steps, why = C._llm_plan("anything")
    assert steps == []
    assert "no model provider" in why


# ------------------------------------------------- the four empties stay distinct
def _plane_empty(monkeypatch, chat):
    """Run `run()` with a rule plan that matches nothing and a given model reply."""
    import kairos.llm as L

    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: True)
    monkeypatch.setattr(L.ROUTER, "chat", chat)
    monkeypatch.setattr(C, "plan", lambda text: {
        "intent": "none", "steps": [], "symbols": [], "movement": False})
    return C.run("zzz", use_model=False)


def test_a_dead_provider_is_reported_as_dead_not_as_a_hard_question(monkeypatch):
    """`ok=False` is an account or an outage — something the operator can fix.
    One sentence for all four empty answers made an outage read as a question
    outside the toolset, which is the difference between calling the exchange
    and rewriting the prompt."""
    out = _plane_empty(monkeypatch, lambda prompt, **kw: _Resp("", ok=False))
    assert out["planner"] == "model-empty"
    assert "did not answer" in out["planner_note"]
    assert "did not answer" in out["answer"]


def test_an_answer_that_is_not_json_is_named_as_such(monkeypatch):
    out = _plane_empty(monkeypatch, lambda prompt, **kw: _Resp("I cannot help"))
    assert "not as a step list" in out["planner_note"]


def test_a_model_that_proposes_only_unknown_tools_is_not_the_same_as_silence(monkeypatch):
    out = _plane_empty(
        monkeypatch,
        lambda prompt, **kw: _Resp('{"steps": [{"tool": "place_order", "args": {}}]}'))
    assert "does not have" in out["planner_note"]


def test_a_crashed_planner_says_it_crashed(monkeypatch):
    """`except Exception: return []` was the worst of the four: a bug in the
    planner looked exactly like a model that had nothing to say."""
    def boom(prompt, **kw):
        raise RuntimeError("socket went away")

    out = _plane_empty(monkeypatch, boom)
    assert "RuntimeError" in out["planner_note"]
    assert out["planner"] == "model-empty"


def test_the_rule_plan_never_blames_the_model(monkeypatch):
    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: True)
    monkeypatch.setattr(C, "plan", lambda text: {
        "intent": "record", "steps": [{"tool": "record", "args": {}, "why": "x"}],
        "symbols": [], "movement": False})
    monkeypatch.setattr(C, "execute", lambda steps, budget: [])
    out = C.run("how is your record", use_model=False)
    assert out["planner"] == "rule" and out["planner_note"] == ""


def test_the_doctrine_family_reaches_the_rule_router():
    """The router named the schools — Wyckoff, Dalio, CAN SLIM — but not the
    plain question, so "what rules are you bound by?" fell to the model and the
    desk credited the model with an answer the rules table had given."""
    for q in ("what rules are you bound by?", "which rules do you enforce?",
              "what are your risk limits?", "is there a position cap?",
              "your doctrine", "what governs your entries?"):
        p = C.plan(q)
        assert [s["tool"] for s in p["steps"]] == ["doctrine"], q


def test_a_forecast_question_is_not_stolen_by_the_doctrine_branch():
    """Widening the doctrine triggers must not eat the ordinary ask."""
    for q in ("is OGDC a buy this week", "where is gold heading",
              "best PSX names to trade", "check your health"):
        assert "doctrine" != C.plan(q)["intent"], q


def test_a_placeholder_key_is_not_counted_as_a_provider(monkeypatch):
    """`enabled: true` next to `sk-or-v1-YOUR-KEY-1` is every stranger's first
    run, not a council. Treat it as one and the plane pays a provider round
    trip for a reply that can only come back as the tool's own sentence."""
    from kairos.config import CFG

    monkeypatch.setattr(CFG, "openrouter_keys", lambda: [])
    monkeypatch.setattr(CFG, "kimi_key", lambda: "")
    assert C._a_provider_is_enabled() is False
    monkeypatch.setattr(CFG, "kimi_key", lambda: "sk-" + "0" * 24)
    assert C._a_provider_is_enabled() is True


def test_the_voice_pass_is_dropped_once_the_budget_is_spent(monkeypatch):
    """The budget is the operator's promise, and the experts get it first: a
    plan that used its 25 seconds must not add a rephrase on top."""
    monkeypatch.setattr(C, "_a_provider_is_enabled", lambda: True)
    line = ("Gold (GOLD) at 2,412.10 USD: BEARISH, -1.20% over 5d, P(up) 38%, "
            "confidence 71%, regime trending-down, suggested size 4.0%")

    def boom(*a, **k):
        raise AssertionError("asked a provider after the budget was gone")

    monkeypatch.setattr("kairos.llm.ROUTER.chat", boom)
    assert C._voice("where is gold this week", line, 0.5) == line

    monkeypatch.setattr("kairos.llm.ROUTER.chat",
                        lambda *a, **k: type("R", (), {"ok": True, "text": "  softer  "})())
    assert C._voice("where is gold this week", line, 20.0) == "softer"


# ------------------------------------------------------------------ drift
def test_the_registry_declares_exactly_the_arguments_its_tools_accept():
    """A renamed parameter would silently make a tool unplannable."""
    for name, tool in C.TOOLS.items():
        assert set(inspect.signature(tool.run).parameters) == set(tool.args), name


def test_every_tool_can_render_its_own_output():
    """The answer is built from tool lines, so a tool that cannot describe its
    own result would drop its evidence out of the reply."""
    for name, tool in C.TOOLS.items():
        assert callable(tool.line)
        assert tool.what and tool.seconds > 0, name


# ------------------------------------------------------------------ the loop
def test_the_plane_reports_the_loop_beat_not_a_process_local_boolean():
    """The autopilot loop lives in the daemon process. A CLI or API call in
    another process loads the persisted state with `running` forced off — safe
    to do, misleading to say. The heartbeat is the fact."""
    import datetime as dt
    base = {"ok": True, "running": False, "cycles": 182, "interval_s": 600,
            "armed_live": False, "kill_switch": False, "equity": 999992.0,
            "positions": 0, "realized_pnl": 0.0, "posture": {}}
    fresh = C._l_posture({**base, "cycle_age_s": 40.0})
    assert "alive" in fresh and "not in this process" in fresh
    assert "quiet" in C._l_posture({**base, "cycle_age_s": 9000.0})
    assert "never cycled" in C._l_posture({**base, "cycle_age_s": None})
    assert C._cycle_age(None) is None
    assert C._cycle_age("not a timestamp") is None
    assert C._cycle_age(dt.datetime.now().isoformat()) < 5.0


# ----------------------------------------------------------- the desk surfaces
def test_a_chained_screen_ask_reaches_the_plane_not_the_flat_scan_branch(monkeypatch):
    """The desk's keyword branches predate the plane and still answer a bare
    "best PSX stocks" with a leaderboard card. An ask that also wants a decision
    is two experts deep, so it must be handed over — otherwise the richest path
    is unreachable from the surface most people use."""
    from kairos.api import server as S

    asked = []
    monkeypatch.setattr(C, "run", lambda text, **kw: asked.append(text) or {
        "intent": "scan", "answer": "screened, conducted, checked the posture",
        "steps": [{"tool": "scan", "args": {}, "why": "rank"}], "trace": [],
        "planner": "rule", "ms": 7})
    def flat_scan(**kw):
        raise AssertionError("the flat scan branch answered a chained ask")

    monkeypatch.setattr(S, "scan", flat_scan)
    out = S.chat(S.ChatReq(message="screen the PSX and tell me which name "
                                   "is worth trading"))
    assert out["action"] == "command" and out["data"]["kind"] == "command"
    assert asked and "worth trading" in asked[0]


def test_a_bare_screen_keeps_its_leaderboard_card(monkeypatch):
    from kairos.api import server as S

    def should_not_plan(*a, **kw):
        raise AssertionError("a plain ranking ask should not pay for a plan")

    monkeypatch.setattr(C, "run", should_not_plan)
    monkeypatch.setattr(S, "scan", lambda **kw: {
        "ok": True, "universe": 484,
        "top": [{"name": "OGDC", "sector": "E&P", "composite_score": 2.1,
                 "momentum_1m": 4.0}]})
    out = S.chat(S.ChatReq(message="best Pakistani stocks"))
    assert out["action"] == "scan" and out["data"]["rows"][0]["name"] == "OGDC"
