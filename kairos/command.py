"""The Fugu command plane — plan, run, report.

Fugu AI's doctrine is that an assistant is worth nothing until you can see the
steps it took. This is the same idea pointed at a desk: a request arrives in
plain language, the planner composes one to four steps from a registry of
quant tools, the executor runs them against the live engine with a wall-clock
budget, and the answer carries the trace so the operator can see which expert
said what. A question like "screen the PSX and tell me which name is worth
trading today" is two experts deep — the cross-section feeds the conductor's
verdict on its top name — and the reference is resolved from the first step's
own output, never guessed.

Two properties are load-bearing:

* **Read-only by construction.** The registry contains no tool that can place,
  cancel, arm or switch anything, so no plan — typed by a human, compiled by the
  deterministic planner, or invented by the language model — can move money. The
  verbs that ask for movement are refused before planning, because a model that
  is having a bad day should still not find an order button.
* **Numbers come from tools.** Every sentence in the answer is rendered by the
  tool that produced it, from its own fields. The language model is invited to
  change the voice, never the figures, and the answer is complete without it —
  a suspended provider must not be able to silence the desk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_STEPS = 4
DEFAULT_BUDGET_S = 30.0
_MAX_ROWS = 6

# Movement lives behind the desk's double-key staging and the arming keys. If a
# sentence asks for any of it, the plane answers with where to go instead of
# planning, so a hallucinating planner never sees a mutating tool to choose.
_MUTATION = (
    "place order", "place an order", "put an order", "submit order",
    "submit an order", "send order", "send an order", "execute the order",
    "execute it", "square off", "arm live", "arm the live", "enable live",
    "go live", "switch to live", "start live", "trade with real money",
    "real money", "kill switch", "cancel my order", "cancel the order",
    "withdraw", "transfer funds", "place a trade", "take the position",
    "open a position for me",
)

# "buy HBL now" is an instruction; "should I buy HBL now?" is a question the desk
# answers with a verdict. The first is refused, the second is served — and both
# are safe either way, because the registry below has nothing to execute with.
_COMMAND_VERB = re.compile(r"^(?:please\s+)?(?:kairos[,:\s]+)?"
                           r"(?:buy|sell|purchase|book|load|dump)\b")
_ASKING = re.compile(r"\b(?:should|would|could|can|may|might|is|are|was|were|how|"
                     r"what|why|worth|good|better|think|chance|opportunity)\b|[?]")

_REFUSAL = ("I advise, I do not execute — this plane has no order, arming or "
            "kill-switch tool to call. Staging a live order needs the desk's "
            "Trade Like a Pro panel and both confirmation keys; starting the "
            "loop is `kairos autopilot --start`. Nothing here can move money.")


@dataclass(frozen=True)
class Tool:
    name: str
    what: str
    run: Callable[..., dict]
    line: Callable[[dict], str]
    args: tuple[str, ...] = ()
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    seconds: float = 1.0


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pct(v: Any) -> str:
    return f"{_num(v) * 100:+.2f}%"


# ------------------------------------------------------------------ the tools
def _t_predict(symbol: str, horizon: int = 5) -> dict:
    from .engine import ENGINE
    res = ENGINE.predict(symbol, horizon=int(horizon))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "prediction failed")}
    s, rk = res["signal"], res["risk"]
    return {"ok": True, "symbol": res["symbol"], "name": res["name"],
            "price": res["price"], "currency": res.get("currency"),
            "direction": s["direction"], "score": s["score"],
            "confidence": s["confidence"], "expected_return": s["expected_return"],
            "probability_up": res.get("probability_up"),
            "horizon_days": s["horizon_days"], "regime": res["regime"]["regime"],
            "weight": rk["suggested_weight"], "cached": res.get("cached", False),
            "predict_ms": res.get("predict_ms")}


def _l_predict(d: dict) -> str:
    return (f"{d['name']} ({d['symbol']}) at {d['price']:,.2f} {d['currency'] or ''}: "
            f"{d['direction']}, {_pct(d['expected_return'])} over {d['horizon_days']}d, "
            f"P(up) {_num(d['probability_up']) * 100:.0f}%, confidence "
            f"{_num(d['confidence']) * 100:.0f}%, regime {d['regime']}, "
            f"suggested size {_num(d['weight']) * 100:.1f}%")


def _t_conduct(symbol: str) -> dict:
    from . import fugu
    r = fugu.conduct(symbol)
    return {"ok": True, **r}


def _l_conduct(d: dict) -> str:
    cal = d.get("calibration") or {}
    extra = (f" {d['rr']:.1f}R geometry" if d.get("rr") else "") + (
        f", anchored on {cal.get('n_resolved')} realized calls"
        if cal.get("n_resolved") else "")
    return (f"Conductor says {d['action']} on {d['symbol']} — {d['direction']}, "
            f"conviction {d['conviction']:.2f} against a {d['threshold']:.2f} bar, "
            f"P(win) {d['p_hit'] * 100:.0f}%, edge {d['edge_r']:+.2f}R"
            f"{extra}. Why: {'; '.join(d['reasons'][:3])}")


def _t_board(limit: int = 5) -> dict:
    from . import fugu
    rows = fugu.board()[:int(limit)]
    return {"ok": True, "rows": rows}


def _l_board(d: dict) -> str:
    rows = d.get("rows") or []
    if not rows:
        return "The conductor board has no rows — the watchlist is empty."
    best = rows[0]
    ranked = ", ".join(f"{r['symbol']} {r['edge_r']:+.2f}R" for r in rows[:3])
    return (f"Conductor board (top {len(rows)} by edge): {ranked}. "
            f"Sharpest is {best['symbol']}: {best['action']}, {best['direction']}, "
            f"conviction {best['conviction']:.2f} vs {best['threshold']:.2f} "
            f"({'all experts concur' if best['action'] == 'EXECUTE' else '; '.join(best['reasons'][:2])})")


def _t_scan(top: int = 5, market: str = "PSX") -> dict:
    from .engine import ENGINE
    r = ENGINE.scan(market=market, top=int(top))
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error", "scan failed")}
    return {"ok": True, "universe": r.get("universe"), "top": r["top"][:_MAX_ROWS]}


def _l_scan(d: dict) -> str:
    top = d.get("top") or []
    if not top:
        return "The PSX cross-section returned nothing."
    lines = [f"{i + 1}. {t.get('name')} — composite {_num(t.get('composite_score')):+.2f}"
             for i, t in enumerate(top[:3])]
    return (f"PSX screen over {d.get('universe')} names; strongest factor basket: "
            + "; ".join(lines).replace("\n", " "))


def _t_backtest(symbols: list[str], days: int = 60) -> dict:
    # deliberately walk_forward() and not run_paper_session(): the session
    # helper also fires a demo order through the paper broker, and a plane that
    # answers questions must not leave footprints in the book it is reporting on
    from .backtest import walk_forward
    from .engine import ENGINE
    from .universe import resolve
    rows = []
    for sym in list(symbols)[:3]:
        inst = resolve(sym)
        if inst is None:
            continue
        try:
            df = ENGINE.md.history(inst)
            bt = walk_forward(df, inst.key, test_days=int(days))
            rows.append({"symbol": inst.key, "name": getattr(inst, "name", inst.key),
                         "verdict": bt.verdict, "trades": bt.trades,
                         "hit_rate": bt.hit_rate, "return": bt.total_return,
                         "benchmark": bt.benchmark_return, "sharpe": bt.sharpe,
                         "max_dd": bt.max_drawdown, "bars": bt.bars})
        except ValueError as exc:
            rows.append({"symbol": inst.key, "name": getattr(inst, "name", inst.key),
                         "error": str(exc)})
    wins = sum(1 for r in rows if r.get("verdict") == "PROFITABLE")
    return {"ok": bool(rows), "days": int(days), "results": rows,
            "summary": {"instruments": len(rows), "profitable": wins}}


def _l_backtest(d: dict) -> str:
    rows = [r for r in (d.get("results") or []) if not r.get("error")]
    if not rows:
        return "The walk-forward found no instrument with enough history."
    wins = sum(1 for r in rows if r.get("verdict") == "PROFITABLE")
    lines = [f"{r['symbol']} {_pct(r.get('return'))} vs buy-and-hold "
             f"{_pct(r.get('benchmark'))} over {r.get('bars')} bars, "
             f"{int(_num(r.get('trades')))} trades, verdict {r.get('verdict')}"
             for r in rows[:3]]
    return (f"Walk-forward, costs modelled ({d['days']}-bar test windows): "
            f"{wins} of {len(rows)} profitable — " + "; ".join(lines))


def _t_portfolio(symbols: list[str], capital: float = 1_000_000.0) -> dict:
    from .engine import ENGINE
    r = ENGINE.portfolio(list(symbols), capital=float(capital))
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error", "portfolio failed")}
    w = sorted(r["weights"].items(), key=lambda kv: -abs(kv[1]))[:_MAX_ROWS]
    return {"ok": True, "solver": r.get("solver"), "sharpe": r.get("sharpe"),
            "expected_return": r.get("expected_return"),
            "expected_vol": r.get("expected_vol"),
            "weights": {k: v for k, v in w}}


def _l_portfolio(d: dict) -> str:
    w = ", ".join(f"{k} {v * 100:.1f}%" for k, v in (d.get("weights") or {}).items()
                  if _num(v) > 0.001)
    return (f"Annealed allocation over {len(d.get('weights') or {})} names "
            f"({d.get('solver')}), expected {_pct(d.get('expected_return'))} at "
            f"Sharpe {_num(d.get('sharpe')):.2f}: {w}")


def _t_record() -> dict:
    from .feedback import track_record
    tr = track_record() or {}
    return {"ok": True, **tr}


def _l_record(d: dict) -> str:
    n = int(_num(d.get("directional_resolved")))
    if not n:
        resolved = int(_num(d.get("resolved")))
        return (f"{resolved} predictions have hit their horizon, and every one was "
                f"a non-trade — a directional hit rate does not exist yet. "
                f"Confidence is pinned to 1.0 until 10 real calls mature.")
    return (f"{n} directional calls resolved: hit rate {_num(d.get('hit_rate')) * 100:.1f}%"
            f", strong signals {_num(d.get('strong_signal_hit_rate')) * 100:.0f}%, "
            f"mean absolute error {_num(d.get('mean_abs_error')):.4f}")


def _cycle_age(last_cycle: Any) -> float | None:
    """Seconds since the loop's last cycle, or None if it never ran.

    The loop lives in the daemon process. A CLI or API call in another process
    loads the persisted state with `running` forced to False — which is the safe
    thing to *do* and a misleading thing to *say*. The heartbeat is the fact the
    operator needs, so the plane reports it instead of a process-local boolean.
    """
    if not last_cycle:
        return None
    import datetime as dt
    try:
        then = dt.datetime.fromisoformat(str(last_cycle))
    except ValueError:
        return None
    return max(0.0, (dt.datetime.now() - then).total_seconds())


def _t_posture() -> dict:
    from .autopilot import AUTOPILOT
    from .execution import PaperBroker
    st = AUTOPILOT.status() or {}
    acc = PaperBroker().account()
    age = _cycle_age(st.get("last_cycle"))
    return {"ok": True, "running": st.get("running"), "mode": st.get("mode"),
            "armed_live": st.get("armed_live"), "kill_switch": st.get("kill_switch"),
            "cycles": st.get("cycles"), "cycle_age_s": age,
            "interval_s": st.get("interval_s"),
            "posture": st.get("risk_posture"),
            "equity": acc.get("equity"), "realized_pnl": acc.get("realized_pnl"),
            "day_pnl": acc.get("day_pnl"), "positions": len(PaperBroker().open_positions())}


def _l_posture(d: dict) -> str:
    p = d.get("posture") or {}
    age = d.get("cycle_age_s")
    interval = _num(d.get("interval_s")) or 600.0
    if age is None:
        loop = "never cycled"
    else:
        beating = age <= 2.5 * interval
        loop = (f"last cycle {int(age)}s ago ({'alive' if beating else 'quiet'})")
    return (f"Autopilot {'running' if d.get('running') else 'not in this process'}, "
            f"{loop}, {int(_num(d.get('cycles')))} cycles; live arming "
            f"{'ON' if d.get('armed_live') else 'OFF'}, kill switch "
            f"{'ENGAGED' if d.get('kill_switch') else 'clear'}, equity "
            f"{_num(d.get('equity')):,.0f} with {int(_num(d.get('positions')))} open, "
            f"realized {_num(d.get('realized_pnl')):+,.2f}. "
            f"Posture: {p.get('note') or p.get('state') or 'no constraint'}")


def _t_health() -> dict:
    from .health import system_health
    h = system_health()
    return {"ok": True, "data_online": h.get("data_online"),
            "market_data": h.get("market_data"), "llm": h.get("llm"),
            "wsl": bool((h.get("wsl") or {}).get("models_ready")),
            "universe_size": h.get("universe_size"), "uptime_s": h.get("uptime_s")}


def _l_health(d: dict) -> str:
    llm = d.get("llm") or {}
    md = ", ".join(f"{k} {v}" for k, v in (d.get("market_data") or {}).items())
    unanswered = llm.get("unanswered") or {}
    prov = f"providers {llm.get('provider')}"
    if unanswered:
        prov += " (configured but not answering: " + "; ".join(
            f"{k} — {v}" for k, v in unanswered.items()) + ")"
    dead = llm.get("dead_seats") or []
    return (f"Data {'online' if d.get('data_online') else 'OFFLINE'} — {md}. "
            f"{prov} over {int(_num(llm.get('models')))} council seats, "
            f"{len(dead)} dead seats, universe {d.get('universe_size')}, "
            f"up {int(_num(d.get('uptime_s')) / 60)} min.")


def _t_livebook() -> dict:
    from . import livebook
    rows = livebook.review() or []
    return {"ok": True, "rows": rows[:_MAX_ROWS], "count": len(rows)}


def _l_livebook(d: dict) -> str:
    rows = d.get("rows") or []
    if not rows:
        return ("No JS Global book is on file — add holdings under Trade Like a "
                "Pro and the engine will watch them with stops and targets.")
    tot = sum(_num(r.get("pnl")) for r in rows)
    lines = "; ".join(f"{r['symbol']} {r.get('status', '—')} "
                      f"{_num(r.get('pnl_pct')):+.1f}%" for r in rows[:3])
    return f"Live book: {d.get('count')} position(s), unrealised {tot:+,.0f} — {lines}"


def _t_news(limit: int = 4) -> dict:
    from .engine import ENGINE
    items = ENGINE.news_items()[:int(limit)]
    return {"ok": True, "items": [{"title": getattr(i, "title", ""),
                                   "source": getattr(i, "source", "")} for i in items]}


def _l_news(d: dict) -> str:
    items = d.get("items") or []
    if not items:
        return "The news feeds are unreachable right now."
    return "On the radar: " + "; ".join(f"{i['title']} ({i['source']})" for i in items[:3])


def _t_doctrine() -> dict:
    from .rules import describe
    return {"ok": True, "doctrine": describe()}


def _l_doctrine(d: dict) -> str:
    return f"Duty plan: {d['doctrine']}"


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("predict", "full dossier for one instrument: direction, expected return, "
         "P(up), confidence, regime, suggested size — args: symbol, horizon (days)",
         _t_predict, _l_predict, ("symbol", "horizon"), {"horizon": (1, 60)}, 3.0),
    Tool("conduct", "the fused expert verdict on one instrument: conviction against "
         "the adaptive bar, probability-weighted edge in R, and why — args: symbol",
         _t_conduct, _l_conduct, ("symbol",), {}, 3.0),
    Tool("board", "conductor board over the watchlist, ranked by edge — args: limit",
         _t_board, _l_board, ("limit",), {"limit": (1, 10)}, 8.0),
    Tool("scan", "PSX cross-sectional factor leaderboard — args: top, market",
         _t_scan, _l_scan, ("top", "market"), {"top": (1, 10)}, 4.0),
    Tool("backtest", "pure walk-forward backtest, costs modelled, no orders placed "
         "— args: symbols, days",
         _t_backtest, _l_backtest, ("symbols", "days"), {"days": (20, 250)}, 20.0),
    Tool("portfolio", "annealed portfolio weights — args: symbols, capital",
         _t_portfolio, _l_portfolio, ("symbols", "capital"), {}, 6.0),
    Tool("record", "realized track record straight from the journal",
         _t_record, _l_record, (), {}, 1.0),
    Tool("posture", "autopilot state: mode, arming, kill switch, equity, risk posture",
         _t_posture, _l_posture, (), {}, 1.0),
    Tool("health", "subsystem health, including which providers actually answer",
         _t_health, _l_health, (), {}, 2.0),
    Tool("livebook", "the operator's JS Global book with engine verdicts on each holding",
         _t_livebook, _l_livebook, (), {}, 5.0),
    Tool("news", "top headlines from the news radar — args: limit",
         _t_news, _l_news, ("limit",), {"limit": (1, 6)}, 1.0),
    Tool("doctrine", "the trading rules the system is currently bound by",
         _t_doctrine, _l_doctrine, (), {}, 0.5),
)}

# A plan may only name symbols the universe knows. Anything else is a typo or a
# hallucination, and a typo on a trading desk is not a small thing.
_SYMBOL_ARGS = ("symbol", "symbols")


# -------------------------------------------------------------------- planning
def _asks_for_movement(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in _MUTATION):
        return True
    stripped = low.strip()
    return bool(_COMMAND_VERB.match(stripped)) and not _ASKING.search(stripped)


def _symbols(text: str) -> list[str]:
    """Uppercase-looking tokens that resolve in the universe, in mention order."""
    from .universe import resolve
    out: list[str] = []
    for tok in re.findall(r"\b[A-Z][A-Z0-9.]{1,14}\b", text.upper()):
        key = tok.strip(".")
        if resolve(key) is not None and key not in out:
            out.append(key)
    if out:
        return out
    # no ticker typed — try the universe's own display names ("gold", "bitcoin")
    from .universe import universe
    low = f" {text.lower()} "
    for inst in universe():
        name = str(getattr(inst, "name", "") or "").lower()
        key = str(getattr(inst, "key", "") or "").lower()
        if key and f" {key} " in low:
            out.append(key.upper())
        elif len(name) > 3 and f" {name} " in low:
            out.append(str(inst.key).upper())
    return out


def _horizon(text: str) -> int | None:
    m = re.search(r"\b(\d{1,3})[\s-]*(?:day|d|session)s?\b", text.lower())
    if m:
        return max(1, min(60, int(m.group(1))))
    if any(w in text.lower() for w in ("month", "4 week", "a month")):
        return 21
    if "week" in text.lower():
        return 5
    if "tomorrow" in text.lower() or "overnight" in text.lower():
        return 1
    return None


def _days(text: str) -> int | None:
    """A walk-forward window, which may be far longer than a forecast horizon."""
    m = re.search(r"\b(\d{2,4})[\s-]*(?:day|d|bar|session)s?\b", text.lower())
    return max(20, min(250, int(m.group(1)))) if m else None


def _refs(text: str) -> bool:
    return "$" in text


def plan(text: str) -> dict[str, Any]:
    """Compile plain language into steps. Deterministic; no model required."""
    low = text.lower()
    syms = _symbols(text)
    s1 = syms[0] if syms else None
    steps: list[dict[str, Any]] = []

    def add(tool: str, why: str, **args) -> None:
        if len(steps) >= MAX_STEPS:
            return
        steps.append({"tool": tool, "args": args, "why": why})

    def has(*words) -> bool:
        return any(w in low for w in words)

    if has("js global", "my book", "my positions", "my holdings", "live book",
           "livebook", "what am i holding"):
        add("livebook", "read the operator's real book")
        if s1:
            add("conduct", "then run the fused verdict on it", symbol=s1)
    elif has("track record", "hit rate", "accuracy", "how reliable",
             "how good are you", "are you working", "proven"):
        add("record", "grade the system on realized outcomes")
        add("posture", "and what that has done to the entry bar")
    elif has("health", "status", "is it running", "uptime", "providers",
             "what is down", "anything broken"):
        add("health", "check every subsystem, including provider truth")
        add("posture", "and the execution posture")
    elif has("doctrine", "your rules", "risk rules", "discipline", "wyckoff",
             "dow theory", "canslim", "buffett", "dalio", "lynch"):
        add("doctrine", "state the rules the system is bound by")
    elif has("news", "headline", "what happened"):
        add("news", "read the radar")
    elif has("allocate", "allocat", "portfolio", "weights", "diversif",
             "split my", "rebalance"):
        add("portfolio", "optimise weights across the named basket",
            symbols=syms or ["GOLD", "WTI", "EURUSD", "SPX", "BTC"])
    elif has("backtest", "walk forward", "walk-forward", "would it have worked",
             "prove it", "history"):
        add("backtest", "walk-forward, costs modelled",
            **({"days": _days(text)} if _days(text) else {}),
            symbols=syms or ["GOLD", "WTI", "HBL"])
    elif has("which name", "best stock", "best name", "screen", "scan",
             "top psx", "what should i trade", "worth trading", "leaderboard",
             "momentum leader"):
        add("scan", "rank the PSX cross-section", top=5)
        add("conduct", "run the conductor on the top name from that screen",
            symbol="$0.top.0.name")
        add("posture", "and check the book may act on it at all")
    elif has("board", "conductor", "watchlist", "what are you watching",
             "rank the watchlist", "edge"):
        add("board", "rank the watchlist by probability-weighted edge", limit=5)
    elif s1 and has("why", "reason", "explain", "pass", "fail", "rejected",
                    "blocked"):
        add("conduct", "the conductor records the reason for this exact verdict",
            symbol=s1)
    elif s1:
        hz = _horizon(text)
        add("predict", "the dossier for this instrument",
            **({"horizon": hz} if hz else {}), symbol=s1)
        add("conduct", "then the fused expert verdict and its edge", symbol=s1)
    if not steps and has("buy", "sell", "heading", "outlook", "forecast",
                         "predict", "going", "target", "chance"):
        add("board", "no instrument resolved — rank the watchlist instead", limit=5)
    return {"intent": steps[0]["tool"] if steps else "none",
            "steps": steps, "symbols": syms, "movement": _asks_for_movement(text)}


# -------------------------------------------------------------------- executor
def _dig(obj: Any, path: list[str]) -> Any:
    for part in path:
        if isinstance(obj, list):
            if not part.lstrip("-").isdigit():
                return None
            i = int(part)
            obj = obj[i] if -len(obj) <= i < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _ref(spec: str, results: list[dict]) -> Any:
    """`$0.top.0.name` — step 0's *output*, then that path into it.

    A step reference walks the data the tool returned, not the trace record, so
    a plan can pipe one expert's leaderboard into the next expert's verdict.
    """
    parts = spec[1:].split(".")
    if not parts or not parts[0].isdigit():
        raise ValueError(f"reference {spec} must name a step index")
    i = int(parts[0])
    if i >= len(results):
        raise ValueError(f"reference {spec} points past the steps run so far")
    return _dig(results[i].get("data"), parts[1:])


def _resolve_args(tool: Tool, args: dict, results: list[dict]) -> dict:
    """Validate one step's arguments against the registry, refs included."""
    out: dict[str, Any] = {}
    for name, value in (args or {}).items():
        if name not in tool.args:
            raise ValueError(f"{tool.name} takes no argument {name!r}")
        if isinstance(value, str) and value.startswith("$"):
            value = _ref(value, results)
            if value is None:
                raise ValueError(f"{tool.name}: reference {args[name]} resolved to nothing")
        if name in _SYMBOL_ARGS:
            items = value if isinstance(value, list) else [value]
            keys = [str(v).upper() for v in items if _resolvable(str(v))]
            if not keys:
                raise ValueError(f"{tool.name}: no known instrument in {value!r}")
            value = keys if isinstance(value, list) else keys[0]
        lo_hi = tool.limits.get(name)
        if lo_hi is not None:
            value = max(lo_hi[0], min(lo_hi[1], _num(value)))
            value = int(value) if float(value).is_integer() else value
        out[name] = value
    for name in tool.args:
        if name not in out and name in _SYMBOL_ARGS:
            raise ValueError(f"{tool.name} needs a symbol")
    return out


def _resolvable(sym: str) -> bool:
    from .universe import resolve
    return resolve(sym.upper().strip(".")) is not None


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (str, int, float)):
        return v
    try:
        import numpy as np
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
    except Exception:
        pass
    return str(v)


def execute(steps: list[dict], budget_s: float = DEFAULT_BUDGET_S) -> list[dict]:
    """Run the plan, recording every step — including the ones that failed."""
    results: list[dict] = []
    # a caller-supplied budget arrives as JSON, and NaN passes an ordinary
    # comparison while never expiring — fall back to the default instead
    budget = _num(budget_s)
    if budget <= 0 or budget > 1e6:
        budget = DEFAULT_BUDGET_S
    t_start = time.time()
    for i, step in enumerate(steps):
        tool = TOOLS.get(str(step.get("tool") or ""))
        rec = {"step": i, "tool": step.get("tool"), "args": step.get("args"),
               "why": step.get("why")}
        if tool is None:
            results.append({**rec, "ok": False,
                            "error": f"{step.get('tool')!r} is not in the registry"})
            continue
        left = budget - (time.time() - t_start)
        if left <= 0:
            results.append({**rec, "ok": False, "error": "budget exhausted"})
            continue
        t0 = time.perf_counter()
        try:
            args = _resolve_args(tool, dict(step.get("args") or {}), results)
            data = _plain(tool.run(**args) or {})
            rec |= {"ok": bool(data.get("ok", True)), "data": data,
                    "error": data.get("error")}
        except Exception as exc:
            rec |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        rec["ms"] = int((time.perf_counter() - t0) * 1000)
        results.append(rec)
    return results


def _say(results: list[dict]) -> str:
    lines: list[str] = []
    for r in results:
        tool = TOOLS.get(str(r.get("tool") or ""))
        if r.get("ok") and tool:
            try:
                lines.append(tool.line(r.get("data") or {}))
            except Exception:
                pass
        elif r.get("tool"):
            lines.append(f"{r['tool']} did not run: {r.get('error')}")
    return " ".join(lines)


def _a_provider_is_enabled() -> bool:
    """True when some LLM leg has a key. The plane works fully without one."""
    try:
        from .config import CFG
        apis = CFG.apis or {}
        return any(bool((apis.get(name) or {}).get("enabled"))
                   for name in ("openrouter", "moonshot", "kimi"))
    except Exception:
        return False


def _voice(text: str, answer: str) -> str:
    """Optional rephrasing. The figures are already in `answer`; the model may
    only change how they are said, so a dead provider costs polish, not truth."""
    if not answer or len(answer) < 40 or not _a_provider_is_enabled():
        return answer
    try:
        from .config import CFG
        from .llm import ROUTER
        model = CFG.apis.get("openrouter", {}).get(
            "fast_model", "deepseek/deepseek-chat-v3-0324")
        prompt = (f"You are KAIROS, a concise quant desk analyst. The operator asked: "
                  f"\"{text[:300]}\". These are the numbers your experts produced — "
                  f"repeat them exactly, invent nothing, add nothing: {answer[:1400]}\n"
                  f"Answer in 2-4 sentences with a clear verdict.")
        r = ROUTER.chat(prompt, model=model, max_tokens=260, use_cache=False)
        if r.ok and r.text and r.text.strip():
            return r.text.strip()
    except Exception:
        pass
    return answer


def guard(text: str) -> dict[str, Any] | None:
    """The refusal for an instruction to move money, or None if it is a question.

    Every surface — desk chat, CLI, `/api/command`, any future front-end — asks
    this one call, so the rule and the wording exist in exactly one place and a
    front-end cannot forget to be strict.
    """
    if not _asks_for_movement(text):
        return None
    return {"ok": True, "intent": "refused", "answer": _REFUSAL,
            "steps": [], "trace": [], "ms": 0, "planner": "guard"}


def run(text: str, budget_s: float = DEFAULT_BUDGET_S,
        use_model: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    refused = guard(text)
    if refused is not None:
        return refused
    p = plan(text)
    steps = p["steps"]
    planner = "rule"
    if not steps:
        steps = _llm_plan(text)
        planner = "model"
        if steps:
            p["intent"] = str(steps[0].get("tool") or "model")
    if not steps:
        answer = ("I could not turn that into a step on my tools. Ask about an "
                  "instrument (\"is OGDC a buy this week\"), the screen (\"best PSX "
                  "names to trade\"), the board (\"what are you watching\"), a "
                  "backtest, a portfolio, my realized record, or my health.")
        return {"ok": True, "intent": "none", "answer": answer, "steps": [],
                "trace": [], "ms": int((time.perf_counter() - t0) * 1000),
                "planner": planner}
    trace = execute(steps, budget_s)
    answer = _say(trace)
    if not answer:
        answer = "Every step in that plan failed — see the trace."
    elif use_model:
        answer = _voice(text, answer)
    return {"ok": True, "intent": p["intent"], "answer": answer,
            "steps": [{"tool": s["tool"], "args": s["args"], "why": s["why"]}
                      for s in steps],
            "trace": [{"step": r["step"], "tool": r["tool"], "ok": r["ok"],
                       "ms": r.get("ms", 0), "error": r.get("error"),
                       "args": r.get("args")} for r in trace],
            "ms": int((time.perf_counter() - t0) * 1000), "planner": planner}


# --------------------------------------------------------- model-planned steps
def _llm_plan(text: str) -> list[dict]:
    """Last resort when no rule compiled a plan: ask a model to compose one.

    It chooses only from the registry, and every choice is re-validated — an
    unknown tool, an argument the tool does not take, or a symbol the universe
    does not have is dropped here rather than executed.
    """
    try:
        from .llm import ROUTER, _extract_json
        if not _a_provider_is_enabled():
            return []
        menu = "\n".join(f"- {t.name}: {t.what}" for t in TOOLS.values())
        prompt = (
            "You are the planner for a read-only quant trading desk. Compose at "
            f"most {MAX_STEPS} steps from this registry, or an empty list if none "
            f"fits:\n{menu}\nReply with JSON only: "
            "{\"steps\":[{\"tool\":\"name\",\"args\":{},\"why\":\"short\"}]}\n"
            f"Operator request: \"{text[:300]}\"")
        r = ROUTER.chat(prompt, max_tokens=300, use_cache=False)
        if not r.ok or not r.text:
            return []
        data = _extract_json(r.text)
        raw = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for s in raw[:MAX_STEPS]:
            if not isinstance(s, dict):
                continue
            tool = TOOLS.get(str(s.get("tool") or ""))
            args = s.get("args") if isinstance(s.get("args"), dict) else {}
            if tool is None or any(k not in tool.args for k in args):
                continue
            out.append({"tool": tool.name, "args": args,
                        "why": str(s.get("why") or "planned by the model")[:80]})
        return out
    except Exception:
        return []
