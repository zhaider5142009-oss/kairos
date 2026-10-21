"""KAIROS control tower — FastAPI backend + single-page human-friendly UI."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from kairos import __version__ as KAIROS_VERSION
from kairos.config import CFG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

from kairos.engine import ENGINE  # noqa: E402
from kairos.universe import resolve, universe  # noqa: E402

app = FastAPI(title="KAIROS Control Tower", version=KAIROS_VERSION)
BOOT_TS = time.time()


# ------------------------------------------------------------ remote deploy auth
# When KAIROS_TOKEN is set, the whole API locks (reads too — the live book
# carries real holdings) except /api/health. KAIROS_AUTH=writes relaxes GETs
# to public. Locally, leave the token unset for a fully open desk.
AUTH_TOKEN = os.environ.get("KAIROS_TOKEN", "")
AUTH_MODE = os.environ.get("KAIROS_AUTH", "all").lower()


@app.middleware("http")
async def _token_gate(request, call_next):
    if AUTH_TOKEN and request.url.path != "/api/health":
        mutating = request.method in ("POST", "PUT", "DELETE", "PATCH")
        gated_read = (AUTH_MODE == "all"
                      and request.url.path.startswith("/api/"))
        if mutating or gated_read:
            tok = (request.headers.get("x-kairos-token")
                   or request.query_params.get("token") or "")
            if not secrets.compare_digest(tok, AUTH_TOKEN):
                return JSONResponse(status_code=401,
                                    content={"ok": False, "error": "unauthorized"})
    return await call_next(request)


@app.on_event("startup")
def _auto_resume():  # paper only, by doctrine — live never resurrects itself
    ap = (CFG.section("autopilot") or {}) if hasattr(CFG, "section") else {}
    if ap.get("auto_start_paper", True):
        try:
            from kairos.autopilot import AUTOPILOT
            AUTOPILOT.start("paper")
        except Exception:
            pass


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


def _safe(obj):
    """Recursively coerce numpy scalars/dates into JSON-encodable Python."""
    import datetime as _dt
    import math as _math

    import numpy as _np
    import pandas as _pd
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return [_safe(v) for v in obj.tolist()]
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not _math.isfinite(obj):
        return None
    if isinstance(obj, (_dt.datetime, _dt.date, _pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, _pd.DataFrame):
        return _safe(obj.to_dict(orient="records"))
    return obj


# ---------------------------------------------------------------- health
@app.get("/api/health")
def health():
    from kairos.health import system_health
    try:
        out = system_health()
        out["uptime_s"] = int(time.time() - BOOT_TS)
        return out
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------- universe
@app.get("/api/universe")
def get_universe():
    out = []
    for i in universe():
        out.append({"key": i.key, "name": i.name, "asset_class": i.asset_class,
                    "market": i.market, "sector": i.sector, "currency": i.currency})
    return {"ok": True, "instruments": out}


# ---------------------------------------------------------------- predict
@app.get("/api/predict")
def predict(symbol: str, horizon: int = Query(5, ge=1, le=21),
            council: bool = False, wsl: bool = True, news: bool = True):
    res = ENGINE.predict(symbol, horizon=horizon, use_llm=council,
                         use_wsl=wsl, include_news=news)
    if not res.get("ok"):
        raise HTTPException(404, res.get("error"))
    return _safe(res)


# ---------------------------------------------------------------- scan
@app.get("/api/scan")
def scan(top: int = Query(20, ge=5, le=100)):
    return _safe(ENGINE.scan(top=top))


# ---------------------------------------------------------------- portfolio
class PortfolioReq(BaseModel):
    symbols: list[str]
    capital: float = 1_000_000.0
    cardinality: int | None = None


@app.post("/api/portfolio")
def portfolio(req: PortfolioReq):
    res = ENGINE.portfolio(req.symbols, capital=req.capital, k=req.cardinality)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error"))
    return _safe(res)


@app.get("/api/portfolio")
def portfolio_get(symbols: str, capital: float = 1_000_000.0,
                  cardinality: int | None = None):
    return portfolio(PortfolioReq(symbols=[s.strip() for s in symbols.split(",")],
                                  capital=capital, cardinality=cardinality))


# ---------------------------------------------------------------- news
@app.get("/api/news")
def news(limit: int = 30):
    items = ENGINE.news_items()
    return {"ok": True, "items": [{
        "title": getattr(i, "title", ""), "source": getattr(i, "source", ""),
        "published": str(getattr(i, "published", "")),
        "url": getattr(i, "url", ""), "score": getattr(i, "score", 0)}
        for i in items[:limit]]}


# ---------------------------------------------------------------- paper order
class PaperReq(BaseModel):
    symbol: str
    notional: float = 10_000.0
    force: bool = False


@app.post("/api/paper")
def paper_order(req: PaperReq):
    from kairos.execution import OrderIntent, PaperBroker, RiskGate
    r = ENGINE.predict(req.symbol, horizon=5, include_news=False)
    if not r.get("ok"):
        raise HTTPException(404, r.get("error"))
    s = r["signal"]
    if abs(s["score"]) < 0.18 and not req.force:
        return {"ok": False, "reason": "signal_below_threshold",
                "score": s["score"], "message":
                f"Signal {s['score']:+.2f} below the 0.18 trade threshold — "
                "KAIROS stood down. Use force=true to override."}
    side = "BUY" if s["score"] > 0 else "SELL"
    qty = max(1.0, round(req.notional / r["price"], 2))
    broker = PaperBroker()
    intent = OrderIntent(symbol=r["symbol"], side=side, quantity=qty, venue="PAPER")
    allowed, why = RiskGate().check(intent, broker.account(), broker.open_positions())
    if not allowed:
        return {"ok": False, "reason": "risk_gate", "message": why}
    fill = broker.submit(intent, r["price"])
    return {"ok": True, "fill": _safe(fill.__dict__), "signal": _safe(s),
            "account": broker.account()}


@app.get("/api/paper/account")
def paper_account():
    from kairos.execution import PaperBroker
    return {"ok": True, "account": PaperBroker().account(),
            "positions": PaperBroker().open_positions()}


# ---------------------------------------------------------------- backtest
@app.get("/api/backtest")
def backtest(symbols: str, days: int = Query(60, ge=20, le=250)):
    from kairos.backtest import run_paper_session
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "no symbols given")
    return _safe(run_paper_session(ENGINE.md, syms, ENGINE, days=days))


# ---------------------------------------------------------------- mt5
@app.get("/api/mt5")
def mt5(symbols: str = "XAUUSD,XAGUSD,EURUSD", benchmark: bool = False):
    from kairos.execution import get_mt5
    br = get_mt5()
    connected = br.ensure()
    out = {"ok": connected, "connected": connected, "ticks": [],
           "error": "" if connected else br.last_error,
           "attach_only": True,
           "terminal_running": br._terminal_running()}
    if connected:
        for sym in [s.strip() for s in symbols.split(",") if s.strip()]:
            t = br.tick(sym)
            if t:
                t["symbol"] = sym
                out["ticks"].append(t)
        if benchmark:
            out["benchmark"] = br.benchmark_latency("XAUUSD", n=3)
    return out


# ---------------------------------------------------------------- autopilot
class AutopilotReq(BaseModel):
    action: str = "status"          # status | start | stop | configure
    mode: str = "paper"             # paper | live
    confirm_live: bool = False
    symbols: list[str] | None = None
    interval_s: int | None = None
    entry_threshold: float | None = None
    max_trades_per_cycle: int | None = None


@app.get("/api/autopilot")
def autopilot_get():
    from kairos.autopilot import AUTOPILOT
    return _safe(AUTOPILOT.status())


@app.post("/api/autopilot")
def autopilot_post(req: AutopilotReq):
    from kairos.autopilot import AUTOPILOT
    from kairos.execution import JOURNAL
    act = req.action.lower()
    if act == "configure":
        AUTOPILOT.configure(req.symbols, req.interval_s,
                            req.entry_threshold, req.max_trades_per_cycle)
        JOURNAL.log("AUTOPILOT_CONFIGURED", {"by": "api",
                                             "symbols": req.symbols,
                                             "interval_s": req.interval_s,
                                             "entry_threshold": req.entry_threshold})
        return _safe(AUTOPILOT.status())
    if act == "start":
        out = AUTOPILOT.start(mode=req.mode, confirm_live=req.confirm_live)
        # an arming ATTEMPT is itself an audit event, including the refused
        # ones — whoever tried, and what stopped them
        JOURNAL.log("AUTOPILOT_ARM_REQUEST", {"by": "api", "mode": req.mode,
                                             "confirm_live": req.confirm_live,
                                             "ok": bool(out.get("ok")),
                                             "error": (out.get("error") or "")[:200]})
        return _safe(out)
    if act == "stop":
        JOURNAL.log("AUTOPILOT_STOP_REQUEST", {"by": "api"})
        return _safe(AUTOPILOT.stop())
    if act == "cycle":
        JOURNAL.log("AUTOPILOT_CYCLE_REQUEST", {"by": "api"})
        return _safe(AUTOPILOT.cycle())
    return _safe(AUTOPILOT.status())


@app.post("/api/killswitch")
def killswitch(engage: bool = True, reason: str = "api"):
    from kairos.execution import JOURNAL, KILL
    if engage:
        KILL.engage(reason)
        JOURNAL.log("KILL_SWITCH", {"by": "api", "reason": reason})
        return {"ok": True, "engaged": True}
    KILL.release()
    # releasing the brake deserves the same record as stepping on it
    JOURNAL.log("KILL_SWITCH_RELEASE", {"by": "api", "reason": reason})
    return {"ok": True, "engaged": False}


@app.get("/api/desk")
def desk():
    """One-call snapshot for the Pro Desk tab."""
    from kairos.autopilot import AUTOPILOT
    from kairos.execution import JOURNAL
    st = AUTOPILOT.status()
    st["journal"] = [j for j in JOURNAL.tail(40)
                     if str(j.get("event", "")).startswith(("AUTOPILOT", "PAPER",
                                                            "MT5_", "PSX_", "KILL"))]
    return _safe(st)


# ---------------------------------------------------------------- live signals
@app.get("/api/live")
def live_signals():
    """Shareable live signal board from the autopilot's latest cycle."""
    from kairos.autopilot import AUTOPILOT
    st = AUTOPILOT.status()
    rows = [dict(r, share=f"/api/signal-card?symbol={r['symbol']}")
            for r in st.get("signals") or []]
    return _safe({"ok": True, "as_of": st.get("last_cycle"),
                  "running": st.get("running"), "mode": st.get("mode"),
                  "signals": rows})


@app.get("/api/signal-card")
def signal_card(symbol: str, caption: bool = False):
    """Instagram-square PNG signal card; ?caption=1 returns the share text."""
    from fastapi.responses import FileResponse
    try:
        from kairos import signalcard
    except ImportError as e:
        # the free image ships no matplotlib on purpose; that has to be a
        # sentence about an extra, not a 500 about a missing module
        raise HTTPException(501, f"signal cards need `pip install kairos[card]` ({e})") from e
    try:
        path, cap = signalcard.card_for(symbol)
    except Exception as e:
        raise HTTPException(404, f"no signal for {symbol!r}: {e}") from e
    if caption:
        return {"ok": True, "caption": cap}
    return FileResponse(path, media_type="image/png",
                        filename=f"kairos-{symbol.upper()}-signal.png")


# ---------------------------------------------------------------- live book
class LivebookReq(BaseModel):
    action: str                 # add | remove | stop | ticket
    symbol: str | None = None
    qty: float | None = None
    entry: float | None = None
    stop: float | None = None


@app.get("/api/livebook")
def livebook_get(refresh: bool = False):
    """Real JS Global holdings reviewed through the engine."""
    from kairos import livebook
    return _safe({"ok": True, "positions": livebook.review(force=refresh)})


@app.post("/api/livebook")
def livebook_post(req: LivebookReq):
    from kairos import livebook
    try:
        if req.action == "add":
            livebook.add(req.symbol or "", req.qty or 0, req.entry or 0)
        elif req.action == "remove":
            livebook.remove(req.symbol or "")
        elif req.action == "stop":
            if not livebook.set_stop(req.symbol or "", req.stop):
                raise ValueError("symbol not in the live book")
        elif req.action == "sync":
            from kairos import jsgsync
            return _safe({"ok": True, "report": jsgsync.sync()})
        elif req.action == "ticket":
            return _safe({"ok": True,
                          "ticket": livebook.make_sell_ticket(req.symbol or "")})
        else:
            raise ValueError(f"unknown action {req.action!r}")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        from kairos.jsgsync import JsGlobalError
        if isinstance(e, JsGlobalError):
            raise HTTPException(502, str(e)) from e
        raise
    return _safe({"ok": True, "positions": livebook.review()})


@app.get("/api/jsg/book")
def jsg_book(force: bool = False):
    """The funded account as the broker sees it: free cash + MTM holdings.

    Read-only, and the number the live risk posture is sized from — a portal
    outage is a 502 here, never a zero.
    """
    from kairos import jsgsync
    try:
        return _safe(jsgsync.real_book(force=force))
    except jsgsync.JsGlobalError as e:
        raise HTTPException(502, str(e)) from e


# ---------------------------------------------------------- JS Global orders
# Real money moves only through these four keys turning:
#   js_global.allow_orders (standing yes)  +  the trading PIN (per order)
#   + the kill switch clear               +  the broker's own price band
# A PIN is accepted on input and never echoed, logged or journalled.

class JsgPlaceReq(BaseModel):
    symbol: str
    side: str                     # BUY | SELL | SHORT
    qty: int
    price: float
    order_type: str = "LIMIT"     # LIMIT | MARKET | STOP_LOSS | FOK | MIT
    market: str | None = None
    note: str = "desk"


class JsgIdReq(BaseModel):
    id: str
    pin: str | None = None


class JsgCancelReq(BaseModel):
    order_no: str


def _jsg_guard(fn, *args, **kwargs):
    from kairos.jsgsync import JsGlobalAuthError, JsGlobalError
    try:
        return fn(*args, **kwargs)
    except JsGlobalAuthError as e:
        raise HTTPException(502, str(e)) from e
    except JsGlobalError as e:
        raise HTTPException(400, str(e)) from e


class JsgPinReq(BaseModel):
    pin: str


@app.post("/api/jsg/pin")
def jsg_pin_store(req: JsgPinReq):
    """Save the broker's trading PIN so the autopilot can act unattended.

    The value goes into git-ignored config.yaml and nowhere else: it is not
    echoed, not returned, and only its length is journalled. Without a stored
    PIN the loop stays a ticket generator — this is the keystroke that turns
    it into an executor, so it is deliberately something a human does.
    """
    from kairos.config import CFG
    from kairos.execution import JOURNAL
    pin = (req.pin or "").strip()
    if not (pin.isdigit() and 4 <= len(pin) <= 8):
        raise HTTPException(400, "a trading PIN is 4-8 digits")
    if not CFG.store_secret("js_global", "trading_pin", pin):
        raise HTTPException(500, "no config.yaml to store the PIN in")
    JOURNAL.log("JSG_PIN_STORED", {"chars": len(pin)})
    from kairos import jsgtrade
    return _safe({"ok": True, "has_pin": True,
                  "orders_enabled": jsgtrade.enabled()})


@app.delete("/api/jsg/pin")
def jsg_pin_clear():
    """Forget the stored PIN — instant return to human-confirmed tickets."""
    from kairos.config import CFG
    from kairos.execution import JOURNAL
    if not CFG.store_secret("js_global", "trading_pin", ""):
        raise HTTPException(500, "no config.yaml to clear")
    JOURNAL.log("JSG_PIN_CLEARED", {})
    return _safe({"ok": True, "has_pin": False})


@app.get("/api/jsg/orders")
def jsg_orders(band_symbol: str | None = None, market: str | None = None):
    """Gate status, staged orders and (optionally) a scrip's broker price band."""
    from kairos import jsgtrade
    out = _safe(jsgtrade.state())
    out["pending"] = jsgtrade.pending()
    if band_symbol:
        out["band"] = _jsg_guard(jsgtrade.band, band_symbol, market)
    return out


@app.post("/api/jsg/orders")
def jsg_stage(req: JsgPlaceReq):
    """Stage a JS Global order. Nothing reaches the broker until it is acked."""
    from kairos import jsgtrade
    return _jsg_guard(jsgtrade.stage, req.symbol, req.side, req.qty, req.price,
                      req.order_type, req.market, req.note)


@app.post("/api/jsg/orders/ack")
def jsg_ack(req: JsgIdReq):
    """Release a staged order to the broker — THIS EXECUTES REAL MONEY."""
    from kairos import jsgtrade
    res = _jsg_guard(jsgtrade.ack, req.id, req.pin)
    return {k: v for k, v in res.items() if k != "pin"}


@app.post("/api/jsg/orders/drop")
def jsg_drop(req: JsgIdReq):
    from kairos import jsgtrade
    return _jsg_guard(jsgtrade.drop, req.id)


@app.post("/api/jsg/orders/cancel")
def jsg_cancel(req: JsgCancelReq):
    """Cancel a LIVE broker order by its number."""
    from kairos import jsgtrade
    return _jsg_guard(jsgtrade.cancel, req.order_no)


# ---------------------------------------------------------------- fugu conductor
@app.get("/api/fugu")
def fugu_board():
    from kairos import fugu
    return _safe({"ok": True, "board": fugu.board()})


@app.get("/api/fugu/{symbol}")
def fugu_one(symbol: str):
    from kairos import fugu
    return _safe(fugu.conduct(symbol))


# ---------------------------------------------------------------- chat
class ChatReq(BaseModel):
    message: str


def _find_instrument(msg: str, name_tokens: bool = True):
    import re
    up = msg.upper()
    for w in re.findall(r"\b[A-Z][A-Z0-9.]{1,14}\b", up):
        inst = resolve(w.strip("."))
        if inst is not None:
            return inst
    if not name_tokens:
        return None
    for inst in universe():
        toks = [t for t in re.split(r"[^A-Z0-9]+", (inst.name or "").upper())
                if len(t) >= 5 and t not in ("FUTURES", "INDEX", "EXCHANGE")]
        for t in toks:
            if re.search(rf"\b{re.escape(t)}\b", up):
                return inst
    return None


def _horizon_from(msg: str) -> int:
    import re
    m = re.search(r"(\d{1,2})\s*(day|d\b|week)", msg.lower())
    if m:
        n = int(m.group(1)) * (5 if m.group(2) == "week" else 1)
        return max(1, min(21, n))
    low = msg.lower()
    if "today" in low or "tomorrow" in low or "overnight" in low:
        return 1
    if "week" in low or "month" in low:
        return 5
    return 5


def _plane_reply(r: dict) -> dict:
    """Map a command-plane result into the chat envelope the desk already renders.

    The trace travels with it so the operator can see which expert produced which
    number instead of taking a sentence on faith.
    """
    action = "refused" if r.get("intent") == "refused" else "command"
    return _safe({"ok": True, "reply": r.get("answer", ""), "action": action,
                  "data": {"kind": action, "intent": r.get("intent"),
                           "steps": r.get("steps") or [],
                           "trace": r.get("trace") or [],
                           "planner": r.get("planner"), "ms": r.get("ms")}})


# ---------------------------------------------------------------- command plane
class CommandReq(BaseModel):
    message: str
    budget_s: float = 30.0
    use_model: bool = True


@app.post("/api/command")
def command(req: CommandReq):
    """Plain language in, plan → tools → trace → numbers out.

    The plane carries no order, arming or kill-switch tool, so the worst a
    confused plan can do is run an analysis. The budget is clamped because the
    caller supplies it and a hung desk must not be able to pin the daemon.
    """
    from kairos import command as plane
    msg = req.message.strip()
    if not msg:
        return _safe({"ok": True, "intent": "help",
                      "answer": "Ask about an instrument, the PSX screen, the "
                                "conductor board, a backtest, a portfolio, my "
                                "realized record, or my health.",
                      "steps": [], "trace": [], "ms": 0, "planner": "none"})
    return _safe(plane.run(msg, budget_s=min(max(req.budget_s, 1.0), 120.0),
                           use_model=bool(req.use_model)))


@app.post("/api/chat")
def chat(req: ChatReq):
    import re
    msg = req.message.strip()
    if not msg:
        return {"ok": True, "reply": "Ask me anything — e.g. \"where is gold heading this week?\"",
                "action": "help", "data": None}
    low = msg.lower()
    facts = None

    def has(*words):
        return any(w in low for w in words)

    from kairos import command as plane
    refused = plane.guard(msg)
    if refused is not None:
        # one guard for every surface: an instruction to move money is refused
        # here, before a ticker is found and turned into an advisory card
        return _plane_reply(refused)

    # portfolio intent before symbol detection (lists of tickers)
    if has("my real", "js global", "live book", "livebook", "my stock account",
           "my actual positions", "my holdings", "my positions"):
        from kairos import livebook
        rows = livebook.review()
        if not rows:
            return {"reply": "You haven't given me your JS Global positions yet. "
                             "Open Trade Like a Pro → Your JS Global live book and "
                             "add each holding (symbol, quantity, average price) — "
                             "I'll watch them with stops, targets and exit tickets.",
                    "card": None}
        tot = sum(r.get("pnl") or 0 for r in rows)
        lines = [f"YOUR JS GLOBAL BOOK — {len(rows)} position(s), "
                 f"unrealised P&L {tot:+,.0f}:"]
        for r in rows:
            lines.append(
                f"• {r['symbol']}: {r.get('status', '—')} · last "
                f"{r.get('last', 0):,.2f} ({r.get('pnl_pct', 0):+.1f}%) · "
                f"engine {r.get('direction', '—')} score "
                f"{(r.get('score') or 0):+.3f}")
        return {"reply": "\n".join(lines), "card": None}

    if has("portfolio", "allocate", "allocat", "diversif", "split my", "weights for"):
        syms = re.findall(r"\b[A-Z][A-Z0-9.]{1,6}\b", msg.upper())
        picked = [s for s in syms if resolve(s) is not None]
        if len(picked) < 3:
            picked = ["GOLD", "WTI", "EURUSD", "SPX", "OGDC", "HBL"]
        r = ENGINE.portfolio(picked, capital=1_000_000.0)
        if not r.get("ok"):
            return {"ok": True, "reply": f"Portfolio engine could not run: {r.get('error')}",
                    "action": "error", "data": None}
        top_w = sorted(r["weights"].items(), key=lambda kv: -abs(kv[1]))[:6]
        lines = "\n".join(f"• {s} — {w*100:.1f}% (PKR/USD {round(r['allocations'][s]):,})"
                          for s, w in top_w if w > 0.001)
        reply = (f"Here is the annealed allocation for {len(r['weights'])} names — "
                 f"expected return {r['expected_return']*100:.2f}%, "
                 f"Sharpe {r['sharpe']}:\n{lines}\n"
                 "Open the Portfolio tab for the full profile.")
        return _safe({"ok": True, "reply": reply, "action": "portfolio",
                      "data": {"kind": "portfolio", "solver": r.get("solver"),
                               "weights": r["weights"], "allocations": r["allocations"],
                               "expected_return": r["expected_return"],
                               "expected_vol": r["expected_vol"], "sharpe": r["sharpe"]}})

    if has("trading rule", "your rule", "doctrine", "risk rule", "discipline",
           "wyckoff", "dow theory", "canslim", "buffett", "dalio", "lynch"):
        from kairos import rules as R
        for name, corpus in (("wyckoff", R.WYCKOFF_PHASES),
                             ("dow", R.DOW_THEORY), ("canslim", R.CANSLIM),
                             ("buffett", R.BUFFETT_FILTERS),
                             ("dalio", R.DALIO_PRINCIPLES),
                             ("lynch", R.LYNCH_CATEGORIES)):
            if name in low:
                lines = "\n".join(f"• {k}: {v}" for k, v in corpus.items())
                return {"ok": True,
                        "reply": f"{name.upper()} — a framework KAIROS carries:\n"
                                 f"{lines}\n\nMy live doctrine: {R.describe()}",
                        "action": "rules", "data": None}
        return {"ok": True, "reply": f"{R.describe()} I also encode Wyckoff, Dow "
                                     "theory, CANSLIM, Buffett, Dalio and Lynch — "
                                     "ask me about any of them.",
                "action": "rules", "data": None}

    if has("track record", "hit rate", "accuracy", "how good are you", "how reliable"):
        from kairos.feedback import track_record
        tr = track_record()
        if tr.get("resolved"):
            reply = (f"So far {tr['resolved']} of my predictions have hit their horizon. "
                     f"Directional hit rate {tr['hit_rate']*100:.1f}% "
                     f"(strong signals {tr.get('strong_signal_hit_rate') or 0:.0%}). "
                     f"Mean absolute return error {tr.get('mean_abs_error')}.")
        else:
            reply = ("I journal every prediction and grade myself once the horizon elapses — "
                     "no outcomes have matured yet. Ask again in a few days.")
        return {"ok": True, "reply": reply, "action": "feedback", "data": None}

    if has("news", "headline", "what happened", "what's going on"):
        r = news(limit=5)
        items = r.get("items", [])
        if not items:
            return {"ok": True, "reply": "The news feeds are unreachable right now.",
                    "action": "error", "data": None}
        lines = "\n".join(f"• {i['title']} — {i['source']}" for i in items)
        return {"ok": True, "reply": f"Top of the radar right now:\n{lines}",
                "action": "news", "data": None}

    inst = _find_instrument(msg, name_tokens=False)

    # A screen that also asks for a decision is two experts deep — ranking the
    # cross-section is only half the answer. Hand those to the plane, which
    # pipes the screen's own top name into the conductor and the risk posture;
    # a bare "best PSX stocks" keeps its leaderboard card.
    if inst is None and has("scan", "screen", "which name", "best stock", "psx") and \
            has("worth trading", "to trade", "trade today", "worth buying",
                "should i trade", "verdict"):
        return _plane_reply(plane.run(msg))

    if inst is None and has("scan", "stock", "top name", "which name",
                            "momentum leader", "psx", "screen", "kse"):
        r = scan(top=5)
        top = r.get("top", [])
        lines = "\n".join(
            f"{i+1}. {t['name']} — composite {t['composite_score']:.2f}, "
            f"1m momentum {(t['momentum_1m'] or 0):.2f}z"
            for i, t in enumerate(top))
        return _safe({"ok": True,
                      "reply": f"PSX cross-section ({r.get('universe')} names scanned). "
                               f"Today's strongest factor basket:\n{lines}\n"
                               "Open PSX Scan for the full leaderboard.",
                      "action": "scan", "data": {"kind": "scan", "rows": top}})

    inst = inst or _find_instrument(msg)

    if inst is None:
        # Nothing the keyword router owns — hand the ask to the command plane,
        # which plans over the tool registry and answers with a visible trace.
        try:
            r = plane.run(msg)
        except Exception as exc:
            r = {"intent": "none", "answer": f"plane failed: {exc}"}
        if r.get("intent") not in ("none", None):
            return _plane_reply(r)
        return {"ok": True,
                "reply": ("I can predict any instrument (\"is OGDC a buy?\", \"where is gold "
                          "this week\"), scan the PSX (\"best Pakistani stocks\"), build a "
                          "portfolio (\"allocate 1M across GOLD WTI SPX\"), or check my track "
                          "record. What would you like to know?"),
                "action": "help", "data": None}

    # ---- default: prediction for the detected instrument ----
    horizon = _horizon_from(msg)
    res = ENGINE.predict(inst.key, horizon=horizon, include_news=True)
    if not res.get("ok"):
        return {"ok": True, "reply": f"I couldn't run {inst.key}: {res.get('error')}",
                "action": "error", "data": None}
    s, rk = res["signal"], res["risk"]
    d = s["direction"]
    ev = res.get("event_risk") or {}
    caveats = []
    if ev.get("position_multiplier", 1.0) < 1.0:
        caveats.append(f"{ev['next_event']} on {ev['next_date']} — I am sizing down ×"
                       f"{ev['position_multiplier']}")
    if res.get("kse_regime_multiplier", 1.0) < 1.0:
        caveats.append("KSE-100 regime is cautious — PSX sizes cut")
    facts = (f"{res['name']} ({res['symbol']}) at {res['price']:,.2f} {res['currency']}: "
             f"call is {d}, expected return {(s['expected_return'] or 0)*100:+.2f}% over "
             f"{s['horizon_days']} days, P(up) {(res['probability_up'] or 0)*100:.0f}%, "
             f"confidence {(s['confidence'] or 0)*100:.0f}%, regime {res['regime']['regime']}, "
             f"suggested position {(rk['suggested_weight'] or 0)*100:.1f}%.")
    reply = facts + (" Note: " + "; ".join(caveats) + "." if caveats else "")

    # optional LLM phrasing — keep the numbers, change only the voice
    try:
        from kairos.config import CFG
        from kairos.llm import ROUTER
        if CFG.apis.get("openrouter", {}).get("enabled", False):
            model = CFG.apis.get("openrouter", {}).get("fast_model",
                                                       "deepseek/deepseek-chat-v3-0324")
            prompt = (f"You are KAIROS, a concise quant desk analyst. The user asked: "
                      f"\"{msg}\"\nYour engine produced these exact numbers — do not "
                      f"change or invent numbers, just answer in 2-4 natural sentences "
                      f"with a clear verdict and the suggested sizing:\n{facts}\n"
                      + ("Cautions: " + "; ".join(caveats) if caveats else ""))
            r = ROUTER.chat(prompt, model=model, max_tokens=220, use_cache=False)
            if r.ok and r.text:
                reply = r.text.strip()
    except Exception:
        pass
    return _safe({"ok": True, "reply": reply, "action": "predict",
                  "data": {"kind": "predict", "symbol": res["symbol"],
                           "name": res["name"], "price": res["price"],
                           "direction": s["direction"], "score": s["score"],
                           "confidence": s["confidence"],
                           "expected_return": s["expected_return"],
                           "probability_up": res["probability_up"],
                           "horizon_days": s["horizon_days"],
                           "weight": rk["suggested_weight"],
                           "regime": res["regime"]["regime"]}})



@app.get("/api/journal")
def journal(last: int = 50):
    from kairos.config import CFG
    p = CFG.data_dir / "audit_journal.jsonl"
    if not p.exists():
        return {"ok": True, "entries": []}
    lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    out = []
    for line in lines[-last:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return {"ok": True, "entries": out}


# ---------------------------------------------------------------- feedback
@app.get("/api/feedback")
def feedback(resolve_first: bool = False):
    from kairos.engine import ENGINE
    from kairos.feedback import confidence_adjustment, resolve_outcomes, track_record
    newly = resolve_outcomes(ENGINE.md) if resolve_first else 0
    tr = track_record()
    tr["newly_resolved"] = newly
    tr["confidence_adjustment"] = confidence_adjustment()
    return tr


# ---------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (ROOT / "api" / "ui.html").read_text(encoding="utf-8")
