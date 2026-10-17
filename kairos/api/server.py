"""KAIROS control tower — FastAPI backend + single-page human-friendly UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

from kairos.engine import ENGINE  # noqa: E402
from kairos.universe import resolve, universe  # noqa: E402

app = FastAPI(title="KAIROS Control Tower", version="2.0.0")


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
        return system_health()
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
    from kairos.execution import Mt5Bridge
    br = Mt5Bridge()
    connected = br.connect()
    out = {"ok": connected, "connected": connected, "ticks": []}
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
    from kairos.execution import JOURNAL, KILL
    act = req.action.lower()
    if act == "configure":
        AUTOPILOT.configure(req.symbols, req.interval_s,
                            req.entry_threshold, req.max_trades_per_cycle)
        return _safe(AUTOPILOT.status())
    if act == "start":
        out = AUTOPILOT.start(mode=req.mode, confirm_live=req.confirm_live)
        return _safe(out)
    if act == "stop":
        return _safe(AUTOPILOT.stop())
    if act == "cycle":
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
    from kairos import signalcard
    try:
        path, cap = signalcard.card_for(symbol)
    except Exception as e:
        raise HTTPException(404, f"no signal for {symbol!r}: {e}")
    if caption:
        return {"ok": True, "caption": cap}
    return FileResponse(path, media_type="image/png",
                        filename=f"kairos-{symbol.upper()}-signal.png")


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

    # portfolio intent before symbol detection (lists of tickers)
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
