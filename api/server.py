"""KAIROS control tower — FastAPI backend + single-page human-friendly UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis.engine import ENGINE  # noqa: E402
from jarvis.universe import universe  # noqa: E402

app = FastAPI(title="KAIROS Control Tower", version="2.0.0")


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


# ---------------------------------------------------------------- health
@app.get("/api/health")
def health():
    from jarvis.llm import health_probe
    from jarvis.wsl_bridge import WSL
    try:
        h = ENGINE.md.health()
    except Exception as exc:
        h = {"market-data": f"error: {exc}"}
    try:
        llm = health_probe()
    except Exception:
        llm = {"ok": False, "provider": "down"}
    return {"ok": True, "market_data": h,
            "llm": {"ok": llm.get("ok", False), "provider": llm.get("provider"),
                    "models": llm.get("models", 0)},
            "wsl": {"available": WSL.wsl_available(),
                    "models_ready": all(WSL.models_ready().values())},
            "universe_size": len(universe())}


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
    return res


# ---------------------------------------------------------------- scan
@app.get("/api/scan")
def scan(top: int = Query(20, ge=5, le=100)):
    return ENGINE.scan(top=top)


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
    return res


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
    from jarvis.execution import OrderIntent, PaperBroker, RiskGate
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
    return {"ok": True, "fill": fill.__dict__, "signal": s,
            "account": broker.account()}


@app.get("/api/paper/account")
def paper_account():
    from jarvis.execution import PaperBroker
    return {"ok": True, "account": PaperBroker().account(),
            "positions": PaperBroker().open_positions()}


# ---------------------------------------------------------------- backtest
@app.get("/api/backtest")
def backtest(symbols: str, days: int = Query(60, ge=20, le=250)):
    from jarvis.backtest import run_paper_session
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "no symbols given")
    return run_paper_session(ENGINE.md, syms, ENGINE, days=days)


# ---------------------------------------------------------------- mt5
@app.get("/api/mt5")
def mt5(symbols: str = "XAUUSD,XAGUSD,EURUSD", benchmark: bool = False):
    from jarvis.execution import Mt5Bridge
    br = Mt5Bridge()
    connected = br.connect()
    out = {"ok": connected, "connected": connected, "ticks": []}
    if connected:
        for sym in [s.strip() for s in symbols.split(",") if s.strip()]:
            t = br.tick(sym)
            if t:
                out["ticks"].append(t)
        if benchmark:
            out["benchmark"] = br.benchmark_orders("XAUUSD", n=3)
    return out


# ---------------------------------------------------------------- journal
@app.get("/api/journal")
def journal(last: int = 50):
    p = ROOT / "data" / "audit_journal.jsonl"
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


# ---------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (ROOT / "api" / "ui.html").read_text(encoding="utf-8")
