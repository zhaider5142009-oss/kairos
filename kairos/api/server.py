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
from kairos.universe import universe  # noqa: E402

app = FastAPI(title="KAIROS Control Tower", version="2.0.0")


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


def _safe(obj):
    """Recursively coerce numpy scalars/dates into JSON-encodable Python."""
    import datetime as _dt

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
        return float(obj)
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
                out["ticks"].append(t)
        if benchmark:
            out["benchmark"] = br.benchmark_latency("XAUUSD", n=3)
    return out


# ---------------------------------------------------------------- journal
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
