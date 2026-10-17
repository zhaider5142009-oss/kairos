"""Live book — the user's REAL JS Global (PSX) holdings, watched by KAIROS.

JS Global has no retail order API, so KAIROS cannot see or touch these
positions directly. Enter them once (symbol, quantity, average price) and
every review runs them through the same institutional layer as the paper
book: live price, engine verdict, recommended stop, and an alert state.
When a position should be exited, KAIROS produces a broker-ready JS Global
ticket for human confirmation — it never pretends to have a live PSX feed.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from .config import CFG

CACHE_S = 300.0
_cache: tuple[float, list[dict[str, Any]]] | None = None


def _path():
    return CFG.data_dir / "livebook.json"


def positions() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data.get("positions") or {}
    except Exception:
        return {}


def _save(positions: dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"positions": positions,
                             "updated": datetime.now().isoformat(timespec="seconds")},
                            indent=2), encoding="utf-8")


def add(symbol: str, qty: float, entry: float) -> dict[str, Any]:
    from .universe import resolve
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol required")
    if resolve(sym) is None:
        raise ValueError(f"{sym} is not in the KAIROS universe")
    if qty <= 0 or entry <= 0:
        raise ValueError("qty and entry must be positive")
    pos = positions()
    pos[sym] = {"qty": int(float(qty)), "entry": float(entry),
                "added": datetime.now().isoformat(timespec="seconds"),
                "stop": None}
    _save(pos)
    _invalidate()
    return pos[sym]


def remove(symbol: str) -> bool:
    pos = positions()
    sym = (symbol or "").strip().upper()
    if sym in pos:
        pos.pop(sym)
        _save(pos)
        _invalidate()
        return True
    return False


def set_stop(symbol: str, stop: float | None) -> bool:
    pos = positions()
    sym = (symbol or "").strip().upper()
    if sym not in pos:
        return False
    pos[sym]["stop"] = float(stop) if stop else None
    _save(pos)
    _invalidate()
    return True


def _invalidate() -> None:
    global _cache
    _cache = None


# ------------------------------------------------------------------ review
def review(force: bool = False) -> list[dict[str, Any]]:
    """Engine verdict over every real holding (cached ~5 minutes)."""
    global _cache
    if _cache and not force and time.time() - _cache[0] < CACHE_S:
        return _cache[1]
    from .engine import ENGINE
    rows: list[dict[str, Any]] = []
    for sym, p in positions().items():
        row: dict[str, Any] = {"symbol": sym, "qty": p["qty"], "entry": p["entry"],
                               "stop": p.get("stop")}
        try:
            res = ENGINE.predict(sym)
            s = res["signal"]
            bands = res.get("confidence_band") or {}
            rec_stop = bands.get("p10")
            price = float(res["price"])
            stop = float(p.get("stop") or rec_stop or 0)
            pnl = (price - float(p["entry"])) * float(p["qty"])
            row.update({
                "name": res["name"], "last": price,
                "pnl": round(pnl, 2),
                "pnl_pct": round((price / float(p["entry"]) - 1) * 100, 2),
                "direction": s["direction"], "score": s["score"],
                "confidence": s["confidence"],
                "expected_return": s["expected_return"],
                "rec_stop": rec_stop,
                "target": (res.get("targets") or {}).get(f"{s['horizon_days']}d"),
                "status": _status(price, float(p["entry"]), stop, s["score"]),
            })
        except Exception as exc:
            row.update({"status": "NO DATA", "error": str(exc)[:120]})
        rows.append(row)
    rows.sort(key=lambda r: r.get("pnl_pct") or 0)
    _cache = (time.time(), rows)
    return rows


def _status(price: float, entry: float, stop: float, score: float) -> str:
    if stop and price <= stop:
        return "STOP HIT — exit"
    if score <= -0.15:
        return "EXIT SIGNAL"
    if score >= 0.15:
        return "HOLD — conviction"
    if price > entry:
        return "HOLD — in profit"
    return "WATCH — weak tape"


# ------------------------------------------------------------------ tickets
def make_sell_ticket(symbol: str) -> dict[str, Any]:
    """Broker-ready JS Global SELL ticket for a real holding (or a part)."""
    from .execution import JOURNAL, JsGlobalTicket, OrderIntent
    sym = (symbol or "").strip().upper()
    p = positions().get(sym)
    if not p:
        raise ValueError(f"{sym} is not in the live book")
    res = None
    try:
        from .engine import ENGINE
        res = ENGINE.predict(sym)
    except Exception:
        res = None
    price = float((res or {}).get("price") or 0)
    intent = OrderIntent(symbol=sym, side="SELL", quantity=int(p["qty"]),
                         order_type="MARKET", venue="PSX_JSGLOBAL",
                         comment=f"livebook exit @ {price:,.2f}")
    ticket = JsGlobalTicket().build_ticket(intent)
    JOURNAL.log("LIVEBOOK_TICKET", {"symbol": sym, "ticket": ticket["ticket_id"]})
    _invalidate()
    return ticket
