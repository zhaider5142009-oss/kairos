"""JS Global order placement — the double-keyed live money path.

Mapped from the broker's own Web Trade Cast (site.js + the live order modal):

  POST /Home/PlaceOrder  {Account, BuySell, Market, OrderType, Volume,
                          Script, Exchange:"KSE", Price, PIN, LimitPrice}
  POST /Home/CancelOrder {orignalorderno}
  GET  /Home/GetUpperLowerCap -> [{symbol, market, upperCap, lowerLock}]

Portal quirks this module absorbs so callers can stay sane:
  * BuySell is broker code, not English: BUY / SEL / SHS (short sell).
  * OrderType is sent as the select VALUE on the buy side ("StopLoss") but
    the LABEL on the sell side ("Stop Loss") — both spellings map per side.
  * Account is the account NUMBER (accountNo), e.g. "53035".
  * The broker rejects a price outside the scrip's upper cap / lower lock;
    the portal only checks that client-side, so we check it here first.

The trading PIN is the broker's own second factor; KAIROS re-honours it with
the ack-token doctrine merged from JARVIS v1 (trading_signals.py): an order is
STAGED first (idempotent, journalled, 15-minute TTL), then ACKed either by a
human at the desk typing the PIN, or by an explicitly armed autopilot holding
the operator's PIN on file. Two independent keys must turn for real money to
move:

  1. config/env `js_global.allow_orders = true`  (the operator's standing yes)
  2. the trading PIN                             (per-order human knowledge)

Fail-closed extras: the kill switch blocks every stage/ack/placed/cancel;
per-order notional cap; broker price band; PINs are never logged, never
journalled, never returned by any endpoint. PMX is not wired here — PMX flow
remains tickets only.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import JOURNAL, KILL
from .jsgsync import JsGlobalError, _conf, _net, _sess

PENDING_PATH = CFG.data_dir / "jsg_pending.json"
TTL_S = 900.0
CAPS_TTL_S = 120.0

VALID_MARKETS = ("REG", "FUT", "ODL", "SIF", "SQR", "FSR", "LB", "BNB")
SIDES = {"BUY": "BUY", "SELL": "SEL", "SEL": "SEL", "SHORT": "SHS", "SHS": "SHS"}
# canonical order type -> (buy-side wire value, sell-side wire value)
ORDER_TYPES = {
    "LIMIT": ("Limit", "Limit"),
    "STOP_LOSS": ("StopLoss", "Stop Loss"),
    "MARKET": ("Market", "Market"),
    "FOK": ("FOK", "FOK"),
    "MIT": ("MIT", "MIT"),
}


def enabled() -> bool:
    """Both standing keys turned — the autopilot's auto-place precondition."""
    try:
        c = _conf()
    except JsGlobalError:
        return False
    return bool(c.get("allow_orders")) and bool(c.get("trading_pin"))


def state() -> dict[str, Any]:
    """What the desk shows before anyone risks a click."""
    try:
        c = _conf()
    except JsGlobalError as exc:
        return {"configured": False, "error": str(exc), "allow_orders": False,
                "has_pin": False, "pending": len(pending())}
    return {"configured": True, "allow_orders": bool(c.get("allow_orders")),
            "has_pin": bool(c.get("trading_pin")),
            "order_account": str(c.get("order_account") or "auto (portal accountNo)"),
            "max_order_notional": c.get("max_order_notional"),
            "kill_switch": KILL.engaged(), "pending": len(pending())}


# ------------------------------------------------------------------ staging
def _load() -> list[dict[str, Any]]:
    try:
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(rows: list[dict[str, Any]]) -> None:
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def _prune() -> list[dict[str, Any]]:
    return [r for r in _load() if time.time() - float(r.get("t", 0)) < TTL_S]


def pending() -> list[dict[str, Any]]:
    return _prune()


def drop(pending_id: str) -> dict[str, Any]:
    """Discard a staged order. No broker call is ever made for it."""
    rows = _prune()
    row = next((r for r in rows if r["id"] == pending_id), None)
    if row is None:
        raise JsGlobalError(f"no pending order {pending_id}")
    _save([r for r in rows if r["id"] != pending_id])
    JOURNAL.log("JSG_DROPPED", {"id": pending_id, "symbol": row["symbol"],
                                "side": row["side"], "qty": row["qty"],
                                "price": row["price"]})
    return {"ok": True, "dropped": pending_id}


def band(symbol: str, market: str | None = None) -> dict[str, Any]:
    """The broker's own price band for a scrip — shown before staging."""
    c = _conf()
    mkt = (market or c.get("market") or "REG").upper()
    b = _cap_for(symbol.strip().upper(), mkt)
    if not b:
        return {"symbol": symbol.strip().upper(), "market": mkt, "known": False}
    return {"symbol": symbol.strip().upper(), "market": mkt, "known": True,
            "lower_lock": b[0], "upper_cap": b[1]}


def stage(symbol: str, side: str, qty: float, price: float,
          order_type: str = "LIMIT", market: str | None = None,
          note: str = "KAIROS") -> dict[str, Any]:
    """Register a pending order request. No broker call yet."""
    c = _conf()
    if not c.get("allow_orders"):
        raise JsGlobalError("order placement disabled — set "
                            "js_global.allow_orders (or JSG_ALLOW_ORDERS)")
    if KILL.engaged():
        raise JsGlobalError("kill switch engaged — order staging blocked")

    sym = symbol.strip().upper()
    bside = SIDES.get(side.strip().upper())
    if bside is None:
        raise JsGlobalError("side must be BUY, SELL or SHORT")
    otype = ORDER_TYPES.get(order_type.strip().upper().replace("-", "_"))
    if otype is None:
        raise JsGlobalError(f"unknown order type {order_type} "
                            f"(want {'/'.join(ORDER_TYPES)})")
    qty, price = int(float(qty)), round(float(price), 2)
    if qty <= 0:
        raise JsGlobalError("quantity must be a positive whole number of shares")
    if price <= 0:
        raise JsGlobalError("price must be positive")

    mkt = (market or c.get("market") or "REG").upper()
    if mkt not in VALID_MARKETS:
        raise JsGlobalError(f"unknown PSX market segment {mkt}")

    cap = float(c.get("max_order_notional") or 250_000)
    if qty * price > cap:
        raise JsGlobalError(f"notional {qty * price:,.0f} over per-order "
                            f"cap {cap:,.0f}")

    band = _cap_for(sym, mkt)
    if band:
        lo, hi = band
        if not (lo <= price <= hi):
            raise JsGlobalError(f"{sym} price {price} is outside the broker's "
                                f"band {lo}–{hi} (lower lock / upper cap)")

    wire = otype[1] if bside in ("SEL", "SHS") else otype[0]
    row = {"id": uuid.uuid4().hex[:8], "t": time.time(),
           "ts": datetime.now().isoformat(timespec="seconds"),
           "symbol": sym, "side": bside, "side_in": side.strip().upper(),
           "qty": qty, "price": price, "order_type": wire, "market": mkt,
           "note": note[:60]}
    _save(_prune() + [row])
    JOURNAL.log("JSG_STAGED", {k: row[k] for k in
                               ("id", "symbol", "side", "qty", "price",
                                "order_type", "market", "note")})
    return row


# ------------------------------------------------------------------ placing
def ack(pending_id: str, pin: str | None = None) -> dict[str, Any]:
    """Release the staged order to the broker. This moves real money."""
    c = _conf()
    if not c.get("allow_orders"):
        raise JsGlobalError("order placement disabled")
    if KILL.engaged():
        raise JsGlobalError("kill switch engaged — order blocked")
    rows = _prune()
    row = next((r for r in rows if r["id"] == pending_id), None)
    if row is None:
        raise JsGlobalError(f"no pending order {pending_id} (stale after 15 min)")
    use_pin = str(pin or c.get("trading_pin") or "").strip()
    if not use_pin:
        raise JsGlobalError("trading PIN required — type it on the desk")
    try:
        out = _post_order(c, row, use_pin)
    except JsGlobalError as exc:
        # A POST that died in transit may still have been executed by the
        # broker. Drop the staged copy so nobody re-acks it into a duplicate.
        _save([r for r in rows if r["id"] != pending_id])
        JOURNAL.log("JSG_UNCONFIRMED", {"id": pending_id, "symbol": row["symbol"],
                                        "side": row["side"], "qty": row["qty"],
                                        "price": row["price"],
                                        "error": str(exc)[:200]})
        raise JsGlobalError(f"{exc} — this order may or may not have reached "
                            "the broker; check your order list before staging "
                            "it again") from exc
    _save([r for r in rows if r["id"] != pending_id])
    _forget_book()
    return out


def _forget_book() -> None:
    """Cash and holdings just moved at the broker — the cached real book is
    now a stale picture of a live account."""
    try:
        from . import jsgsync
        jsgsync.invalidate_book()
    except Exception:
        pass


def place_now(symbol: str, side: str, qty: float, price: float,
              order_type: str = "LIMIT", market: str | None = None,
              note: str = "KAIROS-auto") -> dict[str, Any]:
    """stage + ack with the stored PIN — the explicitly armed autopilot path."""
    row = stage(symbol, side, qty, price, order_type, market, note)
    return ack(row["id"])


def cancel(order_no: str | int) -> dict[str, Any]:
    c = _conf()
    if not c.get("allow_orders"):
        raise JsGlobalError("order placement disabled")
    if KILL.engaged():
        raise JsGlobalError("kill switch engaged — cancel blocked")
    r = _net(_sess().post, c["url"] + "/Home/CancelOrder",
             data={"orignalorderno": str(order_no)}, timeout=20)
    resp = r.text[:300]
    _forget_book()
    JOURNAL.log("JSG_CANCEL", {"order_no": str(order_no), "http": r.status_code,
                               "resp": resp})
    return {"ok": r.status_code == 200, "order_no": str(order_no),
            "response": resp}


# --------------------------------------------------------- broker price band
_caps: tuple[float, dict[str, tuple[float, float]]] | None = None
_caps_retry_at = 0.0
CAPS_RETRY_S = 300.0


def _cap_map(force: bool = False) -> dict[str, tuple[float, float]]:
    """symbol|market -> (lower lock, upper cap), cached like the portal does.

    A failed fetch also gets cached (for CAPS_RETRY_S): every retry here
    means another portal login, and this account locks out on repetition.
    """
    global _caps, _caps_retry_at
    now = time.time()
    if _caps and not force and now - _caps[0] < CAPS_TTL_S:
        return _caps[1]
    if not force and now < _caps_retry_at:
        return _caps[1] if _caps else {}
    try:
        c = _conf()
        r = _sess().get(c["url"] + "/Home/GetUpperLowerCap", timeout=25)
        rows = r.json()
        if not isinstance(rows, list):
            raise ValueError("unexpected caps payload")
    except Exception as exc:
        _caps_retry_at = now + CAPS_RETRY_S
        JOURNAL.log("JSG_CAPS_FAIL", {"error": str(exc)[:160],
                                      "retry_in_s": int(CAPS_RETRY_S)})
        return _caps[1] if _caps else {}
    _caps_retry_at = 0.0
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        try:
            key = f"{str(row['symbol']).upper()}|{str(row['market']).upper()}"
            out[key] = (float(row["lowerLock"]), float(row["upperCap"]))
        except (KeyError, TypeError, ValueError):
            continue
    _caps = (time.time(), out)
    return out


def _cap_for(symbol: str, market: str) -> tuple[float, float] | None:
    return _cap_map().get(f"{symbol.upper()}|{market.upper()}")


# ------------------------------------------------------------------ the POST
def _post_order(c: dict[str, Any], row: dict[str, Any],
                pin: str) -> dict[str, Any]:
    acct = str(c.get("order_account") or "") or _first_account(c)
    payload = {"Account": acct, "BuySell": row["side"], "Market": row["market"],
               "OrderType": row["order_type"], "Volume": row["qty"],
               "Script": row["symbol"], "Exchange": "KSE",
               "Price": row["price"], "PIN": pin,
               "LimitPrice": row["price"] if row["order_type"] == "Limit" else 0}
    r = _net(_sess().post, c["url"] + "/Home/PlaceOrder",
               data=payload, timeout=20)
    resp = _scrub(r.text[:300], pin)
    order_no = _order_no(resp)
    ok = r.status_code == 200 and not _rejected(resp)
    JOURNAL.log("JSG_PLACED", {"id": row["id"], "symbol": row["symbol"],
                               "side": row["side"], "qty": row["qty"],
                               "price": row["price"], "market": row["market"],
                               "order_type": row["order_type"], "account": acct,
                               "http": r.status_code, "order_no": order_no,
                               "accepted": ok, "resp": resp,
                               "note": row.get("note", "")})
    return {"ok": ok, "symbol": row["symbol"], "side": row["side"],
            "qty": row["qty"], "price": row["price"], "account": acct,
            "order_no": order_no, "response": resp}


def _first_account(c: dict[str, Any]) -> str:
    from . import jsgsync
    for acc in jsgsync._accounts(_sess()):
        no = str(acc.get("accountNo") or acc.get("accountno") or "")
        if no:
            return no
    raise JsGlobalError("no broker account configured and none returned "
                        "by the portal (js_global.order_account)")


def _order_no(resp: str) -> str | None:
    m = re.search(r"order\s*(?:no|number|id|#)[^0-9]{0,4}(\d{4,})", resp, re.I)
    if m:
        return m.group(1)
    m = re.fullmatch(r"\s*(\d{4,})\s*", resp)
    return m.group(1) if m else None


def _rejected(resp: str) -> bool:
    t = resp.lower()
    return any(w in t for w in ("invalid", "rejected", "not allowed",
                                "unsuccess", "failed", "error", "blocked"))


def _scrub(text: str, pin: str) -> str:
    """Never let a broker echo of the PIN reach logs, APIs or the desk.

    Whole-token only: a 4-digit PIN is routinely a substring of an order
    number, and blind replacement would corrupt the receipt we parse.
    """
    pin = (pin or "").strip()
    if len(pin) < 4:
        return text
    return re.sub(rf"(?<!\w){re.escape(pin)}(?!\w)", "***", text)
