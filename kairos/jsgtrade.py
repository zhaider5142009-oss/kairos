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
  * Account is the account NUMBER (accountNo), e.g. "90001".
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
per-order notional cap; broker price band — a stage the band table cannot
cover is journalled as unchecked rather than passing silently; qty and price
must be finite numbers; the staged-ticket file is rewritten atomically and a
queue that cannot be read is refused, not rebuilt from an empty read; a ticket
the TTL drops is journalled. PINs are never logged, never journalled, never
returned by any endpoint. PMX is not wired here — PMX flow remains tickets
only.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import JOURNAL, KILL
from .jsgsync import JsGlobalError, _conf, _net, _sess
from .store import atomic_write, dumps_strict

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
                "has_pin": False, "pending": len(pending()),
                "queue": queue_health()}
    return {"configured": True, "allow_orders": bool(c.get("allow_orders")),
            "has_pin": bool(c.get("trading_pin")),
            "order_account": str(c.get("order_account") or "auto (portal accountNo)"),
            "max_order_notional": c.get("max_order_notional"),
            "kill_switch": KILL.engaged(), "pending": len(pending()),
            "queue": queue_health()}


# ------------------------------------------------------------------ staging
_queue_lock = threading.RLock()
_last_queue_error: str | None = None


def queue_health() -> dict[str, Any]:
    """Whether the staged-ticket file can actually be answered.

    The desk has to print this beside the ticket list: an unreadable queue and
    a queue with nothing in it both display as "no staged orders", and the
    first means a ticket someone is waiting on may exist and simply not be
    visible.
    """
    return {"ok": _last_queue_error is None, "error": _last_queue_error,
            "path": str(PENDING_PATH)}


def _read_rows() -> list[dict[str, Any]] | None:
    """The tickets on disk, or None when the file could not be answered.

    None is not "no tickets". Every write here rebuilds the whole file from
    this read, so an unreadable file answered as `[]` would let the next stage
    or drop replace the queue with a list holding none of the tickets that
    survived in it — real order intent, gone, with nothing on the page and
    nothing in the trail to say it was ever there.
    """
    global _last_queue_error
    _last_queue_error = None
    try:
        raw = PENDING_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        _last_queue_error = f"could not be read ({str(exc)[:120]})"
        return None
    try:
        rows = json.loads(raw)
    except ValueError as exc:
        _last_queue_error = f"is not valid JSON ({str(exc)[:120]})"
        return None
    if not isinstance(rows, list):
        _last_queue_error = f"holds {type(rows).__name__}, not a list"
        return None
    return [r for r in rows if isinstance(r, dict) and r.get("id")]


def _load() -> list[dict[str, Any]]:
    """The tickets, or a refusal — for every path that is about to rewrite them."""
    rows = _read_rows()
    if rows is None:
        raise JsGlobalError(f"pending order queue at {PENDING_PATH} {_last_queue_error} "
                            f"— the file is left untouched and nothing is "
                            f"written over it")
    return rows


def _save(rows: list[dict[str, Any]]) -> None:
    """Replace the queue in one atomic step.

    A plain write_text here used to truncate the file before rewriting it, so a
    crash, a forced power cut or a full disk left a queue no parser accepted —
    and the read above turned that into "nothing staged".
    """
    with _queue_lock:
        PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write(PENDING_PATH, dumps_strict(rows, indent=1))
        except (OSError, ValueError) as exc:
            raise JsGlobalError(f"pending order queue at {PENDING_PATH} could not be "
                                f"written ({str(exc)[:120]}) — order refused") from exc


def _age_s(row: dict[str, Any], now: float) -> float | None:
    """Seconds since a ticket was staged, or None if it carries no usable time."""
    try:
        return now - float(row.get("t"))
    except (TypeError, ValueError):
        pass
    try:
        return (datetime.now()
                - datetime.fromisoformat(str(row.get("ts")))).total_seconds()
    except (TypeError, ValueError):
        return None


def _journal_gone(gone: list[tuple[dict[str, Any], float | None]],
                  now: float) -> None:
    """Record every ticket the TTL took out of the queue.

    An expired ticket used to vanish on the next write with no event behind it:
    the ack then said "no pending order (stale after 15 min)" and the trail
    showed a stage that simply stopped existing.
    """
    for row, age in gone:
        JOURNAL.log("JSG_TICKET_EXPIRED",
                    {"id": row.get("id"), "symbol": row.get("symbol"),
                     "side": row.get("side"), "qty": row.get("qty"),
                     "price": row.get("price"), "staged_ts": row.get("ts"),
                     "age_s": None if age is None else round(age),
                     "ttl_s": TTL_S})


def _prune(rows: list[dict[str, Any]],
           now: float | None = None) -> list[dict[str, Any]]:
    """The tickets still inside their TTL; the rest are journalled and dropped."""
    if not rows:
        return []
    now = time.time() if now is None else now
    live: list[dict[str, Any]] = []
    gone: list[tuple[dict[str, Any], float | None]] = []
    for row in rows:
        age = _age_s(row, now)
        if age is None or not math.isfinite(age) or age >= TTL_S:
            gone.append((row, age))
        else:
            live.append(row)
    if gone:
        _save(live)
        _journal_gone(gone, now)
    return live


def pending() -> list[dict[str, Any]]:
    """The staged tickets, for display. Never raises — see queue_health()."""
    global _last_queue_error
    rows = _read_rows()
    if rows is None:
        return []
    try:
        return _prune(rows)
    except JsGlobalError as exc:
        _last_queue_error = (f"tickets expired but the queue could not be rewritten "
                             f"({str(exc)[:120]})")
        return rows


def drop(pending_id: str) -> dict[str, Any]:
    """Discard a staged order. No broker call is ever made for it."""
    with _queue_lock:
        rows = _prune(_load())
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
    # Refuse before anything else spends a portal login or journalls a band
    # check for an order that will not be written: a queue this process cannot
    # read is a queue it must not replace.
    _load()

    sym = symbol.strip().upper()
    bside = SIDES.get(side.strip().upper())
    if bside is None:
        raise JsGlobalError("side must be BUY, SELL or SHORT")
    otype = ORDER_TYPES.get(order_type.strip().upper().replace("-", "_"))
    if otype is None:
        raise JsGlobalError(f"unknown order type {order_type} "
                            f"(want {'/'.join(ORDER_TYPES)})")
    qty, price = float(qty), float(price)
    # Checked before anything else, and not by comparison: every guard below
    # asks a question of the form "is this number past that one", and a NaN
    # answers no. A hole in the price feed therefore cleared the positivity
    # check, the notional cap and the band at once and went to the broker.
    if not (math.isfinite(qty) and math.isfinite(price)):
        raise JsGlobalError(f"order needs finite numbers — qty {qty!r}, "
                            f"price {price!r}")
    qty, price = int(qty), round(price, 2)
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
    else:
        # The caps table is a portal call, and a failed one is cached for five
        # minutes because each retry means another login and this account locks
        # out on repetition — so an order can legitimately go out with the band
        # unread, the broker enforcing it instead. It must not go out looking,
        # in the trail, as though the guard had run.
        JOURNAL.log("JSG_BAND_UNCHECKED", {"symbol": sym, "market": mkt,
                                           "price": price})

    wire = otype[1] if bside in ("SEL", "SHS") else otype[0]
    row = {"id": uuid.uuid4().hex[:8], "t": time.time(),
           "ts": datetime.now().isoformat(timespec="seconds"),
           "symbol": sym, "side": bside, "side_in": side.strip().upper(),
           "qty": qty, "price": price, "order_type": wire, "market": mkt,
           "note": note[:60]}
    # read-modify-write as one step: an armed autopilot stages while a human is
    # at the desk, and two unguarded rebuilds of the same file lose one of them
    with _queue_lock:
        _save(_prune(_load()) + [row])
    JOURNAL.log("JSG_STAGED", {k: row[k] for k in
                               ("id", "symbol", "side", "qty", "price",
                                "order_type", "market", "note")})
    return row


# ------------------------------------------------------------------ placing
def _remove(pending_id: str) -> str | None:
    """Take one ticket out of the queue, from the queue as it is now.

    The list read before the POST is stale by the time the POST returns — an
    order can sit in the broker for seconds, and a rewrite built from it would
    put back anything staged meanwhile. Returns what went wrong, if anything:
    callers must not treat a failed removal as a removal.
    """
    with _queue_lock:
        rows = _read_rows()
        if rows is None:
            return f"queue at {PENDING_PATH} {_last_queue_error}"
        try:
            _save([r for r in rows if r.get("id") != pending_id])
        except JsGlobalError as exc:
            return str(exc)
    return None


def ack(pending_id: str, pin: str | None = None) -> dict[str, Any]:
    """Release the staged order to the broker. This moves real money."""
    c = _conf()
    if not c.get("allow_orders"):
        raise JsGlobalError("order placement disabled")
    if KILL.engaged():
        raise JsGlobalError("kill switch engaged — order blocked")
    rows = _prune(_load())
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
        # broker. The record comes first — if the queue write below then fails,
        # the trail still says an order went out with an unknown outcome — and
        # the staged copy is dropped so nobody re-acks it into a duplicate.
        JOURNAL.log("JSG_UNCONFIRMED", {"id": pending_id, "symbol": row["symbol"],
                                        "side": row["side"], "qty": row["qty"],
                                        "price": row["price"],
                                        "error": str(exc)[:200]})
        stuck = _remove(pending_id)
        note = "" if stuck is None else (f" — and it is STILL in the staging queue "
                                         f"({stuck}); do not ack it again")
        raise JsGlobalError(f"{exc} — this order may or may not have reached "
                            "the broker; check your order list before staging "
                            f"it again{note}") from exc
    stuck = _remove(pending_id)
    _forget_book()
    if stuck is not None:
        # The broker has the order. Saying so beats a clean-looking answer that
        # leaves a re-ackable ticket behind for the same shares.
        JOURNAL.log("JSG_QUEUE_STUCK", {"id": pending_id, "symbol": row["symbol"],
                                        "order_no": out.get("order_no"),
                                        "error": stuck[:200]})
        out = {**out, "queue_warning": f"order went out but the staged copy could "
                                       f"not be removed ({stuck}) — do not ack it again"}
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
    if JOURNAL.log("JSG_INTENT", {"symbol": symbol, "side": side, "qty": qty,
                                  "price": price, "order_type": order_type,
                                  "market": market, "note": note}) is not True:
        # Same rule as the MT5 door: an order the trail cannot record is an
        # order that cannot later be audited or reversed. Refuse it before the
        # ticket exists, not after the broker has it.
        raise JsGlobalError(f"audit trail is not accepting writes "
                            f"({JOURNAL.last_error}) — order refused")
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
