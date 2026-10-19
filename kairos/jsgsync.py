"""JS Global account sync — READ-ONLY bridge to Web Trade Cast.

KAIROS cannot see the Java TradeCast desktop app, but JS Global also serves
the same account over its web terminal (wt.jsglobalonline.com). This module
logs in with the git-ignored `js_global:` section of config.yaml, reads the
real portfolio, and imports net holdings into the live book so the engine
can review them like any other position.

Hard rules:
  * Only read endpoints are ever called here (GetUserData, GetOpenPosition,
    GetCollaterals, GetPortfolioHoldings, GetAccountBalance, GetActivityLog,
    GetUpperLowerCap). Order placement lives in kairos/jsgtrade.py, which is
    double-keyed (js_global.allow_orders AND the trading PIN) and never
    reaches the broker unless the operator turned both keys.
  * Exactly ONE login attempt per sync call; on failure we surface the error
    instead of risking an account lockout.
  * Credentials never appear in code, logs, git or returned payloads.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import requests

from .config import CFG

SESSION_TTL_S = 600.0
_TIMEOUT = 25
# a live cycle reads the real book more than once (posture + each order); the
# portal is the same session either way, so serve repeats from a short cache
_BOOK_TTL = 45.0
_lock = threading.Lock()
_session: tuple[float, "requests.Session"] | None = None
_book: dict[str, Any] | None = None
_book_ts = 0.0


class JsGlobalError(RuntimeError):
    pass


class JsGlobalAuthError(JsGlobalError):
    pass


def _conf() -> dict[str, Any]:
    """js_global config, with environment overrides for headless deploys.

    A free cloud server carries no config.yaml, so the whole bridge is
    drivable from secrets: JSG_PORTAL / JSG_LOGIN / JSG_PASSWORD and, for the
    order path, JSG_PIN / JSG_ALLOW_ORDERS / JSG_ORDER_ACCOUNT.
    """
    sec = dict(CFG.section("js_global") or {})
    env = os.environ
    url = str(env.get("JSG_PORTAL") or sec.get("portal_url")
              or "https://wt.jsglobalonline.com").rstrip("/")
    login = str(env.get("JSG_LOGIN") or sec.get("login") or "")
    pw = str(env.get("JSG_PASSWORD") or sec.get("password") or "")
    if not (login and pw):
        raise JsGlobalAuthError("js_global.login/password not set "
                                "(config.yaml or JSG_LOGIN/JSG_PASSWORD)")
    allow = env.get("JSG_ALLOW_ORDERS", sec.get("allow_orders", False))
    if isinstance(allow, str):
        allow = allow.strip().lower() in ("1", "true", "yes", "on")
    return {"url": url, "login": login, "password": pw,
            "portal_url": url,
            "allow_orders": bool(allow),
            "trading_pin": str(env.get("JSG_PIN") or sec.get("trading_pin") or ""),
            "order_account": str(env.get("JSG_ORDER_ACCOUNT")
                                 or sec.get("order_account") or ""),
            "market": str(sec.get("market") or "REG").upper(),
            "max_order_notional": float(sec.get("max_order_notional") or 250_000)}


def _digit_boxes(html: str) -> list[int]:
    boxes = []
    for tag in re.findall(r"<input[^>]*>", html):
        m = re.search(r'id="Digit(\d+)"', tag)
        if m and "disabled" not in tag:
            boxes.append(int(m.group(1)))
    return sorted(boxes)


def _net(fn, *a, **kw):
    """Translate a transport failure into one honest JsGlobalError.

    Never retries: a second login attempt against this portal risks an
    account lockout, and a retried PlaceOrder risks a duplicate real order.
    """
    try:
        return fn(*a, **kw)
    except requests.RequestException as exc:
        raise JsGlobalError(f"portal unreachable: {type(exc).__name__}") from exc


def _login(c: dict[str, Any]) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                      "X-Requested-With": "XMLHttpRequest",
                      "Accept": "application/json, text/javascript, */*; q=0.01"})
    r = _net(s.get, c["url"], timeout=_TIMEOUT)
    pw = c["password"]
    data: dict[str, str] = {"UserName": c["login"]}
    for i in _digit_boxes(r.text):
        if i > len(pw):
            raise JsGlobalError(f"portal asks for password char {i} but stored password is shorter "
                                "— update js_global.password in config.yaml")
        data[f"Digit{i}"] = pw[i - 1]
    if len(data) == 1:
        raise JsGlobalError("login form parse failed (no password boxes found)")
    _net(s.post, c["url"] + "/Home/_Login", data=data, timeout=_TIMEOUT)
    _net(s.get, c["url"] + "/Home/Index", timeout=_TIMEOUT)
    if not _accounts(s):
        raise JsGlobalAuthError("login rejected — check js_global credentials in config.yaml")
    return s


def _sess() -> requests.Session:
    """Reuse a warm session; fresh login otherwise (single attempt)."""
    global _session
    with _lock:
        if _session and time.time() - _session[0] < SESSION_TTL_S:
            return _session[1]
        s = _login(_conf())
        _session = (time.time(), s)
        return s


def _get(path: str, params: dict | None = None) -> Any:
    global _session
    s = _sess()
    r = _net(s.get, _conf()["url"] + path, params=params or {}, timeout=_TIMEOUT)
    if r.status_code in (401, 403) or "UserName" in r.text[:2000] and "Digit1" in r.text[:2000]:
        with _lock:
            _session = None
        raise JsGlobalAuthError("session expired — retry the sync")
    if not r.text.strip():
        return None
    try:
        import json as _json
        j = _json.loads(r.text)
        return _json.loads(j) if isinstance(j, str) else j
    except ValueError:
        return r.text


def _accounts(s: requests.Session) -> list[dict[str, Any]]:
    """Accounts behind the current session; [] when the portal refuses them.

    A transport failure is NOT swallowed here — otherwise an outage would be
    reported as "login rejected" and the operator would rotate good
    credentials for no reason.
    """
    try:
        j = _net(s.get, _conf()["url"] + "/Home/GetUserData",
                 timeout=_TIMEOUT).json()
    except JsGlobalError:
        raise
    except Exception:
        return []
    return j if isinstance(j, list) else []


def accounts() -> list[dict[str, Any]]:
    return _accounts(_sess())


def open_positions(account_no: str, account_name: str) -> list[dict[str, Any]]:
    """Normalized net holdings from both broker views.

    GetOpenPosition keys on the account NAME; GetCollaterals (the margin /
    portfolio book where JS Global actually shows holdings) keys on the
    account NUMBER. Row shapes differ — collapse to symbol/qty/avg/mtm.
    """
    rows: list[dict[str, Any]] = []
    try:
        data = _get("/Home/GetOpenPosition", {"account": account_name})
    except JsGlobalAuthError:
        raise
    except Exception:
        data = None
    if isinstance(data, list):
        for r in data:
            try:
                rows.append({"symbol": str(r.get("symbol") or "").upper(),
                             "qty": float(str(r.get("netQty") or 0).replace(",", "")),
                             "avg": float(str(r.get("average") or 0).replace(",", "")),
                             "pending_sell": 0.0, "mtm": None, "market": r.get("market"),
                             "source": "open"})
            except (TypeError, ValueError):
                continue
    try:
        data = _get("/Home/GetCollaterals", {"account": account_no})
    except JsGlobalAuthError:
        raise
    except Exception:
        data = None
    if isinstance(data, list):
        for r in data:
            try:
                rows.append({"symbol": str(r.get("symbol") or "").upper(),
                             "qty": float(str(r.get("quantityTotal") or 0).replace(",", "")),
                             "avg": float(str(r.get("avgRateBuy") or 0).replace(",", "")),
                             "pending_sell": float(str(r.get("pendingSell") or 0).replace(",", "")),
                             "mtm": r.get("mtmPrice"), "market": r.get("market"),
                             "source": "collateral"})
            except (TypeError, ValueError):
                continue
    return rows


def ledger_balance(account_no: str) -> float | None:
    """Free cash in PKR.

    GetAccountBalance keys on the account NUMBER. Passing the holder name
    returns "0.0" instead of an error, which would silently look like an
    empty wallet, so callers must never use a name here.
    """
    try:
        v = _get("/Home/GetAccountBalance", {"account": account_no})
        return float(str(v).replace(",", "")) if v is not None else None
    except Exception:
        return None


def real_book(force: bool = False) -> dict[str, Any]:
    """The REAL account in one read-only shot: free cash + MTM holdings.

    The live risk posture runs on these numbers. It must never fall back to
    an estimate: a transport or login failure raises and the caller stops
    trading that venue.
    """
    global _book, _book_ts
    with _lock:
        if _book is not None and not force and time.time() - _book_ts < _BOOK_TTL:
            return dict(_book)
    accs = accounts()
    if not accs:
        raise JsGlobalAuthError("no accounts returned after login")
    holdings: list[dict[str, Any]] = []
    cash_total = 0.0
    names: list[str] = []
    for acc in accs:
        no = str(acc.get("accountNo") or "")
        names.append(str(acc.get("accountName") or no))
        if not no:
            continue
        bal = ledger_balance(no)
        if bal is None:
            raise JsGlobalError(f"balance unavailable for account {no}")
        cash_total += bal
        for r in open_positions(no, str(acc.get("accountName") or no)):
            qty = float(r.get("qty") or 0)
            if qty <= 0:
                continue
            mtm = _num(r.get("mtm")) or float(r.get("avg") or 0)
            holdings.append({"symbol": r["symbol"], "qty": qty,
                             "avg": float(r.get("avg") or 0), "mtm": mtm,
                             "value": round(mtm * qty, 2),
                             "pending_sell": float(r.get("pending_sell") or 0),
                             "account": no})
    book_value = round(sum(h["value"] for h in holdings), 2)
    out = {"ok": True, "venue": "JS_GLOBAL", "accounts": names,
           "cash": round(cash_total, 2), "holdings_value": book_value,
           "equity": round(cash_total + book_value, 2),
           "equity_basis": "free cash + mark-to-market holdings",
           "holdings": holdings, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with _lock:
        _book, _book_ts = out, time.time()
    return dict(out)


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def invalidate_book() -> None:
    """Forget the cached real book — call after anything that moves cash."""
    global _book, _book_ts
    with _lock:
        _book = None
        _book_ts = 0.0


def pending_orders(account: str) -> list[dict[str, Any]]:
    try:
        data = _get("/Home/GetActivityLog", {"account": account})
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [o for o in data if str(o.get("action", "")).upper() in ("PEN", "PENDING")]


def sync(force: bool = False) -> dict[str, Any]:
    """Import the broker's real holdings into the live book (upsert)."""
    from . import livebook
    from .universe import resolve

    global _session, _book
    if force:
        with _lock:
            _session = None
            _book = None

    accs = accounts()
    if not accs:
        raise JsGlobalAuthError("no accounts returned after login")

    imported: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    rows_all: list[dict[str, Any]] = []
    for acc in accs:
        no = str(acc.get("accountNo") or "")
        name = str(acc.get("accountName") or no)
        bal = ledger_balance(no)
        rows = sorted(open_positions(no, name),
                      key=lambda r: 0 if r["source"] == "collateral" else 1)
        rows_all.extend(rows)
        for row in rows:
            sym, qty, avg = row["symbol"], row["qty"], row["avg"]
            if not sym or qty <= 0 or sym in seen:
                continue
            seen.add(sym)
            if resolve(sym) is None:
                skipped.append(sym)
                continue
            livebook.add(sym, qty, avg if avg > 0 else 1.0)
            imported.append({"symbol": sym, "qty": int(qty), "avg": avg,
                             "pending_sell": row["pending_sell"], "mtm": row["mtm"],
                             "account": name, "balance": bal})

    pend: list[dict[str, Any]] = []
    for acc in accs:
        for o in pending_orders(str(acc.get("accountNo") or "")):
            pend.append({"scrip": o.get("scrip"), "side": o.get("type"),
                         "qty": o.get("totalVolume"), "price": o.get("price"),
                         "time": o.get("time")})

    book = livebook.positions()
    broker_syms = {r["symbol"] for r in rows_all if r["qty"] > 0 or r["pending_sell"] > 0}
    # only prune when the broker actually reported a position set — an empty
    # or failed read must never wipe the live book
    released = [s for s in book if s.upper() not in broker_syms] if rows_all else []
    for s in released:
        livebook.remove(s)
    missing = [s for s in book if s not in {i["symbol"] for i in imported}]

    report = {"ok": True, "accounts": [a.get("accountName") for a in accs],
              "imported": imported, "skipped_not_in_universe": skipped,
              "released_sold": released,
              "pending_orders": pend, "manual_only_in_livebook": missing,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        from .execution import JOURNAL
        JOURNAL.log("JSG_SYNC", {"accounts": report["accounts"],
                                 "imported": [i["symbol"] for i in imported]})
    except Exception:
        pass
    livebook._invalidate()
    return report
