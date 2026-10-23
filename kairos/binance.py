"""Binance spot — the third venue: tape, account read, and a double-keyed order door.

KAIROS now serves three execution venues, and each one gets the same shape:
MT5/PMEX for commodities and FX, JS Global for PSX equities, Binance for crypto.
None of them is special-cased in the loop; each exposes `connected`-style health,
an account read that returns `None` rather than a flattering zero, and an order
path that refuses unless every key in front of it is turned.

What was measured on this box (2026-10-23, read-only probes, no order sent):
`/api/v3/ping` 0.59 s, `klines` 1.36 s cold, a signed `/api/v3/account` 0.31 s
with 866 assets and no non-zero balance, and `apiRestrictions` saying
`enableReading: true, enableSpotAndMarginTrading: false, ipRestrict: false`.
So this key *cannot* place spot orders, and that is reported as the reason rather
than discovered as a rejection.

Two security decisions live here deliberately:

* The signature travels in the query string, so a signed URL is a credential.
  Every failure message is redacted before it can reach a log, a journal row or
  an operator's terminal — the shared `_http_get()` in `data.py` embeds the full
  URL in its `ConnectionError`, which is fine for public endpoints and not for
  this one.
* The key material is read from the git-ignored `config.yaml` (`binance:`) or the
  environment, and never written anywhere else. `state()` reports only whether
  each half is present.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from .config import CFG
from .execution import Fill, JOURNAL, KILL, OrderIntent

VENUE = "BINANCE"
DEFAULT_BASE = "https://api.binance.com"
RECV_WINDOW_MS = 5000
# A dead host must not be re-tried every cycle by every symbol: the breaker
# window is one full autopilot interval, and `alive()` is what health reports.
DOWN_PENALTY_S = 900.0
# The quote side of every pair KAIROS trades, and the set treated as cash rather
# than as a position. A USDT balance is not a bet on USDT.
QUOTE_ASSET = "USDT"
QUOTE_ASSETS = ("USDT", "FDUSD", "BUSD", "USDC", "TUSD", "USDP")


def pair_for(symbol: str) -> str | None:
    """Map a name to a Binance spot pair — the venue's symbol, or the symbol itself.

    The rest of KAIROS speaks in canonical keys ("BTC", "OGDC"); the exchange only
    answers to "BTCUSDT". The pair comes from the registry, not from string
    surgery here, so a key the registry does not list on this venue has no pair and
    the order is refused before anything is signed.
    """
    from .universe import resolve
    inst = resolve(str(symbol or "").upper())
    if inst is not None and inst.binance:
        return inst.binance
    s = str(symbol or "").upper()
    return s if s.endswith(QUOTE_ASSET) and len(s) > len(QUOTE_ASSET) else None


class BinanceError(Exception):
    """A refusal that is safe to show: no key, no signature, no full URL."""


def _conf() -> dict[str, Any]:
    c = dict(CFG.section("binance") or {})
    env = {"api_key": "BINANCE_API_KEY", "secret_key": "BINANCE_SECRET",
           "base_url": "BINANCE_BASE_URL", "allow_orders": "BINANCE_ALLOW_ORDERS"}
    for field, var in env.items():
        if os.environ.get(var):
            c[field] = os.environ[var]
    if isinstance(c.get("allow_orders"), str):
        c["allow_orders"] = c["allow_orders"].strip().lower() in ("1", "true", "yes", "on")
    return c


def configured() -> bool:
    """Both halves of the key pair present — the door's first lock."""
    c = _conf()
    return bool(c.get("api_key")) and bool(c.get("secret_key"))


def enabled() -> bool:
    """The order door's two keys, mirroring `jsgtrade.enabled()`: the operator
    must arm it in config *and* hold a usable key pair. Either half alone places
    nothing."""
    try:
        return bool(_conf().get("allow_orders")) and configured()
    except Exception:
        return False


def _redact(text: str) -> str:
    """Strip the signature (and any key echo) from anything about to be shown."""
    out = re.sub(r"signature=[0-9a-fA-F]+", "signature=[redacted]", str(text))
    return out[:300]


def _num(value: Any) -> str:
    """A price or quantity as plain decimal text — no exponent, no trailing dot.

    `str(1e-05)` sends "1e-05" and the exchange reads that as an illegal
    character; `f"{9999.0:.8f}"` keeps the digits but trimming the zeros leaves
    "9999.", which is rejected the same way. Both are refusals that look like a
    sizing bug, so the number is put in the venue's own shape here.
    """
    text = f"{float(value):.8f}".rstrip("0")
    return text[:-1] if text.endswith(".") else text


class BinanceClient:
    """Public klines, signed account reads and gated orders against one host."""

    def __init__(self) -> None:
        self.error: str | None = None
        self.last_latency_s: float | None = None
        self._down_until = 0.0
        self._clock_skew_ms = 0
        self._skew_checked = False
        self._perms: dict[str, Any] | None = None
        self._perms_at = 0.0
        self.outcomes: dict[str, str] = {}

    # Credentials are read on every request, not frozen at construction, for the
    # same reason `jsgtrade` re-reads `_conf()`: the arming key and the API key
    # are expected to change while the tower is running — and a rotated key that
    # only takes effect after a restart is a key that is not rotated.
    @property
    def base(self) -> str:
        return str(_conf().get("base_url") or DEFAULT_BASE).rstrip("/")

    @property
    def key(self) -> str:
        return str(_conf().get("api_key") or "")

    @property
    def secret(self) -> str:
        return str(_conf().get("secret_key") or "")

    @property
    def timeout(self) -> int:
        return int(_conf().get("timeout_s") or 15)

    # ------------------------------------------------------------------ transport
    def _send(self, path: str, params: dict[str, Any] | None = None,
              signed: bool = False, method: str = "GET") -> Any:
        if signed and not self.secret:
            # Recorded before raising: `account()` turns a BinanceError into None,
            # and a silent None from a missing key is the same shape as a silent
            # None from an outage. The desk needs to be able to tell them apart.
            self.error = f"{path} needs a key pair; none is configured"
            raise BinanceError(self.error)
        if signed and not self._skew_checked:
            # A rejected timestamp looks exactly like a rejected order; measuring
            # the box clock once per process is cheaper than that ambiguity.
            self.sync_clock()
        q = dict(params or {})
        if signed:
            q["timestamp"] = int(time.time() * 1000) + self._clock_skew_ms
            q["recvWindow"] = RECV_WINDOW_MS
            body = urllib.parse.urlencode(q)
            q["signature"] = hmac.new(self.secret.encode(), body.encode(),
                                      hashlib.sha256).hexdigest()
        query = urllib.parse.urlencode(q)
        url = self.base + path + (f"?{query}" if query else "")
        hdrs = {"X-MBX-APIKEY": self.key} if self.key else {}
        req = urllib.request.Request(url, method=method, headers=hdrs)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8", "replace") or "{}")
            self.last_latency_s = round(time.time() - t0, 3)
            self._down_until = 0.0
            return payload
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read().decode("utf-8", "replace"))
            self.error = f"{path} HTTP {exc.code}: {detail}"
            if exc.code in (418, 429):        # rate-limited: back off, do not hammer
                self._down_until = time.time() + DOWN_PENALTY_S
            raise BinanceError(self.error) from None
        except Exception as exc:
            # `path`, never the URL: a signed URL is a credential.
            self.error = f"{path} unreachable ({type(exc).__name__}: {_redact(exc)})"
            raise BinanceError(self.error) from None

    def sync_clock(self) -> int:
        """Binance rejects a timestamp more than recvWindow from server time.

        The box clock is the operator's, not ours, so the skew is measured once
        per process instead of assumed away.
        """
        try:
            server = int(self._send("/api/v3/time")["serverTime"])
        except BinanceError:
            return self._clock_skew_ms
        self._clock_skew_ms = server - int(time.time() * 1000)
        self._skew_checked = True
        return self._clock_skew_ms

    def alive(self) -> bool:
        if time.time() < self._down_until:
            return False
        try:
            self._send("/api/v3/ping")
            return True
        except BinanceError:
            self._down_until = time.time() + DOWN_PENALTY_S
            return False

    # ------------------------------------------------------------------ market data
    def klines(self, symbol: str, interval: str = "1d", limit: int = 500) -> pd.DataFrame | None:
        """Daily (or any-interval) bars in the store's own column order."""
        try:
            rows = self._send("/api/v3/klines",
                              {"symbol": symbol.upper(), "interval": interval,
                               "limit": int(limit)})
        except BinanceError:
            return None
        if not isinstance(rows, list) or not rows:
            return None
        out = pd.DataFrame([{"date": pd.to_datetime(r[0], unit="ms").tz_localize(None).date(),
                             "open": float(r[1]), "high": float(r[2]),
                             "low": float(r[3]), "close": float(r[4]),
                             "volume": float(r[5])} for r in rows])
        return out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    def tick(self, symbol: str) -> float | None:
        try:
            return float(self._send("/api/v3/ticker/price",
                                    {"symbol": symbol.upper()})["price"])
        except (BinanceError, KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ account
    def permissions(self, force: bool = False) -> dict[str, Any]:
        """What the key itself is allowed to do — cached, because it is a property
        of the key, not of the market. Checked before an order is attempted so the
        operator is told *which* door is shut instead of eating a broker rejection."""
        if self._perms is not None and not force and time.time() - self._perms_at < 300:
            return self._perms
        try:
            self._perms = dict(self._send("/sapi/v1/account/apiRestrictions",
                                          signed=True))
            self._perms_at = time.time()
        except BinanceError:
            self._perms = {}
        return self._perms

    def tradeable(self) -> tuple[bool, str]:
        p = self.permissions()
        if not p:
            return False, ("the key's permissions could not be read, so whether it "
                           "may trade is unknown — nothing was sent")
        if not p.get("enableSpotAndMarginTrading"):
            return False, ("the API key itself has enableSpotAndMarginTrading=false; "
                           "recreate it with Spot & Margin Trading enabled to trade")
        if not p.get("enableReading"):
            return False, "the key cannot read the account it would trade"
        return True, "spot trading permitted"

    def account(self) -> dict[str, Any] | None:
        """Venue-contract account read: `None` means unknown, never flat.

        A zero here would size an order as though the account were empty, which
        is the failure the MT5 and JS Global reads already refuse to produce.
        """
        try:
            raw = self._send("/api/v3/account", signed=True)
        except BinanceError:
            return None
        if not isinstance(raw, dict) or "balances" not in raw:
            self.error = "account read returned no balances field"
            return None
        held = {}
        prices: dict[str, float] = {}
        unpriced: list[str] = []
        equity = 0.0
        for b in raw["balances"]:
            free, locked = float(b.get("free") or 0), float(b.get("locked") or 0)
            if not (free or locked):
                continue
            asset = str(b["asset"])
            held[asset] = {"free": free, "locked": locked}
            px = (1.0 if asset in QUOTE_ASSETS
                  else self.tick(asset + QUOTE_ASSET))
            if px is None:
                unpriced.append(asset)
                continue
            prices[asset] = px
            equity += (free + locked) * px
        self.error = None
        return {"venue": VENUE, "currency": QUOTE_ASSET,
                "equity": round(equity, 2), "balance": round(equity, 2),
                "day_pnl": None, "open_positions": len(held), "held": held,
                "prices": prices,
                "can_trade": bool(raw.get("canTrade")),
                "unvalued": unpriced,
                "note": ("holdings not fully valued: no price for " + ",".join(unpriced)
                         if unpriced else "")}

    def positions(self) -> list[dict[str, Any]] | None:
        """Spot holdings in the position shape the risk engine reads, or None.

        Spot has no position objects: a non-zero base-asset balance *is* the
        position and quote currency is cash, so USDT is not counted as a bet on
        USDT. The venue reports no entry price and no unrealised P&L for a spot
        balance, so those are None rather than 0 — `_risk_posture` already treats
        a held row with no readable price as blind, which is the honest answer,
        instead of a free one, which is not.
        """
        acc = self.account()
        if acc is None:
            return None
        out: list[dict[str, Any]] = []
        for asset, h in (acc.get("held") or {}).items():
            qty = float(h.get("free") or 0) + float(h.get("locked") or 0)
            if qty <= 0 or asset in QUOTE_ASSETS:
                continue
            out.append({"symbol": asset + QUOTE_ASSET, "side": "BUY", "qty": qty,
                        "entry": None, "last": (acc.get("prices") or {}).get(asset),
                        "pnl": None, "venue": VENUE})
        return out

    # ------------------------------------------------------------------ orders
    def _outcome(self, cid: str, value: str) -> None:
        """Remember how this client order id ended, keyed by the id not by a
        "last call" field: two threads sharing one client could otherwise label
        a possibly-live order as a refusal, which is the mistake this record
        exists to prevent. Read it back with `outcome()`."""
        self.outcomes[cid] = value
        if len(self.outcomes) > 200:
            for old in list(self.outcomes)[:len(self.outcomes) - 200]:
                self.outcomes.pop(old, None)

    def outcome(self, client_order_id: str) -> str:
        """REFUSED | UNCONFIRMED | PLACED | REJECTED, or UNKNOWN if it was never
        this client's order. `UNKNOWN` is not `REFUSED`: the answer is not known."""
        return self.outcomes.get(client_order_id, "UNKNOWN")

    def place_order(self, intent: OrderIntent) -> Fill | None:
        """Signed order POST — reachable only through every gate in front of it.

        The gates, in the order they refuse: armed config, a kill switch that is
        clear, a key pair that exists, a key that is actually permitted to trade,
        a name this venue has a market for, finite numbers, and the audit journal
        accepting the intent *before* the request goes out. Same rule as the JS
        Global and MT5 doors: an order we cannot record is an order we do not send.

        A transport failure after the POST is `UNCONFIRMED`, not `REJECTED`: the
        request may have reached the matching engine. Callers must not treat None
        as "nothing happened" — read `outcome(coid)`.
        """
        def refuse(reason: str) -> None:
            self._outcome(intent.client_order_id, "REFUSED")
            JOURNAL.log("BINANCE_REFUSED", {"symbol": intent.symbol,
                                            "side": intent.side, "reason": reason})

        if not enabled():
            refuse("order placement disabled — set binance.allow_orders "
                   "(or BINANCE_ALLOW_ORDERS) and hold a key pair")
            return None
        if KILL.engaged():
            refuse("kill switch engaged — order blocked")
            return None
        ok, why = self.tradeable()
        if not ok:
            refuse(why)
            return None
        pair = pair_for(intent.symbol)
        if not pair:
            # Not a rejected order — a name this venue has no market for. Sending
            # it would return an exchange error that reads like a sizing problem.
            refuse(f"'{intent.symbol}' has no Binance spot pair, so this venue "
                   f"cannot hold it")
            return None
        if not (math.isfinite(float(intent.quantity)) and intent.quantity > 0):
            refuse(f"quantity {intent.quantity!r} is not a positive finite number")
            return None
        params = {"symbol": pair, "side": intent.side.upper(),
                  "type": intent.order_type.upper(),
                  "newClientOrderId": intent.client_order_id}
        qty_txt = _num(intent.quantity)
        px_txt = ""
        if intent.order_type.upper() == "LIMIT":
            if not intent.limit_price:
                refuse("LIMIT order with no limit price")
                return None
            px_txt = _num(intent.limit_price)
        # A number below the venue's own precision would otherwise go out as "0"
        # — an order for nothing at all, or a limit at free.
        if float(qty_txt) <= 0 or (px_txt and float(px_txt) <= 0):
            refuse(f"order numbers are smaller than this venue can express "
                   f"(quantity {qty_txt}, price {px_txt or '-'})")
            return None
        if JOURNAL.last_error:
            refuse(f"audit trail is failing: {JOURNAL.last_error}")
            return None
        if not JOURNAL.log("BINANCE_INTENT", {"symbol": pair,
                                              "key": intent.symbol,
                                              "side": intent.side,
                                              "qty": float(qty_txt),
                                              "type": intent.order_type,
                                              "cid": intent.client_order_id}):
            refuse(f"intent could not be journaled: {JOURNAL.last_error}")
            return None
        if px_txt:
            params["timeInForce"] = "GTC"
            params["price"] = px_txt
        params["quantity"] = qty_txt
        try:
            raw = self._send("/api/v3/order", params, signed=True, method="POST")
        except BinanceError as exc:
            # The request may have been matched. It is not a refusal.
            self._outcome(intent.client_order_id, "UNCONFIRMED")
            JOURNAL.log("BINANCE_UNCONFIRMED", {"cid": intent.client_order_id,
                                                "error": _redact(exc)})
            return None
        status = str(raw.get("status") or "UNKNOWN") if isinstance(raw, dict) else "UNKNOWN"
        self._outcome(intent.client_order_id,
                      "PLACED" if status in ("FILLED", "PARTIALLY_FILLED", "NEW")
                      else "REJECTED")
        fills = (raw.get("fills") or []) if isinstance(raw, dict) else []
        px = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
              / max(sum(float(f["qty"]) for f in fills), 1e-12)) if fills else 0.0
        qty = float(raw.get("executedQty") or intent.quantity)
        fill = Fill(client_order_id=intent.client_order_id,
                    broker_order_id=str(raw.get("orderId", "")),
                    symbol=str(raw.get("symbol") or pair).upper(),
                    side=intent.side.upper(),
                    quantity=qty, price=px,
                    commission=sum(float(f.get("commission") or 0) for f in fills),
                    slippage=0.0,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"), venue=VENUE,
                    status=status)
        JOURNAL.log("BINANCE_PLACED", {"cid": fill.client_order_id,
                                       "order_id": fill.broker_order_id,
                                       "status": fill.status, "price": fill.price,
                                       "qty": fill.quantity,
                                       "commission": fill.commission})
        return fill

    def state(self) -> dict[str, Any]:
        """Desk-facing summary. Reports presence of secrets, never the secrets.

        Reads no network: `permissions()` is a signed call with a 15 s timeout,
        and a desk poll must not block on the venue. The cached answer is shown
        as-is, with `perms_probed` saying whether there is an answer at all.
        """
        c = _conf()
        p = self._perms or {}
        return {"venue": VENUE, "configured": configured(),
                "allow_orders": bool(c.get("allow_orders")),
                "armed": enabled(),
                "probed": self.last_latency_s is not None,
                "connected": self.error is None and self.last_latency_s is not None,
                "error": self.error, "latency_s": self.last_latency_s,
                "clock_skew_ms": self._clock_skew_ms,
                "perms_probed": self._perms is not None,
                "key_can_read": p.get("enableReading"),
                "key_can_trade": p.get("enableSpotAndMarginTrading"),
                "ip_restricted": p.get("ipRestrict"),
                "down_until": round(self._down_until - time.time(), 1)
                if time.time() < self._down_until else 0}


BINANCE = BinanceClient()
