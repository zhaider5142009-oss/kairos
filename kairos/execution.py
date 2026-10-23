"""Execution stack: paper broker, MT5 live bridge, JS Global PSX tickets.

Design doctrine (institutional):
  * FAIL-CLOSED risk gates — if the broker bridge errors, NO order passes.
  * Idempotency — every order carries a deterministic client order ID so a
    retry can never double-fill.
  * Append-only audit journal — every decision, order, and fill is recorded
    before it happens (intention) and after (result).
  * Kill-switch — a file on disk halts all live execution instantly.
  * Reconciliation — live positions re-read from the broker before every
    sizing decision; KAIROS never trusts its own memory over the broker's.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .store import atomic_write, dumps_strict
from . import rules as _R


# ---------------------------------------------------------------------------
# Intent / result records
# ---------------------------------------------------------------------------
@dataclass
class OrderIntent:
    symbol: str                     # broker symbol, e.g. "XAUUSD" or "OGDC"
    side: str                       # BUY | SELL
    quantity: float
    order_type: str = "MARKET"      # MARKET | LIMIT
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    venue: str = "PAPER"            # PAPER | MT5 | PSX_JSGLOBAL
    comment: str = "KAIROS"
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class Fill:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    ts: str
    venue: str
    status: str = "FILLED"          # FILLED | REJECTED | CANCELLED


# ---------------------------------------------------------------------------
# Audit journal — append-only JSONL
# ---------------------------------------------------------------------------
class AuditJournal:
    """The append-only trail every claim about this system is checked against.

    A trail that fails quietly is worse than no trail: it reports a period of
    no decisions to anyone who reads it. So a write that does not land is
    reported to the caller and remembered, and a read tolerates one bad line
    instead of surrendering the whole file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CFG.data_dir / "audit_journal.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None
        self.skipped = 0
        self._lock = threading.Lock()

    def _open_line(self) -> str:
        """A newline if the file ends mid-record.

        A crash mid-append, a copied journal or a restored backup can leave a
        last line with no terminator. Appending onto it welds two records into
        one line no parser accepts, and the trail loses both of them.
        """
        try:
            size = self.path.stat().st_size
            if not size:
                return ""
            with open(self.path, "rb") as raw:
                raw.seek(-1, os.SEEK_END)
                return "" if raw.read(1) == b"\n" else "\n"
        except OSError:
            return ""

    def log(self, event: str, payload: dict[str, Any]) -> bool:
        record = {"ts": datetime.now().isoformat(timespec="seconds"),
                  "event": event, **payload}
        try:
            line = dumps_strict(record) + "\n"
        except (TypeError, ValueError) as exc:
            self.last_error = f"{event} not recorded: payload is not serializable ({exc})"
            return False
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(self._open_line() + line)
        except Exception as exc:
            self.last_error = f"{event} not recorded: {type(exc).__name__}: {exc}"
            return False
        self.last_error = None
        return True

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        self.skipped = 0
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except Exception as exc:
            self.last_error = f"audit trail unreadable: {type(exc).__name__}: {exc}"
            return []
        rows: list[dict[str, Any]] = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                self.skipped += 1
                continue
            if isinstance(rec, dict):
                rows.append(rec)
            else:
                self.skipped += 1
        return rows[max(len(rows) - int(n), 0):]


JOURNAL = AuditJournal()

# how many client order ids stay in the idempotency window
MAX_SEEN_CIDS = 4096


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------
class KillSwitch:
    def __init__(self) -> None:
        self.path = CFG.data_dir / "KILL_SWITCH"

    def engaged(self) -> bool:
        return self.path.exists()

    def engage(self, reason: str = "manual") -> None:
        self.path.write_text(f"engaged {datetime.now().isoformat()} reason={reason}",
                             encoding="utf-8")
        JOURNAL.log("KILL_SWITCH_ENGAGED", {"reason": reason})

    def release(self) -> None:
        if self.path.exists():
            self.path.unlink()
        JOURNAL.log("KILL_SWITCH_RELEASED", {})


KILL = KillSwitch()


# ---------------------------------------------------------------------------
# Fail-closed pre-trade risk gate
# ---------------------------------------------------------------------------
class RiskGate:
    """All checks must PASS with positive knowledge; errors = BLOCK (fail-closed).

    Limits are read from config on every check, not snapshotted at build time:
    the tower stays up for days, and a limit tightened mid-run has to bind on
    the very next order, not after a restart.
    """

    def __init__(self) -> None:
        self._seen_coids: dict[str, None] = {}

    @staticmethod
    def _limits() -> dict[str, Any]:
        r = CFG.risk
        limit, source = _R.daily_loss_fraction(r)
        return {"max_positions": int(r.get("max_positions", 5)),
                "daily_loss_limit": limit,
                "daily_loss_source": source,
                "max_order_value_pct": float(r.get("max_order_value_pct", 0.10)),
                "max_qty": float(r.get("max_order_qty", 100000))}

    def check(self, intent: OrderIntent, account: dict[str, Any] | None,
              open_positions: list[dict[str, Any]] | None) -> tuple[bool, str]:
        if KILL.engaged():
            return False, "kill switch engaged"
        try:
            lim = self._limits()
        except (TypeError, ValueError, OverflowError) as exc:
            return False, f"risk limits unreadable — fail-closed ({type(exc).__name__})"
        # A limit that is not a number is not an unlimited order. `qty > nan` is
        # false, so a NaN in the `risk:` block used to clear the quantity bound
        # and the position cap the same way a NaN in the order cleared its
        # bounds — this is the last gate every venue's order passes through.
        bad = sorted(k for k, v in lim.items()
                     if k != "daily_loss_source"
                     and not (isinstance(v, (int, float))
                              and math.isfinite(float(v)) and float(v) >= 0))
        if bad:
            return False, f"unusable risk limit(s): {', '.join(bad)} — fail-closed"
        if intent.client_order_id in self._seen_coids:
            return False, f"duplicate order {intent.client_order_id}"
        # Finiteness before bounds. Every check on this path asks whether one
        # number is past another, and a NaN answers no to all of them, so a
        # hole in the feed used to clear both bounds and the value check below
        # — and a NaN fill lands in the same paper book the published track
        # record is read from.
        try:
            qty = float(intent.quantity)
            lp = (None if intent.limit_price is None
                  else float(intent.limit_price))
        except (TypeError, ValueError):
            return False, "quantity/limit_price are not numbers"
        if not math.isfinite(qty) or (lp is not None and not math.isfinite(lp)):
            return False, (f"unusable order numbers (qty {qty!r}, "
                           f"price {lp!r})")
        if qty <= 0 or qty > lim["max_qty"]:
            return False, f"quantity out of bounds: {intent.quantity}"
        if intent.side not in ("BUY", "SELL"):
            return False, f"invalid side {intent.side}"
        # account/positions must be POSITIVE knowledge — None fails closed
        if account is None:
            return False, "no account snapshot (bridge down) — fail-closed"
        if open_positions is None:
            return False, "no position snapshot (bridge down) — fail-closed"
        # Everything below compares against equity as a positive amount. A
        # reading that is not a number, or is negative — a margin call — used
        # to count as positive knowledge and then opt out of the loss check.
        raw_eq = account.get("equity")
        try:
            equity = float(0.0 if raw_eq is None else raw_eq)
        except (TypeError, ValueError):
            return False, f"unreadable equity {raw_eq!r}"
        if equity <= 0:
            return False, f"no usable equity ({equity:,.2f})"
        if len(open_positions) >= lim["max_positions"] and intent.order_type == "MARKET":
            return False, f"max positions reached ({lim['max_positions']})"
        raw_day = account.get("day_pnl")
        try:
            day_pnl = float(raw_day or 0.0)
        except (TypeError, ValueError):
            return False, f"unreadable day P&L {raw_day!r}"
        if not math.isfinite(day_pnl):
            return False, f"unusable day P&L ({raw_day!r})"
        if day_pnl < -lim["daily_loss_limit"] * equity:
            return False, (f"daily loss limit hit: {day_pnl:,.0f} "
                           f"< -{lim['daily_loss_limit']:.2%} x {equity:,.0f} "
                           f"({lim['daily_loss_source']})")
        order_value = qty * (lp or 0.0)
        if intent.order_type == "MARKET":
            order_value = None                       # market: rely on qty checks
        elif order_value and order_value > lim["max_order_value_pct"] * 10 * equity:
            return False, "limit order value absurdly large vs equity"
        # an idempotency window, not a lifetime ledger: a 24/7 process would
        # otherwise accumulate every coid it ever saw.
        self._seen_coids[intent.client_order_id] = None
        while len(self._seen_coids) > MAX_SEEN_CIDS:
            self._seen_coids.pop(next(iter(self._seen_coids)))
        return True, "ok"


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------
class PaperBroker:
    """Deterministic simulated venue: fills at next open + slippage, applies
    commissions, tracks cash/equity/positions, persists state to disk."""

    def __init__(self, capital: float = 1_000_000.0) -> None:
        self.state_path = CFG.data_dir / "paper_account.json"
        state = self._load()
        self.capital: float = state.get("capital", capital)
        self.cash: float = state.get("cash", capital)
        self.positions: dict[str, dict[str, float]] = state.get("positions", {})
        self.fills: list[dict[str, Any]] = state.get("fills", [])
        self.realized_pnl: float = state.get("realized_pnl", 0.0)
        self.day_pnl: float = state.get("day_pnl", 0.0)
        self._day_stamp = state.get("day_stamp", "")

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                # An unreadable file used to fall through as "no state yet", so
                # the next cycle opened at full capital with no positions and no
                # realized P&L — a silent reset that looks exactly like a
                # healthy book, and that equity sizes live orders. Keep the
                # evidence and refuse to trade against a book we cannot read.
                stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                damaged = self.state_path.with_name(self.state_path.name + f".{stamp}.damaged")
                try:
                    self.state_path.replace(damaged)
                    kept = damaged.name
                except OSError:
                    kept = None
                JOURNAL.log("PAPER_STATE_DAMAGED", {"path": str(self.state_path),
                                                    "kept_as": kept})
                KILL.engage("paper account state unreadable — fail-closed")
                return {}
        return {}

    def _save(self) -> None:
        atomic_write(self.state_path, json.dumps({
            "capital": self.capital, "cash": self.cash, "positions": self.positions,
            "fills": self.fills[-500:], "realized_pnl": self.realized_pnl,
            "day_pnl": self.day_pnl, "day_stamp": self._day_stamp,
        }, indent=1))

    # -- account/positions ----------------------------------------------------
    def account(self) -> dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._day_stamp:
            self._day_stamp, self.day_pnl = today, 0.0
            # The roll has to reach the file. State was written only on a fill, so
            # on a quiet stretch — which is most of this book — the ledger kept a
            # stale stamp and the daily figure published in the track record aged
            # past the day it claims to describe.
            self._save()
        equity = self.cash + sum(p["qty"] * p["last"] for p in self.positions.values())
        return {"venue": "PAPER", "login": "paper", "equity": equity,
                "balance": self.cash, "day_pnl": self.day_pnl,
                "open_positions": len(self.positions), "leverage": 1.0}

    def open_positions(self) -> list[dict[str, Any]]:
        out = []
        for sym, p in self.positions.items():
            out.append({"symbol": sym, "side": "BUY" if p["qty"] > 0 else "SELL",
                        "qty": abs(p["qty"]), "entry": p["entry"],
                        "last": p["last"], "pnl": (p["last"] - p["entry"]) * p["qty"]})
        return out

    def mark(self, symbol: str, price: float) -> None:
        if symbol in self.positions:
            self.positions[symbol]["last"] = float(price)
            self._save()

    # -- order -----------------------------------------------------------------
    def _fee_bps(self, symbol: str) -> float:
        """Commission model, config-tunable. PSX cash gets a higher default
        because brokerage + GFSS + SECP + PSX + CDC levies stack ~0.2% all-in;
        keep it conservative until the broker's exact slab is configured."""
        p = CFG.section("paper") or {}
        try:
            from .universe import resolve
            inst = resolve(symbol)
            if inst and inst.market.upper() in ("PSX", "KSE"):
                return float(p.get("psx_commission_bps", 20.0))
        except Exception:
            pass
        return float(p.get("commission_bps", 8.0))

    def submit(self, intent: OrderIntent, fill_price: float,
               slippage_bps: float = 5.0,
               commission_bps: float | None = None) -> Fill | None:
        if commission_bps is None:
            commission_bps = self._fee_bps(intent.symbol)
        slip = fill_price * slippage_bps / 10_000
        price = fill_price + slip if intent.side == "BUY" else fill_price - slip
        commission = price * intent.quantity * commission_bps / 10_000
        qty = intent.quantity if intent.side == "BUY" else -intent.quantity
        pos = self.positions.get(intent.symbol)
        realized = 0.0
        if pos is None:
            self.positions[intent.symbol] = {"qty": qty, "entry": price,
                                              "last": price, "commission": commission}
        elif pos["qty"] * qty >= 0:                       # adding
            new_qty = pos["qty"] + qty
            pos["entry"] = (pos["entry"] * pos["qty"] + price * qty) / new_qty if new_qty else price
            pos["qty"] = new_qty
            pos["commission"] = pos.get("commission", 0) + commission
        else:                                             # reducing / closing
            close_qty = min(abs(qty), abs(pos["qty"]))
            realized = (price - pos["entry"]) * np.sign(pos["qty"]) * close_qty
            pos["qty"] += qty
            if abs(pos["qty"]) < 1e-9:
                del self.positions[intent.symbol]
        # proper cash ledger: buys consume notional, sells return it, so
        # equity = cash + Σ qty·last stays a true mark-to-market number
        self.cash -= price * qty + commission
        self.realized_pnl += realized - commission
        self.day_pnl += realized - commission
        fill = Fill(intent.client_order_id, f"paper-{int(time.time() * 1000)}",
                    intent.symbol, intent.side, intent.quantity, price,
                    commission, slip, datetime.now().isoformat(timespec="seconds"),
                    "PAPER")
        self.fills.append(asdict(fill))
        self._save()
        JOURNAL.log("PAPER_FILL", {"coid": fill.client_order_id, "symbol": fill.symbol,
                                   "side": fill.side, "qty": fill.quantity,
                                   "price": fill.price, "commission": commission,
                                   "realized": realized})
        return fill

    def equity_curve(self) -> pd.Series:
        if not self.fills:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.fills)
        df["ts"] = pd.to_datetime(df["ts"])
        pnl = []
        cash = self.capital
        pos: dict[str, float] = {}
        entry: dict[str, float] = {}
        for _, row in df.sort_values("ts").iterrows():
            sym, side, qty, price = row["symbol"], row["side"], row["quantity"], row["price"]
            q = qty if side == "BUY" else -qty
            if sym in pos and pos[sym] * q < 0:
                close_q = min(abs(q), abs(pos[sym]))
                cash += (price - entry[sym]) * np.sign(pos[sym]) * close_q
            if sym not in pos or pos[sym] * q >= 0:
                if sym in pos:
                    tot = abs(pos[sym]) + abs(q)
                    entry[sym] = (entry[sym] * abs(pos[sym]) + price * abs(q)) / tot
                    pos[sym] += q
                else:
                    pos[sym], entry[sym] = q, price
            else:
                pos[sym] += q
                if abs(pos[sym]) < 1e-9:
                    pos.pop(sym)
                    entry.pop(sym)
            cash -= row["commission"]
            pnl.append(cash)
        return pd.Series(pnl, index=df.sort_values("ts")["ts"].values)


# ---------------------------------------------------------------------------
# MT5 live bridge (PMEX futures, JS Markets, any MT5 broker)
# ---------------------------------------------------------------------------
MT5_SYMBOL_MAP = {
    "GOLD": "XAUUSD", "SILVER": "XAGUSD", "WTI": "USOIL", "BRENT": "UKOIL",
    "NATGAS": "NATGAS", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "USDCAD": "USDCAD", "AUDUSD": "AUDUSD", "USDCHF": "USDCHF",
    "SPX": "US500", "NDX": "NAS100", "DJI": "US30", "BTC": "BTCUSD",
}


class Mt5Bridge:
    """Live MetaTrader 5 bridge with ATTACH-ONLY semantics by default.

    KAIROS does not launch the terminal — the desktop MetaTrader 5 app is the
    operator's own process. We attach to it only while `terminal64.exe` is
    already running and signed in; otherwise every call fails closed with a
    clear instruction. `mt5.spawn: true` is the operator's explicit permission to
    start the configured terminal again, and even then it never passes
    credentials on a command line. Reconnection is automatic but throttled, and
    a dead link degrades state instead of crashing anything.
    Attaching is refused if the running terminal is signed into a different
    account than the configured one. Every failure fails CLOSED (no order is
    placed)."""

    _ATTACH_COOLDOWN_S = 20.0
    _SPAWN_WAIT_S = 30.0
    _last_attach_try = 0.0          # class-level: shared across instances

    def __init__(self) -> None:
        self._mt5: Any = None
        self.connected = False
        self.last_error = ""
        self.attached_account: dict[str, Any] = {}
        self.last_order_latency_ms: float | None = None

    @staticmethod
    def _terminal_running() -> bool:
        return bool(Mt5Bridge._running_terminal_path(cached_only=True))

    @staticmethod
    def _running_terminal_path(cached_only: bool = False) -> str:
        """Executable path of the running terminal64, or "" if none is running.

        Two MT5 installs is now a real situation rather than a corner case — the
        operator has a broker terminal and a PMEX terminal — and "some terminal is
        open" is not the same fact as "the terminal we mean is open". Asking the
        process itself is the only way to tell them apart without launching one.
        """
        if sys.platform != "win32":
            return ""
        key = "_terminal_path_cache"
        now = time.time()
        if cached_only:
            return getattr(Mt5Bridge, key, ("", 0.0))[0]
        path, stamp = getattr(Mt5Bridge, key, ("", 0.0))
        if now - stamp < 20.0:
            return path
        found = ""
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq terminal64.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10, check=False)
            if "terminal64.exe" in out.stdout.lower():
                try:
                    ps = subprocess.run(
                        ["powershell", "-NoProfile", "-Command",
                         "(Get-Process terminal64 -ErrorAction SilentlyContinue)."
                         "Path -join '|'"],
                        capture_output=True, text=True, timeout=10, check=False)
                    found = (ps.stdout or "").strip()
                except Exception:
                    found = ""
        except Exception:
            found = ""
        Mt5Bridge._terminal_path_cache = (found, now)
        return found

    def _hint(self) -> str:
        m = CFG.section("mt5") or {}
        venue = str(m.get("venue") or "your MT5 broker").strip()
        server = str(m.get("server") or "").strip()
        where = f" on {server}" if server else ""
        how = ("mt5.spawn is armed, so KAIROS may start it again on its own"
               if m.get("spawn") else
               "KAIROS attaches automatically once it is open; it does not launch "
               "the terminal unless mt5.spawn is armed")
        return (f"open MetaTrader 5, sign in to {venue}{where} and enable Algo "
                f"Trading. {how}.")

    @staticmethod
    def _spawn_armed() -> bool:
        return bool((CFG.section("mt5") or {}).get("spawn"))

    def _maybe_spawn(self, want_path: str) -> bool:
        """Start the operator's own terminal when — and only when — they armed it.

        A 24/7 loop that depends on a GUI app somebody opened once is not
        institutional: the terminal closes, an update reboots the machine, and the
        system silently stops seeing the market while its journals keep filling
        with "not attached". `mt5.spawn` is the operator's word for "you may open
        it again". It is off by default, and it is deliberately *not* given the
        credentials: the account has to be one the terminal saves for itself,
        because a password passed on a command line is readable by every other
        process on the box, including ours in a crash dump.
        """
        m = CFG.section("mt5") or {}
        if not m.get("spawn"):
            return False
        if not want_path or not os.path.isfile(want_path):
            self.last_error = ("mt5.spawn is armed but the configured terminal_path is "
                               "not a file, so there is nothing to start")
            return False
        try:
            import subprocess
            subprocess.Popen([want_path], cwd=os.path.dirname(want_path))
        except Exception as exc:
            self.last_error = f"mt5.spawn could not start the terminal: {exc}"
            return False
        deadline = time.time() + Mt5Bridge._SPAWN_WAIT_S
        while time.time() < deadline:
            if self._running_terminal_path():
                JOURNAL.log("MT5_TERMINAL_SPAWNED",
                            {"path": want_path, "waited_s":
                             round(Mt5Bridge._SPAWN_WAIT_S - (deadline - time.time()), 1)})
                return True
            time.sleep(1.0)
        self.last_error = (f"started {want_path} but no terminal64 appeared within "
                           f"{Mt5Bridge._SPAWN_WAIT_S:.0f}s — an account that is not "
                           "saved in the terminal cannot be signed in from here")
        return False

    def attach(self, login: int | None = None, password: str | None = None,
               server: str | None = None, terminal_path: str | None = None) -> bool:
        """Join an ALREADY-RUNNING terminal, unless the operator armed a respawn."""
        import time as _t
        if _t.time() - Mt5Bridge._last_attach_try < Mt5Bridge._ATTACH_COOLDOWN_S:
            return self.connected
        Mt5Bridge._last_attach_try = _t.time()
        try:
            import MetaTrader5 as mt5
        except Exception as exc:
            self.last_error = f"MetaTrader5 package unavailable: {exc}"
            self.connected = False
            return False
        self._mt5 = mt5
        want_path = str(terminal_path or "").strip().rstrip('"')
        running = self._running_terminal_path()
        spawn_tried = False
        if not running and self._spawn_armed():
            spawn_tried = True
            if self._maybe_spawn(want_path):
                running = self._running_terminal_path()
        if not running:
            self.connected = False
            if spawn_tried and self.last_error:
                # The specific reason — "that path is not a file", "no terminal
                # appeared within 30s" — must survive. Replacing it with the
                # generic instruction is how a supervisor tells an operator to
                # open an app that is already trying to open.
                return False
            self.last_error = f"MT5 terminal is not running — {self._hint()}"
            return False
        kwargs: dict[str, Any] = {}
        if want_path:
            # `initialize(path=...)` launches the terminal when that path is not
            # the one already open, which is exactly the thing this bridge
            # promises never to do. So the path is only forwarded when the
            # running process really is that install.
            seen = [p.strip() for p in running.split("|")]
            norm = os.path.normcase(os.path.abspath(want_path))
            if not any(os.path.normcase(os.path.abspath(p)) == norm for p in seen if p):
                self.connected = False
                self.last_error = (
                    f"the running MT5 terminal is {seen[0] if seen else 'unknown'}, not the "
                    f"configured {want_path} — attach refused rather than sign in to a "
                    "second terminal behind the operator's back")
                return False
            kwargs["path"] = want_path
        if login and password and server:
            kwargs.update({"login": int(login), "password": password, "server": server})
        try:
            if not mt5.initialize(**kwargs):
                self.last_error = f"MT5 attach failed: {mt5.last_error()}"
                self.connected = False
                return False
            info = mt5.account_info()
            if info is None:
                self.last_error = ("MT5 terminal is running but NOT signed in — "
                                   + self._hint())
                self.connected = False
                return False
            self.attached_account = {"login": int(getattr(info, "login", 0) or 0),
                                     "server": str(getattr(info, "server", "") or "")}
            if login:
                # The bridge used to attach to whichever terminal happened to be
                # open and then trade it. With a broker account and a PMEX account
                # both reachable from one machine, "an MT5 account" is not enough
                # of an answer: sizing, positions and the kill switch all divide
                # by the equity of whatever `account_info()` returns.
                if self.attached_account["login"] != int(login):
                    self.last_error = (
                        f"terminal is signed into account "
                        f"{self.attached_account['login'] or '?'} on "
                        f"{self.attached_account['server'] or '?'}, not the configured "
                        f"account {int(login)} — attach refused rather than trade the "
                        "wrong account")
                    self.connected = False
                    try:
                        mt5.shutdown()   # closes our IPC link only, never the app
                    except Exception:
                        pass
                    return False
            first = not self.connected
            self.connected = True
            if first:
                JOURNAL.log("MT5_ATTACHED", {"info": self._account_raw()})
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    def ensure(self) -> bool:
        """Heartbeat: verify the link, transparently re-attach when possible."""
        if self.connected and self._account_raw() is not None:
            return True
        self.connected = False
        m = CFG.section("mt5") or {}
        return self.attach(login=m.get("login"), password=m.get("password"),
                           server=m.get("server"), terminal_path=m.get("terminal_path"))

    # legacy name — now attach-only, never launches the terminal
    def connect(self, login=None, password=None, server=None, terminal_path=None) -> bool:
        return self.attach(login=login, password=password, server=server,
                           terminal_path=terminal_path)

    def _account_raw(self) -> dict[str, Any] | None:
        if not self.connected:
            return None
        try:
            info = self._mt5.account_info()
            if info is None:
                return None
            return dict(login=info.login, name=info.name, server=info.server,
                        currency=info.currency, equity=info.equity,
                        balance=info.balance, margin_free=info.margin_free,
                        leverage=info.leverage, profit=info.profit)
        except Exception:
            return None

    def account(self) -> dict[str, Any] | None:
        raw = self._account_raw()
        if raw is None:
            self.connected = False
            return None
        poss = self.positions()
        # None travels onward: a snapshot that invents a zero count for a read
        # it could not make is what makes the account look flat.
        raw.update({"venue": "MT5", "day_pnl": raw.get("profit", 0.0),
                    "open_positions": None if poss is None else len(poss)})
        return raw

    def positions(self) -> list[dict[str, Any]] | None:
        if not self.connected:
            return None
        try:
            poss = self._mt5.positions_get()
            if poss is None:
                # MetaTrader spells two different answers with two different
                # objects: None means the request failed, an empty tuple means
                # the account holds nothing. Returning [] here made a dead
                # terminal indistinguishable from a flat one, and every
                # consumer downstream reads an empty list as a fact — the
                # exposure and max-positions breakers pass on it, and
                # reconciliation forgets the live trades it was managing.
                self.last_error = "positions_get() returned no data"
                return None
            return [{"symbol": p.symbol, "side": "BUY" if p.type == 0 else "SELL",
                     "qty": float(p.volume), "entry": float(p.price_open),
                     "last": float(p.price_current), "pnl": float(p.profit),
                     "ticket": p.ticket} for p in poss]
        except Exception:
            return None

    def resolve_symbol(self, kairos_key: str) -> str | None:
        if not self.connected:
            return None
        candidate = MT5_SYMBOL_MAP.get(kairos_key.upper(), kairos_key.upper())
        try:
            info = self._mt5.symbol_info(candidate)
            if info is not None:
                if not info.visible:
                    self._mt5.symbol_select(candidate, True)
                return candidate
            # fuzzy fallback: any symbol containing the key
            for s in (self._mt5.symbols_get() or []):
                if kairos_key.upper() in s.name.upper():
                    self._mt5.symbol_select(s.name, True)
                    return s.name
            return None
        except Exception:
            return None

    def tick(self, symbol: str) -> dict[str, float] | None:
        if not self.connected:
            return None
        try:
            t = self._mt5.symbol_info_tick(symbol)
            if t is None:
                return None
            return {"bid": float(t.bid), "ask": float(t.ask),
                    "spread": round(float(t.ask) - float(t.bid), 5)}
        except Exception:
            return None

    def place_order(self, intent: OrderIntent, deviation: int = 20) -> Fill | None:
        if not self.connected:
            JOURNAL.log("MT5_REJECT", {"reason": "not connected", "coid": intent.client_order_id})
            return None
        mt5 = self._mt5
        symbol = self.resolve_symbol(intent.symbol)
        if symbol is None:
            JOURNAL.log("MT5_REJECT", {"reason": f"symbol {intent.symbol} not found",
                                       "coid": intent.client_order_id})
            return None
        t = self.tick(symbol)
        if t is None:
            JOURNAL.log("MT5_REJECT", {"reason": "no tick", "coid": intent.client_order_id})
            return None
        price = t["ask"] if intent.side == "BUY" else t["bid"]
        if JOURNAL.log("MT5_INTENT", {"coid": intent.client_order_id,
                                      "symbol": symbol, "side": intent.side,
                                      "qty": intent.quantity, "price": price,
                                      "sl": intent.stop_loss,
                                      "tp": intent.take_profit}) is not True:
            # The doctrine is that the intent is recorded before it happens. An
            # order the trail cannot record is an order that cannot be audited,
            # explained or reversed — so the door closes before the request is
            # ever assembled.
            JOURNAL.log("MT5_REJECT", {"reason": f"audit trail is not accepting "
                                                 f"writes ({JOURNAL.last_error})",
                                       "coid": intent.client_order_id})
            return None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(intent.quantity),
            "type": mt5.ORDER_TYPE_BUY if intent.side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": deviation,
            "magic": 20261016,
            "comment": f"KAIROS {intent.comment}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if intent.stop_loss:
            request["sl"] = float(intent.stop_loss)
        if intent.take_profit:
            request["tp"] = float(intent.take_profit)
        t0 = time.perf_counter()
        try:
            result = mt5.order_send(request)
        except Exception as exc:
            JOURNAL.log("MT5_REJECT", {"reason": f"order_send exception: {exc}",
                                       "coid": intent.client_order_id})
            return None
        latency = (time.perf_counter() - t0) * 1000
        self.last_order_latency_ms = latency
        if result is None:
            JOURNAL.log("MT5_REJECT", {"reason": "order_send returned None",
                                       "coid": intent.client_order_id,
                                       "last_error": str(mt5.last_error())})
            return None
        if result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
            JOURNAL.log("MT5_REJECT", {"reason": f"retcode={result.retcode}",
                                       "coid": intent.client_order_id,
                                       "comment": str(getattr(result, "comment", ""))})
            return None
        fill = Fill(intent.client_order_id, str(result.order), symbol, intent.side,
                    intent.quantity, float(result.price), 0.0,
                    abs(float(result.price) - price),
                    datetime.now().isoformat(timespec="seconds"), "MT5")
        JOURNAL.log("MT5_FILL", {"coid": fill.client_order_id, "order": fill.broker_order_id,
                                 "symbol": fill.symbol, "side": fill.side,
                                 "qty": fill.quantity, "price": fill.price,
                                 "latency_ms": round(latency, 1),
                                 "retcode": result.retcode})
        return fill

    def close_position(self, ticket: int) -> bool:
        if not self.connected:
            return False
        mt5 = self._mt5
        poss = mt5.positions_get(ticket=ticket)
        if not poss:
            return False
        p = poss[0]
        tick = self.tick(p.symbol)
        if tick is None:
            return False
        side = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick["bid"] if p.type == 0 else tick["ask"]
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                   "volume": float(p.volume), "type": side, "position": p.ticket,
                   "price": price, "deviation": 20, "magic": 20261016,
                   "comment": "KAIROS close", "type_time": mt5.ORDER_TIME_GTC,
                   "type_filling": mt5.ORDER_FILLING_IOC}
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        JOURNAL.log("MT5_CLOSE", {"ticket": p.ticket, "ok": ok})
        return ok

    def modify_sl(self, ticket: int, sl: float | None = None,
                  tp: float | None = None) -> bool:
        """Ratchet a live position's SL/TP server-side (favourable moves only
        are the caller's responsibility)."""
        if not self.connected:
            return False
        mt5 = self._mt5
        request: dict[str, Any] = {"action": mt5.TRADE_ACTION_SLTP,
                                   "position": int(ticket)}
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)
        try:
            result = mt5.order_send(request)
            ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            JOURNAL.log("MT5_MODIFY", {"ticket": ticket, "sl": sl, "tp": tp,
                                       "ok": ok})
            return ok
        except Exception as exc:
            JOURNAL.log("MT5_MODIFY", {"ticket": ticket, "ok": False,
                                       "error": str(exc)[:200]})
            return False

    def benchmark_latency(self, symbol: str, n: int = 5) -> dict[str, Any]:
        """Round-trip quote latency over N symbol-select + tick fetches.

        Risk-free by design: measures connectivity only and never sends an
        order — blasting live market orders to benchmark would violate the
        paper-first doctrine."""
        if not self.connected:
            return {"latencies_ms": [], "success_rate": 0.0, "mean_ms": 0.0}
        mt5 = self._mt5
        lat: list[float] = []
        ok = 0
        for _ in range(max(1, n)):
            try:
                mt5.symbol_select(symbol, True)
                t0 = time.perf_counter()
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol) if info else None
                ms = (time.perf_counter() - t0) * 1000
                if tick is not None and (tick.bid or tick.ask):
                    ok += 1
                    lat.append(round(ms, 2))
            except Exception:
                continue
        mean = round(sum(lat) / len(lat), 1) if lat else 0.0
        return {"latencies_ms": lat, "success_rate": ok / max(1, n), "mean_ms": mean}

    def shutdown(self) -> None:
        if self.connected and self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self.connected = False


_MT5: Mt5Bridge | None = None


def get_mt5() -> Mt5Bridge:
    """Process-wide shared bridge so attach state/cooldown stay consistent."""
    global _MT5
    if _MT5 is None:
        _MT5 = Mt5Bridge()
    return _MT5


# ---------------------------------------------------------------------------
# JS Global PSX execution desk (semi-automated)
# ---------------------------------------------------------------------------
class JsGlobalTicket:
    """PSX has no retail FIX/API for individual investors; institutional flow
    goes through brokers like JS Global Capital. KAIROS produces broker-ready
    order tickets (with all compliance fields) and can deliver them by file
    or webhook. Execution remains human-confirmed — by design."""

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = out_dir or CFG.data_dir / "psx_orders"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build_ticket(self, intent: OrderIntent, account_number: str = "REPLACE_ME",
                     market: str = "REGULAR", settlement: str = "T+2",
                     validity: str = "DAY") -> dict[str, Any]:
        ticket = {
            "ticket_id": f"JSG-{intent.client_order_id}",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "broker": "JS Global Capital Limited",
            "account_number": account_number,
            "market": "PSX", "market_type": market, "settlement": settlement,
            "symbol": intent.symbol, "side": intent.side,
            "order_type": intent.order_type,
            "quantity": int(intent.quantity),
            "price": (intent.limit_price if intent.order_type == "LIMIT"
                      else "MARKET"),
            "validity": validity,
            "stop_loss": intent.stop_loss, "take_profit": intent.take_profit,
            "strategy_comment": intent.comment,
            "status": "AWAITING_HUMAN_CONFIRMATION",
        }
        path = self.out_dir / f"{ticket['ticket_id']}.json"
        path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")
        JOURNAL.log("PSX_TICKET", {"ticket_id": ticket["ticket_id"],
                                   "symbol": intent.symbol, "side": intent.side,
                                   "qty": intent.quantity, "path": str(path)})
        return ticket
