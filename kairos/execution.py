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
import re
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
from .store import _non_finite_path, atomic_write, dumps_strict
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

# The money a paper book is counted in. The desk and the track record used to
# print a literal `Rs` over whatever number they were handed, so a book funded
# in dollars would have published dollars under a rupee label; every one of them
# now takes its label from this code. One book, one money: a holding this book
# cannot count in its own currency is a hole in the ledger, and the old ledger
# did not have it. GOLD is
# quoted in dollars and PSX scrips in rupees, and `submit` used to deduct
# `price * qty` from the cash whichever price the tape printed, so 1.2 oz of
# gold at $4,183 charged a Rs 1,000,000 book Rs 5,020 — 0.5% of capital for a
# position worth 139% of it — and the same 277x understatement landed in
# `day_pnl`, which is the number the daily-loss brake divides by equity, and in
# the `realized P&L Rs` line the track record publishes.
PAPER_CURRENCY = "PKR"

CURRENCY_SYMBOLS = {"PKR": "Rs", "USD": "$", "EUR": "€", "GBP": "£",
                    "INR": "₹", "JPY": "¥"}


def money_label(code: str | None) -> str:
    """The label a statement prints in front of an amount of `code`.

    The label is part of the money, not part of the sentence around it. The track
    record, the pack it feeds and the desk all wrote "Rs" as a literal, so a book
    opened in dollars — a supported `paper.currency` — would have published
    dollars under a rupee label, in the one document whose job is to be
    checkable. An unknown code prints as the code itself: a guessed symbol is a
    worse label than none.
    """
    ccy = str(code or PAPER_CURRENCY).strip().upper()
    return CURRENCY_SYMBOLS.get(ccy, ccy)


def _paper_currency() -> str:
    ccy = str((CFG.section("paper") or {}).get("currency") or "").strip().upper()
    return ccy or PAPER_CURRENCY


def _instrument_currency(symbol: str, home: str) -> str:
    """The money a name is printed in, or the book's own when the registry is silent.

    A name outside the registry has no second currency in it, so it is counted at
    par rather than refused: the universe holds exactly PKR and USD names, and
    anything else is a hand-typed ticket or a fixture. Counting the unknown as
    *foreign* would be the more expensive mistake — it would brick the book on a
    name that was never mixed money to begin with.
    """
    try:
        from .universe import resolve
        ccy = str(getattr(resolve(symbol), "currency", "") or "").strip().upper()
    except Exception:  # noqa: BLE001 — a registry that cannot answer is a silent one
        return home
    return ccy or home


class PaperBroker:
    """Deterministic simulated venue: fills at next open + slippage, applies
    commissions, tracks cash/equity/positions, persists state to disk.

    The book holds one money. Every figure it reports — capital, cash, equity,
    realized, day P&L — is in `currency`, and a fill in another currency is
    converted at the published cross before it touches the ledger."""

    def __init__(self, capital: float = 1_000_000.0) -> None:
        self.state_path = CFG.data_dir / "paper_account.json"
        state = self._load()
        # The file wins over the config: the numbers already in the ledger were
        # counted in whatever money the book was opened with, and relabelling a
        # rupee ledger as dollars is the same mixing in a new place.
        self.currency: str = str(state.get("currency") or _paper_currency()).upper()
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
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                # The writer refuses a NaN; our own reader is lenient about one,
                # so a legacy or hand-edited file still parses and every figure
                # under it goes quietly non-finite. The next `_save()` — which
                # `account()` calls on the desk's read path — would then raise out
                # of a *successful* read. Read it with the same rule it was
                # written with, and let a ledger that breaks the rule follow the
                # path below instead of poisoning the arithmetic.
                where = _non_finite_path(state, "$")
                if where is not None:
                    raise ValueError(f"{where} is not a finite number")
                return state
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
        atomic_write(self.state_path, dumps_strict({
            "currency": self.currency,
            "capital": self.capital, "cash": self.cash, "positions": self.positions,
            "fills": self.fills[-500:], "realized_pnl": self.realized_pnl,
            "day_pnl": self.day_pnl, "day_stamp": self._day_stamp,
        }, indent=1))

    # -- money -----------------------------------------------------------------
    def rate(self, frm: str) -> float | None:
        """Book money per one unit of `frm`.

        One for the book's own currency without asking the tape, the published
        cross for the other one, and None when the cross has no usable print —
        which is a refusal, not a rate of one. This desk holds exactly two
        currencies, so the same reader that settled the cluster budget (#96)
        answers every mixing this book does.
        """
        from .data import fx_rate
        frm = str(frm or "").strip().upper()
        if not frm or frm == self.currency:
            return 1.0
        return fx_rate(frm, self.currency)

    def _ccy(self, symbol: str, pos: dict[str, Any] | None = None) -> str:
        """The money a holding is printed in, fixed on the row when it was opened.

        The stored value wins so a position keeps the currency it was filled in
        even if the registry later starts (or stops) knowing about the name.
        """
        return (str((pos or {}).get("ccy") or "").strip().upper()
                or _instrument_currency(symbol, self.currency))

    def units_for(self, symbol: str, notional: float, price: float) -> float | None:
        """How many units of `symbol` `notional` of book money buys, or None.

        A hand-typed ticket states its size in the money the desk works in and the
        tape prints the instrument's, and the two are not the same for a dollar
        name on a rupee book: dividing one by the other without the cross is the
        277x sizing error the autopilot had, in the one place a human enters it.
        None says there is no cross to do it with, which is a refusal, not a size.
        """
        if not (notional > 0) or not (price > 0):
            return None
        fx = self.rate(self._ccy(symbol))
        return None if fx is None else notional / (price * fx)

    def _mark(self, symbol: str, pos: dict[str, Any]) -> float | None:
        """What a holding is worth in book money, or None when it cannot be marked.

        A missing price or a missing cross is blindness, not zero: this figure
        feeds equity, and equity is what every breaker on the desk divides by.
        """
        try:
            qty = float(pos["qty"])
            last = float(pos["last"])
        except (TypeError, ValueError, KeyError):
            return None
        if not (math.isfinite(qty) and math.isfinite(last)):
            return None
        if qty == 0:
            return 0.0
        if last <= 0:
            return None
        fx = self.rate(self._ccy(symbol, pos))
        return None if fx is None else qty * last * fx

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
        marked, unmarked = 0.0, []
        for sym, p in self.positions.items():
            mv = self._mark(sym, p)
            if mv is None:
                unmarked.append(sym)
            else:
                marked += mv
        # None, not a smaller number: an equity that cannot be counted is one the
        # gate refuses to size against and the breakers report as unreadable.
        # Adding the dollar holdings to the rupee cash instead — which is what
        # this line used to do — is how the book grew equity it never had.
        equity = None if unmarked else self.cash + marked
        return {"venue": "PAPER", "login": "paper", "currency": self.currency,
                "equity": equity, "unmarked": unmarked,
                "balance": self.cash, "day_pnl": self.day_pnl,
                "open_positions": len(self.positions), "leverage": 1.0}

    def open_positions(self) -> list[dict[str, Any]]:
        out = []
        for sym, p in self.positions.items():
            out.append({"symbol": sym, "side": "BUY" if p["qty"] > 0 else "SELL",
                        "qty": abs(p["qty"]), "entry": p["entry"],
                        "last": p["last"], "pnl": (p["last"] - p["entry"]) * p["qty"],
                        # the money `entry`, `last` and `pnl` are printed in. A
                        # row of a dollar holding next to a row of a rupee one is
                        # only readable if the reader can tell which is which.
                        "currency": self._ccy(sym, p)})
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
        ccy = self._ccy(intent.symbol, self.positions.get(intent.symbol))
        fx = self.rate(ccy)
        if fx is None:
            # The venue would take this trade, and this book cannot count what it
            # costs. Refusing is the only answer that leaves the ledger in one
            # money; booking it at par is the bug this path was written to close.
            JOURNAL.log("PAPER_NO_FX", {"symbol": intent.symbol, "side": intent.side,
                                        "currency": ccy, "book": self.currency})
            return None
        commission = price * intent.quantity * commission_bps / 10_000
        qty = intent.quantity if intent.side == "BUY" else -intent.quantity
        pos = self.positions.get(intent.symbol)
        realized = 0.0
        # A cash book cannot buy what it cannot fund. Nothing else on this path
        # checks it: the gate's order-value bound only reads LIMIT orders, and the
        # autopilot's own caps keep an entry under a fraction of equity, so the
        # orders that can reach here bigger than the account are the ones a human
        # types in. Before the cross, a Rs 150,000 ticket on a dollar name cost
        # 150,000 of a million-rupee book and never looked near the edge; counted
        # in rupees it is 41.5 million, and the answer is a refusal, not a book
        # that goes quietly negative and calls the difference leverage.
        # A buy that closes a short is not a purchase — it hands cash back — so
        # only the order that opens or adds to the long side needs the money.
        cost = (price * qty + commission) * fx
        if qty > 0 and (pos is None or pos["qty"] >= 0) and cost > self.cash:
            JOURNAL.log("PAPER_NO_BUYING_POWER",
                        {"symbol": intent.symbol, "qty": intent.quantity,
                         "cost": round(cost, 2), "cash": round(self.cash, 2),
                         "book": self.currency})
            return None
        if pos is None:
            self.positions[intent.symbol] = {"qty": qty, "entry": price,
                                              "last": price, "commission": commission,
                                              "ccy": ccy}
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
        # equity = cash + Σ qty·last·fx stays a true mark-to-market number — in
        # the book's own money. A leg closed is realised at the cross it closed
        # on, which is the convention a rupee-funded desk reports its dollar
        # trades by; the rate used is on the journal line with the fill.
        self.cash -= cost
        self.realized_pnl += (realized - commission) * fx
        self.day_pnl += (realized - commission) * fx
        fill = Fill(intent.client_order_id, f"paper-{int(time.time() * 1000)}",
                    intent.symbol, intent.side, intent.quantity, price,
                    commission, slip, datetime.now().isoformat(timespec="seconds"),
                    "PAPER")
        self.fills.append(asdict(fill))
        self._save()
        JOURNAL.log("PAPER_FILL", {"coid": fill.client_order_id, "symbol": fill.symbol,
                                   "side": fill.side, "qty": fill.quantity,
                                   "price": fill.price, "commission": commission,
                                   "realized": realized, "currency": ccy,
                                   "book": self.currency, "fx": fx})
        return fill

    def equity_curve(self) -> pd.Series:
        """The book's value after each fill, in the book's own money.

        Rebuilt from `fills` because the state file keeps only the last 500. Each
        leg is counted at the cross the book answers with today: `fills` stores
        the venue's own print and no rate, so a historical curve of a mixed book
        can only be restated at par or at today's money, and today's money is the
        one that keeps a dollar trade the size of a dollar trade.
        """
        if not self.fills:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.fills)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
        pnl: list[float] = []
        kept: list = []
        cash = self.capital
        pos: dict[str, float] = {}
        entry: dict[str, float] = {}
        for row in df.itertuples():
            sym, side = row.symbol, row.side
            qty, price = float(row.quantity), float(row.price)
            fx = self.rate(_instrument_currency(sym, self.currency))
            if fx is None:
                # A leg that cannot be counted is not a leg that cost nothing:
                # leaving it out understates the curve and says so in the line
                # above, while adding it at par would redraw the exact mixing
                # this method exists to stop doing.
                continue
            q = qty if side == "BUY" else -qty
            if sym in pos and pos[sym] * q < 0:
                close_q = min(abs(q), abs(pos[sym]))
                cash += (price - entry[sym]) * np.sign(pos[sym]) * close_q * fx
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
            cash -= row.commission * fx
            pnl.append(cash)
            kept.append(row.ts)
        # The index is the rows that were counted, not the rows that were read:
        # a mixed book with no cross skips a leg, and pairing a shorter series
        # with the full timestamp index raised out of the desk's own equity chart.
        return pd.Series(pnl, index=pd.DatetimeIndex(kept), dtype=float)


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

    _AUTH_RE = re.compile(
        r"'(\d+)':\s*(?:authori[sz]ed on (?P<server>\S+)"
        r"|authori[sz]ation on (?P<refused>\S+) failed \((?P<why>[^)]*)\))")
    _AUTH_CACHE: tuple[float, list[dict[str, Any]]] = (0.0, [])

    @staticmethod
    def terminal_dirs(root: str | None = None) -> list[dict[str, str]]:
        """Every MT5 install with a data directory on this machine.

        `Terminal/<hash>/origin.txt` records the install a directory belongs to,
        so "the demo terminal" and "the broker terminal" can be told apart
        without starting either one.
        """
        base = root or os.path.join(os.environ.get("APPDATA") or "", "MetaQuotes",
                                    "Terminal")
        out: list[dict[str, str]] = []
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return out
        for name in names:
            d = os.path.join(base, name)
            logs = os.path.join(d, "logs")
            if not os.path.isdir(logs):
                continue
            install = ""
            try:
                raw = Path(os.path.join(d, "origin.txt")).read_bytes()
                text = (raw.decode("utf-16")
                        if raw[:2] in (b"\xff\xfe", b"\xfe\xff")
                        else raw.decode("utf-8", "replace"))
                install = text.strip("\x00\r\n ")
            except OSError:
                pass
            out.append({"id": name, "install": install, "logs": logs})
        return out

    @staticmethod
    def _read_auth(root: str | None = None, files: int = 8) -> list[dict[str, Any]]:
        """The terminal's own record of every sign-in it attempted.

        `initialize()` answers a refused sign-in with one integer, and that
        integer names neither the broker nor the reason. The log MetaTrader 5
        writes does: `'<account>': authorization on <network> failed (<why>)`.
        The account number is carried ONLY internally, to match a configured
        login; every caller-facing shape masks it to its last three digits.
        """
        out: list[dict[str, Any]] = []
        for term in Mt5Bridge.terminal_dirs(root):
            try:
                names = [n for n in os.listdir(term["logs"])
                         if n[:1].isdigit() and n.endswith(".log")]
            except OSError:
                continue
            found = False
            for name in sorted(names, reverse=True)[:files]:
                try:
                    raw = Path(os.path.join(term["logs"], name)).read_bytes()
                except OSError:
                    continue
                text = (raw.decode("utf-16")
                        if raw[:2] in (b"\xff\xfe", b"\xfe\xff")
                        else raw.decode("utf-8", "replace"))
                for line in reversed(text.splitlines()):
                    m = Mt5Bridge._AUTH_RE.search(line)
                    if not m:
                        continue
                    out.append({"terminal": term["id"], "install": term["install"],
                                "login": int(m.group(1)),
                                "server": m.group("server") or m.group("refused"),
                                "ok": bool(m.group("server")),
                                "reason": (m.group("why") or "").strip(),
                                "when": name[:-4]})
                    found = True
                if found:
                    break       # the newest day with a sign-in is the answer
        out.sort(key=lambda r: r["when"], reverse=True)
        return out

    @staticmethod
    def auth_history(root: str | None = None,
                     login: int | None = None) -> list[dict[str, Any]]:
        """Sign-in attempts across every terminal here, newest first, masked."""
        rows = Mt5Bridge._scan_auth(root)
        if login:
            rows = [r for r in rows if r["login"] == int(login)]
        return [{k: v for k, v in r.items() if k != "login"} | {
            "account": f"…{str(r['login'])[-3:]}"} for r in rows]

    @staticmethod
    def _scan_auth(root: str | None = None) -> list[dict[str, Any]]:
        now = time.time()
        if not root:
            stamp, rows = Mt5Bridge._AUTH_CACHE
            if now - stamp < 60.0:
                return rows
        rows = Mt5Bridge._read_auth(root)
        if not root:
            Mt5Bridge._AUTH_CACHE = (now, rows)
        return rows

    @staticmethod
    def signin_clue(login: int | None, root: str | None = None) -> str:
        """One sentence about the last broker answer for this account, or ""."""
        if not login:
            return ""
        rows = [r for r in Mt5Bridge._scan_auth(root) if r["login"] == int(login)]
        if not rows:
            return ""
        last = rows[0]
        tail = f"…{str(int(login))[-3:]}"
        if last["ok"]:
            return f"the terminal last signed {tail} in on {last['server']}"
        why = f" ({last['reason']})" if last["reason"] else ""
        return (f"the terminal last tried {tail} on {last['server']} and was "
                f"refused{why}")

    @staticmethod
    def preflight(root: str | None = None) -> dict[str, Any]:
        """The facts outside the IPC link that decide whether a link is possible.

        Three of the failures that actually block a live PMEX link never reach
        `initialize()`: the configured trade server is blank, the only installed
        terminal belongs to another broker, and the last sign-in for the account
        was already refused by that broker's network. Reporting them before an
        attach is the difference between "not connected" and one instruction the
        operator can follow.
        """
        m = CFG.section("mt5") or {}
        venue = str(m.get("venue") or "").strip()
        server = str(m.get("server") or "").strip()
        login = int(m.get("login") or 0) or None
        rows = Mt5Bridge._scan_auth(root)
        mine = [r for r in rows if login and r["login"] == login]
        networks = sorted({r["server"] for r in rows if r.get("server")})
        on_venue = [s for s in networks if venue and venue.lower() in s.lower()]
        blockers: list[str] = []
        if login and not server:
            blockers.append(
                "mt5.server is blank — a sign-in cannot be aimed at a named network, "
                "so the terminal is left to reuse whichever account it already saved")
        if venue and networks and not on_venue:
            blockers.append(
                f"no terminal here has ever reached a {venue} network; it has used "
                + ", ".join(networks))
        last = mine[0] if mine else None
        newest = rows[0] if rows else None
        if last and not last["ok"]:
            blockers.append(
                f"the last sign-in for this account was refused by {last['server']}"
                + (f" ({last['reason']})" if last["reason"] else ""))

        def _attempt(row: dict[str, Any] | None) -> dict[str, Any] | None:
            if not row:
                return None
            return {"account": f"…{str(row['login'])[-3:]}", "server": row["server"],
                    "ok": row["ok"], "reason": row["reason"], "when": row["when"]}
        return {"venue": venue, "server": server,
                "login_tail": f"…{str(login)[-3:]}" if login else "",
                "terminal_running": Mt5Bridge._running_terminal_path(cached_only=True),
                "terminals": [{"id": t["id"], "install": t["install"]}
                              for t in Mt5Bridge.terminal_dirs(root)],
                "networks": networks,
                # The newest broker answer on this machine is evidence even when
                # it belongs to another saved account — it is how "the only
                # terminal here is a demo one" becomes checkable rather than
                # inferred.
                "last_attempt": _attempt(newest),
                "last_attempt_for_this_account": _attempt(last),
                "blockers": blockers}

    def _hint(self) -> str:
        m = CFG.section("mt5") or {}
        venue = str(m.get("venue") or "your MT5 broker").strip()
        server = str(m.get("server") or "").strip()
        where = f" on {server}" if server else ""
        networks = sorted({r["server"] for r in self._scan_auth() if r.get("server")})
        gap = ""
        if venue and networks and not any(venue.lower() in s.lower() for s in networks):
            # Telling an operator to sign in to PMEX from a terminal that has
            # only ever reached a demo network is advice they cannot follow.
            gap = (f" No terminal on this machine has ever reached a {venue} network "
                   f"(it has used {', '.join(networks)}), so the {venue} terminal "
                   f"may not be installed here at all.")
        how = ("mt5.spawn is armed, so KAIROS may start it again on its own"
               if m.get("spawn") else
               "KAIROS attaches automatically once it is open; it does not launch "
               "the terminal unless mt5.spawn is armed")
        return (f"open MetaTrader 5, sign in to {venue}{where} and enable Algo "
                f"Trading. {how}.{gap}")

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

        def fail(msg: str) -> bool:
            """Fail closed, and add the terminal's own account of the last sign-in.

            "not signed in" is the state; "the network refused this account" is
            the cause, and only the terminal's log knows it.
            """
            clue = self.signin_clue(login)
            self.last_error = f"{msg} — {clue}" if clue else msg
            self.connected = False
            return False

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
            clue = self.signin_clue(login)
            if clue:
                self.last_error = f"{self.last_error} Last sign-in: {clue}."
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
                return fail(
                    f"the running MT5 terminal is {seen[0] if seen else 'unknown'}, not the "
                    f"configured {want_path} — attach refused rather than sign in to a "
                    "second terminal behind the operator's back")
            kwargs["path"] = want_path
        if login and password and server:
            kwargs.update({"login": int(login), "password": password, "server": server})
        try:
            if not mt5.initialize(**kwargs):
                return fail(f"MT5 attach failed: {mt5.last_error()}")
            info = mt5.account_info()
            if info is None:
                return fail("MT5 terminal is running but NOT signed in — "
                            + self._hint())
            self.attached_account = {"login": int(getattr(info, "login", 0) or 0),
                                     "server": str(getattr(info, "server", "") or "")}
            if login:
                # The bridge used to attach to whichever terminal happened to be
                # open and then trade it. With a broker account and a PMEX account
                # both reachable from one machine, "an MT5 account" is not enough
                # of an answer: sizing, positions and the kill switch all divide
                # by the equity of whatever `account_info()` returns.
                if self.attached_account["login"] != int(login):
                    try:
                        mt5.shutdown()   # closes our IPC link only, never the app
                    except Exception:
                        pass
                    return fail(
                        f"terminal is signed into account "
                        f"{self.attached_account['login'] or '?'} on "
                        f"{self.attached_account['server'] or '?'}, not the configured "
                        f"account {int(login)} — attach refused rather than trade the "
                        "wrong account")
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

    def spec(self, symbol: str) -> dict[str, float] | None:
        """The terminal's own trading rules for one symbol.

        A lot is not a fixed thing: contract size, the smallest order and the step
        between them are per-symbol settings the broker owns, and they differ
        across a book (a 100-oz gold CFD and a 1,000-barrel oil CFD are not the
        same hundred dollars). Sizing from an assumption instead of the spec puts
        every order on that symbol out by the ratio between the two — and it puts
        it out in whichever direction the broker chose.
        """
        if not self.connected or not symbol:
            return None
        try:
            info = self._mt5.symbol_info(symbol)
        except Exception:
            return None
        if info is None:
            return None
        out = {"contract_size": float(info.trade_contract_size),
               "volume_min": float(info.volume_min),
               "volume_step": float(info.volume_step)}
        # A zero in any of the three is not a rule, it is a half-filled struct,
        # and each one would either divide by zero or floor the order to nothing.
        return out if all(v > 0 and math.isfinite(v) for v in out.values()) else None

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
        # DONE_PARTIAL is a accepted order too, and its whole message is that the
        # venue struck less than was asked for. `result.volume` is the dealt size;
        # the request is only a fallback for a bridge that reports no volume at
        # all, because a fill sized at the order names a trade nobody made.
        dealt = float(getattr(result, "volume", 0) or 0) or intent.quantity
        fill = Fill(intent.client_order_id, str(result.order), symbol, intent.side,
                    dealt, float(result.price), 0.0,
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
        path.write_text(dumps_strict(ticket, indent=2), encoding="utf-8")
        JOURNAL.log("PSX_TICKET", {"ticket_id": ticket["ticket_id"],
                                   "symbol": intent.symbol, "side": intent.side,
                                   "qty": intent.quantity, "path": str(path)})
        return ticket
