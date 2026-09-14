"""
PaperBridge — crash-safe simulated broker ("Paper Money") on top of real MT5
market data. Subclass of MT5Bridge: market-data methods (rates / tick /
indicators) hit the live MetaTrader terminal read-only, while every
trade-affecting method is simulated against a paper account with real-time
mark-to-market.

Robustness model (investment-bank flavoured):
  - append-only JSONL trade journal with fsync — source of truth
  - boot-time journal replay restores open positions after a crash/restart
  - reconciliation: journal vs in-memory state cross-check on every write
  - idempotent fills: duplicate signal_id / duplicate symbol+direction refused
  - SL/TP auto-trigger off live ticks (stop-loss wins over take-profit)
  - consecutive-loss circuit breaker + optional cooldown
  - equity curve journaled (drawdown / Sharpe / profit-factor analysis)
  - thread-safe: order path may run on HTTP threads, marking on the loop thread
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .config import JarvisConfig
from .mt5_bridge import MT5Bridge, SendResult, JarvisMT5Error

LOGGER = logging.getLogger("jarvis.paper")


class PaperPosition:
    """Minimal position object exposing the MT5 attribute surface that the
    strategy engine + UI consume (ticket/symbol/type/volume/price_open/...)."""

    __slots__ = (
        "ticket", "symbol", "type", "volume", "price_open", "price_current",
        "sl", "tp", "time", "profit", "swap", "commission", "magic", "comment",
        "direction", "signal_id", "opened_at",
    )

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperPosition {self.symbol} {'BUY' if self.type == 0 else 'SELL'} id={self.ticket} entry={self.price_open} now={self.price_current}>"


class PaperBridge(MT5Bridge):
    """Simulated execution broker sharing the real MT5 data surface."""

    def __init__(
        self,
        *,
        terminal_path,
        symbols,
        timeframe,
        history_bars,
        magic,
        default_lot,
        poll_seconds: float = 2.0,
        paper_cfg: dict | None = None,
        journal_dir: str | os.PathLike = "",
    ) -> None:
        super().__init__(
            terminal_path=terminal_path,
            symbols=symbols,
            timeframe=timeframe,
            history_bars=history_bars,
            magic=magic,
            default_lot=default_lot,
            poll_seconds=poll_seconds,
        )
        cfg = paper_cfg or {}
        self.mode = "paper"
        self.balance = float(cfg.get("starting_balance", 10000.0))
        self.starting_balance = self.balance
        self.leverage = float(cfg.get("leverage", 100))
        self.commission_per_lot = float(cfg.get("commission_per_lot", 0.0))
        self.equity_snapshot_every = float(cfg.get("equity_snapshot_every", 3.0))
        self.breaker_consecutive_losses = int(cfg.get("breaker_consecutive_losses", 3))
        self.breaker_cooldown_seconds = float(cfg.get("breaker_cooldown_seconds", 600.0))
        self.paper_max_positions = int(cfg.get("max_positions", 3))

        self._journal_dir = Path(journal_dir) if journal_dir else None
        self._journal_file: Any = None
        self._ticket_seq = int(cfg.get("ticket_start", 9000000))
        self._positions: dict[int, PaperPosition] = {}
        self._consecutive_losses = 0
        self._breaker_until = 0.0
        self._realized_day = 0.0
        self._last_price: dict[str, float] = {}
        self._last_mark_ts = 0.0
        self._last_equity_ts = 0.0
        self._day_start_balance = self.balance
        self._lock = threading.RLock()
        self._state_file = None
        self._verify_counter = 0
        self._sym_info_cache: dict[str, tuple[float, dict]] = {}

        if self._journal_dir is not None:
            self._init_journal()

    # ------------------------------------------------------------ lifecycle

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "PaperBridge":
        s = config.section("mt5")
        paper_cfg = dict(config.section("paper"))
        # Path resolution: journal_dir == "auto" → {cache_dir}/paper
        journal = ""
        try:
            cache_dir = Path(config.section("paths")["cache_dir"])
            journal = cache_dir / "paper"
        except Exception:
            journal = None
        return cls(
            terminal_path=s["terminal_path"],
            symbols=s["symbols"],
            timeframe=s["timeframe"],
            history_bars=s["history_bars"],
            magic=s["magic"],
            default_lot=s["order"]["default_lot"],
            poll_seconds=s.get("poll_seconds", 2.0),
            paper_cfg=paper_cfg,
            journal_dir=journal or "",
        )

    def _init_journal(self) -> None:
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._journal_dir / "paper_state.json"
        self._journal_file = open(
            self._journal_dir / "ledger.jsonl", "a", encoding="utf-8"
        )
        self._replay()

    def _replay(self) -> None:
        """Rebuild in-memory state from the append-only journal (source of
        truth). If a state snapshot also exists, cross-check for drift."""
        ledger_path = self._journal_dir / "ledger.jsonl"
        if not ledger_path.exists():
            return
        positions: dict[int, PaperPosition] = {}
        realized = 0.0
        losses = 0
        day_start = self.balance
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k = e.get("kind")
            if k == "open":
                p = PaperPosition(
                    ticket=int(e["ticket"]), symbol=e["symbol"], type=int(e["type"]),
                    volume=float(e["volume"]), price_open=float(e["entry"]),
                    price_current=float(e["entry"]), sl=float(e.get("sl") or 0.0),
                    tp=float(e.get("tp") or 0.0), time=float(e["ts"]),
                    profit=0.0, swap=0.0, commission=0.0,
                    magic=self.magic, comment=e.get("comment", ""),
                    direction=int(e["direction"]), signal_id=e.get("signal_id", ""),
                    opened_at=float(e["ts"]),
                )
                positions[int(e["ticket"])] = p
                self._ticket_seq = max(self._ticket_seq, int(e["ticket"]))
            elif k == "close":
                positions.pop(int(e["ticket"]), None)
                realized += float(e.get("net_pnl") or 0.0)
                if float(e.get("net_pnl") or 0.0) < 0:
                    losses += 1
                else:
                    losses = 0
            elif k == "modify":
                p = positions.get(int(e["ticket"]))
                if p is not None:
                    if e.get("sl") is not None:
                        p.sl = float(e["sl"])
                    if e.get("tp") is not None:
                        p.tp = float(e["tp"])
            elif k == "breaker":
                losses = int(e.get("count", losses))
        self._positions = positions
        self._realized_day = realized
        self._consecutive_losses = losses
        self._day_start_balance = self.balance - realized
        LOGGER.info(
            "Paper journal replay: %d open position(s), day pnl=%.2f, consecutive losses=%d",
            len(positions), realized, losses,
        )
        # Reconcile with state snapshot (if any) — drift => journal wins, warn.
        if self._state_file and self._state_file.exists():
            try:
                snap = json.loads(self._state_file.read_text(encoding="utf-8"))
                snap_ids = {int(p["ticket"]) for p in snap.get("positions", [])}
                journal_ids = set(positions)
                if snap_ids != journal_ids:
                    LOGGER.warning(
                        "Paper reconciliation drift detected (snapshot=%d vs journal=%d) "
                        "— journal is authoritative",
                        len(snap_ids), len(journal_ids),
                    )
            except Exception:
                pass
        self._write_state()

    def _write_state(self) -> None:
        """Persist a crash-tolerant snapshot of current paper state."""
        if not self._state_file:
            return
        try:
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({
                    "ts": time.time(),
                    "balance": round(self.balance, 2),
                    "ticket_seq": self._ticket_seq,
                    "consecutive_losses": self._consecutive_losses,
                    "breaker_until": self._breaker_until,
                    "positions": [p.to_dict() for p in self._positions.values()],
                }),
                encoding="utf-8",
            )
            tmp.replace(self._state_file)
        except Exception:
            LOGGER.exception("paper state write failed")

    def _journal(self, kind: str, **data: Any) -> None:
        if self._journal_file is None:
            return
        line = json.dumps({"ts": time.time(), "kind": kind, **data}, default=str)
        try:
            self._journal_file.write(line + "\n")
            self._journal_file.flush()
            os.fsync(self._journal_file.fileno())
        except Exception:
            LOGGER.exception("paper journal write failed")

    # --------------------------------------------------------- paper account

    def paper_snapshot(self) -> dict:
        """Snapshot for the status/UI payload (account panel + paper mode)."""
        with self._lock:
            open_pos = [p.to_dict() for p in self._positions.values()]
            equity = self._calc_equity(open_pos)
            recon_ok = True
            recon_error = ""
            try:
                recon_ok = self._verify()
            except Exception as e:
                recon_ok = False
                recon_error = str(e)[:80]
        return {
            "mode": self.mode,
            "balance": round(self.balance, 2),
            "equity": round(equity, 2),
            "starting_balance": round(self.starting_balance, 2),
            "day_pnl": round(equity - self._day_start_balance, 2),
            "realized_day_pnl": round(self._realized_day, 2),
            "unrealized": round(equity - self.balance, 2),
            "open_positions": len(open_pos),
            "max_positions": self.paper_max_positions,
            "consecutive_losses": self._consecutive_losses,
            "breaker_active": time.time() < self._breaker_until,
            "breaker_until": self._breaker_until,
            "recon_ok": recon_ok,
            "recon_error": recon_error,
            "leverage": self.leverage,
            "commission_per_lot": self.commission_per_lot,
            "ticker": dict(self._last_price),
        }

    def _calc_equity(self, open_pos: list[dict]) -> float:
        return self.balance + sum(
            self._pnl_usd(p) for p in open_pos if isinstance(p, dict)
        )

    def _pnl_usd(self, pos: dict) -> float:
        info = self._sym_info(pos["symbol"])
        tick_size = float(info.get("trade_tick_size") or 1e-5)
        tick_value = float(info.get("trade_tick_value") or 1.0)
        entry = float(pos["price_open"])
        cur = float(pos.get("price_current") or entry)
        return (cur - entry) * int(pos["direction"]) * tick_value * float(pos["volume"]) / tick_size if tick_size else 0.0

    def _sym_info(self, symbol: str) -> dict:
        now = time.time()
        cached = self._sym_info_cache.get(symbol)
        if cached and now - cached[0] < 60.0:
            return cached[1]
        info = self.get_symbol_info(symbol) or {}
        self._sym_info_cache[symbol] = (now, info)
        return info

    # -------------------------------------------------------------- override

    def equity(self) -> float:
        with self._lock:
            open_pos = [p.to_dict() for p in self._positions.values()]
        return self._calc_equity(open_pos)

    def balance(self) -> float:
        return self.balance

    def daily_pnl(self) -> float:
        return self.equity() - self._day_start_balance

    def open_positions(self) -> list:
        """Return PaperPosition objects (attribute surface ~ MT5 tuples), so
        callers using getattr(...) work unchanged in paper mode."""
        with self._lock:
            return list(self._positions.values())

    def _open_pos_dicts(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._positions.values()]

    def account_info(self) -> dict:
        info = super().account_info()
        with self._lock:
            open_pos = [p.to_dict() for p in self._positions.values()]
        equity = round(self._calc_equity(open_pos), 2)
        margin = self._calc_margin(open_pos)
        return {
            **info,
            "login": info.get("login"),
            "server": info.get("server") or "PAPER",
            "company": "JARVIS PAPER",
            "currency": info.get("currency") or "USD",
            "leverage": int(self.leverage),
            "balance": round(self.balance, 2),
            "equity": equity,
            "margin": round(margin, 2),
            "margin_free": round(max(0.0, equity - margin), 2),
            "margin_level": round(equity / margin * 100.0, 2) if margin > 0 else 0.0,
            "profit": round(equity - self.balance, 2),
            "mode": self.mode,
        }

    def _calc_margin(self, open_pos: list[dict]) -> float:
        total = 0.0
        for p in open_pos:
            info = self._sym_info(p["symbol"])
            contract = float(info.get("trade_contract_size") or 1.0)
            price = float(p.get("price_open") or 0.0)
            total += (float(p["volume"]) * contract * price) / self.leverage
        return total

    def terminal_info(self) -> dict:
        ti = super().terminal_info()
        ti["trade_allowed"] = True
        ti["name"] = (ti.get("name") or "MetaTrader 5") + " [PAPER]"
        return ti

    # -------------------------------------------------------------- trading

    def send_market_order(
        self,
        *,
        symbol: str,
        direction: int,
        lot: float | None = None,
        sl: float,
        tp: float,
        comment: str = "jarvis",
    ) -> SendResult:
        if direction not in (-1, +1):
            return SendResult(False, None, None, "Invalid direction")
        with self._lock:
            # Idempotency: never re-open an identical signal twice.
            oid = comment.removeprefix("jarvis:")
            for p in self._positions.values():
                if p.signal_id and p.signal_id == comment and p.symbol == symbol:
                    return SendResult(False, None, None, "Duplicate signal (idempotent)")
            # One position per symbol+direction (hedging policy handled by strategy).
            for p in self._positions.values():
                if p.symbol == symbol and p.direction == direction and not self._hedge_policy_on():
                    return SendResult(False, None, None, "Duplicate direction on symbol")
            if len(self._positions) >= self.paper_max_positions:
                return SendResult(False, None, None, "Paper max positions reached")
            if time.time() < self._breaker_until:
                return SendResult(False, None, None, "Circuit breaker cooldown active")
            # Ensure market data availability for a realistic fill price.
            try:
                if not self.ensure_connected():
                    return SendResult(False, None, None, "MT5 not connected (no live fill)")
                tick = self.tick(symbol)
            except JarvisMT5Error as e:
                return SendResult(False, None, None, f"fill unavailable: {e}")

            entry = tick.ask if direction == 1 else tick.bid
            if entry <= 0:
                return SendResult(False, None, None, "No live price")
            vol = float(lot or self.default_lot)
            self._ticket_seq += 1
            ticket = self._ticket_seq
            pos = PaperPosition(
                ticket=ticket, symbol=symbol, type=(0 if direction == 1 else 1),
                volume=vol, price_open=entry, price_current=entry, sl=float(sl),
                tp=float(tp), time=time.time(), profit=0.0, swap=0.0,
                commission=0.0, magic=self.magic, comment=comment[:31],
                direction=direction, signal_id=comment, opened_at=time.time(),
            )
            self._positions[ticket] = pos
            self._last_price[symbol] = entry
            commission = self.commission_per_lot * vol
            self.balance -= commission
            self._journal("open", ticket=ticket, symbol=symbol, type=pos.type,
                          direction=direction, volume=vol, entry=entry, sl=float(sl),
                          tp=float(tp), signal_id=comment, comment=comment[:31],
                          commission=commission)
            self._write_state()
            rec = self._verify()

        LOGGER.info("PAPER OPEN %s %s id=%s entry=%.5f sl=%.5f tp=%.5f vol=%.2f",
                    symbol, "BUY" if direction == 1 else "SELL", ticket, entry, sl, tp, vol)
        return SendResult(True, ticket, 10009, "Paper fill (simulated)", {
            "paper": True, "entry": entry, "symbol": symbol,
        })

    def close_position(self, position_id: int) -> SendResult:
        with self._lock:
            pos = self._positions.get(int(position_id))
            if pos is None:
                return SendResult(False, None, None, "Position not found (paper)")
            try:
                if not self.ensure_connected():
                    return SendResult(False, None, None, "MT5 not connected (no live exit)")
                tick = self.tick(pos.symbol)
            except JarvisMT5Error:
                exit_price = pos.price_open
                reason_ext = "no-tick"
            else:
                exit_price = tick.bid if pos.type == 0 else tick.ask
                reason_ext = "market"

            pnl = self._close_boxed(pos, exit_price, "manual_close")
            self._positions.pop(int(position_id), None)
            self._write_state()
            self._verify()
        return SendResult(True, position_id, 10009, f"Paper close ({reason_ext}) pnl={pnl:.2f}")

    def close_all(self) -> list[SendResult]:
        out = []
        for p in self.open_positions():
            r = self.close_position(int(p.ticket))
            out.append(r)
        return out

    def modify_position(self, position_id: int, sl: float | None = None, tp: float | None = None) -> SendResult:
        with self._lock:
            pos = self._positions.get(int(position_id))
            if pos is None:
                return SendResult(False, None, None, "Position not found (paper)")
            if sl is not None:
                pos.sl = float(sl)
            if tp is not None:
                pos.tp = float(tp)
            self._journal("modify", ticket=int(position_id),
                          sl=sl if sl is not None else pos.sl,
                          tp=tp if tp is not None else pos.tp)
        return SendResult(True, position_id, 10009, "Paper modify (simulated)")

    def execute_order(self, *, symbol: str, order_type: int, volume: float,
                      price: float, sl: float, tp: float, magic: int = 0,
                      comment: str = "") -> dict:
        direction = 1 if order_type == 0 else -1
        r = self.send_market_order(symbol=symbol, direction=direction, lot=volume,
                                   sl=sl, tp=tp, comment=comment)
        return {"success": r.success, "order_id": r.order_id, "retcode": r.retcode,
                "comment": r.comment, "price": price, "sl": sl, "tp": tp, "volume": volume}

    # ------------------------------------------------------------- bubbling

    def poll_paper_state(self) -> int:
        """Mark-to-market + SL/TP auto-trigger + equity journal. Call each
        main-loop cycle. Returns number of auto-closed positions."""
        if not self.ensure_connected():
            return 0
        now = time.time()
        closed = 0
        with self._lock:
            positions = list(self._positions.values())
        for pos in positions:
            try:
                tick = self.tick(pos.symbol)
            except JarvisMT5Error:
                continue
            if pos.type == 0:
                pos.price_current = tick.bid
            else:
                pos.price_current = tick.ask
            self._last_price[pos.symbol] = pos.price_current

            trigger = self._sl_tp_trigger(pos, tick)
            if trigger:
                with self._lock:
                    if self._positions.get(pos.ticket) is not None:
                        self._close_boxed(pos, trigger["price"], trigger["reason"])
                        self._positions.pop(pos.ticket, None)
                        self._write_state()
                closed += 1

        if now - self._last_equity_ts >= self.equity_snapshot_every:
            self._last_equity_ts = now
            equity = self.equity()
            self._journal("equity", e=round(equity, 2), b=round(self.balance, 2),
                          n=len(self._positions))
        return closed

    def _sl_tp_trigger(self, pos: PaperPosition, tick) -> dict | None:
        """Return the exit reason+price when a live tick crossed SL/TP.
        Conservative: when both crossed, the stop-loss (loss-limit) wins."""
        if pos.type == 0:  # long
            if pos.sl > 0 and tick.bid <= pos.sl:
                return {"reason": "stop_loss", "price": pos.sl}
            if pos.tp > 0 and tick.ask >= pos.tp:
                return {"reason": "take_profit", "price": pos.tp}
        else:  # short
            if pos.sl > 0 and tick.ask >= pos.sl:
                return {"reason": "stop_loss", "price": pos.sl}
            if pos.tp > 0 and tick.bid <= pos.tp:
                return {"reason": "take_profit", "price": pos.tp}
        return None

    def _close_boxed(self, pos: PaperPosition, exit_price: float, reason: str) -> float:
        """Realize PnL + commission for a paper position (must hold self._lock
        or be sole owner). Returns net PnL in USD."""
        info = self._sym_info(pos.symbol)
        tick_size = float(info.get("trade_tick_size") or 1e-5)
        tick_value = float(info.get("trade_tick_value") or 1.0)
        pnl = (exit_price - pos.price_open) * pos.direction * tick_value * pos.volume / tick_size if tick_size else 0.0
        commission_out = self.commission_per_lot * pos.volume
        net = pnl - commission_out
        self.balance += net
        self._realized_day += net
        self._journal("close", ticket=pos.ticket, symbol=pos.symbol,
                      exit=exit_price, reason=reason, pnl=round(pnl, 2),
                      net_pnl=round(net, 2), commission=round(commission_out, 2))
        self._update_breaker(net)
        LOGGER.info("PAPER CLOSE %s %s reason=%s pnl=%.2f net=%.2f",
                    pos.symbol, "BUY" if pos.type == 0 else "SELL", reason, pnl, net)
        return net

    def _update_breaker(self, net_pnl: float) -> None:
        if self.breaker_consecutive_losses <= 0:
            return
        if net_pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.breaker_consecutive_losses:
                self._breaker_until = time.time() + self.breaker_cooldown_seconds
                self._journal("breaker", count=self._consecutive_losses,
                              until=self._breaker_until,
                              cooldown=self.breaker_cooldown_seconds)
                LOGGER.warning(
                    "PAPER CIRCUIT BREAKER: %d consecutive losses — trading halted until %s",
                    self._consecutive_losses,
                    datetime.fromtimestamp(self._breaker_until).strftime("%H:%M:%S"),
                )
        else:
            self._consecutive_losses = 0

    # ----------------------------------------------------------- verification

    def verify_state(self) -> tuple[bool, str]:
        """Public reconciliation check — returns (ok, error_message)."""
        ok = self._verify()
        if ok:
            return True, ""
        return False, f"journal vs memory mismatch (drift #{self._verify_counter})"

    def _verify(self) -> bool:
        """Reconciliation guard — journal must answer the same position set
        we hold in memory (counting 'open' minus 'close' entries)."""
        if not self._journal_dir:
            return True
        try:
            opened: set[int] = set()
            closed: set[int] = set()
            for line in (self._journal_dir / "ledger.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("kind") == "open":
                    opened.add(int(e["ticket"]))
                elif e.get("kind") == "close":
                    closed.add(int(e["ticket"]))
            journal_open = opened - closed
            mem_open = set(self._positions)
            if journal_open != mem_open:
                self._verify_counter += 1
                LOGGER.error(
                    "Paper reconciliation FAILED: journal=%d vs memory=%d (drift #%d)",
                    len(journal_open), len(mem_open), self._verify_counter,
                )
                return False
            return True
        except Exception:
            return False

    def _hedge_policy_on(self) -> bool:
        try:
            return bool(self._hedge_enabled)
        except AttributeError:
            return False

    def set_hedge_policy(self, enabled: bool) -> None:
        self._hedge_enabled = bool(enabled)


__all__ = ["PaperBridge", "PaperPosition"]