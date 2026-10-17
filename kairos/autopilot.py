"""Autopilot — autonomous opportunity scanning & execution loop.

Doctrine: paper is the default, live must be armed explicitly, everything
is fail-closed and journaled. The loop never trusts a stale bridge — before
every order it re-reads account and positions, honours RiskGate and the
kill switch, and PSX flow always remains a human-confirmed JS Global ticket.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import (Fill, JOURNAL, KILL, JsGlobalTicket, Mt5Bridge,
                        OrderIntent, PaperBroker, RiskGate)

DEFAULT_WATCHLIST = ["GOLD", "WTI", "EURUSD", "BTC", "SPX"]
MAX_ACTIONS_KEPT = 80


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Autopilot:
    """A single supervised thread that repeatedly:
       scan watchlist -> engine verdicts -> size -> gate -> execute -> journal.

    mode:
      paper  — fills against live prices in the persistent PaperBroker book.
      live   — MT5 instruments route to the live bridge (requires a connected
               terminal + explicit confirm); PSX names ALWAYS become JS Global
               tickets awaiting human confirmation, never silent fills.
    """

    def __init__(self) -> None:
        self.state_path = CFG.data_dir / "autopilot.json"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.gate = RiskGate()
        self.mt5 = Mt5Bridge()
        self.jsg = JsGlobalTicket()
        self.st = self._load()

    # ------------------------------------------------------------- state io
    def _defaults(self) -> dict[str, Any]:
        return {"running": False, "mode": "paper", "armed_live": False,
                "symbols": list(DEFAULT_WATCHLIST), "interval_s": 300,
                "entry_threshold": 0.18, "exit_threshold": 0.05,
                "max_trades_per_cycle": 2, "horizon_days": 5,
                "cycles": 0, "last_cycle": None, "last_error": None,
                "signals": [], "actions": [], "held": {}}

    def _load(self) -> dict[str, Any]:
        st = self._defaults()
        try:
            saved = _read_json(self.state_path)
            st.update({k: v for k, v in saved.items() if k in st})
        except Exception:
            pass
        st["running"] = False   # a restart never resurrects a live loop
        return st

    def _save(self) -> None:
        try:
            self.state_path.write_text(__import__("json").dumps(self.st, default=str),
                                       encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------- controls
    def _mt5_connect(self) -> bool:
        m = CFG.section("mt5") or {}
        return self.mt5.connect(login=m.get("login"), password=m.get("password"),
                                server=m.get("server"),
                                terminal_path=m.get("terminal_path"))

    def configure(self, symbols: list[str] | None = None,
                  interval_s: int | None = None,
                  entry_threshold: float | None = None,
                  max_trades_per_cycle: int | None = None) -> None:
        with self._lock:
            if symbols:
                self.st["symbols"] = [s.upper().strip() for s in symbols if s.strip()]
            if interval_s:
                self.st["interval_s"] = max(30, int(interval_s))
            if entry_threshold is not None:
                self.st["entry_threshold"] = min(0.6, max(0.05, float(entry_threshold)))
            if max_trades_per_cycle:
                self.st["max_trades_per_cycle"] = min(10, int(max_trades_per_cycle))
            self._save()

    def start(self, mode: str = "paper", confirm_live: bool = False) -> dict[str, Any]:
        mode = mode.lower()
        note = ""
        if mode == "live":
            if not confirm_live:
                return {"ok": False, "error": "live mode requires confirm_live=true"}
            if KILL.engaged():
                return {"ok": False, "error": "kill switch is engaged — cannot arm live"}
            if not (self.mt5.connected or self._mt5_connect()):
                return {"ok": False,
                        "error": f"MT5 terminal not reachable: {self.mt5.last_error} "
                                 "— open MetaTrader 5, sign in to JS Markets and "
                                 "enable Algo Trading"}
            note = "live armed — MT5 instruments execute real orders; PSX stays ticket-only"
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "error": "autopilot already running"}
            self.st.update({"running": True, "mode": mode, "armed_live": mode == "live",
                            "last_error": None})
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="kairos-autopilot")
            self._thread.start()
        self._save()
        JOURNAL.log("AUTOPILOT_START", {"mode": mode, "symbols": self.st["symbols"]})
        return {"ok": True, "mode": mode, "note": note,
                "symbols": self.st["symbols"], "interval_s": self.st["interval_s"]}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        with self._lock:
            th = self._thread
        if th and th.is_alive():
            th.join(timeout=10)
        with self._lock:
            self.st.update({"running": False, "armed_live": False, "mode": "paper"})
            self._save()
        JOURNAL.log("AUTOPILOT_STOP", {})
        return {"ok": True, "running": False}

    # ------------------------------------------------------------- the loop
    def _loop(self) -> None:
        # first cycle shortly after start, then every interval
        delay = 15
        while not self._stop.wait(delay):
            try:
                self.cycle()
            except Exception as exc:
                with self._lock:
                    self.st["last_error"] = str(exc)[:400]
                    self._save()
                JOURNAL.log("AUTOPILOT_ERROR", {"error": str(exc)[:400]})
            delay = self.st["interval_s"]

    def cycle(self) -> dict[str, Any]:
        from .engine import ENGINE
        from .universe import resolve

        st = self.st
        broker = PaperBroker()
        live = st["mode"] == "live" and self.mt5.connected
        acct = (self.mt5.account() if live else None) or broker.account()
        pos_list = self.mt5.positions() if live else broker.open_positions()
        held = {p["symbol"].upper(): p for p in (pos_list or [])}

        rows: list[dict[str, Any]] = []
        trades = 0
        for sym in st["symbols"]:
            inst = resolve(sym)
            if inst is None:
                continue
            try:
                res = ENGINE.predict(inst.key, horizon=st["horizon_days"])
            except Exception as exc:
                rows.append({"symbol": inst.key, "name": inst.name,
                             "error": str(exc)[:120]})
                continue
            if not res.get("ok"):
                continue
            s, rk = res["signal"], res["risk"]
            score, price = s["score"], res["price"]
            is_psx = inst.market.upper() in ("PSX", "KSE")
            action = "hold"
            detail = ""

            # ---- exit existing position if the call flipped / matured ----
            hp = held.get((self._venue_symbol(inst.key, is_psx, live)).upper())
            if hp is not None and trades < st["max_trades_per_cycle"]:
                flip = abs(score) >= st["exit_threshold"] and \
                    (score * (1 if hp["side"] == "BUY" else -1)) < 0
                mature = st["held"].get(inst.key, {}).get("started", "") and \
                    self._age_days(st["held"][inst.key]["started"]) >= st["horizon_days"]
                if flip or mature:
                    side = "SELL" if hp["side"] == "BUY" else "BUY"
                    qty = float(hp["qty"])
                    act, detail = self._execute(inst, side, qty, price, res,
                                                broker, live, close=True)
                    action = act
                    trades += 1
                    if act.startswith("CLOSED"):
                        st["held"].pop(inst.key, None)

            # ---- open a fresh position on a strong call ----
            elif abs(score) >= st["entry_threshold"] and hp is None and \
                    trades < st["max_trades_per_cycle"]:
                eq = float(acct["equity"] or 0)
                notional = min((rk["suggested_weight"] or 0) * eq, 0.15 * eq)
                # doctrine sizing cap: risk-unit position from the stop
                bands = res.get("confidence_band") or {}
                sl0 = bands.get("p10") if score > 0 else bands.get("p90")
                if sl0 and eq > 0:
                    from . import rules as _R
                    ps = _R.position_size(eq, price, sl0)
                    if ps.get("position_value"):
                        notional = min(notional, float(ps["position_value"]))
                if notional > 10:
                    qty = notional / price if not live else \
                        round(max(0.01, notional / price / 100), 2)
                    side = "BUY" if score > 0 else "SELL"
                    act, detail = self._execute(inst, side, qty, price, res,
                                                broker, live, close=False)
                    action = act
                    trades += 1
                    if act in ("OPENED", "TICKET"):
                        st["held"][inst.key] = {"started": _now(),
                                                "side": side, "qty": qty}

            rows.append({"symbol": res["symbol"], "name": res["name"],
                         "price": price, "direction": s["direction"],
                         "score": score, "confidence": s["confidence"],
                         "expected_return": s["expected_return"],
                         "probability_up": res["probability_up"],
                         "weight": rk["suggested_weight"],
                         "regime": res["regime"]["regime"],
                         "action": action, "detail": detail})

        rows.sort(key=lambda r: -abs(r.get("score") or 0))
        with self._lock:
            st.update({"cycles": st["cycles"] + 1, "last_cycle": _now(),
                       "signals": rows[:12]})
            self._save()
        return {"trades": trades, "signals": len(rows)}

    # ------------------------------------------------------------- execution
    def _venue_symbol(self, key: str, is_psx: bool, live: bool) -> str:
        if live and not is_psx:
            return self.mt5.resolve_symbol(key) or key
        return key

    def _age_days(self, iso: str) -> float:
        try:
            return (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 86400
        except Exception:
            return 0.0

    def _execute(self, inst, side: str, qty: float, price: float,
                 res: dict[str, Any], broker: PaperBroker, live: bool,
                 close: bool) -> tuple[str, str]:
        if KILL.engaged():
            self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                          "action": "BLOCKED", "why": "kill switch"})
            self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
            return "BLOCKED", "kill switch engaged"

        is_psx = inst.market.upper() in ("PSX", "KSE")
        bands = res.get("confidence_band") or {}
        tgt = res.get("targets") or {}
        hkey = f"{res['signal']['horizon_days']}d"
        if side == "BUY":
            sl, tp = bands.get("p10"), tgt.get(hkey)
        else:
            sl, tp = bands.get("p90"), tgt.get(hkey)

        venue = "PAPER"
        if live and not is_psx:
            venue = "MT5"
        acc = (self.mt5.account() if venue == "MT5" else broker.account())
        pos = (self.mt5.positions() if venue == "MT5" else broker.open_positions())

        # ---- doctrine plan check (opens only): R:R, risk-per-trade,
        # confidence, stop side. Exits must always be allowed.
        from . import rules as _R
        if not close:
            plan = _R.check_plan(price, sl, tp,
                                 float((acc or {}).get("equity") or 0),
                                 (res.get("signal") or {}).get("confidence"),
                                 1 if side == "BUY" else -1, qty=qty)
            if not plan["passed"]:
                why = "; ".join(f"{c['rule']} {c['value']} (needs {c['required']})"
                                for c in plan["checks"] if not c["passed"])
                self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                              "action": "PLANCHECK", "why": why})
                self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
                JOURNAL.log("AUTOPILOT_PLANCHECK", {"symbol": inst.key, "why": why})
                return "PLANCHECK", why

        intent = OrderIntent(symbol=inst.key, side=side,
                             quantity=round(qty, 4), venue=venue,
                             stop_loss=sl, take_profit=tp,
                             comment="autopilot-close" if close else "autopilot")
        allowed, why = self.gate.check(intent, acc, pos)
        if not allowed:
            self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                          "action": "GATED", "why": why})
            self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
            JOURNAL.log("AUTOPILOT_GATED", {"coid": intent.client_order_id,
                                            "symbol": inst.key, "why": why})
            return "GATED", why

        if venue == "MT5":
            fill = self.mt5.place_order(intent)
            act = "LIVE-FILLED" if fill else "LIVE-REJECT"
            if fill and close:
                self._mt5_close_one(fill.symbol)
        elif is_psx and live:
            ticket = self.jsg.build_ticket(intent)
            act = "TICKET"
            fill = None
            self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                          "action": "TICKET", "why": ticket["ticket_id"]})
            self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
            return act, ticket["ticket_id"]
        else:
            fill = broker.submit(intent, price)
            act = "CLOSED" if close else "OPENED"
            if close and fill:
                broker.mark(inst.key, price)
        self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                      "action": act, "side": side,
                                      "qty": round(qty, 4), "price": price,
                                      "coid": intent.client_order_id})
        self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
        JOURNAL.log("AUTOPILOT_" + act.replace("-", "_"),
                    {"coid": intent.client_order_id, "symbol": inst.key,
                     "side": side, "qty": qty, "price": price, "venue": venue})
        return act, (fill.broker_order_id if isinstance(fill, Fill) else "")

    def _mt5_close_one(self, symbol: str) -> None:
        try:
            for p in (self.mt5.positions() or []):
                if p["symbol"] == symbol:
                    self.mt5.close_position(int(p["ticket"]))
        except Exception:
            pass

    # ------------------------------------------------------------- reporting
    def status(self) -> dict[str, Any]:
        broker = PaperBroker()
        return {"ok": True, **{k: self.st[k] for k in
                ("running", "mode", "armed_live", "symbols", "interval_s",
                 "entry_threshold", "max_trades_per_cycle", "horizon_days",
                 "cycles", "last_cycle", "last_error", "signals", "actions")},
                "kill_switch": KILL.engaged(),
                "mt5": {"connected": self.mt5.connected,
                        "error": self.mt5.last_error},
                "paper_account": broker.account(),
                "paper_positions": broker.open_positions()}


def _read_json(path) -> dict[str, Any]:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


AUTOPILOT = Autopilot()
