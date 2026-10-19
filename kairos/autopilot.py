"""Autopilot — autonomous opportunity scanning & execution loop.

Doctrine: paper is the default, live must be armed explicitly, everything
is fail-closed and journaled. The loop never trusts a stale bridge — before
every order it re-reads account and positions, honours RiskGate and the
kill switch, and PSX flow always remains a human-confirmed JS Global ticket.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import (Fill, JOURNAL, KILL, JsGlobalTicket, get_mt5,
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
        self.mt5 = get_mt5()
        self.jsg = JsGlobalTicket()
        self._posture: dict[str, Any] | None = None
        self.st = self._load()

    # ------------------------------------------------------------- state io
    def _defaults(self) -> dict[str, Any]:
        return {"running": False, "mode": "paper", "armed_live": False,
                "symbols": list(DEFAULT_WATCHLIST), "interval_s": 300,
                "entry_threshold": 0.18, "exit_threshold": 0.05,
                "max_trades_per_cycle": 2, "horizon_days": 5,
                "cycles": 0, "last_cycle": None, "last_error": None,
                "signals": [], "actions": [], "held": {}, "consec_losses": 0,
                "equity_peak": 0.0, "peak_since": None,
                "last_rotation": None, "rotated": []}

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
        return self.mt5.attach(login=m.get("login"), password=m.get("password"),
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
        from . import rules as _R
        from .engine import ENGINE
        from .universe import resolve

        st = self.st
        broker = PaperBroker()
        live = st["mode"] == "live" and self.mt5.ensure()
        acct = (self.mt5.account() if live else None) or broker.account()
        pos_list = self.mt5.positions() if live else broker.open_positions()
        held = {p["symbol"].upper(): p for p in (pos_list or [])}
        # the venue is the source of truth: drop managed entries it no
        # longer reports (manual close, broker-side stop hit, restart drift)
        self._reconcile(held, live)
        posture = self._risk_posture(acct, pos_list)
        with self._lock:
            self._posture = posture

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

            # ---- manage an existing position: stops/targets/trail/flip/age.
            # Risk reduction is never rate-limited by max_trades_per_cycle.
            hp = held.get((self._venue_symbol(inst.key, is_psx, live)).upper())
            if hp is not None:
                hb = st["held"].get(inst.key, {})
                reason = self._manage_exit(inst, hp, hb, price, score)
                if reason:
                    side = "SELL" if hp["side"] == "BUY" else "BUY"
                    qty = float(hp["qty"])
                    act, _d = self._execute(inst, side, qty, price, res,
                                            broker, live, close=True)
                    action, trades = act, trades + 1
                    if act in ("CLOSED", "LIVE-FILLED"):
                        self._book_close(inst.key, hp, hb, price, reason)
                        detail = f"{reason} exit"

            # ---- open a fresh position on a strong call ----
            elif abs(score) >= st["entry_threshold"] and hp is None and \
                    trades < st["max_trades_per_cycle"]:
                if posture["block_entries"]:
                    action, detail = "BREAKER", posture["blockers"][0]
                elif is_psx and (liq := self._psx_liquid(inst)):
                    action, detail = "LIQUIDITY", liq
                else:
                    # model-agreement gate: the battery must broadly concur
                    models = res.get("models") or []
                    dirn = 1 if score > 0 else -1
                    agree = (sum(1 for m in models
                                 if (m.get("direction") or 0) * dirn > 0)
                             / len(models)) if models else 1.0
                    if agree < float(_R.get_rule("min_model_agreement", 0.55)):
                        action = "AGREEMENT"
                        detail = f"only {agree:.0%} of models concur"
                    else:
                        # FUGU conductor: probability-weighted edge must be real
                        from . import fugu as _F
                        verdict = _F.conduct(inst.key, res=res, posture=posture)
                        st.setdefault("fugu", {})[inst.key] = {
                            "action": verdict["action"], "edge_r": verdict["edge_r"],
                            "conviction": verdict["conviction"], "p_hit": verdict["p_hit"],
                            "reasons": verdict["reasons"][:2]}
                        if verdict["action"] != "EXECUTE":
                            action = "FUGU-PASS"
                            detail = "; ".join(verdict["reasons"][:2])
                        else:
                            eq = float(acct["equity"] or 0)
                            notional = min((rk["suggested_weight"] or 0) * eq,
                                           0.15 * eq)
                            bands = res.get("confidence_band") or {}
                            sl0 = bands.get("p10") if score > 0 else bands.get("p90")
                            tgt0 = (res.get("targets") or {}).get(f"{s['horizon_days']}d")
                            if sl0 and eq > 0:
                                ps = _R.position_size(eq, price, sl0)
                                if ps.get("position_value"):
                                    notional = min(notional, float(ps["position_value"]))
                            if notional > 10:
                                qty = notional / price if not live else \
                                    round(max(0.01, notional / price / 100), 2)
                                side = "BUY" if score > 0 else "SELL"
                                act, detail = self._execute(inst, side, qty, price,
                                                            res, broker, live,
                                                            close=False)
                                action = act
                                trades += 1
                                if act in ("OPENED", "TICKET"):
                                    st["held"][inst.key] = {
                                        "started": _now(), "side": side,
                                        "qty": qty, "entry": price,
                                        "stop": sl0, "stop0": sl0, "target": tgt0}

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
        # self-improvement pulse: resolve realized outcomes so Fugu's
        # calibration anchor and the adaptive entry bar keep tracking truth
        if st["cycles"] % 12 == 0:
            try:
                from . import feedback as _FB
                n = _FB.resolve_outcomes(ENGINE.md)
                if n:
                    JOURNAL.log("FEEDBACK_RESOLVED", {"newly_resolved": n})
            except Exception:
                pass
        self._maybe_rotate_psx()
        return {"trades": trades, "signals": len(rows)}

    # ------------------------------------------------------- PSX sophistication
    def _psx_liquid(self, inst) -> str | None:
        """Microstructure screen: reject illiquid or penny PSX scrips — on the
        real exchange these are the names where tickets die unfilled or get
        hit by price bands. Returns a rejection reason, or None if tradable."""
        from . import rules as _R
        from .engine import ENGINE
        try:
            df = ENGINE.md.history(inst)
        except Exception:
            return None
        if df is None or len(df) < 20:
            return "insufficient PSX trading history"
        px = float(df["close"].iloc[-1])
        min_px = float(_R.get_rule("psx_min_price", 5.0))
        if px < min_px:
            return f"price {px:.2f} under Rs {min_px:.0f} — penny-scrip risk"
        if "volume" in df.columns:
            v = pd.to_numeric(df["volume"], errors="coerce").tail(20).mean()
            min_v = float(_R.get_rule("psx_min_adv_shares", 50000))
            if v is not None and not np.isnan(float(v)) and float(v) < min_v:
                return f"20d volume {float(v):,.0f}/day under {min_v:,.0f}"
        return None

    def _maybe_rotate_psx(self) -> None:
        """Daily scan-driven watchlist refresh: the cross-sectional factor
        leaders join the autopilot's own symbol set (and are swapped out when
        they fall off, unless a position is open)."""
        ap = CFG.section("autopilot") or {}
        if not ap.get("psx_auto_watch", True):
            return
        st = self.st
        last = st.get("last_rotation")
        if last and self._age_days(last) < 1.0:
            return
        try:
            from .engine import ENGINE
            scan = ENGINE.scan(top=15)
        except Exception:
            return
        if not scan.get("ok"):
            return
        from . import rules as _R
        from .universe import register_psx
        min_px = float(_R.get_rule("psx_min_price", 5.0))
        min_v = float(_R.get_rule("psx_min_adv_shares", 50000))
        fresh: list[str] = []
        for row in scan.get("top") or []:
            sym = str(row.get("name") or "").upper().strip()
            if not sym or sym in fresh:
                continue
            try:
                px = float(row.get("close") or 0)
                vol = float(row.get("volume") or 0)
            except (TypeError, ValueError):
                continue
            if px < min_px:                      # penny-scrip screen
                continue
            if vol and vol < min_v:              # liquidity screen
                continue
            if register_psx(sym, str(row.get("sector") or "")) is None:
                continue                         # collides with a non-PSX key
            fresh.append(sym)
            if len(fresh) == 6:
                break
        if not fresh:
            return
        rotated = set(st.get("rotated") or [])
        keep = [s for s in st["symbols"]
                if s not in rotated or s in st["held"]]
        st["symbols"] = keep + [s for s in fresh[:6] if s not in keep]
        st["rotated"] = fresh[:6]
        st["last_rotation"] = _now()
        self._act("WATCHLIST", "ROTATE", "PSX factor leaders: "
                  + ", ".join(fresh[:6]))
        JOURNAL.log("AUTOPILOT_ROTATION", {"added": fresh[:6]})
        self._save()

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

    def _act(self, symbol: str, action: str, why: str) -> None:
        self.st["actions"].insert(0, {"ts": _now(), "symbol": symbol,
                                      "action": action, "why": why})
        self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]

    # ------------------------------------------------- reconciliation & posture
    def _reconcile(self, held: dict[str, Any], live: bool) -> None:
        """Trust the venue over our memory: a managed entry the venue no
        longer reports is dropped (manual close / broker-side stop hit)."""
        from .universe import resolve
        for key in list(self.st["held"].keys()):
            inst = resolve(key)
            if inst and inst.market.upper() in ("PSX", "KSE"):
                continue          # PSX flow is tickets, never venue positions
            vsym = (self._venue_symbol(key, False, live) or key).upper()
            if vsym not in held:
                self.st["held"].pop(key, None)
                self._act(key, "RECONCILE", "venue no longer holds it")
                JOURNAL.log("AUTOPILOT_RECONCILE", {"symbol": key})

    def _risk_posture(self, acct: dict | None,
                      pos_list: list | None) -> dict[str, Any]:
        """Circuit breakers computed from fresh venue data every cycle."""
        from . import rules as _R
        r = _R.rules()
        eq = float((acct or {}).get("equity") or 0)
        day = float((acct or {}).get("day_pnl") or 0)
        gross = sum(abs(float(p.get("qty") or 0))
                    * float(p.get("last") or p.get("entry") or 0)
                    for p in (pos_list or []))
        exposure_pct = gross / eq * 100 if eq > 0 else 0.0
        consec = int(self.st.get("consec_losses") or 0)
        blockers = []
        if eq > 0 and day < -float(r["max_daily_loss_pct"]) / 100 * eq:
            blockers.append(f"daily loss {day:,.0f} breached the "
                            f"-{r['max_daily_loss_pct']}% breaker")
        if exposure_pct > float(r["max_total_exposure_pct"]):
            blockers.append(f"gross exposure {exposure_pct:.0f}% past the "
                            f"{r['max_total_exposure_pct']}% cap")
        if consec >= int(r["max_consecutive_losses"]):
            blockers.append(f"{consec} consecutive losses — cooling off entries")
        # cross-day equity drawdown guard (merged from JARVIS v1
        # profit_engine.DrawdownGuard): persistent peak tracker, de-risk after
        # a multi-day trough even when each individual day looked fine.
        peak = float(self.st.get("equity_peak") or 0)
        dd_pct = 0.0
        if eq > 0:
            if eq > peak:
                self.st["equity_peak"] = round(eq, 2)
                self.st["peak_since"] = _now()
            elif peak > 0:
                dd_pct = (1 - eq / peak) * 100
                if dd_pct > float(_R.get_rule("max_equity_drawdown_pct", 8.0)):
                    blockers.append(f"equity {dd_pct:.1f}% under peak "
                                    f"{peak:,.0f} — drawdown guard active")
        return {"equity": eq, "day_pnl": day,
                "exposure_pct": round(exposure_pct, 1),
                "consec_losses": consec, "n_positions": len(pos_list or []),
                "equity_peak": round(max(peak, eq), 2),
                "dd_from_peak_pct": round(dd_pct, 2),
                "block_entries": bool(blockers), "blockers": blockers}

    # ------------------------------------------------------ position management
    def _manage_exit(self, inst, hp: dict, hb: dict, price: float,
                     score: float) -> str | None:
        """Returns an exit reason for this position, or None to keep holding.
        Priority: hard stop > target > trailing ratchet > signal flip > maturity."""
        from . import rules as _R
        st = self.st
        dirn = 1 if hp["side"] == "BUY" else -1
        entry = float(hb.get("entry") or hp.get("entry") or 0)
        stop, tp = hb.get("stop"), hb.get("target")
        if stop and (price - float(stop)) * dirn <= 0:
            return "STOP"
        if tp and entry and (price - float(tp)) * dirn >= 0 and \
                (price - entry) * dirn > 0:
            return "TARGET"
        # trailing ratchet in R multiples — only ever moves with the trend
        stop0 = hb.get("stop0") or stop
        if stop0 and entry:
            r = abs(entry - float(stop0))
            profit = (price - entry) * dirn
            if r > 0 and profit >= float(_R.get_rule("trail_after_r_multiples", 1)) * r:
                k = int(profit // r)
                want = entry + dirn * r * max(0, k - 1)
                if stop is None or (want - float(stop)) * dirn > 0:
                    hb["stop"] = round(want, 6)
                    self._act(inst.key, "TRAIL",
                              f"stop ratcheted to {want:,.2f} (+{k - 1}R locked)")
                    JOURNAL.log("AUTOPILOT_TRAIL", {"symbol": inst.key,
                                                    "stop": hb["stop"]})
                    if hp.get("ticket") is not None and self.mt5.connected:
                        self.mt5.modify_sl(int(hp["ticket"]), sl=hb["stop"])
        flip = abs(score) >= st["exit_threshold"] and (score * dirn) < 0
        mature = hb.get("started") and \
            self._age_days(hb["started"]) >= st["horizon_days"]
        if flip:
            return "FLIP"
        if mature:
            return "MATURE"
        return None

    def _book_close(self, key: str, hp: dict, hb: dict, price: float,
                    reason: str) -> None:
        dirn = 1 if hp["side"] == "BUY" else -1
        entry = float(hp.get("entry") or hb.get("entry") or price)
        pnl = (price - entry) * dirn * float(hp.get("qty") or 0)
        st = self.st
        st["consec_losses"] = int(st.get("consec_losses") or 0) + 1 if pnl < 0 else 0
        st["held"].pop(key, None)
        JOURNAL.log("AUTOPILOT_BOOK_CLOSE",
                    {"symbol": key, "reason": reason, "pnl_est": round(pnl, 2),
                     "consec_losses": st["consec_losses"]})

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
            # live entry spread guard: never pay an absurd crossing cost
            if not close:
                t = self.mt5.tick(self.mt5.resolve_symbol(inst.key) or inst.key)
                msp = float(_R.get_rule("max_spread_pct", 0.6))
                if t and price > 0 and t["spread"] / price * 100 > msp:
                    why = f"spread {t['spread'] / price * 100:.2f}% > {msp}%"
                    self._act(inst.key, "SPREAD", why)
                    JOURNAL.log("AUTOPILOT_SPREAD", {"symbol": inst.key, "why": why})
                    return "SPREAD", why
            fill = self.mt5.place_order(intent)
            act = "LIVE-FILLED" if fill else "LIVE-REJECT"
            if fill and close:
                self._mt5_close_one(fill.symbol)
            if fill and not close:
                # never trust the ack — confirm the position exists venue-side
                time.sleep(1.0)
                syms = [str(p.get("symbol", "")).upper()
                        for p in (self.mt5.positions() or [])]
                if fill.symbol.upper() not in syms:
                    act = "LIVE-UNVERIFIED"
                    JOURNAL.log("AUTOPILOT_UNVERIFIED",
                                {"coid": intent.client_order_id,
                                 "symbol": fill.symbol})
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
        acct = broker.account()
        pos = broker.open_positions()
        posture = getattr(self, "_posture", None) or self._risk_posture(acct, pos)
        return {"ok": True, **{k: self.st[k] for k in
                ("running", "mode", "armed_live", "symbols", "interval_s",
                 "entry_threshold", "max_trades_per_cycle", "horizon_days",
                 "cycles", "last_cycle", "last_error", "signals", "actions",
                 "held", "consec_losses")},
                "risk_posture": posture,
                "kill_switch": KILL.engaged(),
                "mt5": {"connected": self.mt5.connected,
                        "error": self.mt5.last_error},
                "paper_account": acct,
                "paper_positions": pos}


def _read_json(path) -> dict[str, Any]:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


AUTOPILOT = Autopilot()
