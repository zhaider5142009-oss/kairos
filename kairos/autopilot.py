"""Autopilot — autonomous opportunity scanning & execution loop.

Doctrine: paper is the default, live must be armed explicitly, everything
is fail-closed and journaled. The loop never trusts a stale bridge — before
every order it re-reads account and positions, honours RiskGate and the
kill switch, and PSX flow is a human-confirmed JS Global ticket unless the
operator turns every order key (autopilot.auto_place + js_global.allow_orders
+ the broker PIN).
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import (Fill, JOURNAL, KILL, JsGlobalTicket, get_mt5,
                        OrderIntent, PaperBroker, RiskGate)
from .store import atomic_write, dumps_strict

DEFAULT_WATCHLIST = ["GOLD", "WTI", "EURUSD", "BTC", "SPX"]
MAX_ACTIONS_KEPT = 80


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jsg_auto_ok() -> bool:
    """allow_orders AND a stored PIN — the two broker-side order keys."""
    try:
        from . import jsgtrade
        return jsgtrade.enabled()
    except Exception:
        return False


def _psx_live_ready() -> bool:
    """auto_place AND the broker-side keys: the only combination that makes a
    PSX verdict a real order, so it is also what makes PSX a live venue."""
    return bool((CFG.section("autopilot") or {}).get("auto_place")) and _jsg_auto_ok()


def _binance_live_ready() -> bool:
    """`binance.allow_orders` AND a usable key pair — the two keys in front of a
    real crypto order, mirroring `_jsg_auto_ok`.

    This one flag decides whether Binance is a venue at all: the loop may name it,
    count its balances in the exposure picture, and forget a managed entry it no
    longer reports only when the operator has armed it. Unarmed, a crypto verdict
    keeps going where it goes today.
    """
    try:
        from . import binance as _BN
        return _BN.enabled()
    except Exception:
        return False


def _figure(raw: Any) -> float:
    """A risk reading as a number, or zero when it is not one.

    Zero is the value the posture already reports as "no usable figure", so an
    unreadable account reading becomes a blocker instead of an exception that
    ends the cycle — and a NaN, which answers no comparison at all, never
    reaches the guards.
    """
    try:
        v = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _reading(raw: Any) -> float | None:
    """A risk figure that is allowed to have no answer.

    For the readings where an absence must stay visible: the day's P&L. `_figure`
    turns a hole into zero, and zero is the one value the daily loss brake cannot
    afford to invent — it is "flat day, keep opening", which is the fail-open this
    whole path exists to close. `None` here becomes a blocker that says so.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class Autopilot:
    """A single supervised thread that repeatedly:
       scan watchlist -> engine verdicts -> size -> gate -> execute -> journal.

    mode:
      paper  — fills against live prices in the persistent PaperBroker book.
      live   — each of the three venues runs on its own bridge: MT5 (commodities,
               FX) needs a connected terminal, JS Global (PSX cash) needs
               auto_place + allow_orders + the broker PIN, Binance (crypto) needs
               binance.allow_orders + a key pair that the key itself permits to
               trade. Any one of them alone is enough to make the loop live; a
               venue that drops mid-loop stops taking entries there while the
               others keep working, and its open trades are kept rather than
               forgotten. PSX flow without all three keys stays a human-confirmed
               JS Global ticket.
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
        self._bn_armed = False
        # this cycle's live venue snapshots — never persisted, the real book
        # carries account numbers
        self._live_acct: dict[str, Any] | None = None
        self._psx_live: dict[str, Any] | None = None
        self._psx_down = False
        self.st = self._load()

    # ------------------------------------------------------------- state io
    def _defaults(self) -> dict[str, Any]:
        return {"running": False, "mode": "paper", "armed_live": False,
                "symbols": list(DEFAULT_WATCHLIST), "interval_s": 300,
                "entry_threshold": 0.18, "exit_threshold": 0.05,
                "max_trades_per_cycle": 2, "horizon_days": 5,
                "cycles": 0, "last_cycle": None, "last_cycle_s": None,
                "last_error": None, "feedback_error": None,
                "state_saved": None, "state_error": None,
                "signals": [], "actions": [], "held": {}, "consec_losses": 0,
                "equity_peak": 0.0, "peak_since": None,
                "last_rotation": None, "rotated": [], "psx_day_start": None,
                "macro": None, "news_desk": None}

    def _load(self) -> dict[str, Any]:
        st = self._defaults()
        try:
            saved = _read_json(self.state_path)
            st.update({k: v for k, v in saved.items() if k in st})
        except Exception as exc:
            # An unreadable state file used to be indistinguishable from a first
            # run: full capital, no positions, peak reset. The loop trades on that
            # as an account, and the desk calls it healthy.
            st["state_error"] = (f"state file unreadable ({type(exc).__name__}: "
                                 f"{str(exc)[:140]}) — running on defaults")
        st["running"] = False   # a restart never resurrects a live loop
        # and it must never *report* one either: the desk shows LIVE straight
        # from these fields, so a stale arming in the file would tell the
        # operator the opposite of what the process is doing.
        st["mode"] = "paper"
        st["armed_live"] = False
        return st

    def _save(self) -> None:
        """Persist the live state, and keep the failure visible.

        `except Exception: pass` used to sit here. The desk reads `cycles`,
        `equity_peak`, `held` and `consec_losses` out of this file, so a stopped
        write left the operator reading a state from hours ago while the loop went
        on trading — and the next restart recomputed the circuit breakers from that
        stale peak. `state_error` rides inside the file when the file can still be
        written, and in `status()` when it cannot.
        """
        self.st["state_error"] = None
        try:
            text = dumps_strict(self.st)
        except Exception as exc:
            self._state_failed("serialize", exc)
            return
        try:
            atomic_write(self.state_path, text)
        except Exception as exc:
            self._state_failed("write", exc)
            return
        self.st["state_saved"] = time.time()

    def _state_failed(self, stage: str, exc: Exception) -> None:
        reason = f"state {stage} failed ({type(exc).__name__}: {str(exc)[:160]})"
        self.st["state_error"] = reason
        if reason != getattr(self, "_state_err", None):
            # one journal row per distinct failure: a full disk fails every cycle,
            # and 288 identical rows a day buries the event it is meant to mark.
            self._state_err = reason
            JOURNAL.log("AUTOPILOT_STATE_" + stage.upper() + "_FAILED",
                        {"error": reason})

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
            mt5_up = self.mt5.connected or self._mt5_connect()
            psx_up = _psx_live_ready()
            if not (mt5_up or psx_up):
                return {"ok": False,
                        "error": "no live venue reachable — MT5: "
                                 f"{self.mt5.last_error or 'not attached'} | "
                                 "JS Global: needs autopilot.auto_place "
                                 "+ js_global.allow_orders + the trading PIN"}
            venues = " + ".join(n for n, ok in (("MT5", mt5_up),
                                                ("JS Global PSX", psx_up)) if ok)
            note = (f"live armed on {venues}; risk posture runs on the real "
                    "book and a venue that goes down mid-loop takes no new "
                    "entries there")
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
            interval = self.st["interval_s"]
            try:
                self.cycle()
            except Exception as exc:
                with self._lock:
                    self.st["last_error"] = str(exc)[:400]
                    self._save()
                JOURNAL.log("AUTOPILOT_ERROR", {"error": str(exc)[:400]})
            # a cycle that outlived its own interval means this box is behind:
            # skip a lap instead of hammering the CPU back-to-back
            delay = interval * 2 if (self.st.get("last_cycle_s") or 0) > interval else interval

    def cycle(self) -> dict[str, Any]:
        started = time.perf_counter()
        return self._cycle(started)

    def _cycle(self, started: float) -> dict[str, Any]:
        from . import rules as _R
        from .engine import ENGINE
        from .universe import resolve

        st = self.st
        broker = PaperBroker()
        want_live = st["mode"] == "live"
        mt5_up = bool(self.mt5.ensure()) if want_live else False
        psx_book = self._psx_book() if want_live else None
        psx_up = psx_book is not None
        # Armed, not merely configured: an unarmed Binance contributes nothing to
        # this picture, so a read-only API key cannot quietly become a third
        # account whose balances the exposure breaker counts as blind rows.
        bn_armed = _binance_live_ready() if want_live else False
        live = want_live and (mt5_up or psx_up or bn_armed)
        # live sizing/braking is REAL money only: never fall back to the
        # paper book (Rs 1,000,000 of simulated equity would justify orders
        # the funded account could not settle)
        acct = self._live_account(mt5_up, psx_book, bn_armed) if live else broker.account()
        pos_list = self._venue_positions(mt5_up, psx_book, bn_armed) if live \
            else broker.open_positions()
        self._live_acct = acct if live else None
        self._psx_live = psx_book
        self._bn_armed = bn_armed
        held = {p["symbol"].upper(): p for p in (pos_list or [])}
        # the venue is the source of truth: drop managed entries it no
        # longer reports (manual close, broker-side stop hit, restart drift)
        self._reconcile(held, live, mt5_up, psx_up, pos_known=pos_list is not None,
                        bn_up=bn_armed)
        posture = self._risk_posture(acct, pos_list)
        with self._lock:
            self._posture = posture

        rows: list[dict[str, Any]] = []
        trades = 0
        macro: dict[str, Any] | None = None
        # One news round trip for the whole watchlist, once a cycle. `predict`
        # only reads the desk cache, because a host that takes 3.7-5.1s per query
        # does not belong on a path that already costs 9.7s per instrument; here
        # the twelve fetches run in parallel and the cycle pays for one slow
        # symbol, not twelve. Measured 2026-10-23: 12 instruments in 8.7s.
        watch = [i for i in (resolve(s) for s in st["symbols"]) if i is not None]
        try:
            readings = ENGINE.refresh_news(watch)
            news_desk = {"asked": len(watch),
                         "available": sum(1 for v in readings.values()
                                          if v.get("available")),
                         "paused": ENGINE.news_paused}
        except Exception as exc:
            # A dead news desk must not cost the cycle: the strategies that do
            # not read news carry on, and "News Sentiment Tilt" abstains. But it
            # has to be *recorded* as dead, or a broken feed looks exactly like
            # a market with no news in it.
            news_desk = {"asked": len(watch), "available": 0,
                         "paused": ENGINE.news_paused,
                         "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        with self._lock:
            self.st["news_desk"] = news_desk
            self._save()
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
            if macro is None:
                # one condition for the whole cycle: every symbol is scored
                # against the same panel, so the desk shows the state that
                # sized the book rather than twelve copies of it
                macro = res.get("macro")
            s, rk = res["signal"], res["risk"]
            score, price = s["score"], res["price"]
            is_psx = inst.market.upper() in ("PSX", "KSE")
            action = "hold"
            detail = ""

            # ---- manage an existing position: stops/targets/trail/flip/age.
            # Risk reduction is never rate-limited by max_trades_per_cycle.
            hp = next((held[s] for s in self._venue_symbols(inst.key, is_psx, live)
                       if s in held), None)
            if hp is not None:
                hb = st["held"].get(inst.key, {})
                reason = self._manage_exit(inst, hp, hb, price, score)
                if reason:
                    side = "SELL" if hp["side"] == "BUY" else "BUY"
                    qty = float(hp["qty"])
                    act, _d = self._execute(inst, side, qty, price, res,
                                            broker, live, close=True)
                    action, trades = act, trades + 1
                    if act in ("CLOSED", "LIVE-FILLED", "PLACED"):
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
                                if live and not is_psx:
                                    # MT5 quotes CFDs in lots, not shares
                                    qty = round(max(0.01,
                                                    notional / price / 100), 2)
                                elif live:
                                    qty = int(notional // price)   # PSX shares
                                else:
                                    qty = notional / price
                                side = "BUY" if score > 0 else "SELL"
                                act, detail = self._execute(inst, side, qty, price,
                                                            res, broker, live,
                                                            close=False)
                                action = act
                                trades += 1
                                if act in ("OPENED", "TICKET", "PLACED"):
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
                         "macro": (res.get("macro") or {}).get("tag"),
                         "news": (res.get("news") or {}).get("score"),
                         "news_source": (res.get("news") or {}).get("source"),
                         "condition": res.get("condition"),
                         "action": action, "detail": detail})

        rows.sort(key=lambda r: -abs(r.get("score") or 0))
        with self._lock:
            st.update({"cycles": st["cycles"] + 1, "last_cycle": _now(),
                       "last_cycle_s": round(time.perf_counter() - started, 1),
                       "signals": rows[:12], "macro": macro})
            self._save()
        # self-improvement pulse: resolve realized outcomes so Fugu's
        # calibration anchor and the adaptive entry bar keep tracking truth
        if st["cycles"] % 12 == 0:
            try:
                from . import feedback as _FB
                n = _FB.resolve_outcomes(ENGINE.md)
                if n:
                    JOURNAL.log("FEEDBACK_RESOLVED", {"newly_resolved": n})
                with self._lock:
                    self.st["feedback_error"] = None
                    self._save()
            except Exception as exc:
                # Never `pass`. A forward record that stopped maturing looks
                # exactly like one that is simply not due yet — the ledger
                # prints its pending queue either way — and the pulse that
                # scores it runs twice an hour in a thread nobody watches.
                with self._lock:
                    self.st["feedback_error"] = str(exc)[:300]
                    self._save()
                JOURNAL.log("FEEDBACK_FAILED", {"error": str(exc)[:300]})
        self._maybe_rotate_psx()
        return {"trades": trades, "signals": len(rows)}

    # ------------------------------------------------------- PSX sophistication
    def _psx_liquid(self, inst) -> str | None:
        """Microstructure screen: reject illiquid or penny PSX scrips — on the
        real exchange these are the names where tickets die unfilled or get
        hit by price bands. Returns a rejection reason, or None if tradable.

        A screen that cannot see answers 'reject'. Both fail-opens below were
        the same mistake: `px < min_px` is False for a NaN, so an unreadable
        scrip cleared the penny-stock check, and a history call that raised was
        swallowed into a None that reads as 'liquid enough'."""
        from . import rules as _R
        from .engine import ENGINE
        try:
            df = ENGINE.md.history(inst)
        except Exception as exc:
            return f"PSX history unreadable: {str(exc)[:80]}"
        if df is None or len(df) < 20:
            return "insufficient PSX trading history"
        px = float(df["close"].iloc[-1])
        min_px = float(_R.get_rule("psx_min_price", 5.0))
        if not math.isfinite(px):
            return f"last price is not a usable number ({px!r})"
        if px < min_px:
            return f"price {px:.2f} under Rs {min_px:.0f} — penny-scrip risk"
        if "volume" in df.columns:
            v = pd.to_numeric(df["volume"], errors="coerce").tail(20).mean()
            # One floor from two keys (`psx_min_adv_shares` and the doctrine's
            # `min_avg_volume`), the tighter of the two, so tightening either one
            # in config actually changes what gets screened out.
            min_v, _floor_src = _R.adv_share_floor()
            if v is None or np.isnan(float(v)):
                # Every volume cell missing is not a liquid name; it is a screen
                # with nothing to read, which this function's own policy treats
                # as a rejection.
                return ("PSX volume unreadable over the last 20 bars — the "
                        "liquidity floor cannot be checked")
            if float(v) < min_v:
                return f"20d volume {float(v):,.0f}/day under {min_v:,.0f} " \
                       f"({_floor_src})"
            return None
        # No volume column at all: this docstring's own policy is that a screen
        # which cannot see answers "reject", and a frame with no volume is the
        # clearest case of not seeing.
        return "PSX history has no volume column — liquidity unverifiable"

    def _maybe_rotate_psx(self) -> None:
        """Daily scan-driven watchlist refresh: the cross-sectional factor
        leaders join the autopilot's own symbol set (and are swapped out when
        they fall off, unless a position is open)."""
        ap = CFG.section("autopilot") or {}
        if not ap.get("psx_auto_watch", True):
            return
        st = self.st
        last = st.get("last_rotation")
        if last:
            age = self._age_days(last)
            if age is not None and age < 1.0:
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
        min_v, _floor_src = _R.adv_share_floor()
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
            # A candidate that reports no volume is not a liquid one: the old
            # reading skipped this screen precisely for the rows whose liquidity
            # was unknown, which is the name the screen exists to catch.
            if not vol or vol < min_v:           # liquidity screen
                continue
            if register_psx(sym, sector=str(row.get("sector") or "")) is None:
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
    def _venue_symbols(self, key: str, is_psx: bool, live: bool) -> list[str]:
        """Every symbol a live venue could report this name under, most likely first.

        A crypto key now has two candidates — the Binance pair on the armed book
        and whatever the MT5 map says — because arming one venue does not retire
        the position the other one already holds. Callers therefore match against
        the set, and reconciliation drops an entry only when no venue that
        answered holds it: forgetting a live trade on the strength of a guessed
        symbol is the exact failure this function would otherwise introduce.
        """
        if not live or is_psx:
            return [key.upper()]
        out: list[str] = []
        if getattr(self, "_bn_armed", False):
            from . import binance as _BN
            pair = _BN.pair_for(key)
            if pair:
                out.append(pair.upper())
        out.append(str(self.mt5.resolve_symbol(key) or key).upper())
        return out

    def _age_days(self, iso: str) -> float | None:
        """Days between now and an ISO stamp, or None when there is no answer.

        Zero here used to mean "brand new", and both callers of an unreadable
        stamp took it: a held entry whose start date could not be read never
        reached maturity, and the watchlist stopped rotating for the day — each
        of which looks exactly like a system working normally.
        """
        try:
            return (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 86400
        except (TypeError, ValueError):
            return None

    def _act(self, symbol: str, action: str, why: str) -> None:
        self.st["actions"].insert(0, {"ts": _now(), "symbol": symbol,
                                      "action": action, "why": why})
        self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]

    # ------------------------------------------------- reconciliation & posture
    def _reconcile(self, held: dict[str, Any], live: bool, mt5_up: bool = True,
                   psx_up: bool = False, pos_known: bool = True,
                   bn_up: bool = False) -> None:
        """Trust the venue over our memory: a managed entry the venue no
        longer reports is dropped (manual close / broker-side stop hit).

        Only venues with a snapshot this cycle are authoritative — forgetting
        a live trade because a terminal was briefly closed is worse than
        keeping it. `pos_known` is the same rule one level finer: a venue that
        was up but whose position read failed hands back an empty `held` that
        says nothing at all, and dropping the entries then leaves a live trade
        being traded with nobody managing its stop.
        """
        if not pos_known:
            return
        from .universe import resolve
        for key in list(self.st["held"].keys()):
            inst = resolve(key)
            is_psx = bool(inst) and inst.market.upper() in ("PSX", "KSE")
            if is_psx and not (live and psx_up):
                continue        # ticket, paper entry or portal outage: not a state
            if not is_psx and live and not (mt5_up or bn_up):
                continue        # no non-PSX venue answered — keep the memory
            if not any(s in held for s in self._venue_symbols(key, is_psx, live)):
                self.st["held"].pop(key, None)
                self._act(key, "RECONCILE", "venue no longer holds it")
                JOURNAL.log("AUTOPILOT_RECONCILE", {"symbol": key})

    def _psx_book(self) -> dict[str, Any] | None:
        """The funded JS Global book, fresh, or None if it cannot be trusted.

        None must never be read as an empty account: an outage that looked
        like zero equity would have let every circuit breaker pass. So a book
        that does not describe a funded account is a venue loss too, even when
        the portal raised nothing on the way out — `ledger_balance` answers a
        mis-keyed query with 0.0 instead of an error, which alone is enough to
        hand back exactly the outage-shaped zero this function exists to stop.
        """
        from . import jsgsync
        try:
            book = jsgsync.real_book()
        except Exception as exc:
            return self._psx_lost(str(exc)[:200])
        if float(book.get("equity") or 0) <= 0:
            return self._psx_lost(
                "portal returned a zero-equity book (cash "
                f"{book.get('cash')!r}, "
                f"{len(book.get('holdings') or [])} holdings)")
        self._psx_down = False
        return book

    def _psx_lost(self, why: str) -> None:
        """Mark the PKR venue lost, once, and hand back None to fail closed."""
        if not getattr(self, "_psx_down", False):
            self._psx_down = True
            self._act("PSX", "VENUE-DOWN", why[:160])
            JOURNAL.log("AUTOPILOT_VENUE_LOSS", {"venue": "JS_GLOBAL",
                                                 "error": why})
        return None

    def _psx_day_pnl(self, book: dict[str, Any]) -> float:
        """Today's move on the real PKR book.

        The portal reports no day P&L, so it is measured against the first
        equity snapshot of the calendar day. Deposits would read as profit —
        the loop never funds the account, so that never happens here.

        Only a positive equity may set that baseline. A zero stamped here is
        sticky: measuring any later reading against it produces no loss figure
        at all, so one degraded read at the open would have kept the daily-loss
        breaker silent for the rest of the session on a book that was losing
        money the whole time.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        eq = round(float(book.get("equity") or 0), 2)
        if eq <= 0:
            return 0.0                       # unusable read; stamp nothing
        snap = self.st.get("psx_day_start") or {}
        if snap.get("date") != today or float(snap.get("equity") or 0) <= 0:
            # The second half also heals a baseline an earlier build latched
            # at zero, which otherwise pinned this figure forever.
            self.st["psx_day_start"] = {"date": today, "equity": eq}
            return 0.0
        return round(eq - float(snap["equity"]), 2)

    def _psx_part(self, book: dict[str, Any]) -> dict[str, Any]:
        """The PKR book in account-snapshot shape (what the gates read)."""
        return {"venue": "JS_GLOBAL", "balance": book["cash"],
                "equity": book["equity"], "day_pnl": self._psx_day_pnl(book),
                "open_positions": len(book["holdings"])}

    def _live_account(self, mt5_up: bool, book: dict[str, Any] | None,
                      bn_up: bool = False) -> dict[str, Any] | None:
        """Real-currency account snapshot across whichever live venues are up.

        None when nothing is reachable — RiskGate and the doctrine check both
        fail closed on None, which is the point.

        The sum below mixes currencies (MT5/Binance quote USD, JS Global quotes
        PKR) and is therefore only an upper bound in rupee terms — it overstates
        a PKR-heavy book against a USD one, which errs toward refusing, never
        toward sizing up. `_execute` never sizes from this: each venue branch
        reads its own account, in its own currency.
        """
        parts: list[dict[str, Any]] = []
        if mt5_up:
            a = self.mt5.account()
            if a:
                parts.append(a)
        if book:
            parts.append(self._psx_part(book))
        if bn_up:
            from . import binance as _BN
            a = _BN.BINANCE.account()
            if a:
                parts.append(a)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        counts = [p.get("open_positions") for p in parts]
        return {"venue": "+".join(str(p.get("venue")) for p in parts),
                "balance": round(sum(float(p.get("balance") or 0) for p in parts), 2),
                "equity": round(sum(float(p.get("equity") or 0) for p in parts), 2),
                "day_pnl": round(sum(float(p.get("day_pnl") or 0) for p in parts), 2),
                # one venue that could not count its positions makes the total
                # unknown; summing it as zero would print a flat account
                "open_positions": (None if any(c is None for c in counts)
                                   else sum(int(c or 0) for c in counts))}

    def _venue_positions(self, mt5_up: bool,
                         book: dict[str, Any] | None,
                         bn_up: bool = False,
                         ) -> list[dict[str, Any]] | None:
        """Every live position, or None when a venue that was up could not read.

        None and [] are different facts and the whole risk picture turns on
        which one it is: [] says the broker answered "flat" and lets the
        exposure breaker pass, None says nobody answered. The terminal used to
        be folded into the first, so a dead MT5 link zeroed gross exposure,
        cleared the position-count cap and let reconciliation forget the live
        trades it was supposed to be managing.
        """
        out: list[dict[str, Any]] = []
        if mt5_up:
            poss = self.mt5.positions()
            if poss is None:
                return None
            out.extend(poss)
        for h in (book or {}).get("holdings", []):
            out.append({"symbol": h["symbol"], "side": "BUY", "qty": h["qty"],
                        "entry": h["avg"], "last": h["mtm"],
                        "pnl": round((h["mtm"] - h["avg"]) * h["qty"], 2),
                        "venue": "JS_GLOBAL"})
        if bn_up:
            from . import binance as _BN
            poss = _BN.BINANCE.positions()
            if poss is None:
                # Armed and unreadable is the same fact a dead terminal is.
                return None
            out.extend(poss)
        return out

    def _risk_posture(self, acct: dict | None,
                      pos_list: list | None) -> dict[str, Any]:
        """Circuit breakers computed from fresh venue data every cycle."""
        from . import rules as _R
        r = _R.rules()
        eq = _figure((acct or {}).get("equity"))
        day = _reading((acct or {}).get("day_pnl"))
        dl_frac, dl_src = _R.daily_loss_fraction(CFG.risk, r)
        pos_unknown = pos_list is None
        rows = [(abs(_figure(p.get("qty"))),
                 _figure(p.get("last") or p.get("entry"))) for p in (pos_list or [])]
        # A held position whose price cannot be read is not a free one: summing
        # zero for it reports an exposure smaller than the book, and the gross
        # cap is the only brake on how much is at risk. A row with no quantity is
        # genuinely nothing at risk (a pending ticket), so it is not blindness.
        blind_rows = sum(1 for q, px in rows if q > 0 and px <= 0)
        gross = sum(q * px for q, px in rows)
        exposure_pct = (None if pos_unknown or eq <= 0
                        else gross / eq * 100)
        consec = int(self.st.get("consec_losses") or 0)
        blockers = []
        if eq <= 0:
            # Every guard below needs a positive amount to compare against, so
            # a cycle that cannot read equity has NO breakers rather than
            # working ones. The gate still refuses the order; the posture says
            # so too, instead of reporting a clean risk picture it never had.
            blockers.append("no usable equity figure — venues unreadable")
        if pos_unknown:
            blockers.append("position list unreadable — exposure unknown, "
                            "not flat")
        elif blind_rows:
            blockers.append(f"{blind_rows} position row(s) unreadable — gross "
                            "exposure understated, not zero")
        if day is None:
            blockers.append("day P&L unreadable — the daily loss brake is blind")
        elif eq > 0 and day < -dl_frac * eq:
            blockers.append(f"daily loss {day:,.0f} breached the "
                            f"-{dl_frac * 100:.2f}% breaker ({dl_src})")
        if exposure_pct is not None and \
                exposure_pct > float(r["max_total_exposure_pct"]):
            blockers.append(f"gross exposure {exposure_pct:.0f}% past the "
                            f"{r['max_total_exposure_pct']}% cap")
        if consec >= int(r["max_consecutive_losses"]):
            blockers.append(f"{consec} consecutive losses — cooling off entries")
        # cross-day equity drawdown guard (merged from JARVIS v1
        # profit_engine.DrawdownGuard): persistent peak tracker, de-risk after
        # a multi-day trough even when each individual day looked fine.
        peak = _figure(self.st.get("equity_peak"))
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
                "daily_loss_limit_pct": round(dl_frac * 100, 2),
                "daily_loss_limit_from": dl_src,
                "exposure_pct": None if exposure_pct is None
                                    else round(exposure_pct, 1),
                "consec_losses": consec,
                "n_positions": None if pos_unknown else len(pos_list),
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
        started = hb.get("started")
        age = self._age_days(started) if started else None
        if started and age is None and not hb.get("age_warned"):
            # Once for the row, and into the trail: a position with no readable
            # start date keeps its other exits, and the desk has to say the
            # maturity exit is off instead of showing an entry that looks
            # simply not due yet.
            hb["age_warned"] = True
            self._act(inst.key, "AGE UNREADABLE",
                      f"start date {str(started)[:20]!r} is not a date — this "
                      "entry has no maturity exit until the state file is fixed")
            JOURNAL.log("AUTOPILOT_AGE_UNREADABLE",
                        {"symbol": inst.key, "started": str(started)[:40]})
        # Two timers name the same instruction: the autopilot's own horizon and
        # the doctrine's `max_hold_days`, which used to bind nothing. The tighter
        # governs, so an operator who sets a 3-day hold cap gets a 3-day hold cap.
        due, _due_src = _R.maturity_days(st["horizon_days"])
        mature = age is not None and age >= due
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
        from . import binance as _BN
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
            # A crypto name has a home venue now, but only once that venue is
            # armed: naming an unarmed one would route the order to a bridge
            # whose only answer is a refusal.
            venue = ("BINANCE" if (getattr(self, "_bn_armed", False)
                                   and _BN.pair_for(inst.key)) else "MT5")
        elif live and is_psx:
            venue = "JS_GLOBAL"
        if venue == "MT5":
            acc, pos = self.mt5.account(), self.mt5.positions()
        elif venue == "BINANCE":
            # USD sizing against the USD book. A PKR figure from the PSX book (or
            # the paper Rs 1,000,000) would misprice every guard on this order for
            # the same reason it misprices a PSX one, in the other direction.
            acc, pos = _BN.BINANCE.account(), _BN.BINANCE.positions()
        elif venue == "JS_GLOBAL":
            # PKR sizing against the PKR book only. A USD MT5 equity figure
            # (or the paper Rs 1,000,000) would misprice every risk check on
            # a real order; with no book this fails closed at the gate.
            b = getattr(self, "_psx_live", None)
            acc = self._psx_part(b) if b else None
            pos = self._venue_positions(False, b)
        else:
            acc, pos = broker.account(), broker.open_positions()

        # ---- doctrine plan check (opens only): R:R, risk-per-trade,
        # confidence, stop side. Exits must always be allowed.
        from . import rules as _R
        if not close:
            # The doctrine's own numbers, from the frame the prediction was made
            # on: the name cap needs to know how many names are already open, and
            # the trend and heat filters need the mean and the RSI. A venue whose
            # book cannot be read yields `n_held=None`, which the cap treats as
            # full — "we do not know how many we hold" is not "we hold none".
            doc = res.get("doctrine") or {}
            n_held = (None if pos is None
                      else max(len(self.st.get("held") or {}), len(pos)))
            plan = _R.check_plan(price, sl, tp,
                                 float((acc or {}).get("equity") or 0),
                                 (res.get("signal") or {}).get("confidence"),
                                 1 if side == "BUY" else -1, qty=qty,
                                 n_held=n_held, sma=doc.get("sma"),
                                 rsi14=doc.get("rsi14"))
            if not plan["passed"]:
                why = plan["failures"] or "plan check refused"
                if plan["blind"]:
                    why += " [not evaluated: " + ", ".join(plan["blind"]) + "]"
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
        elif venue == "BINANCE":
            if not close:
                # A 24/7 tape moves while the daily bar the call was made on does
                # not. If the venue's price and the decided price have come apart
                # past the doctrine's crossing tolerance, the call is stale and
                # the honest action is to send nothing.
                pair = _BN.pair_for(inst.key) or inst.key
                px = _BN.BINANCE.tick(pair)
                msp = float(_R.get_rule("max_spread_pct", 0.6))
                if px is None:
                    why = f"{pair}: the venue gave no price, so this order " \
                          f"cannot be checked against the tape it will execute on"
                    self._act(inst.key, "NOPRICE", why)
                    JOURNAL.log("AUTOPILOT_NOPRICE", {"symbol": inst.key, "why": why})
                    return "NOPRICE", why
                if price > 0 and abs(px - price) / price * 100 > msp:
                    why = (f"venue {px:g} vs decided {price:g} = "
                           f"{abs(px - price) / price * 100:.2f}% > {msp}%")
                    self._act(inst.key, "TAPE-DIVERGE", why)
                    JOURNAL.log("AUTOPILOT_TAPE_DIVERGE",
                                {"symbol": inst.key, "why": why})
                    return "TAPE-DIVERGE", why
            fill = _BN.BINANCE.place_order(intent)
            # None is three different facts on a venue: refused, rejected, or sent
            # and unanswered. The last one may be a live order, so it must not be
            # booked as a refusal.
            oc = _BN.BINANCE.outcome(intent.client_order_id)
            if oc == "UNCONFIRMED":
                act = "LIVE-UNCONFIRMED"
                JOURNAL.log("AUTOPILOT_UNCONFIRMED",
                            {"coid": intent.client_order_id, "symbol": inst.key,
                             "venue": venue})
            else:
                act = "LIVE-FILLED" if fill else "LIVE-REJECT"
            if fill and not close:
                # spot has no position object to close — the sell IS the exit — so
                # the acknowledgement is checked against the balance instead.
                time.sleep(1.0)
                syms = [str(p.get("symbol", "")).upper()
                        for p in (_BN.BINANCE.positions() or [])]
                if fill.symbol.upper() not in syms:
                    act = "LIVE-UNVERIFIED"
                    JOURNAL.log("AUTOPILOT_UNVERIFIED",
                                {"coid": intent.client_order_id,
                                 "symbol": fill.symbol})
        elif is_psx and live:
            placed = self._psx_auto_place(inst, side, qty, price, intent, close)
            if placed:
                return placed
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

    def _psx_auto_place(self, inst, side: str, qty: float, price: float,
                        intent: OrderIntent, close: bool) -> tuple[str, str] | None:
        """The opt-in real-money PSX path. Returns (action, detail), or None to
        fall back to a human-confirmed ticket.

        Three separate deliberate acts must all be turned:
          autopilot.auto_place  +  js_global.allow_orders  +  the broker PIN.
        Anything unusual — a broker refusal, an unread answer, a timeout —
        degrades to a ticket instead of retrying into a duplicate order.
        """
        from . import jsgtrade
        cfg = CFG.section("autopilot") or {}
        if not cfg.get("auto_place") or not jsgtrade.enabled():
            return None
        if getattr(self, "_psx_down", False):
            self._act(inst.key, "TICKET-ONLY", "JS Global portal unreachable this cycle")
            return None
        if side == "BUY":
            book = getattr(self, "_psx_live", None) or {}
            cash = float(book.get("cash") or 0)
            need = qty * price * 1.005            # + commission headroom
            if cash <= 0 or need > cash:
                JOURNAL.log("AUTOPILOT_NO_CASH", {"symbol": inst.key,
                                                  "need": round(need, 2),
                                                  "cash": round(cash, 2)})
                self._act(inst.key, "GATED", f"needs Rs {need:,.0f}, free cash "
                                             f"Rs {cash:,.0f} — ticket only")
                return None
        else:
            # a PKR cash account cannot sell what it does not own: an opening
            # SELL is a short sale (blocked), a close may not exceed the
            # settled quantity net of orders already resting at the broker.
            book = getattr(self, "_psx_live", None) or {}
            rows = [h for h in book.get("holdings", [])
                    if h["symbol"] == inst.key.upper()]
            free = sum(h["qty"] - h["pending_sell"] for h in rows)
            if not close:
                self._act(inst.key, "GATED", "PSX cash account cannot short")
                return None
            if free < qty:
                self._act(inst.key, "GATED", f"only {free:.0f} {inst.key} sellable "
                                             f"(asked {qty:.0f})")
                return None
        try:
            out = jsgtrade.place_now(inst.key, side, qty, price, "LIMIT",
                                     note="autopilot" + ("-close" if close else ""))
        except Exception as exc:
            JOURNAL.log("AUTOPILOT_PLACE_FAIL", {"symbol": inst.key,
                                                 "error": str(exc)[:200]})
            self._act(inst.key, "PLACE-FAIL", str(exc)[:160])
            return None
        if not out.get("ok"):
            self._act(inst.key, "PLACE-REJECT", (out.get("response") or "")[:160])
            JOURNAL.log("AUTOPILOT_PLACE_REJECT",
                        {"symbol": inst.key, "resp": (out.get("response") or "")[:200]})
            return None
        detail = f"order {out.get('order_no') or 'accepted'} at {out.get('account')}"
        self.st["actions"].insert(0, {"ts": _now(), "symbol": inst.key,
                                      "action": "PLACED", "side": side,
                                      "qty": qty, "price": price, "why": detail})
        self.st["actions"] = self.st["actions"][:MAX_ACTIONS_KEPT]
        JOURNAL.log("AUTOPILOT_PLACED", {"coid": intent.client_order_id,
                                         "symbol": inst.key, "side": side,
                                         "qty": qty, "price": price,
                                         "order_no": out.get("order_no"),
                                         "account": out.get("account"),
                                         "close": close})
        return "PLACED", detail

    def _mt5_close_one(self, symbol: str) -> None:
        """Close the terminal-side leg of a position the autopilot just sold.

        A swallowed failure here used to leave the terminal holding a position the
        book had already booked as closed — an open trade with nobody managing its
        stop, and no trace of it in the journal. So a close that does not go out is
        reported, and reported loudly enough that the desk shows it.
        """
        try:
            positions = self.mt5.positions()
        except Exception as exc:
            positions = None
            JOURNAL.log("AUTOPILOT_CLOSE_READ_FAIL", {"symbol": symbol,
                                                      "error": str(exc)[:200]})
        if positions is None:
            self._act(symbol, "CLOSE-UNVERIFIED",
                      f"terminal position list unreadable for {symbol}")
            return
        for p in positions:
            if p.get("symbol") != symbol:
                continue
            try:
                out = self.mt5.close_position(int(p["ticket"]))
            except Exception as exc:
                out = None
                JOURNAL.log("AUTOPILOT_CLOSE_FAIL", {"symbol": symbol,
                                                     "ticket": p.get("ticket"),
                                                     "error": str(exc)[:200]})
            if not out:
                self._act(symbol, "CLOSE-FAILED",
                          f"terminal close for {symbol} ticket {p.get('ticket')} "
                          f"did not come back DONE — the position may still be live")

    # ------------------------------------------------------------- reporting
    def status(self) -> dict[str, Any]:
        from . import binance as _BN
        broker = PaperBroker()
        acct = broker.account()
        pos = broker.open_positions()
        posture = getattr(self, "_posture", None) or self._risk_posture(acct, pos)
        return {"ok": True, **{k: self.st[k] for k in
                ("running", "mode", "armed_live", "symbols", "interval_s",
                 "entry_threshold", "max_trades_per_cycle", "horizon_days",
                 "cycles", "last_cycle", "last_cycle_s", "last_error",
                 "feedback_error", "state_saved", "state_error",
                 "signals", "actions",
                 "held", "consec_losses", "psx_day_start", "macro",
                 "news_desk")},
                "risk_posture": posture,
                "kill_switch": KILL.engaged(),
                "mt5": {"connected": self.mt5.connected,
                        "error": self.mt5.last_error},
                "psx_live": {"keys_on": _psx_live_ready(),
                             "book_this_cycle": self._psx_live is not None},
                "binance": _BN.BINANCE.state(),
                "live_account": self._live_acct,
                "paper_account": acct,
                "paper_positions": pos}


def _read_json(path) -> dict[str, Any]:
    """A missing file is a first run. An unreadable one is a fault, and raises.

    This used to swallow every exception and return `{}`, which is the shape of
    "no state yet": a state file truncated by a full disk or a power cut sent the
    loop back to full capital, no positions and a reset equity peak, and the desk
    read that as the same healthy account it had reported an hour earlier.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"state file held a {type(data).__name__}, not an object")
    return data


AUTOPILOT = Autopilot()
