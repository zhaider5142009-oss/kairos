"""FUGU conductor — the multi-expert command plane (task 22).

Named after the fugu chef's doctrine: serve nothing that has not passed every
safety cut, and portion strictly by demonstrated skill. The conductor fuses
the full expert stack — ensemble vote, model agreement, regime, realized
calibration from the feedback journal, doctrine R:R geometry and the current
risk posture — into ONE verdict carrying a probability-weighted edge in
R-multiples. Nothing reaches the risk gate unless it is expected to make
money after costs, and the entry bar tightens itself when realized accuracy
slips and loosens only when the system has earned it.

Read-only w.r.t. execution: it advises; gates and venues still dispose.
"""

from __future__ import annotations

import time
from typing import Any

from . import feedback
from .rules import check_plan, get_rule

_board_cache: tuple[float, list[dict[str, Any]]] | None = None
_BOARD_TTL = 120.0


def _dir_num(direction: Any) -> int:
    if isinstance(direction, (int, float)):
        return 1 if direction > 0 else -1 if direction < 0 else 0
    d = (str(direction) or "").upper()
    return 1 if d.startswith("L") or "BUY" in d else -1 if d.startswith("S") else 0


def _agreement(res: dict[str, Any]) -> float:
    models = res.get("models") or []
    if not models:
        return 0.0
    want = res["signal"]["direction"]
    same = sum(1 for m in models if _dir_num(m.get("direction")) == _dir_num(want))
    return same / len(models)


def _calibration() -> dict[str, Any]:
    tr = feedback.track_record() or {}
    return {"hit_rate": tr.get("hit_rate"),
            "strong_hit_rate": tr.get("strong_signal_hit_rate"),
            "n_resolved": tr.get("resolved", 0)}


def _adaptive_threshold(base: float, cal: dict[str, Any]) -> float:
    """Self-tuning entry bar: below-break-even realized accuracy raises the
    bar; proven accuracy lowers it gently — never under 0.40."""
    hr = cal.get("hit_rate")
    if not hr or cal.get("n_resolved", 0) < 10:
        return base
    if hr < 0.55:
        return min(0.85, base * (1.0 + (0.55 - hr) * 1.6))
    if hr > 0.62:
        return max(0.40, base * (1.0 - (hr - 0.62) * 0.8))
    return base


def conduct(symbol: str, res: dict[str, Any] | None = None,
            posture: dict[str, Any] | None = None) -> dict[str, Any]:
    """One fused verdict for one instrument."""
    if res is None:
        from .engine import ENGINE
        res = ENGINE.predict(symbol)
    sig = res["signal"]
    price = float(res["price"])
    direction = sig["direction"]
    score = float(sig["score"])
    conf = float(sig["confidence"] or 0.0)
    agr = _agreement(res)
    cal = _calibration()
    regime = str((res.get("regime") or {}).get("regime", "")).lower()

    # R-multiple geometry from the engine's own bands/targets
    bands = res.get("confidence_band") or {}
    tgt = (res.get("targets") or {}).get(f"{sig['horizon_days']}d") or bands.get(
        "p90" if _dir_num(direction) > 0 else "p10")
    stop = bands.get("p10") if _dir_num(direction) > 0 else bands.get("p90")
    rr = ev_gain = ev_loss = 0.0
    if tgt and stop and 0 < abs(tgt - price) and 0 < abs(price - stop):
        rr = abs(tgt - price) / abs(price - stop)
        ev_gain, ev_loss = rr, 1.0

    # probability of profit: honest, conservative, calibration-anchored
    anchor = (cal["hit_rate"] or 0.5) if score != 0 else 0.5
    if cal.get("strong_hit_rate") and abs(score) > 0.3:
        anchor = 0.5 * anchor + 0.5 * cal["strong_hit_rate"]
    p_hit = max(0.30, min(0.78, anchor + 0.32 * abs(score) * agr - (0.0 if agr else 0.1)))

    regime_riskoff = any(k in regime for k in ("risk-off", "bear", "crisis", "high-vol"))
    posture_bad = bool(posture and posture.get("block_entries"))
    edge_r = (p_hit * ev_gain - (1 - p_hit) * ev_loss) if ev_gain else 0.0

    thr_base = float(get_rule("fugu_conviction_floor", 0.50))
    thr = _adaptive_threshold(thr_base, cal)
    conviction = min(1.0, conf * (0.45 + 0.55 * agr))
    min_rr = float(get_rule("min_risk_reward_ratio"))

    reasons: list[str] = []
    if posture_bad:
        reasons.append("risk posture: entries blocked")
    if regime_riskoff and _dir_num(direction) > 0:
        reasons.append(f"regime {regime} vs long")
    if agr < float(get_rule("min_model_agreement")):
        reasons.append(f"model agreement {agr:.0%} below floor")
    if conviction < thr:
        reasons.append(f"conviction {conviction:.2f} < adaptive bar {thr:.2f}")
    if edge_r < 0.15:
        reasons.append(f"expected edge {edge_r:.2f}R below 0.15R")
    if rr < min_rr:
        reasons.append(f"R:R {rr:.1f} below doctrine floor {min_rr:.1f}")
    if _dir_num(direction) == 0:
        reasons.append("no directional call")

    action = "EXECUTE" if not reasons else "PASS"
    plan = None
    if action == "EXECUTE":
        plan = check_plan(entry=price, stop=float(stop), target=float(tgt),
                          equity=float((posture or {}).get("equity") or 1_000_000),
                          confidence=conf, signal_dir=_dir_num(direction))
        if not plan["passed"]:
            action = "PASS"
            reasons.append(f"plan check: {plan['failures']}")

    return {
        "symbol": symbol.upper(), "asof": time.strftime("%H:%M:%S"),
        "action": action, "direction": direction, "side": _dir_num(direction),
        "price": price, "score": score, "confidence": conf,
        "agreement": round(agr, 3), "regime": regime,
        "conviction": round(conviction, 3), "threshold": round(thr, 3),
        "p_hit": round(p_hit, 3), "edge_r": round(edge_r, 3), "rr": round(rr, 2),
        "entry": price, "stop": round(float(stop), 4) if stop else None,
        "target": round(float(tgt), 4) if tgt else None,
        "calibration": cal, "posture_ok": not posture_bad,
        "reasons": reasons or ["all experts concur"],
    }


def board(symbols: list[str] | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Conductor board over the active watchlist (cached 2 min)."""
    global _board_cache
    if not symbols:
        from .autopilot import AUTOPILOT
        symbols = list((AUTOPILOT.status().get("symbols") or [])[:10]) or \
            ["GOLD", "WTI", "EURUSD", "BTC", "SPX"]
    if _board_cache and not force and time.time() - _board_cache[0] < _BOARD_TTL \
            and [r["symbol"] for r in _board_cache[1]] == [s.upper() for s in symbols]:
        return _board_cache[1]
    from .autopilot import AUTOPILOT
    posture = (AUTOPILOT.status() or {}).get("risk_posture")
    rows = [conduct(s, posture=posture) for s in symbols]
    rows.sort(key=lambda r: -r["edge_r"])
    _board_cache = (time.time(), rows)
    return rows
