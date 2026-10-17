"""KAIROS trading doctrine — the great rules, encoded as parameters.

Ported and upgraded from JARVIS v1 `trading_rules.py` during the single-
project merge. The classics it encodes:

  1. Cut losses short, let winners run          7. Risk:Reward minimum 1:2
  2. Never risk more than 1-2% per trade        8. Don't fight the Fed / macro
  3. Trend is your friend                       9. Diversification matters
  4. Buy on fear, sell on greed                10. Size = f(conviction, vol)
  5. Volume precedes price                     11. Irrationality outlasts you
  6. Support/resistance are real               12. Your edge is discipline

Every value is overridable: config.yaml `rules:` section, or a
data/rules.json file at runtime. Nothing here is a profit guarantee —
it is the discipline layer that keeps an autonomous system alive.
"""

from __future__ import annotations

import json
from typing import Any

from .config import CFG

DEFAULT_RULES: dict[str, Any] = {
    "max_risk_per_trade_pct": 2.0,       # % of equity risked per trade
    "min_risk_reward_ratio": 2.0,        # minimum reward : risk
    "max_position_pct": 15.0,            # max % of equity in one name
    "max_open_positions": 8,             # simultaneous holdings
    "stop_loss_atr_multiplier": 2.0,     # stop = entry - 2*ATR (long)
    "take_profit_atr_multiplier": 6.0,   # target = entry + 6*ATR (long)
    "trailing_stop_atr_multiplier": 3.0, # trail once 3*ATR in profit
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "volume_min_ratio": 1.2,             # volume vs 20-day average
    "trend_sma_period": 50,
    "require_above_sma50": True,         # longs only above trend
    "max_correlation": 0.7,              # portfolio correlation cap
    "min_market_cap_pkr": 5_000_000_000, # PSX: skip micro caps
    "min_avg_volume": 50_000,            # PSX: liquidity floor
    "earnings_blackout_days": 3,         # stand aside around prints
    "max_hold_days": 60,
    "min_confidence": 50,                # engine confidence floor (%)
    "fee_pct": 0.15,                     # PSX brokerage
    "tax_pct": 0.10,                     # capital gains
    # ---- institutional overlays (ultimate form) ----
    "max_daily_loss_pct": 3.0,           # breaker: stop opening after -3% day
    "max_total_exposure_pct": 60.0,      # breaker: gross exposure cap
    "max_consecutive_losses": 4,         # breaker: pause entries after 4 red
    "min_model_agreement": 0.55,         # entry needs 55% of models agreeing
    "max_spread_pct": 0.60,              # live MT5 entry spread guard
    "trail_after_r_multiples": 1,        # start ratcheting stop at +1R
}

# ---------------------------------------------------------------- wisdom corpus
WYCKOFF_PHASES = {
    "A": "Stopping action — smart money absorbs selling",
    "B": "Cause building — accumulation/distribution",
    "C": "Test — spring/upthrust shakeout",
    "D": "Mark-up / mark-down begins",
    "E": "Public participation — trend accelerates",
    "F": "Distribution / re-accumulation complete",
}

DOW_THEORY = {
    "primary_trend": "1+ year, the main direction",
    "secondary_trend": "3 weeks - 3 months, corrective",
    "minor_trend": "Day to weeks, noise",
}

CANSLIM = {
    "C": "Current quarterly earnings +25% YoY",
    "A": "Annual earnings +25% over 3-5 years",
    "N": "New product / management / price breakout",
    "S": "Supply & demand: low float, high demand",
    "L": "Leader outperforming its sector",
    "I": "Institutional sponsorship rising",
    "M": "Market direction is up",
}

BUFFETT_FILTERS = {
    "1": "Do I understand this business?",
    "2": "Durable competitive advantage (moat)?",
    "3": "Management honest and competent?",
    "4": "Price attractive vs intrinsic value?",
}

DALIO_PRINCIPLES = {
    "1": "Truth + radical transparency = best decisions",
    "2": "Pain + reflection = progress",
    "3": "Diversification is the holy grail",
    "4": "Know your beta vs your alpha",
    "5": "All-weather beats any single bet",
}

LYNCH_CATEGORIES = {
    "slow_growers": "Mature, low growth, high dividend",
    "stalwarts": "Large, steady 10-12% growth",
    "fast_growers": "20%+ growth, expanding fast",
    "cyclicals": "Boom-bust tied to the economy",
    "turnarounds": "From trouble to recovery",
    "asset_play": "Hidden value in the assets",
}


# ------------------------------------------------------------------ accessors
def _runtime_file():
    return CFG.data_dir / "rules.json"


def rules() -> dict[str, Any]:
    """Effective doctrine: defaults < config.yaml rules: < data/rules.json."""
    merged = dict(DEFAULT_RULES)
    try:
        merged.update(CFG.section("rules") or {})
    except Exception:
        pass
    p = _runtime_file()
    if p.exists():
        try:
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return merged


def get_rule(name: str, default: Any = None) -> Any:
    return rules().get(name, DEFAULT_RULES.get(name, default))


def save_overrides(overrides: dict[str, Any]) -> None:
    p = _runtime_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(overrides)
    p.write_text(json.dumps(cur, indent=2), encoding="utf-8")


# ------------------------------------------------------------------- sizing
def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Full Kelly; callers should quarter it (fractional Kelly discipline)."""
    if win_loss_ratio <= 0:
        return 0.0
    f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    return max(0.0, f)


def position_size(equity: float, entry: float, stop: float,
                  risk_pct: float | None = None) -> dict[str, Any]:
    """Risk-unit sizing: qty = (equity x risk%) / |entry - stop|, capped at
    max_position_pct of equity so a tight stop can't blow past the name cap."""
    r = rules()
    rp = (risk_pct if risk_pct is not None
          else float(r["max_risk_per_trade_pct"])) / 100.0
    risk_per_unit = abs(entry - stop)
    if entry <= 0 or risk_per_unit <= 0:
        return {"qty": 0.0, "error": "stop too close to entry"}
    qty = (equity * rp) / risk_per_unit
    cap_qty = (equity * float(r["max_position_pct"]) / 100.0) / entry
    qty = min(qty, cap_qty)
    return {"qty": round(qty, 4),
            "risk_amount": round(risk_per_unit * qty, 2),
            "position_value": round(qty * entry, 2),
            "position_pct": round(qty * entry / equity * 100, 2)}


# -------------------------------------------------------------- pre-trade plan
def check_plan(entry: float, stop: float | None, target: float | None,
               equity: float, confidence: float | None = None,
               signal_dir: int = 0, qty: float | None = None) -> dict[str, Any]:
    """Validate a concrete trade plan against the doctrine.

    Returns {passed, checks:[...]}. A missing stop/target skips that check
    rather than failing — the signal horizon bands usually supply them."""
    r = rules()
    checks: list[dict[str, Any]] = []
    passed = True

    def add(rule: str, value: str, required: str, ok: bool) -> None:
        nonlocal passed
        checks.append({"rule": rule, "value": value, "required": required,
                       "passed": ok})
        if not ok:
            passed = False

    if stop is not None and target is not None and entry > 0:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0.0
        add("Risk:Reward", f"1:{rr:.2f}",
            f">= 1:{r['min_risk_reward_ratio']}",
            rr >= float(r["min_risk_reward_ratio"]))
    if stop is not None and entry > 0 and equity > 0 and qty:
        # account risk of THIS sized order, not raw stop distance
        risk_pct = abs(entry - stop) * float(qty) / equity * 100
        add("Risk/trade", f"{risk_pct:.2f}%",
            f"<= {r['max_risk_per_trade_pct']}%",
            risk_pct <= float(r["max_risk_per_trade_pct"]))
    if confidence is not None:
        add("Confidence", f"{confidence * 100:.0f}%",
            f">= {r['min_confidence']}%",
            confidence * 100 >= float(r["min_confidence"]))
    if stop is not None and signal_dir:
        aligned = (signal_dir > 0 and stop < entry) or \
                  (signal_dir < 0 and stop > entry)
        add("Trend alignment", "yes" if aligned else "stop on wrong side",
            "stop beyond the trend", aligned)
    return {"passed": passed, "checks": checks}


def describe() -> str:
    """One-paragraph doctrine summary for chat/report use."""
    r = rules()
    return (f"Doctrine: max {r['max_risk_per_trade_pct']}% equity risk per trade, "
            f"minimum reward:risk 1:{r['min_risk_reward_ratio']}, "
            f"positions capped at {r['max_position_pct']}% of equity and "
            f"{r['max_open_positions']} simultaneous names, "
            f"stops {r['stop_loss_atr_multiplier']}x ATR, targets "
            f"{r['take_profit_atr_multiplier']}x ATR, "
            f"minimum engine confidence {r['min_confidence']}%, "
            f"max hold {r['max_hold_days']} days, earnings blackout "
            f"{r['earnings_blackout_days']} days.")
