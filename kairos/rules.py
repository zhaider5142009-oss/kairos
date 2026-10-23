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
import math
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

# The keys above that no code reads, each with the reason it cannot be read yet.
#
# This list exists because `describe()` used to print every key as though it
# governed the book, and the CLI doctrine panel, the command plane and the LLM
# council prompt all quote `describe()` — so an operator (and a model) could read
# "stops 2.0x ATR" with confidence while the autopilot took its stop from a
# percentile band and nothing anywhere looked at the multiplier. A limit that
# binds nothing is not a limit; it is a description of a risk control the reader
# believes they have. `tests/test_doctrine_bound.py` fails if a declared key ends
# up in neither this dict nor `ENFORCED`.
NOT_ENFORCED: dict[str, str] = {
    "stop_loss_atr_multiplier":
        "the entry stop comes from the prediction's 10th/90th percentile band, "
        "not an ATR multiple; no code reads this key",
    "take_profit_atr_multiplier":
        "targets come from the horizon return bands in `res['targets']`",
    "trailing_stop_atr_multiplier":
        "the ratchet advances in R multiples (`trail_after_r_multiples`), not ATR",
    "volume_min_ratio":
        "a signal-confirmation shape, exercised by the volume-breakout "
        "strategies, not by the entry gate",
    "min_market_cap_pkr":
        "no fundamentals source: `universe.Instrument` carries no market cap, so "
        "the floor cannot be evaluated for any name",
    "max_correlation":
        "a portfolio cap needs a return matrix across every held name plus the "
        "candidate at gate time; the cycle would pay that in history reads per "
        "candidate, so it is wired next as a cached cross-section rather than "
        "pretended to here",
    "earnings_blackout_days":
        "`calendar.py` holds macro event windows only (no company print dates), "
        "so a per-symbol blackout cannot be computed",
    "fee_pct": "costs are taken per venue from `execution.commission_bps`",
    "tax_pct": "no tax model in the paper book or the walk-forward",
}

# Where each enforced key is actually read, for `audit()`. The test checks these
# claims against the source, so an entry here cannot outlive the code it names.
ENFORCED: dict[str, tuple[str, ...]] = {
    "max_risk_per_trade_pct": ("rules.py",),
    "min_risk_reward_ratio": ("rules.py", "fugu.py"),
    "max_position_pct": ("rules.py",),
    "min_confidence": ("rules.py",),
    "max_open_positions": ("rules.py",),
    "require_above_sma50": ("rules.py",),
    "trend_sma_period": ("engine.py", "rules.py"),
    "rsi_oversold": ("rules.py",),
    "rsi_overbought": ("rules.py",),
    "min_avg_volume": ("rules.py",),
    "max_hold_days": ("rules.py",),
    "max_daily_loss_pct": ("rules.py",),
    "max_total_exposure_pct": ("autopilot.py",),
    "max_consecutive_losses": ("autopilot.py",),
    "min_model_agreement": ("autopilot.py", "fugu.py"),
    "max_spread_pct": ("autopilot.py",),
    "trail_after_r_multiples": ("autopilot.py",),
    "max_equity_drawdown_pct": ("autopilot.py",),
    "psx_min_price": ("autopilot.py",),
    # read once, by `adv_share_floor`, which is what the autopilot now calls
    "psx_min_adv_shares": ("rules.py",),
    "fugu_conviction_floor": ("fugu.py",),
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


def daily_loss_fraction(risk_cfg: dict[str, Any] | None = None,
                        rules_cfg: dict[str, Any] | None = None) -> tuple[float, str]:
    """The daily loss brake as a fraction of equity, and which knob governs.

    One instruction has two names in this project: `rules.max_daily_loss_pct`
    is a percentage and used to stop the autopilot from opening anything after a
    bad day, while `risk.daily_loss_limit_pct` is a fraction and is what the
    pre-trade gate refuses an order on. Two brakes on one number is how an
    operator ends up believing they tightened a limit they did not touch, so
    the tighter reading governs at both places and the answer names the key that
    produced it — a refusal can then say which brake actually fired.

    A reading that is not a positive finite number is a limit of zero. An
    unreadable risk figure has to stop the next order, not disappear into a
    default; and the two units differ by a factor of one hundred, so a value
    pasted from one key into the other lands on the conservative side.
    """
    rc = CFG.risk if risk_cfg is None else risk_cfg
    rl = rules() if rules_cfg is None else rules_cfg
    readings: list[tuple[float, str]] = []
    for name, raw, as_percent in (
            ("rules.max_daily_loss_pct", rl.get("max_daily_loss_pct",
                                                DEFAULT_RULES["max_daily_loss_pct"]), True),
            ("risk.daily_loss_limit_pct", rc.get("daily_loss_limit_pct",
                                                 DEFAULT_RULES["max_daily_loss_pct"] / 100.0),
             False)):
        try:
            value = float(raw) / (100.0 if as_percent else 1.0)
        except (TypeError, ValueError):
            return 0.0, f"{name} is not a readable limit"
        if not math.isfinite(value) or value <= 0:
            return 0.0, f"{name} is not a usable limit"
        readings.append((value, name))
    return min(readings)


def adv_share_floor(r: dict[str, Any] | None = None) -> tuple[float, str]:
    """The PSX liquidity floor as one number, and which key governs it.

    `rules.min_avg_volume` ("PSX: liquidity floor", 50,000 shares) and
    `rules.psx_min_adv_shares` ("20d avg volume floor", 50,000 shares) are the
    same instruction written twice. Only one of them was ever read, so an
    operator who tightened the other watched nothing change — the two-names-one-
    brake shape `daily_loss_fraction` documents, resolved the same way: the
    tighter reading governs at both places and the answer names the key, so a
    rejection can say which knob the operator is actually holding.

    An unreadable floor is `math.inf`, not zero: a share floor that cannot be
    parsed screens every name out rather than letting all of them through.
    """
    rl = rules() if r is None else r
    readings: list[tuple[float, str]] = []
    for name in ("psx_min_adv_shares", "min_avg_volume"):
        if name not in rl:
            continue
        try:
            value = float(rl[name])
        except (TypeError, ValueError):
            return math.inf, f"{name} is not a readable share floor"
        if not math.isfinite(value) or value < 0:
            return math.inf, f"{name} is not a usable share floor"
        readings.append((value, name))
    if not readings:
        return math.inf, "no share floor configured"
    return min(readings)


def open_slot(n_held: int | None,
              r: dict[str, Any] | None = None) -> dict[str, Any]:
    """May this book take one more name? `max_open_positions`, enforced.

    An unreadable count refuses the entry. "How many positions do I hold" and
    "am I at the cap" are the same question, and a gate that answers the second
    one by assuming zero when the first one failed is a cap that only binds when
    the venue is cooperative — the same asymmetry this file refuses on the loss
    brake and `_psx_liquid` refuses on history.

    A name already held does not consume a new slot, so callers ask this only
    when they are about to open something they do not already own.
    """
    rl = rules() if r is None else r
    raw = rl.get("max_open_positions", DEFAULT_RULES["max_open_positions"])
    try:
        cap = int(float(raw))
    except (TypeError, ValueError):
        return {"allowed": False, "cap": None, "held": n_held,
                "why": f"max_open_positions is not a readable count ({raw!r})"}
    if cap <= 0:
        return {"allowed": False, "cap": cap, "held": n_held,
                "why": f"max_open_positions is {cap} — no new name is allowed"}
    if n_held is None:
        return {"allowed": False, "cap": cap, "held": None,
                "why": "the book could not be read, so the name cap cannot be "
                       "checked — refused, not assumed to fit"}
    if n_held >= cap:
        return {"allowed": False, "cap": cap, "held": n_held,
                "why": f"holding {n_held} of {cap} names — the cap is full"}
    return {"allowed": True, "cap": cap, "held": n_held, "why": ""}


def maturity_days(horizon_days: float,
                  r: dict[str, Any] | None = None) -> tuple[float, str]:
    """When this entry is due to be closed, on the tighter of the two brakes.

    The autopilot matures a position on its own `horizon_days` (default 5) while
    the doctrine declares `max_hold_days` (default 60) and nothing read the
    second one. The horizon is the useful number and the cap is the safety net,
    so the smaller governs — and if the safety net is unreadable, the horizon
    still binds: a position that cannot see its own cap keeps the only exit
    timer it has rather than inheriting an infinite hold.
    """
    rl = rules() if r is None else r
    try:
        horizon = float(horizon_days)
    except (TypeError, ValueError):
        horizon = 5.0
    if not math.isfinite(horizon) or horizon <= 0:
        horizon = 5.0
    raw = rl.get("max_hold_days", DEFAULT_RULES["max_hold_days"])
    try:
        cap = float(raw)
    except (TypeError, ValueError):
        return horizon, f"max_hold_days is unreadable ({raw!r}); horizon governs"
    if not math.isfinite(cap) or cap <= 0:
        return horizon, f"max_hold_days is not usable ({cap}); horizon governs"
    if cap < horizon:
        return cap, "rules.max_hold_days"
    return horizon, "autopilot.horizon_days"


def _trend_rows(r: dict[str, Any], signal_dir: int, entry: float,
               sma: float | None, rsi14: float | None,
               blind: list[str]) -> list[dict[str, Any]]:
    """The doctrine's entry filters: trend for longs, heat for both sides.

    These are rules 3 and 4 of the file header — "trend is your friend", "buy on
    fear, sell on greed" — declared for eighteen months as `require_above_sma50`,
    `trend_sma_period`, `rsi_oversold` and `rsi_overbought`, and read by nothing.

    A filter whose number is missing refuses the entry, the same policy
    `_psx_liquid` states for history: a screen that cannot see is not a screen
    that passed. The short side is symmetric only on the RSI heat check, because
    the doctrine only ever declared the trend filter for longs; `describe()` says
    so rather than implying a general trend rule the code does not have.
    """
    rows: list[dict[str, Any]] = []
    if signal_dir > 0 and bool(r.get("require_above_sma50", True)):
        need = float(r.get("trend_sma_period", DEFAULT_RULES["trend_sma_period"]))
        if sma is None or not math.isfinite(sma) or entry <= 0:
            blind.append("Trend")
            rows.append({"rule": "Trend", "value": "no readable SMA",
                         "required": f"close above {need:.0f}-bar mean",
                         "passed": False})
        else:
            rows.append({"rule": "Trend", "value": f"{entry:,.2f} vs {sma:,.2f}",
                         "required": f"close above {need:.0f}-bar mean",
                         "passed": entry >= sma})
    over = float(r.get("rsi_overbought", DEFAULT_RULES["rsi_overbought"]))
    under = float(r.get("rsi_oversold", DEFAULT_RULES["rsi_oversold"]))
    if signal_dir and (over < 100 or under > 0):
        if rsi14 is None or not math.isfinite(rsi14):
            blind.append("Heat")
            rows.append({"rule": "Heat", "value": "no readable RSI",
                         "required": f"not {over:.0f}+ for a long, {under:.0f}- "
                                     "for a short",
                         "passed": False})
        elif signal_dir > 0 and rsi14 > over:
            rows.append({"rule": "Heat", "value": f"RSI {rsi14:.0f}",
                         "required": f"<= {over:.0f} to buy", "passed": False})
        elif signal_dir < 0 and rsi14 < under:
            rows.append({"rule": "Heat", "value": f"RSI {rsi14:.0f}",
                         "required": f">= {under:.0f} to sell short",
                         "passed": False})
        else:
            rows.append({"rule": "Heat", "value": f"RSI {rsi14:.0f}",
                         "required": f"<= {over:.0f} / >= {under:.0f}",
                         "passed": True})
    return rows


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


class _NoBook:
    """The caller has no view of the book at all — distinct from a book that
    could not be read. `check_plan` refuses on the second and abstains on the
    first: an advisor asked to price a single name has never held the position
    list, so answering "cap cannot be checked, entry refused" would not be a
    safety rule, it would be a control that only ever passes callers that lie
    about having a book."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<no book view>"


NO_BOOK = _NoBook()


# -------------------------------------------------------------- pre-trade plan
def check_plan(entry: float, stop: float | None, target: float | None,
               equity: float, confidence: float | None = None,
               signal_dir: int = 0, qty: float | None = None, *,
               n_held: Any = NO_BOOK, sma: float | None = None,
               rsi14: float | None = None) -> dict[str, Any]:
    """Validate a concrete trade plan against the doctrine.

    Returns {passed, checks, failures, blind}. `blind` names the rules that could
    not be evaluated for this order, so a "passed" plan reads as the distance it
    actually travelled rather than as a clean bill from controls that never ran.

    An opening order with no stop fails. That used to be a skip: the band
    percentile was treated as decoration, so a prediction whose 10th percentile
    was missing reached the venue with `stop_loss=None`, `check_plan` reported
    nothing, and `_manage_exit` — which reads `if stop and …` — had no hard stop
    to test and no `stop0` for the ratchet to start from, leaving only a signal
    flip or a maturity date to close the position. Rule 1 of this file's header is
    "cut losses short", and the only way to cut a loss without a stop is to
    discover the loss first.

    A missing *target* stays a skip, because the reward leg genuinely can be
    unknowable while the risk leg must never be; it is reported in `blind`. The
    name cap is a skip too for a caller that never has a book view (`NO_BOOK`,
    the default) and a refusal for one whose book failed to read (`None`) — see
    `_NoBook`.
    """
    r = rules()
    checks: list[dict[str, Any]] = []
    blind: list[str] = []
    passed = True

    def add(rule: str, value: str, required: str, ok: bool) -> None:
        nonlocal passed
        checks.append({"rule": rule, "value": value, "required": required,
                       "passed": ok})
        if not ok:
            passed = False

    if signal_dir:
        add("Stop defined", "none" if stop is None else f"{stop:,.2f}",
            "an opening order must carry a stop", stop is not None)
    if stop is not None and target is not None and entry > 0:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0.0
        add("Risk:Reward", f"1:{rr:.2f}",
            f">= 1:{r['min_risk_reward_ratio']}",
            rr >= float(r["min_risk_reward_ratio"]))
    elif signal_dir:
        blind.append("Risk:Reward")
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
    if n_held is NO_BOOK:
        blind.append("Name cap")
    else:
        slot = open_slot(n_held, r)
        # The cap is reported whenever a count was supplied, and always when it
        # refuses: a caller that has no idea how many names it holds is exactly
        # the caller the cap exists to stop.
        if n_held is not None or not slot["allowed"]:
            add("Name cap",
                f"{slot['held'] if slot['held'] is not None else '?'} of "
                f"{slot['cap'] if slot['cap'] is not None else '?'}",
                f"under {slot['cap']} simultaneous names" if slot["cap"]
                else "a readable max_open_positions", slot["allowed"])
    # Through `add`, not `checks.extend`: the verdict is accumulated inside add,
    # so a filter appended afterwards printed its refusal and still passed the
    # order. The first run of these gates failed exactly that test.
    for row in _trend_rows(r, signal_dir, entry, sma, rsi14, blind):
        add(row["rule"], row["value"], row["required"], row["passed"])
    failures = "; ".join(f"{c['rule']} {c['value']} (needs {c['required']})"
                         for c in checks if not c["passed"])
    return {"passed": passed, "checks": checks, "failures": failures,
            "blind": blind}


def audit() -> dict[str, Any]:
    """Every declared rule, and whether anything reads it.

    A doctrine file that lists limits the code ignores is worse than no doctrine
    file, because the reader substitutes it for having checked. This reports the
    split, with the module each enforced key is read in, so an operator can see
    that `max_hold_days` binds while `min_market_cap_pkr` does not — and so that a
    new key cannot quietly join the silent middle group: the test suite requires
    every declared key to be classified here.
    """
    r = rules()
    declared = set(DEFAULT_RULES) | set(r)
    enforced = [{"key": k, "value": r.get(k), "read_in": ENFORCED[k]}
                for k in sorted(declared) if k in ENFORCED]
    inert = [{"key": k, "value": r.get(k), "why": NOT_ENFORCED[k]}
             for k in sorted(declared) if k in NOT_ENFORCED]
    unclassified = sorted(declared - set(ENFORCED) - set(NOT_ENFORCED))
    return {"enforced": enforced, "declared_not_enforced": inert,
            "unclassified": unclassified,
            "counts": {"declared": len(declared), "enforced": len(enforced),
                       "inert": len(inert), "unclassified": len(unclassified)}}


def describe() -> str:
    """One-paragraph doctrine summary for chat/report use.

    Every clause is a limit some code path applies. Until 2026-10-23 this
    paragraph claimed "stops 2.0x ATR, targets 6.0x ATR" while the autopilot took
    both from percentile bands, and printed a 60-day hold cap and an earnings
    blackout that nothing read — then handed the same paragraph to the LLM council
    as the system's own rules. A claim about *method* is the one a reader cannot
    check from a number, so it is the one this file may not get wrong.
    """
    r = rules()
    return (f"Doctrine: max {r['max_risk_per_trade_pct']}% equity risk per trade, "
            f"minimum reward:risk 1:{r['min_risk_reward_ratio']}, "
            f"positions capped at {r['max_position_pct']}% of equity and "
            f"{r['max_open_positions']} simultaneous names, every opening order "
            f"must carry a stop (from the prediction's 10th/90th percentile band, "
            f"reward leg from the horizon target), minimum engine confidence "
            f"{r['min_confidence']}%, breakers at {r['max_daily_loss_pct']}% day "
            f"loss, {r['max_total_exposure_pct']}% gross exposure and "
            f"{r['max_consecutive_losses']} consecutive losses. "
            + _filter_sentence(r))


def _filter_sentence(r: dict[str, Any]) -> str:
    """The entry filters the gate applies, plus the truth about the rest."""
    parts = []
    if r.get("require_above_sma50", True):
        parts.append("longs only at or above the "
                     f"{r['trend_sma_period']}-bar mean")
    parts.append(f"no long above RSI {r['rsi_overbought']}, no short below "
                 f"RSI {r['rsi_oversold']}")
    floor, src = adv_share_floor(r)
    parts.append("PSX names under "
                 + ("an unreadable share floor" if math.isinf(floor)
                    else f"{floor:,.0f} 20-day average shares")
                 + f" screened out ({src})")
    cap, cap_src = maturity_days(5.0, r)
    parts.append(f"holds mature at {cap:g} days ({cap_src})")
    return ("Applied: " + "; ".join(parts) + ". Declared and NOT enforced: "
            + ", ".join(sorted(NOT_ENFORCED))
            + " — `kairos rules` lists the reason for each.")
