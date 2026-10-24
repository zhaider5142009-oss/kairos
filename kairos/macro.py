"""Cross-asset macro axis: the market condition one instrument's own bars cannot see.

`features.detect_regime` reads the instrument's own series. A PSX blue chip in a
quiet uptrend is therefore "Bull-Quiet" whether the dollar is being squeezed
worldwide, PKR is sliding against it, or neither — and the attribution buckets
keyed on that label credit a strategy with skill it may only have had because
the macro backdrop happened to suit it, then arm the same strategies unchanged
when the backdrop flips. This module supplies the missing axis: seven
cross-asset prints, one bounded score, one bucket tag.

Three rules, the same shapes the rest of this codebase uses:

* **The tag is a function of what the walk-forward can also see.** Every
  component below is computed from the intermarket panel — the same nine daily
  series `ENGINE.intermarket()` already fetches for the strategy library, which
  `tools/strategy_attribution.py` already truncates by date. Live-only context
  (PSX breadth, news tone) is reported as `extras` and is deliberately *not* in
  the score: a bucket tag computed from inputs the measurement never had would
  make every weight fire against evidence that does not describe it.
* **An absent print is not a neutral print.** Each component answers or returns
  None, the composite is the weighted mean over the ones that answered, and
  `coverage` is how much of the panel actually spoke. Thin coverage is its own
  tag ("Unknown") rather than a mid-range score, because "the VIX and the dollar
  both disagree, so call it neutral" and "nothing answered" are different states
  of the world that the old single global multiplier could not tell apart.
* **The scales are judgment, and they are declared.** The divisors below are
  fixed constants, not fitted to this sample — no backtest was allowed to choose
  them, so nothing here is overfit by construction. What is measured, and
  reported per bucket, is whether arming strategies under each tag paid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Bucket tags, in vocabulary order. This string is part of an attribution key,
# so renaming a value invalidates the artifact — it must then be re-measured
# with tools/strategy_attribution.py, not patched around.
RISK_ON = "Risk-On"
NEUTRAL = "Neutral"
RISK_OFF = "Risk-Off"
UNKNOWN = "Unknown"
TAGS = (RISK_ON, NEUTRAL, RISK_OFF, UNKNOWN)

# |score| at or above this is called risk-on / risk-off. Deliberately not tiny:
# a tag that flips every third day produces buckets that are a calendar artefact
# rather than a condition, and the 100-strategy library is not deep enough in
# any one bucket to be judged on those.
TAG_BAND = 0.15

# Below this share of the panel weight, the state is UNKNOWN: no tag, no
# size multiplier above the cautious floor.
MIN_COVERAGE = 0.55
# Two prints out of seven is a coincidence, not a macro read.
MIN_COMPONENTS = 3

# Position sizing by macro state. Risk-Off cuts; UNKNOWN cuts too, because a
# system that cannot see the tape does not get the benefit of the doubt; Risk-On
# is allowed a modest lift, and only that — the lift is bounded so a heuristic
# score can never lever the book the way a measured edge in `attribution` can.
SIZE_MULTIPLIER = {RISK_ON: 1.10, NEUTRAL: 1.00, RISK_OFF: 0.75, UNKNOWN: 0.90}


@dataclass
class Component:
    """One cross-asset print, and whether it actually answered."""
    key: str
    what: str
    weight: float
    score: float | None
    value: float | None
    why: str = ""

    @property
    def present(self) -> bool:
        # bool(), not the bare expression: `np.isfinite()` answers with
        # numpy.bool_, and that leaks into JSON as the *string* "True" through
        # this codebase's default=str serializer — so a desk row would read
        # present="True" and an auditor could not filter on it.
        return self.score is not None and bool(np.isfinite(self.score))

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "what": self.what, "weight": round(self.weight, 3),
                "present": self.present,
                "score": None if self.score is None else _num(self.score, 3),
                "value": None if self.value is None else _num(self.value, 4),
                "why": self.why}


@dataclass
class MacroState:
    tag: str
    score: float
    coverage: float
    size_multiplier: float
    components: list[Component] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    as_of: str = ""

    @property
    def n_present(self) -> int:
        return sum(1 for c in self.components if c.present)

    def label(self) -> str:
        """One line for a human: the state, the number and how much saw it."""
        missing = [c.key for c in self.components if not c.present]
        line = (f"{self.tag} (score {self.score:+.2f}, "
                f"{self.n_present}/{len(self.components)} prints, "
                f"{self.coverage:.0%} of panel weight)")
        if missing:
            line += " — absent: " + ", ".join(missing)
        return line

    def as_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "score": _num(self.score, 3),
                "coverage": _num(self.coverage, 3),
                "size_multiplier": _num(self.size_multiplier, 3),
                "n_present": self.n_present, "n_components": len(self.components),
                "as_of": self.as_of, "label": self.label(),
                "components": [c.as_dict() for c in self.components],
                "extras": self.extras}


# ------------------------------------------------------------------- helpers
def _clip(x: float) -> float | None:
    """Clamp a score into [-1, 1] — and refuse to invent one.

    `min(1.0, nan)` keeps 1.0 and `max(-1.0, 1.0)` keeps 1.0, because both
    helpers drop the argument that cannot be compared. So this clamp, read as
    total, turned a NaN into the loudest possible macro reading: a print that
    answered nothing became "maximum risk appetite", and `present` — which asks
    only that the score be finite — could never see it, because the laundering
    happened first. An unreadable number abstains, which is the other thing the
    panel already does with a feed that did not answer.
    """
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(max(-1.0, min(1.0, x)))


def _positive(c: np.ndarray) -> bool:
    """True when every bar is a number a ratio or a log can be built on.

    A dead vendor serves 0.0, and 0.0 is finite, so it survives the filter in
    `_closes` and lands underneath a division (an infinite price ratio) or a
    logarithm (negative infinity), where it stops being a hole in the data and
    becomes an extreme in the answer.
    """
    return bool(np.all(c > 0.0))


def _dead(c: Component) -> Component:
    """Return the component as an abstention: the window it reads has a hole in it.

    The arithmetic downstream would rather have the hole than lose the print. One
    empty last bar on the dollar turns a 20-day return into -100%, which this
    panel negates and clips into **+1.0 — maximum risk appetite** — while the
    `why` string reports "DXY 20d -100.00%" to an operator as though the dollar
    had collapsed. A zero at the base of the ten-year's difference invents a 4.5
    point yield move out of two prints, one of which is nothing. So the bar that
    cannot be a price costs the component its seat, and the desk is told which.
    """
    c.score, c.value = None, None
    c.why = f"{c.key} window has a zero print"
    return c


def _w(k: float, x: float | None) -> float | None:
    """Scale one leg of a blend, or carry its abstention through the arithmetic."""
    return None if x is None else k * float(x)


def _num(v: Any, nd: int) -> float | None:
    """A JSON number, or null.

    `json.dumps` writes a non-finite float as `Infinity` / `NaN`, and neither is
    JSON — so one of them in the payload is a desk page that stops parsing, the
    same dead white screen a missing bracket produced. Null is also the honest
    answer: an operator reading `value: 0.0` cannot tell an empty print from an
    unreadable one, and the panel already has a word for "did not answer".
    """
    f = float(v)
    return None if not np.isfinite(f) else round(f, nd)


def _closes(df: Any, min_bars: int) -> np.ndarray | None:
    """Trailing closes, finite-only, or None when the series cannot answer.

    Non-finite bars are dropped rather than zero-filled: a vendor gap in the
    middle of the VIX must not be allowed to read as a volatility collapse, and
    dropping it can only ever lose the component its seat, never invent a print.
    """
    if df is None:
        return None
    try:
        raw = np.asarray(df["close"].to_numpy(), dtype=float)
    except (TypeError, ValueError, KeyError):
        return None
    c = raw[np.isfinite(raw)]
    return c if c.size >= min_bars else None


def _ret(c: np.ndarray, n: int) -> float | None:
    """`n`-bar return off a base the size of which actually makes sense."""
    if c.size < n + 1:
        return None
    base = float(c[-1 - n])
    if not np.isfinite(base) or abs(base) < 1e-12:
        return None
    return float(c[-1] / base - 1.0)


# ---------------------------------------------------------------- components
# Each returns a Component, or the same Component with score=None when its
# series is absent. Weights sum to 1.0 over the seven, which is what makes
# `coverage` a percentage of the panel rather than an arbitrary ratio.

def _c_vix(df: Any) -> Component:
    c = Component("VIX", "fear premium", 0.22, None, None)
    x = _closes(df, 30)
    if x is None:
        c.why = "no VIX series"
        return c
    level = float(x[-1])
    if level <= 0:
        # A volatility index cannot print zero. The vendors serve that when a
        # feed dies, and the percentile map below would read it as the calmest
        # tape in the sample — so it abstains instead.
        c.why = f"VIX printed {level:g}, which is not a volatility"
        return c
    # Percentile against its own trailing year where there is one: what counts
    # as frightened is a property of the era, not of an absolute number.
    win = x[-250:]
    if not _positive(win):
        # An empty print inside the year counts as "at or below today" when the
        # percentile is asked, so a dead bar inflates the fear reading: the hole
        # in the tape arrives at the desk as volatility.
        return _dead(c)
    if win.size >= 120 and float(np.std(win)) < 1e-12:
        # A perfectly flat window makes "how often was it at or below this?"
        # equal 1.0 by construction, which would read a dead feed as the most
        # frightened tape on record. A series with no movement has no opinion.
        c.why = f"VIX flat at {level:g} — percentile meaningless"
        return c
    if win.size >= 120:
        pct = float((win <= level).mean())
        score = -(pct - 0.5) * 2.0
        c.why = f"VIX {level:.1f} = {pct:.0%} of the last {win.size} bars"
    else:
        score = _clip((20.0 - level) / 15.0)
        c.why = f"VIX {level:.1f} (short history, absolute scale)"
    d20 = _ret(x, 20)
    spike = _clip(-d20 / 0.50) if d20 is not None else None
    if score is not None and spike is not None:
        # A vol *spike* carries information the level has not priced yet.
        score = 0.65 * score + 0.35 * spike
        c.why += f", 20d {d20:+.0%}"
    c.score, c.value = (_clip(score) if score is not None else None), level
    return c


def _c_dxy(df: Any) -> Component:
    c = Component("DXY", "dollar pressure", 0.18, None, None)
    x = _closes(df, 25)
    if x is None:
        c.why = "no DXY series"
        return c
    if not _positive(x[-21:]):
        # Before the return is computed: a zero at either end of the two prints
        # the move is taken between answers -100% — which the sign flip below
        # turns into the loudest risk-on reading the panel can make — and a zero
        # at the base makes `_ret` abstain for a reason this line would report
        # wrongly as "history too short".
        return _dead(c)
    r = _ret(x, 20)
    if r is None:
        c.why = "DXY history too short for a 20-day move"
        return c
    # A rising dollar drains EM liquidity and is the single most reliable
    # reason a PSX or PMEX book finds itself short of rupees at the same time
    # its export earners look cheap.
    c.score, c.value = _clip(-r / 0.03), float(x[-1])
    c.why = f"DXY 20d {r:+.2%}"
    return c


def _c_rates(df: Any) -> Component:
    c = Component("US10Y", "policy-rate pressure", 0.14, None, None)
    x = _closes(df, 25)
    if x is None:
        c.why = "no US10Y series"
        return c
    if x.size < 21:
        c.why = "US10Y history too short"
        return c
    delta = float(x[-1] - x[-21])   # ^TNX quotes the yield in percent
    if not np.isfinite(delta):
        c.why = "US10Y unreadable"
        return c
    if float(x[-1]) <= 0:
        # The ten-year has not traded at zero in this data's lifetime; a zero
        # print is the feed being empty, and it must not arrive as a neutral
        # opinion about policy rates.
        c.why = f"US10Y printed {x[-1]:g}, which is no data"
        return c
    # The other end of the difference needs the same check, and only for zero:
    # a *negative* yield is a rare finding the vendor is entitled to report, but
    # an empty base makes `x[-1] - 0` read as a 4.5 point policy move that never
    # happened, and the component then says so on the desk.
    if np.any(x[-21:] == 0.0):
        return _dead(c)
    c.score, c.value = _clip(-delta / 0.40), float(x[-1])
    c.why = f"10y {delta:+.2f}pp over 20d (yield {x[-1]:.2f}%)"
    return c


def _c_equity(df: Any) -> Component:
    c = Component("SPX", "global risk beta", 0.16, None, None)
    x = _closes(df, 60)
    if x is None:
        c.why = "no SPX series"
        return c
    if not _positive(x[-60:]):
        # An empty print inside the window drags the 60-bar mean down, so the
        # trend leg reads the hole as the price standing far above its own
        # average — the direction depends on which bar died, and both extremes
        # are score, not abstention.
        return _dead(c)
    sma = float(np.mean(x[-60:]))
    trend = _clip((float(x[-1]) / sma - 1.0) / 0.05) if sma > 0 else None
    r20 = _ret(x, 20)
    mom = _clip(r20 / 0.06) if r20 is not None else None
    parts = [p for p in (_w(0.6, trend), _w(0.4, mom)) if p is not None]
    if not parts:
        c.why = "SPX history too short"
        return c
    c.score = _clip(sum(parts))
    c.value = float(x[-1])
    # The sentence is built from whichever legs actually answered. It used to name
    # the trend leg unconditionally, which was only ever safe while `_clip` could
    # not abstain — and it cannot, since this increment, only *because* of the
    # window guard above: an all-positive 60-bar window has a mean of at least
    # one-sixtieth of its largest bar, so the ratio cannot blow up. That is an
    # accident of arithmetic three lines away, not a contract, and a `None`
    # reaching this format raised out of `macro_state`, which wraps no component
    # in a try, and took the desk endpoint down with it.
    head = "SPX trend unreadable" if trend is None else f"SPX {trend:+.2f} vs 60d mean"
    c.why = head if r20 is None else f"{head}, 20d {r20:+.1%}"
    return c


def _c_growth(cu: Any, au: Any) -> Component:
    c = Component("COPPER/GOLD", "growth vs safety flows", 0.12, None, None)
    x, y = _closes(cu, 25), _closes(au, 25)
    if x is None or y is None:
        c.why = "no copper/gold series"
        return c
    n = min(x.size, y.size)
    if n < 25:
        c.why = "copper/gold overlap too short"
        return c
    if not (_positive(x) and _positive(y)):
        # The guard further down only ever looks at one entry of this window, and
        # the division below reads all of it. A zero gold print answers the ratio
        # with an infinity; a zero *copper* print answers it with exactly zero —
        # and zero ratio, divided by a healthy base, is `-1.0`: the most risk-off
        # growth read the panel can make, produced by a feed that served nothing.
        # A hole in either leg is a hole in the pair.
        return _dead(c)
    ratio = x[-n:] / y[-n:]
    base = float(ratio[-21])
    if not np.isfinite(base) or abs(base) < 1e-12:
        c.why = "copper/gold ratio unreadable"
        return c
    r = float(ratio[-1] / base - 1.0)
    c.score, c.value = _clip(r / 0.06), float(ratio[-1])
    c.why = f"copper/gold 20d {r:+.1%}"
    return c


def _c_btc(df: Any) -> Component:
    c = Component("BTC", "speculative appetite", 0.10, None, None)
    x = _closes(df, 60)
    if x is None:
        c.why = "no BTC series"
        return c
    if not _positive(x[-60:]):
        # Checked before the return is computed, not after: `log(0)` is negative
        # infinity, and a zero sitting at the base of the 20-day move makes
        # `_ret` abstain for a reason this line would report wrongly as "history
        # too short" to an operator reading the desk.
        return _dead(c)
    r = _ret(x, 20)
    if r is None:
        c.why = "BTC history too short"
        return c
    sigma = float(np.std(np.diff(np.log(x[-60:]))))
    if not np.isfinite(sigma) or sigma < 1e-6:
        c.why = "BTC volatility undefined"
        return c
    # BTC is the most levered read on appetite the panel has, so its move is
    # divided by its own vol — otherwise it would shout over every other print.
    z = r / (sigma * np.sqrt(20.0))
    c.score, c.value = _clip(z / 1.5), r
    c.why = f"BTC 20d {r:+.1%} ({z:+.1f} sigma)"
    return c


def _c_pkr(df: Any) -> Component:
    c = Component("USDPKR", "local currency stress", 0.08, None, None)
    x = _closes(df, 25)
    if x is None:
        c.why = "no USDPKR series"
        return c
    if not _positive(x[-21:]):
        return _dead(c)
    r = _ret(x, 20)
    if r is None:
        c.why = "USDPKR history too short"
        return c
    # On a managed peg, 2% in a month is the tape telling you where the
    # interbank actually clears. Rupee depreciation is risk-off for every
    # PSX position: it is the import bill, the FX book, and the repricing of
    # every dollar-denominated obligation at once.
    c.score, c.value = _clip(-r / 0.02), float(x[-1])
    c.why = f"USDPKR 20d {r:+.2%}"
    return c


COMPONENT_WEIGHTS = 1.0


# ---------------------------------------------------------------------- read
def macro_state(intermarket: dict[str, pd.DataFrame] | None,
                extras: dict[str, Any] | None = None) -> MacroState:
    """Score the panel as it stands. Pure, synchronous, no I/O.

    `intermarket` is the same {key: DataFrame} dict the strategy library is
    handed, so passing a date-truncated copy makes this a point-in-time read —
    which is what makes the same function usable by the walk-forward.
    """
    panel = intermarket or {}
    comps = [
        _c_vix(panel.get("VIX")),
        _c_dxy(panel.get("DXY")),
        _c_rates(panel.get("US10Y")),
        _c_equity(panel.get("SPX")),
        _c_growth(panel.get("COPPER"), panel.get("GOLD")),
        _c_btc(panel.get("BTC")),
        _c_pkr(panel.get("USDPKR")),
    ]
    present = [c for c in comps if c.present]
    coverage = sum(c.weight for c in present) / COMPONENT_WEIGHTS
    score = (sum(c.weight * float(c.score) for c in present) /
             sum(c.weight for c in present)) if present else 0.0

    if coverage < MIN_COVERAGE or len(present) < MIN_COMPONENTS:
        tag = UNKNOWN
    elif score >= TAG_BAND:
        tag = RISK_ON
    elif score <= -TAG_BAND:
        tag = RISK_OFF
    else:
        tag = NEUTRAL

    stamps = []
    for s in panel.values():
        try:
            stamps.append(str(s["date"].iloc[-1])[:10])
        except (TypeError, KeyError, IndexError, AttributeError):
            continue
    as_of = max(stamps) if stamps else ""
    return MacroState(tag=tag, score=float(score), coverage=float(coverage),
                      size_multiplier=float(SIZE_MULTIPLIER[tag]),
                      components=comps, extras=dict(extras or {}), as_of=as_of)


def condition_label(micro_regime: str, state: MacroState | str) -> str:
    """The attribution bucket key for "this micro regime, under this macro tag".

    Both producers — the live engine and tools/strategy_attribution.py — call
    this one function. That is the point: if they spelled it differently, every
    live signal would look up a bucket the walk-forward never wrote, the loader
    would answer 1.0 with tag "no coverage", and the system would read as though
    no strategy had an edge when the real event is that the two sides stopped
    describing the world in the same words.
    """
    tag = state.tag if isinstance(state, MacroState) else str(state or UNKNOWN)
    if tag not in TAGS:
        tag = UNKNOWN
    return f"{(micro_regime or 'Unknown').strip()} :: {tag}"


def split_condition(condition: str) -> tuple[str, str]:
    """(micro, macro tag) out of a bucket label; falls back honestly."""
    micro, _, tag = str(condition).partition(" :: ")
    return micro.strip() or "Unknown", (tag.strip() if tag.strip() in TAGS else UNKNOWN)


# ------------------------------------------------------------------ live-only
def psx_breadth(snapshot: pd.DataFrame | None) -> dict[str, Any] | None:
    """How wide the PSX advance is — reported, never scored.

    The cross-section only exists as a live TradingView scan, so the sweep
    cannot reconstruct it for a past date. It belongs beside the macro state
    (a narrow advance under a friendly global tape is a different trade) and
    stays out of `score` so the bucket tag remains a function of inputs the
    measurement had too.
    """
    if snapshot is None or getattr(snapshot, "empty", True):
        return None
    out: dict[str, Any] = {}
    # Each print is guarded on its own: a scan that arrives without the
    # one-month column (a different column list, a vendor that drops it) still
    # owes the desk the trend breadth, and losing one number must not read as
    # "breadth unavailable".
    try:
        close = pd.to_numeric(snapshot["close"], errors="coerce")
        sma50 = pd.to_numeric(snapshot["SMA50"], errors="coerce")
        both = close.notna() & sma50.notna() & (sma50 > 0)
        n = int(both.sum())
        if n >= 30:
            out["symbols"] = n
            out["pct_above_sma50"] = round(float((close[both] > sma50[both]).mean()), 3)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        m1 = pd.to_numeric(snapshot["change|1M"], errors="coerce")
        ok = m1.notna()
        if int(ok.sum()) >= 30:
            out["pct_up_1m"] = round(float((m1[ok] > 0).mean()), 3)
    except (KeyError, TypeError, ValueError):
        pass
    return out or None


def live_state(md: Any, engine: Any = None) -> MacroState:
    """The macro state as it reads right now, for the desk and the risk layer."""
    panel = {}
    note = ""
    try:
        eng = engine if engine is not None else _default_engine()
        panel = eng.intermarket() if eng is not None else {}
    except Exception as exc:
        note = f"panel unavailable: {type(exc).__name__}: {str(exc)[:80]}"
    extras: dict[str, Any] = {"live_only": True}
    if note:
        extras["panel_error"] = note
    try:
        breadth = psx_breadth(md.psx_snapshot())
        if breadth:
            extras["psx_breadth"] = breadth
    except Exception:
        pass
    return macro_state(panel, extras=extras)


def _default_engine():
    from .engine import ENGINE
    return ENGINE
