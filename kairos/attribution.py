"""Per-strategy attribution: which part of the library pays, in which condition.

The 100-strategy library has always been fused as a flat opinion poll — every
vote entered at its own strength, every strategy counted as though it had
never been measured, and no outcome in the journal was ever credited to the
part that produced it. So a strategy that is only right in a trending market
could not be demoted when the market stopped trending, a strategy that raises
on every instrument looked exactly like one that abstains, and the one figure
the system does learn from — the realized track record — updated a single
global confidence multiplier and threw the rest away.

This module is the spine that was missing. It holds *counts*, not opinions:
for every (strategy, horizon, regime) bucket, how many out-of-sample
directional votes it made and how many were right. The weights are a pure
function of those counts, so changing the gain or the shrinkage below does not
require the walk-forward to be re-run, and the evidence a weight came from can
always be shown next to it.

Three rules, all of them the shape the rest of this codebase has settled on:

* **A component that cannot answer is not allowed to answer.** If the artifact
  is missing, unreadable, stale, measured against a different library, or built
  from too few votes to tell skill from noise, `load` refuses and says which.
  It never returns an empty dict and lets the caller read that as "no strategy
  has an edge" — that reading would silently flatten the poll back to uniform,
  which is the claim the file exists to test.
* **Absence of evidence is not evidence of failure.** A strategy with no graded
  votes keeps weight 1.0 and is reported as *no coverage*, not demoted. A
  strategy that raised during measurement is reported as *broken* and keeps
  1.0: it was never given the chance to be wrong.
* **Skill is a difference, not a ratio.** A hit rate is compared with how often
  that horizon rose in that regime over the slices it was earned on, so a
  strategy that is simply long in a bull tape is measured as long, not as
  clever. Where the artifact cannot say how often the tape rose, no weight is
  applied at all and the reason comes back as *no base rate*.
* **A weight that fires is a weight you can find.** `fuse` records how many
  votes it scaled and which bucket produced each one, the engine puts that in
  the prediction, the desk prints it, and the first time attribution becomes
  active it is appended to the audit trail.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import CFG

ARTIFACT_PATH = CFG.data_dir / "strategy_attribution.json"
# Bumped to 2 when the bucket key became a *condition* — micro regime plus the
# cross-asset macro tag from kairos/macro.py. Old schema-1 files still hold real
# counts, but their keys were spelled without the macro half, so every live
# lookup would miss and report "no coverage" while looking like a finding. A
# refusal that names the schema is honest; a flat poll that looks measured is
# not. Re-measure with tools/strategy_attribution.py to produce schema 3.
#
# Bumped to 3 when the denominator became *paired*: schema 2 compared a
# strategy's hits with the condition's up-frequency over every slice of every
# instrument, including the slices that strategy abstained on. A member that
# only speaks in rising tapes therefore collected the promotion its abstentions
# spared it, and a member that spoke in falling tapes was paid for the drift it
# stood against. Each row now carries `base_by_horizon`/`base_by_bucket`, the
# tape over exactly the slices it graded, so schema-2 counts are not silently
# re-read under the new rule.
SCHEMA = 3

# The horizons a thesis is graded at. 5 is what the desk trades; 21 is what an
# investment committee claims to believe. Measuring both is the only way to
# know whether a strategy is a five-day edge wearing a monthly label.
HORIZONS = (5, 21)

# Directional out-of-sample votes a bucket needs before it may move a weight at
# all. Below this a 60% hit rate on twelve votes is a coin flip that landed
# well, and a coin flip must not be allowed to change an order.
MIN_CALLS = 40
# A regime bucket is thinner than the pooled one by design; it may be used a
# little earlier, but only for the regime it was measured in.
MIN_REGIME_CALLS = 20
# Library-wide graded votes before any of it is trusted enough to apply.
MIN_VOTES = 2_000
# A walk-forward measured against yesterday's library is not a description of
# today's. 45 days is generous for a research run on one machine.
MAX_AGE_DAYS = 45

GAIN = 2.0
MIN_WEIGHT = 0.5
MAX_WEIGHT = 1.5

# Slices a bucket needs before its up-frequency may be called a base rate.
MIN_BASE_SLICES = 20

# A strategy that raised in more than this fraction of the slices it was shown
# is not abstaining on judgment, it is failing.
BROKEN_ERROR_RATE = 0.10


# --------------------------------------------------------------------- counts
def new_counts(library_size: int, horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    return {"schema": SCHEMA, "library_size": int(library_size),
            "horizons": [int(h) for h in horizons],
            "instruments": [], "slices": 0, "graded": 0,
            "base": {str(int(h)): {} for h in horizons},
            "strategies": {}}


def ensure_strategy(counts: dict[str, Any], name: str, category: str = "",
                    horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    """One row per strategy, created whether or not it ever votes — an
    abstaining library member has to be distinguishable from a missing one."""
    st = counts["strategies"].get(name)
    if st is None:
        st = {"category": category, "slices_seen": 0, "errors": 0, "votes": 0,
              "strength_sum": 0.0, "by_horizon": {}, "by_bucket": {},
              "base_by_horizon": {}, "base_by_bucket": {},
              "last_error": None}
        counts["strategies"][name] = st
    elif category and not st["category"]:
        st["category"] = category
    for h in horizons:
        st["by_horizon"].setdefault(str(int(h)), [0, 0, 0, 0])
    return st


def note_slice(counts: dict[str, Any], names: list[str],
               errors: dict[str, str] | None = None) -> None:
    """Advance one slice's counters.

    `names` is every strategy the library evaluated on this slice, whether or
    not it spoke, and `errors` the ones that raised. Both are needed because a
    strategy can be silent three different ways — abstained, raised, or not in
    the library at all — and only the first is a judgment.
    """
    for name in names:
        ensure_strategy(counts, name)["slices_seen"] += 1
    for name, msg in (errors or {}).items():
        st = ensure_strategy(counts, name)
        st["errors"] += 1
        st["last_error"] = str(msg)[:160]


def note_outcome(counts: dict[str, Any], regime: str, outcomes: dict[int, float],
                 horizons: tuple[int, ...] = HORIZONS) -> None:
    """Record what the tape did on this slice, whoever voted.

    This is the condition's own up-frequency — what the report prints beside a
    strategy's hit rate so the number can be read as a difference. It is
    deliberately recorded on every slice, including the ones nobody voted on:
    a condition's drift is a property of the condition, and a table that only
    counted slices with a vote in them could not say how much of the tape was
    skipped. The denominator a *weight* is measured against is the paired one
    on each strategy row (`base_by_horizon`/`base_by_bucket`), because a hit
    rate covers only the slices that strategy chose to speak on.
    """
    base = counts.setdefault("base", {})
    for h, realized in outcomes.items():
        if realized is None:
            continue
        hb = base.setdefault(str(int(h)), {})
        rb = hb.setdefault(str(regime), [0, 0])
        ab = hb.setdefault("_all", [0, 0])
        up = 1 if float(realized) > 0 else 0
        rb[0] += 1
        rb[1] += up
        ab[0] += 1
        ab[1] += up


def grade(counts: dict[str, Any], name: str, category: str, direction: int,
          strength: float, regime: str, outcomes: dict[int, float]) -> None:
    """Credit one directional vote against the realized return of each horizon.

    A vote of zero strength is not a claim and is not graded: it would add a
    coin-flip observation to a bucket that is supposed to answer "when this
    strategy spoke, was it right?". Each bucket keeps
    ``[votes, hits, long_votes, short_votes]`` — the split is what lets the
    expected hit rate of a *directional bias* be subtracted from the hit rate
    of a skill.
    """
    if direction == 0 or strength <= 0:
        return
    st = ensure_strategy(counts, name, category)
    st["votes"] += 1
    st["strength_sum"] = round(st["strength_sum"] + float(strength), 4)
    counts["graded"] = counts.get("graded", 0) + 1
    side = 2 if direction > 0 else 3
    for h, realized in outcomes.items():
        if realized is None:
            continue
        correct = 1 if direction * float(realized) > 0 else 0
        bucket = st["by_horizon"].setdefault(str(int(h)), [0, 0, 0, 0])
        _accumulate(bucket, correct, side)
        key = f"{int(h)}|{regime}"
        _accumulate(st["by_bucket"].setdefault(key, [0, 0, 0, 0]), correct, side)
        # The denominator is paired to the votes: how often this horizon rose on
        # the slices *this* strategy spoke on. A member that abstains whenever
        # the tape is falling would otherwise be compared with a market it never
        # sat through and paid for the abstention as though it were a forecast.
        _note_base(st["base_by_horizon"].setdefault(str(int(h)), [0, 0]), realized)
        _note_base(st["base_by_bucket"].setdefault(key, [0, 0]), realized)


def _note_base(pair: list[int], realized: float) -> None:
    pair[0] += 1
    pair[1] += 1 if float(realized) > 0 else 0


def _accumulate(bucket: list[int], correct: int, side: int) -> None:
    """`side` is the bucket's own column index — 2 for a long vote, 3 for a
    short one — so a directional count can never land in the hit column."""
    bucket[0] += 1
    bucket[1] += correct
    bucket[side] += 1


# ------------------------------------------------------------------- weights
def _p_up(pair: list[int] | None, min_slices: int = MIN_BASE_SLICES) -> float | None:
    """How often this bucket rose, or None when too few slices say so."""
    if not pair or int(pair[0]) < min_slices:
        return None
    return float(pair[1]) / float(pair[0])


def _expected(bucket: list[int], p_up: float) -> float:
    """Hits a strategy with this long/short mix would collect with no skill at all.

    Subtracting it is the difference between measuring a forecast and measuring
    the drift of whatever it was pointed at: a bull-market library in which
    every momentum strategy is "right" 65% of the time has not demonstrated
    anything except that bull markets reward being long.
    """
    _n, _hits, n_long, n_short = bucket
    return n_long * p_up + n_short * (1.0 - p_up)


def _weight_from(bucket: list[int], expected: float) -> float:
    """Bounded, shrunk multiplier from one (votes, hits, long, short) count.

    `excess` is hits above the directional base rate per vote, so a strategy
    that is exactly as good as its own bias gets 1.0. The estimate is shrunk
    toward that 1.0 in proportion to how thin the bucket is — `n / (n +
    MIN_CALLS)` — which is the part a raw ratio gets wrong: a 75% hit rate on
    twelve votes must not move a vote the way the same rate on four hundred can.
    """
    n, hits = int(bucket[0]), int(bucket[1])
    if n <= 0:
        return 1.0
    excess = (hits - expected) / (n + 1.0)
    shrink = n / (n + MIN_CALLS)
    return float(max(MIN_WEIGHT, min(MAX_WEIGHT, 1.0 + GAIN * excess * shrink)))


class Attribution:
    """Derived weights over one loaded artifact, plus what it can say about itself."""

    def __init__(self, counts: dict[str, Any]) -> None:
        self.counts = counts
        self.generated = str(counts.get("generated", ""))[:10]
        self._cache: dict[tuple[str, int, str], float] = {}
        self.used = 0
        self.demoted = 0
        self.promoted = 0
        self.no_coverage: list[str] = []
        self.no_base: list[str] = []
        self.broken: list[str] = []

    def weight(self, name: str, horizon: int, regime: str) -> float:
        return self.weight_tag(name, horizon, regime)[0]

    def weight_tag(self, name: str, horizon: int, regime: str) -> tuple[float, str]:
        """The multiplier for one (strategy, horizon, regime) and what produced it.

        Counted per call rather than per derivation: the memo is shared by every
        view of one artifact, so a strategy weighted for the second symbol on
        tonight's watchlist would otherwise vanish from the report of the signal
        that is actually using it.
        """
        ck = (name, int(horizon), regime)
        hit = self._cache.get(ck)
        if hit is None:
            hit = self._derive(name, int(horizon), regime)
            self._cache[ck] = hit
        w, tag = hit
        self.used += 1
        if tag == "no coverage":
            if name not in self.no_coverage:
                self.no_coverage.append(name)
        elif tag == "broken":
            if name not in self.broken:
                self.broken.append(name)
        elif tag == "no base rate":
            if name not in self.no_base:
                self.no_base.append(name)
        elif w < 1.0:
            self.demoted += 1
        elif w > 1.0:
            self.promoted += 1
        return w, tag

    def _derive(self, name: str, horizon: int, regime: str) -> tuple[float, str]:
        st = self.counts["strategies"].get(name)
        if st is None:
            return 1.0, "no coverage"
        if st.get("errors", 0) > BROKEN_ERROR_RATE * max(st["slices_seen"], 1):
            return 1.0, "broken"
        key = f"{horizon}|{regime}"
        bucket = (st.get("by_bucket") or {}).get(key)
        if bucket and int(bucket[0]) >= MIN_REGIME_CALLS:
            # The paired denominator, not the condition's overall up-frequency:
            # a bucket's hits are compared with how often this horizon rose on
            # the slices this strategy was graded on. `is None`, not `or` — a
            # regime in which nothing rose has a base rate of exactly 0.0, and
            # treating that as missing is what once inverted the drifter and the
            # counter-trend caller in a crash tape.
            p_up = _p_up((st.get("base_by_bucket") or {}).get(key))
            if p_up is None:
                return 1.0, "no base rate"
            return _weight_from(bucket, _expected(bucket, p_up)), "regime"
        pooled = (st.get("by_horizon") or {}).get(str(horizon))
        if pooled and int(pooled[0]) >= MIN_CALLS:
            p_up = _p_up((st.get("base_by_horizon") or {}).get(str(horizon)))
            if p_up is None:
                return 1.0, "no base rate"
            return _weight_from(pooled, _expected(pooled, p_up)), "pooled"
        return 1.0, "no coverage"

    def explain(self, name: str, horizon: int, regime: str) -> dict[str, Any]:
        """The evidence behind one weight — for the desk, not the hot path."""
        st = self.counts["strategies"].get(name) or {}
        return {"weight": round(self.weight(name, horizon, regime), 3),
                "votes": st.get("votes", 0), "errors": st.get("errors", 0),
                "slices_seen": st.get("slices_seen", 0),
                "by_horizon": st.get("by_horizon", {}),
                "base_by_horizon": st.get("base_by_horizon", {}),
                "regime_buckets": {k: v for k, v in st.get("by_bucket", {}).items()
                                   if k.startswith(f"{int(horizon)}|")}}

    def view(self) -> "Attribution":
        """A reporting copy that shares these counts.

        The engine caches one Attribution per process, but `status()` is read
        out of it per prediction — and counters that accumulate across a whole
        watchlist would report the process's history rather than this signal.
        """
        clone = Attribution.__new__(Attribution)
        clone.counts = self.counts
        clone.generated = self.generated
        clone._cache = self._cache          # weights are pure; the memo is shared
        clone.used = clone.demoted = clone.promoted = 0
        clone.no_coverage, clone.broken, clone.no_base = [], [], []
        return clone

    def status(self) -> dict[str, Any]:
        return {"applied": True, "generated": self.generated,
                "graded_votes": self.counts.get("graded", 0),
                "instruments": len(self.counts.get("instruments", [])),
                "slices": self.counts.get("slices", 0),
                "min_calls": MIN_CALLS, "gain": GAIN,
                "weighted": self.used, "demoted": self.demoted,
                "promoted": self.promoted, "no_coverage": len(self.no_coverage),
                "no_base_rate": len(self.no_base), "broken": sorted(self.broken)}


# --------------------------------------------------------------------- load
def _as_date(s: Any) -> date | None:
    try:
        return datetime.fromisoformat(str(s)).date()
    except (TypeError, ValueError):
        return None


def load(path: str | Path | None = None, library_size: int | None = None,
         today: date | None = None) -> tuple[Attribution | None, str]:
    """Read the artifact, or refuse and say exactly why. Never an empty answer.

    Every refusal here is a decision, not an error: the caller continues with a
    uniform poll, which is the pre-attribution behaviour, and the reason reaches
    the prediction and the desk so a weight that is *not* firing is as findable
    as one that is.
    """
    path = ARTIFACT_PATH if not path else Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, (f"no attribution artifact at {path} — run "
                      "python tools/strategy_attribution.py")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"attribution artifact at {path} could not be read ({str(exc)[:120]})"
    try:
        counts = json.loads(raw)
    except ValueError as exc:
        return None, f"attribution artifact is not valid JSON ({str(exc)[:120]})"
    if not isinstance(counts, dict) or not isinstance(counts.get("strategies"), dict):
        return None, "attribution artifact does not hold a strategy table"
    if int(counts.get("schema", 0) or 0) != SCHEMA:
        return None, (f"attribution artifact is schema {counts.get('schema')}, "
                      f"this build reads schema {SCHEMA}")
    if library_size is not None and int(counts.get("library_size", -1)) != int(library_size):
        return None, (f"attribution measured {counts.get('library_size')} strategies and "
                      f"the library now holds {library_size} — re-run the walk-forward")
    graded = int(counts.get("graded", 0) or 0)
    if graded < MIN_VOTES:
        return None, (f"attribution graded only {graded} directional votes, need "
                      f"{MIN_VOTES} to separate skill from noise")
    gen = _as_date(counts.get("generated"))
    if gen is None:
        return None, ("attribution artifact carries no readable generation date, so "
                      "its age cannot be checked")
    age = (today or date.today()).toordinal() - gen.toordinal()
    if age > MAX_AGE_DAYS:
        return None, (f"attribution is {age} days old (generated {gen.isoformat()}), "
                      f"limit {MAX_AGE_DAYS}")
    if age < 0:
        return None, f"attribution claims to have been generated {age * -1} days from now"
    return Attribution(counts), ""


# ------------------------------------------------------------- process cache
_LOCK = threading.Lock()
_cache: dict[str, Any] = {"stamp": 0.0, "size": -1, "attr": None,
                          "reason": "not loaded"}


def current(library_size: int | None = None,
            ttl_s: float = 600.0) -> tuple[Attribution | None, str]:
    """Load once per TTL. The walk-forward is a research run, not a hot path,
    so the file it writes is allowed to be read a few times an hour.

    A library that changed size since the load forces a reload, because the
    refusal reason has to name the count the artifact was measured against.
    """
    from .strategies import library_size as _size
    size = int(_size() if library_size is None else library_size)
    with _LOCK:
        now = time.time()
        if now - _cache["stamp"] < ttl_s and size == _cache["size"]:
            return _cache["attr"], _cache["reason"]
        attr, reason = load(library_size=size)
        _cache.update(stamp=now, size=size, attr=attr, reason=reason)
        return attr, reason
