"""Per-instrument news tone that needs no key, no LLM and no rate limit.

Strategy 99, "News Sentiment Tilt", has never once voted. The only tone in the
system was `PredictionEngine.news_sentiment()`, which returns a float from an
OpenRouter call, and the engine set it into the strategy context only when
`apis.openrouter.enabled` — so with the council off, and in every walk-forward
slice (which has no LLM and no historical headlines at all), `ctx` had no
`news_sentiment`, the strategy abstained, and abstention looked exactly like a
component that had read the news and found nothing to say. A signal gated behind
a paid per-call provider is not a signal; it is a bill.

This module supplies the same input without the dependency:

* **The headlines are fetched per instrument, not from a shared feed pool.**
  `data.NewsService` pulls seven fixed feeds and then *matches* the instrument's
  name against whatever arrived. For a mid-cap that never made the front page
  that returns nothing, so the instrument has no tone, which is reported as no
  tone rather than as a calm one.
* **The score is a lexicon, so it is reproducible.** Same headlines, same
  number, on any machine, with no model version to pin and no provider to trust.
  The LLM path stays available and takes precedence when configured; this is the
  floor under it, not a replacement for it.
* **Absence is not neutrality.** An instrument that no returned headline actually
  mentions, or a set of headlines in which no scored word appears, answers
  `score=None` with a `reason`. Averaging unmatched articles toward zero would
  hand strategy 99 a confident "no news" that it would read as a mild signal —
  the same failure this codebase already refuses on the macro axis, where an
  absent print is never counted as a neutral one.

And because a ~4s round trip is not affordable inside a prediction that already
takes 9.7s, the desk is a **cache with an explicit refresh**, not a fetch on the
read path: `refresh()` fills it in parallel for a batch, and `tone()` serves the
last answer with its age attached, so a reader can tell "quiet tape" from "nobody
has asked this symbol yet".
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import CFG
from .data import FEED_MAX_BYTES, _feed_xml, _http_get
from .netio import HttpStatus, close_error
from .store import atomic_write, dumps_strict
from .universe import Instrument, universe

# Kept deliberately small and financial. This is not a general English sentiment
# lexicon: a headline is a title, not prose, and every extra ordinary-language
# word is a word that can fire on a sports page. Two tiers, strong and mild.
STRONG_POS = frozenset(("record", "surge", "soar", "rally", "jump", "spike", "beat", "upgrade",
              "buys", "buyback", "approval", "cleared", "launches", "discovers",
              "expand", "expands", "expansion", "growth", "profit", "profits",
              "surplus",
              "dividend", "award", "won", "contract", "acquires", "merger",
              "strengthens", "highs", "boost", "uplift", "restart", "resumes",
              # "raises" for a payout, guidance or a stake — the shape that
              # probed as "Meezan Bank beats estimates and raises payout" and
              # scored nothing at all. The base form is what the stem lookup
              # below needs; the inflections are reached, not listed.
              "raise"))
STRONG_NEG = frozenset(("plunge", "crash", "slump", "tumble", "sinks", "miss", "misses",
              "missing", "downgrade", "cuts", "loss", "losses", "deficit", "suspend",
              "suspended", "suspension", "shutdown", "halts", "recall", "probe",
              "investigation", "charged", "arrest", "raid", "fine", "penalty",
              "fraud", "breach", "default", "bankrupt", "insolvent", "layoffs",
              "closure", "shortfall", "outage", "theft", "accident", "delay",
              "delayed", "weakens", "lows", "worst", "sell", "sells", "fined",
              "seized", "cancelled", "withdraws",
              # A failed audit, a failed bid, a failed merger. "failed" and
              # "failing" are in `NEGATORS` because they negate the word after
              # them; that was never the same thing as scoring the failure
              # itself, and "The company did not fail its audit" abstained.
              "fail", "fails"))
MILD_POS = frozenset(("rise", "rises", "rose", "grow", "grows", "grew", "gain", "gains",
             "gained", "improve", "improved", "recovery", "rebound", "higher",
             "better", "positive", "strong", "boosted", "extends", "stable",
             "approves", "targets", "plans", "invests", "partnership", "strategic"))

# A concessive marker ("despite", "but", "however") puts everything after it in a
# subordinate clause, and the subordinate clause is not the news: "Bank profits
# grow despite SBP rate cuts" is a report about profits growing. Scoring the tail
# at full strength let a rate cut override the result it was background to, and
# the first version of this module tried to fix that by negating it, which read
# the same headline as profits *falling*. Half weight, no flip: the hedge still
# counts, it just does not outvote the clause it hedges.
CONCESSIVE = frozenset(("despite", "but", "however", "although", "though", "yet"))
CONCESSIVE_WEIGHT = 0.5
MILD_NEG = frozenset(("fall", "falls", "fell", "drop", "drops", "dropped", "decline",
            "declines", "declining", "lower", "worse", "negative", "weak",
            "pressure", "concern", "worries", "uncertain", "stalls", "slows",
            "slowdown", "overdue", "criticised", "criticized", "risk"))

# A negator within this many tokens in front of a scored word flips its sign.
# Only logical negations belong here: "despite" is concessive, and flipping the
# verb after it read "profits grow despite the rate cut" as a report about
# profits *falling*, which is the opposite headline.
NEGATORS = frozenset(("no", "not", "never", "without", "lack", "lacks", "fails",
              "failed", "failing", "denies", "denied", "barely", "hardly"))
NEG_WINDOW = 3

# The last scored word of a headline counts double. A title is noun-then-verb —
# "Lucky Cement profit misses estimates" is about a miss, and "ENGRO posts a
# record loss" is about a loss — so the predicate carries the news and the words
# before it carry the subject. Without this, an earnings miss summed to zero and
# a record loss to exactly nothing: the two shapes a scorer must never emit for a
# headline that plainly says bad news.
PREDICATE_WEIGHT = 2.0

# An article's raw lexicon sum is divided by this before clipping. Sized so one
# strong word reads as a firm-but-not-extreme tilt (≈0.6 with the predicate
# weight) and three in the same direction saturates the headline: a single
# "dividend" in a title is a story, not a full-strength signal, and a scorer that
# saturates on one word hands strategy 99 the loudest voice in the poll.
# Declared judgment, the same convention as macro.py's divisors — nothing here
# was fitted to a sample.
ARTICLE_SCALE = 3.2

# Below this many *matching* headlines the desk declines to answer. One headline
# is a story, not a tape.
MIN_ARTICLES = 2
# And below this share of matching headlines carrying any scored word, the score
# is a rounding of nothing.
MIN_SCORED_SHARE = 0.34

DEFAULT_TTL_S = 900.0
DEFAULT_WINDOW_D = 2
_MAX_WORKERS = 4
# A host that refused once costs the cycle nothing for this long. Google News is
# the second-most-reliable thing in this stack after the broker, and on a shared
# residential IP it does refuse.
DOWN_PENALTY_S = 600.0

# The cache survives its process, bounded by this. `predict` reads and never
# fetches, so a fresh CLI or API process could only answer a tone if the daemon
# that fetched it left the answers on disk — but a headline is a statement about
# a moment, and four TTLs is where "the news the desk saw" becomes "something a
# process once believed", so anything older is dropped rather than adopted.
CACHE_MAX_AGE_S = 4 * DEFAULT_TTL_S
# Roughly the whole PSX watchlist twice over; beyond it the oldest go, because
# the file is read at import and there is no reason to carry a year of Tuesdays.
CACHE_MAX_KEYS = 256

_RSS = "https://news.google.com/rss/search?"
SOURCE = "google news"
_STOP = frozenset(("co", "ltd", "plc", "corp", "corporation", "company", "limited", "the",
         "and", "of", "gas", "oil", "bank", "holders", "global", "energy"))
# A word that appears in headlines this instrument has nothing to do with, so it
# can never identify one on its own. Not a sentiment list and not a dictionary:
# it is the residue of the name-parser below, and a word only has to be in it if
# the desk has actually seen it match the wrong story. "development" is here
# because "Oil & Gas Development Co" reduces to it, and a Pakistani financial
# wire says "development" several times an hour.
_GENERIC = frozenset(("development", "state", "pakistan", "india", "china", "american",
              "national", "general", "global", "international", "united", "first",
              "standard", "common", "commercial", "islamic", "royal", "imperial",
              "power", "cement", "textile", "sugar", "fertilizer", "technology",
              "industries", "industry", "holdings", "holding", "investment",
              "investments", "trading", "services", "foods", "tobacco", "automobile",
              "insurance", "chemicals", "materials", "lucky", "best", "new", "top",
              "big", "group", "house", "works", "line", "street", "share", "shares",
              # Surnames that a Pakistani wire puts in almost every headline it
              # publishes about anybody. DGKC's name reduces to "khan" before its
              # phrase does, and a story about a politician would have scored the
              # cement stock's news for it.
              "khan", "sharif", "zardari", "bilawal", "shehbaz", "imran", "rehan",
              # "Punjab" is the whole reduced name of Bank of Punjab ("Bank" and
              # "of" are stop words), and a Pakistani wire says it several times
              # an hour about a province, not a bank: measured on 2026-10-23
              # "Punjab to extend lockdown across the province" claimed a reading
              # for BANK. It leaves that stock to its symbol, the same trade the
              # desk made for DGKC — recall for a name match that cannot be told
              # apart from a provincial headline.
              "punjab"))

# For a group brand whose whole reduced name is one word — ENGRO is "Engro" once
# "Corporation" is a stop word — the registry cannot say which sibling arm a
# headline means, because `psx_registered.json` keeps only the last forty scan
# leaders: measured on 2026-10-23 the universe held ENGRO and EFERT but no EPCL,
# so "Engro Polymer plans capacity" named no rival the desk knew about and was
# filed under the parent. These words name the arms that live under the brand, so
# a headline that goes further than the brand is about one of them, not this
# instrument. Same policy as `_GENERIC`: an entry is added when the desk has
# actually seen the brand claim another company's story, not before.
_SIBLINGS = {
    "engro": ("polymer", "chemicals", "fertilizers", "rexin", "foods",
              "technologies"),
    "fauji": ("cement", "foundation", "foods", "sugar", "mills"),
    "attock": ("petroleum", "refinery", "cheema", "power"),
    "united": ("foods", "insurance", "auto", "dairy"),
    "islamia": ("sugar", "textile", "mills", "technologies"),
    "arabian": ("energy", "oils", "banking"),
    "lucky": ("cement", "core", "velvet", "motors"),
    "serena": ("hotel", "towers", "mills"),
}

# Not news. A quote vendor's static pages — profile, "share price today", a
# briefing-video stub — are crawled and re-dated constantly, so the feed serves
# them as if they were stories published this morning. They carry the company's
# name and a sentence describing what the company does, which the lexicon scores
# as though the business had just been reported favourably, and a page that has
# said the same thing for a year pins an instrument's tone at one value forever.
# Dropping them costs no recall, because none of them ever contained an event.
_REFERENCE = re.compile(
    r"company profile|share price|price today|stock price|briefing video|"
    r"about us|market snapshot|live updates|\bquote\b", re.I)


def _rival_phrases(head: str, keep: tuple[str, ...], key: str) -> tuple[str, ...]:
    """Other instruments that start with this word but are named further than it.

    "Engro" identifies Engro Corporation only while the headline stops there; a
    headline that says "Engro Polymer" is about a different company that shares
    the brand. So the word is allowed to claim a headline, and these phrases are
    what disallow it again.
    """
    out = []
    own = " ".join(keep[:2])
    for other in universe():
        if other.key == key:
            continue
        k = _kept(other.name)
        if k and k[0] == head and tuple(k[:2]) != tuple(keep[:2]):
            cand = " ".join(k[:2])
            # A sibling that is named *less* than this instrument is not its
            # rival: ENGRO reduces to "engro", and putting that in EFERT's veto
            # list meant every "Engro Fertilizers profit rises" headline was
            # rejected as somebody else's, so the fertiliser arm could only ever
            # be scored from a title that printed EFERT in caps. Measured
            # 2026-10-23; the veto list is only ever for a mention longer than
            # the name being matched.
            if cand and cand not in own:
                out.append(cand)
    return tuple(out)


@dataclass
class Headline:
    title: str
    link: str
    source: str
    age_h: float | None      # None when the feed did not print a readable date

    @property
    def readable(self) -> bool:
        # bool(), not the bare expression — see Component.present in macro.py.
        return self.age_h is not None and bool(np.isfinite(self.age_h))


@dataclass
class ToneReading:
    """The desk's answer for one instrument, with the evidence count beside it."""
    key: str
    score: float | None
    matching: int = 0
    scored: int = 0
    window_d: int = DEFAULT_WINDOW_D
    source: str = ""
    age_s: float = 0.0
    reason: str = ""
    top: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.score is not None and bool(np.isfinite(self.score))

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key,
                "score": None if self.score is None else round(float(self.score), 3),
                "available": self.available, "matching": self.matching,
                "scored": self.scored, "window_d": self.window_d,
                "source": self.source, "age_s": round(self.age_s, 1),
                "reason": self.reason, "top": list(self.top)}


def _tier_table() -> dict[str, float]:
    """One lookup for the four tiers, strongest reading first."""
    out: dict[str, float] = {}
    for words, w in ((STRONG_POS, 1.0), (STRONG_NEG, -1.0),
                     (MILD_POS, 0.5), (MILD_NEG, -0.5)):
        for tok in words:
            out.setdefault(tok, w)
    return out


_TIERS = _tier_table()


def _stem_candidates(tok: str) -> tuple[str, ...]:
    """The whole token first, then the base forms it is inflected from."""
    out = [tok]
    if tok.endswith("es") and len(tok) > 4:
        out.append(tok[:-2])
    if tok.endswith("s") and len(tok) > 3:
        out.append(tok[:-1])
    if tok.endswith("ed") and len(tok) > 4:
        # Both tails: "plunged" is plunge minus a d, "launched" is launch minus
        # two, and a rule that tried only one of them lost a word the other one
        # had. Longer candidate first, so the shallower strip wins when both
        # would match.
        out.append(tok[:-1])
        out.append(tok[:-2])
    elif tok.endswith("d") and len(tok) > 3:
        out.append(tok[:-1])
    return tuple(out)


def _word_weight(tok: str) -> float:
    """The lexicon weight of one headline token, inflections included.

    A headline verb is third-person singular or past tense while the lexicon is
    written in base form. Matching only the exact token abstained on "Meezan
    Bank beats estimates and raises payout" — the most bullish sentence in
    finance, because "beats" is not in the list and "beat" is — and it read
    "loan defaults rise" as one scored word where two had fired, which is how a
    plunge in a stock with rising loan defaults summed to nearly nothing. The
    whole token is tried first, so "losses" is a loss and "misses" is a miss;
    a stem is only reached when the token itself means nothing here.
    """
    for cand in _stem_candidates(tok):
        w = _TIERS.get(cand)
        if w is not None:
            return w
    return 0.0


def _weights(text: str) -> tuple[float, int]:
    """The lexicon sum for one headline, and how many scored words fired."""
    toks = re.findall(r"[a-z0-9']+", text.lower())
    # The default is an index past the end, and it used to be -1 — which made
    # every token look like it came *after* a hedge, so a headline with no
    # "despite" in it had every word halved. Half strength on its own is a
    # quieter signal, but the predicate doubling cancelled against it, so
    # "plunge" and "rise" netted out on a headline that plainly said bad news.
    hedge = max((i for i, t in enumerate(toks) if t in CONCESSIVE), default=len(toks))
    fired: list[float] = []
    main_last: int | None = None
    for i, t in enumerate(toks):
        w = _word_weight(t)
        if w == 0.0:
            continue
        if any(n in toks[max(0, i - NEG_WINDOW):i] for n in NEGATORS):
            w = -w
        if i > hedge:
            w *= CONCESSIVE_WEIGHT
        else:
            main_last = len(fired)
        fired.append(w)
    if not fired:
        return 0.0, 0
    # The predicate is the last word of the main clause. Doubled, it is the news
    # of the headline; when the only scored words sit in the subordinate clause
    # after a hedge, there is no main-clause verb to carry it and the tail takes
    # the weight anyway, at its half value.
    pred = len(fired) - 1 if main_last is None else main_last
    fired[pred] *= PREDICATE_WEIGHT
    return float(sum(fired)), len(fired)


def tone_of(text: str) -> float | None:
    """One headline's tilt in [-1, 1], or None when no scored word appears.

    None rather than 0.0 is the whole point: a headline nobody scored is not a
    headline that said "neutral". A headline that sums to exactly 0.0 *is* that
    claim — two fired words pulled opposite ways — and is answered, not abstained.
    """
    total, fired = _weights(text)
    if not fired:
        return None
    return float(np.clip(total / ARTICLE_SCALE, -1.0, 1.0))


@lru_cache(maxsize=4096)
def _kept(name: str) -> tuple[str, ...]:
    """The distinctive words of an instrument name, in order."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name.lower()) if len(w) > 2]
    return tuple(w for w in words if w not in _STOP)


_CAPS = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,7})(?![A-Za-z0-9])")
_KEYS: tuple[frozenset, int] = (frozenset(), -1)


def _ticker_keys() -> frozenset:
    """Every instrument symbol, for the "does this headline name someone else" test.

    Rebuilt when the universe grows, because the PSX scan registers new symbols
    while the process runs. A false positive here (a headline excluded because it
    contains some other stock's caps acronym) only costs a headline, so the set is
    the whole registry and no effort is spent disambiguating it.
    """
    global _KEYS
    seen, n = _KEYS
    all_keys = frozenset(i.key for i in universe())
    if len(all_keys) != n:
        _KEYS = (all_keys, len(all_keys))
    return _KEYS[0]


def _names_other_stock(key: str, title: str) -> bool:
    """Does the headline print some other instrument's symbol?

    The name-based match is the weak test, and the wires that publish these
    headlines put the *symbol* in the title: "EPCL Company Profile - Engro Polymer
    and Chemicals Ltd" is a page about EPCL, and a matcher that only knew the word
    "Engro" filed it under ENGRO — a wrong number that looks plausible, in the one
    direction that moves a position. This runs only when the headline did not
    contain our own symbol, so an ordinary "HBL, UBL top gainers" title has
    already been claimed by both through the ticker test above.
    """
    return any(t in _ticker_keys() and t != key for t in _CAPS.findall(title))


@dataclass(frozen=True)
class _Identity:
    """The compiled test for "this headline is about that instrument"."""
    ticker: re.Pattern
    phrase: str
    word: re.Pattern | None
    rivals: tuple[str, ...] = ()

    def matches(self, inst_key: str, title: str) -> bool:
        if self.ticker.search(title):
            return True                       # our symbol printed: unambiguously ours
        low = title.lower()
        hit = bool(self.phrase and self.phrase in low) or (
            self.word is not None and self.word.search(low) is not None)
        if not hit:
            return False
        if any(r in low for r in self.rivals):
            return False                      # a sibling that the headline named further
        return not _names_other_stock(inst_key, title)


def _identity(inst: Instrument) -> _Identity:
    """Build the instrument's headline test. Not cached: the PSX universe grows
    while the process runs, and a memoised sibling check would keep answering
    with the registry as it was when the symbol first appeared."""
    keep = _kept(inst.name)
    phrase = " ".join(keep[:2]) if len(keep) >= 2 else ""
    head = keep[0] if keep else ""
    # The single word is what makes "Engro to sell stake" count — a publisher sets
    # a company name in title case, so a caps-only ticker test misses the story it
    # is named in. Three things stop that shortcut from claiming a headline for
    # the wrong stock: `_GENERIC` ("development" is all that survives from Oil &
    # Gas Development Co, and "khan" is all that survives from a politician), the
    # four-character floor, and `_rival_phrases`. Known imprecision, kept on
    # purpose: a group story that names the group without naming the listed arm
    # counts to the arm, and a matcher that split them would be claiming a
    # precision it does not have.
    rivals = _rival_phrases(head, keep, inst.key) if head else ()
    # A brand's own arms are named further than the brand, and `matches` reads
    # these as phrases for the same reason the matcher does: "engro polymer" as a
    # contiguous pair rejects the sibling's story without rejecting an ordinary
    # ENGRO headline that happens to use the word "chemicals" somewhere. Words
    # already in this instrument's name are skipped, or a stock would veto the
    # headlines that identify it — "engro fertilizers" is precisely what EFERT is
    # called, and it is also on the Engro family list.
    own = " ".join(keep)
    declared = tuple(f"{head} {w}" for w in _SIBLINGS.get(head, ())
                     if head and f"{head} {w}" not in own)
    rivals = tuple(dict.fromkeys(rivals + declared))
    word = None
    if (len(head) >= 4 and head not in _GENERIC
            and keep
            and not any(_kept(o.name) == keep
                        for o in universe() if o.key != inst.key)):
        word = re.compile(rf"(?<![a-z0-9]){re.escape(head)}(?![a-z0-9])")
    return _Identity(ticker=re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(inst.key)}(?![A-Za-z0-9])"),
        phrase=phrase, word=word, rivals=rivals)


def _matches(inst: Instrument, title: str) -> bool:
    """Does this headline speak about this instrument?

    The ticker is matched case-sensitively because a four-letter ticker is often
    also an English word — "LUCK" sits in a hundred headlines about luck, and
    charging their tone to Lucky Cement is a wrong number that looks plausible —
    while a financial page that means the stock prints its symbol.
    """
    return _identity(inst).matches(inst.key, title)


def build_query(inst: Instrument, window_d: int) -> str:
    q = f"{inst.name.split('(')[0].strip()} {inst.key} stock"
    if inst.market == "PSX":
        q += " Pakistan"
    return q + f" when:{max(1, int(window_d))}d"


def _parse_date(raw: str) -> float | None:
    """Age in hours, or None. An unreadable date is dropped, never read as fresh:
    `_age_days` once answered 0.0 for a stamp it could not parse and both callers
    took the invention, which stopped PSX rotation for the life of the process."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        try:
            dt = datetime.fromisoformat(raw[:25])
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


class NewsDesk:
    """Per-instrument headline tone, cached, refreshed in parallel."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, ToneReading]] = {}
        self._down_until = 0.0
        self.path = cache_path or (CFG.data_dir / "news_desk.json")
        self.load_error: str | None = None
        self.save_error: str | None = None
        self._load()

    # -- the cache outlives the process -------------------------------------
    def _load(self) -> None:
        """Adopt the readings another process already paid for.

        `predict` reads this cache and never fetches, so without a file behind it
        every CLI call, every API process and every restart answers a tone with
        `not yet fetched` while the daemon that did the fetching is sitting on a
        full desk. The stored timestamp comes with the reading, so an adopted
        answer still says how old it is; `CACHE_MAX_AGE_S` is what stops a
        three-day-old headline being delivered as news.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:
            # Recorded, not raised: a desk that cannot read its cache is worse
            # by one answer per symbol, not dead — and worse than that, an
            # unreadable file that is silently ignored looks like an empty desk,
            # which is a different claim about the market than a broken reader.
            self.load_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return
        now = time.time()
        with self._lock:
            for key, item in (raw.get("readings") or {}).items():
                try:
                    ts, fields = float(item[0]), dict(item[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if now - ts > CACHE_MAX_AGE_S:
                    continue
                self._cache[str(key).upper()] = (ts, ToneReading(**fields))

    def _save(self) -> None:
        """Write the whole desk atomically after a refresh that answered."""
        with self._lock:
            items = {k: [t, dict(r.__dict__)] for k, (t, r) in self._cache.items()}
        if len(items) > CACHE_MAX_KEYS:
            items = dict(sorted(items.items(), key=lambda kv: -kv[1][0])
                         [:CACHE_MAX_KEYS])
        try:
            atomic_write(self.path, dumps_strict({"saved": time.time(),
                                                  "readings": items}))
            self.save_error = None
        except Exception as exc:
            self.save_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    # -- config ------------------------------------------------------------
    def _cfg(self, name: str, default):
        return CFG.section("newsdesk").get(name, default)

    @property
    def paused(self) -> bool:
        """True while the desk is refusing to call the host again after a failure.

        Kept separate from an empty cache on purpose: "nothing in the tape" and
        "we stopped asking because the host would not answer" reach the operator
        as the same absent tone, and only one of them is a fault.
        """
        return time.time() < self._down_until

    def holdings(self) -> list[dict[str, Any]]:
        """Every cached reading, best tone first, asking the network nothing.

        Sorted, because the question an operator arrives with is "what does the
        news say is the most bullish thing I am not holding", and the answer is
        the same rows either way — the ordering is the whole reason to look.
        """
        with self._lock:
            items = [(t, r) for t, r in self._cache.values()]
        now = time.time()
        rows = [ToneReading(**{**r.__dict__, "age_s": now - t}).as_dict()
                for t, r in items]
        rows.sort(key=lambda d: (d["score"] is None, -(d["score"] or 0.0)))
        return rows

    def status(self) -> dict[str, Any]:
        """What the desk holds, asking the network nothing."""
        with self._lock:
            items = list(self._cache.values())
        now = time.time()
        return {"enabled": self.enabled, "paused": self.paused,
                "source": SOURCE, "ttl_s": self.ttl_s, "window_d": self.window_d,
                "timeout_s": self.timeout_s, "held": len(items),
                "available": sum(1 for _t, r in items if r.available),
                "oldest_s": round(now - min((t for t, _r in items), default=now), 1),
                "cache_file": str(self.path.name),
                "load_error": self.load_error, "save_error": self.save_error}

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("enabled", True))

    @property
    def window_d(self) -> int:
        return int(self._cfg("window_d", DEFAULT_WINDOW_D))

    @property
    def ttl_s(self) -> float:
        return float(self._cfg("ttl_s", DEFAULT_TTL_S))

    @property
    def timeout_s(self) -> int:
        # Measured on 2026-10-23: one query takes 3.7-5.1s alone and materially
        # longer when four run at once on this machine's single uplink. A timeout
        # under that is not tight, it is a coin flip, and a coin flip here costs
        # the symbol its tone for ten minutes because the host got paused.
        return int(self._cfg("timeout_s", 9))

    # -- read path: never blocks on the network ------------------------------
    def cached(self, instrument: Instrument) -> ToneReading | None:
        with self._lock:
            hit = self._cache.get(instrument.key.upper())
        if hit is None:
            return None
        ts, reading = hit
        fresh = ToneReading(**{**reading.__dict__, "age_s": time.time() - ts})
        return fresh

    def tone(self, instrument: Instrument, allow_fetch: bool = False) -> ToneReading:
        """The last answer for this instrument.

        `allow_fetch` puts a network call on the caller's thread, so it is off by
        default: the prediction path reads the cache, and the batch refresh that
        fills it runs where a few seconds does not matter. A symbol nobody has
        ever asked about gets an explicit `never fetched`, not a zero.
        """
        if not self.enabled:
            return ToneReading(key=instrument.key, score=None, reason="desk disabled")
        hit = self.cached(instrument)
        if hit is not None:
            if hit.age_s <= self.ttl_s:
                return hit
            if not allow_fetch:
                # Adopted from disk, or simply not due for a refetch: say so in
                # the reason, because an out-of-date tone reaching a strategy is
                # a different fact from a fresh one and the desk row is the only
                # place anybody sees it.
                hit.reason = (f"stale by {int(hit.age_s - self.ttl_s)}s; "
                              f"{hit.reason}")
                return hit
        if not allow_fetch:
            return hit or ToneReading(key=instrument.key, score=None,
                                      reason="not yet fetched")
        return self.refresh([instrument]).get(
            instrument.key.upper(),
            ToneReading(key=instrument.key, score=None, reason="fetch failed"))

    # -- write path ---------------------------------------------------------
    def refresh(self, instruments: list[Instrument],
                force: bool = False) -> dict[str, ToneReading]:
        """Fetch a batch in parallel and put every answer in the cache.

        Each query is its own host round trip on its own socket, so a watchlist
        costs one fetch, not one per symbol — the same arithmetic that cut the
        intermarket panel from ~16s to ~2s. Assembly walks the list in the order
        it was given, so the cached value for a symbol is a deterministic
        function of the responses, not of which thread won.

        A symbol whose reading is still inside `ttl_s` is answered from the
        cache and not re-queried: the autopilot runs a cycle every 300s while the
        desk's own freshness window is 900s, so without this the same twelve
        queries would be paid for twice as often as the answer can change, and
        the host would see a twelve-symbol burst every five minutes forever.
        `force` is for the operator who wants the tape now.
        """
        out: dict[str, ToneReading] = {}
        if not self.enabled or not instruments:
            return out
        if time.time() < self._down_until:
            left = int(self._down_until - time.time())
            now = time.time()
            with self._lock:
                stale = dict(self._cache)
            for inst in instruments:
                hit = stale.get(inst.key.upper())
                if hit is None:
                    out[inst.key.upper()] = ToneReading(
                        key=inst.key, score=None,
                        reason=f"news host paused after a failure, {left}s left")
                    continue
                # The cached answer travels, because refusing to ask is not the
                # same as knowing nothing — but the pause is named in front of
                # the reason, so a tone that stopped being refreshed while the
                # host was down cannot read as a fresh one. The cache stores
                # `(timestamp, reading)`; handing that tuple back is what a
                # caller of `refresh` would then try to read as a tone.
                out[inst.key.upper()] = ToneReading(
                    **{**hit[1].__dict__, "age_s": now - hit[0],
                       "reason": f"host paused, {left}s left; {hit[1].reason}"})
            return out

        ttl = self.ttl_s
        due = list(instruments)
        if not force and ttl > 0:
            now = time.time()
            with self._lock:
                held = dict(self._cache)
            due = []
            for inst in instruments:
                hit = held.get(inst.key.upper())
                if hit is not None and now - hit[0] <= ttl:
                    ts, reading = hit
                    out[inst.key.upper()] = ToneReading(
                        **{**reading.__dict__, "age_s": now - ts})
                else:
                    due.append(inst)
        if not due:
            return out

        workers = max(1, min(int(self._cfg("max_workers", _MAX_WORKERS)),
                             len(due)))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="newsdesk") as ex:
            results = list(ex.map(self._read, due))

        now = time.time()
        with self._lock:
            for reading in results:
                if reading is not None:
                    self._cache[reading.key.upper()] = (now, reading)
                    out[reading.key.upper()] = reading
        for inst in instruments:
            out.setdefault(inst.key.upper(), ToneReading(
                key=inst.key, score=None, reason="fetch failed"))
        if results:
            self._save()
        return out

    def _read(self, inst: Instrument) -> ToneReading | None:
        """One instrument, in a worker thread.

        Nothing in here may escape as an exception: `refresh` maps over a batch,
        and a raise from the tenth symbol's scoring would discard the nine that
        already answered. Nor may it be swallowed into a silent abstention, which
        is how a broken component gets mistaken for a quiet market — so the
        exception's own name and message become the reading's reason.
        """
        base = dict(key=inst.key, window_d=self.window_d, source=SOURCE)
        heads, fault = self._headlines(inst)
        if fault:
            self._down_until = max(self._down_until, time.time() + DOWN_PENALTY_S)
            return ToneReading(**base, score=None,
                               reason=f"news host did not answer ({fault})")
        try:
            who = _identity(inst)
            noise = sum(1 for h in heads if _REFERENCE.search(h.title))
            heads = [h for h in heads if not _REFERENCE.search(h.title)]
            rel = [h for h in heads if who.matches(inst.key, h.title)]
            if not heads:
                return ToneReading(**base, score=None,
                                   reason=(f"host answered with {noise} reference "
                                           f"pages and no news" if noise else
                                           "host answered with no headlines"))
            if not rel:
                return ToneReading(**base, score=None, matching=0,
                                   reason=f"{len(heads)} headlines, none mention it")
            fresh = [h for h in rel if h.readable and h.age_h is not None
                     and h.age_h <= self.window_d * 24.0]
            if not fresh:
                return ToneReading(**base, score=None, matching=len(rel),
                                   reason=f"{len(rel) - len(fresh)} undated or "
                                          f"outside the window")
            reading = self._score(base, fresh, len(rel) - len(fresh))
            if noise:
                reading.reason += f"; {noise} reference pages skipped"
            return reading
        except Exception as exc:
            return ToneReading(**base, score=None,
                               reason=f"desk fault: {type(exc).__name__}: {exc}"[:160])

    def _score(self, base: dict, heads: list[Headline], dropped: int) -> ToneReading:
        scored = [(tone_of(h.title), h) for h in heads]
        used = [(s, h) for s, h in scored if s is not None]
        matching = len(heads)
        if not used:
            return ToneReading(**base, score=None, matching=matching, scored=0,
                               reason=f"{matching} headlines, no scored word")
        share = len(used) / matching
        if share < MIN_SCORED_SHARE:
            return ToneReading(**base, score=None, matching=matching,
                               scored=len(used),
                               reason=f"only {len(used)} of {matching} carry a "
                                      f"scored word")
        if len(used) < MIN_ARTICLES:
            return ToneReading(**base, score=None, matching=matching,
                               scored=len(used),
                               reason=f"{len(used)} scored headline, needs "
                                      f"{MIN_ARTICLES}")
        # Recency-weighted mean: a story from this morning and one from two days
        # ago are not equally about now, and an equal mean let the older one hold
        # the verdict. Weight is linear in remaining window, and every headline
        # here already has a readable date.
        span = float(self.window_d * 24.0)
        wts = np.array([max(0.05, 1.0 - (h.age_h or 0.0) / span) for _s, h in used])
        vals = np.array([s for s, _h in used], dtype=float)
        score = float(np.clip(np.average(vals, weights=wts), -1.0, 1.0))
        order = sorted(used, key=lambda x: -abs(x[0]))
        return ToneReading(**base, score=round(score, 4), matching=matching,
                           scored=len(used),
                           reason=(f"{len(used)}/{matching} scored, {dropped} out of "
                                   f"window, mean {score:+.2f}"),
                           top=tuple(h.title[:110] for _s, h in order[:3]))

    # -- network -----------------------------------------------------------
    def _headlines(self, inst: Instrument) -> tuple[list[Headline], str]:
        """(items, fault). `fault` names the exception when the request itself
        failed, and is "" when the host answered and had nothing to say.

        The two have to stay separate all the way to the reason string: a timeout
        reported as "no headlines" tells an operator the tape is quiet, which is
        the same class of lie as an outage reported as a flat account.
        """
        url = _RSS + urllib.parse.urlencode(
            {"q": build_query(inst, self.window_d), "hl": "en-US", "gl": "US",
             "ceid": "US:en"})
        try:
            raw = _http_get(url, timeout=self.timeout_s, retries=0,
                            max_bytes=FEED_MAX_BYTES)
        except Exception as exc:
            return [], type(exc).__name__
        return _parse_rss(raw), ""


def _parse_rss(raw: bytes) -> list[Headline]:
    root = _feed_xml(raw)
    if root is None:
        return []
    out: list[Headline] = []
    for el in root.iter("item"):
        title = re.sub(r"\s+", " ", (el.findtext("title") or "").strip())
        if not title:
            continue
        src = ""
        tag = el.find("source")
        if tag is not None:
            src = (tag.text or "").strip()
        out.append(Headline(title=title[:300], link=(el.findtext("link") or "").strip(),
                            source=src,
                            age_h=_parse_date(el.findtext("pubDate") or "")))
    return out


# ---------------------------------------------------------------------------
# GDELT DOC 2.0 — reachable where the IP is not rate-limited, silent where it is
# ---------------------------------------------------------------------------
# Measured from this machine on 2026-10-23: every `api.gdeltproject.org` request
# answered HTTP 429, on artlist and tonechart alike, while `data.gdeltproject.org`
# (the raw 15-minute drops) answered in 6s. A deploy on a cloud host is not
# blocked; a residential laptop is. So this is wired to be *used when it answers*
# and to cost nothing when it does not — which is the honest state, rather than a
# table claiming a news feed the process has never once read.
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_PAUSE_S = 1800.0


class GdeltTone:
    """Tonechart aggregate for one query, with a penalty box on refusal."""

    def __init__(self, get=None) -> None:
        # Resolved at call time, not bound here: `GDELT` is a module singleton
        # built at import, so a default of `_http_get` would freeze the function
        # object and every test that swaps the transport would be talking to
        # Google and the internet.
        self._get = get
        self._down_until = 0.0
        self._cache: dict[str, tuple[float, float | None]] = {}
        self.blocked = 0

    def tone(self, query: str, window_d: int = 2,
             ttl_s: float = 900.0) -> float | None:
        key = f"{query}|{window_d}"
        ts, hit = self._cache.get(key, (0.0, None))
        if time.time() - ts < ttl_s and hit is not None:
            return hit
        if time.time() < self._down_until:
            return None
        url = GDELT_DOC + "?" + urllib.parse.urlencode(
            {"query": query, "mode": "tonechart", "format": "json",
             "timespan": f"{max(1, int(window_d))}d"})
        fetch = self._get or _http_get
        try:
            payload = json.loads(fetch(url, timeout=8, retries=0))
        except (HttpStatus, urllib.error.HTTPError) as exc:
            # `close_error` because the second type in that tuple is the raw
            # response object: whatever produced it may have handed it over
            # without putting its socket down.
            close_error(exc)
            if exc.code == 429:
                self.blocked += 1
                self._down_until = time.time() + GDELT_PAUSE_S
            return None
        except Exception:
            return None
        score = _tonechart_mean(payload)
        self._cache[key] = (time.time(), score)
        return score


def _tonechart_mean(payload: Any) -> float | None:
    """Mean GDELT tone over the chart's buckets, or None when there is no chart.

    GDELT's own tone is centred on 0 with ordinary articles near 0 and anything
    newsworthy several points either way, so the raw mean is divided by
    `ARTICLE_SCALE`-like constant below rather than squashed by an invented
    percentile.
    """
    chart = payload.get("timeline", {}).get("chart", []) if isinstance(payload, dict) else []
    vals = [float(row["tone"]) for row in chart
            if isinstance(row, dict) and isinstance(row.get("tone"), (int, float))]
    if not vals:
        return None
    return float(np.clip(np.mean(vals) / GDELT_TONE_SCALE, -1.0, 1.0))


# A single news day rarely moves the aggregate tone past ±2; ±4 is a market event.
GDELT_TONE_SCALE = 4.0

DESK = NewsDesk()
GDELT = GdeltTone()
