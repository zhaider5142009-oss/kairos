"""The news desk — scored, matched, cached, and never asked to invent.

Strategy 99, "News Sentiment Tilt", has never once voted in the life of this
codebase: the only thing that filled `news_sentiment` was an LLM call behind a
config flag that was off, so the news axis was an absent print in every regime
on every instrument. `kairos/news.py` is the floor under it — a keyless headline
tone that a 300 s autopilot cycle can afford to read and a 9.7 s prediction
thread cannot.

What these tests hold shut is mostly the same shape as `test_macro.py`: a state
that could be reported as something it is not.

1. An unscored headline is not a neutral one. `tone_of` answers None, and a
   reading that averaged a None in with its opinions would read "nothing was
   said" as "the tape said nothing either way".
2. A name match that claims another company's story. ENGRO is "Engro" the way
   EPCL is "Engro Polymer", BANK is "Punjab" the way the province is Punjab,
   and a wrong tone that looks plausible is the one direction that moves a
   position.
3. A transport failure reported as a quiet market, and one bad symbol allowed
   to void the nine that answered.
4. A cache that outlives its process: the read path must never fetch, so a
   fresh CLI process can only answer from the file the daemon left — and an
   adopted answer must say it is adopted, and a three-day-old one must be
   refused rather than believed.
5. The scorer's own arithmetic, which was wrong twice over before it was ever
   run against a real headline: a hedge sentinel of -1 halved every word of
   every headline that contained no hedge at all, and the lexicon was written
   in base forms while headlines are written in third person and past tense, so
   "beats estimates" abstained and "plunge … defaults rise" cancelled to zero.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from kairos import engine as E
from kairos import news as N
from kairos import strategies as S
from kairos.config import CFG
from kairos.universe import resolve


# ---------------------------------------------------------------------------
# fixtures: a desk with a transport that is not the internet
# ---------------------------------------------------------------------------
def _item(title: str, age_h: float | None = 2.0, source: str = "Business Today") -> str:
    date = ("" if age_h is None else
            f"<pubDate>{format_datetime(datetime.now(timezone.utc) - timedelta(hours=age_h))}"
            f"</pubDate>")
    safe = title.replace("&", "&amp;").replace("<", "&lt;")
    return (f"<item><title>{safe}</title><link>https://example.invalid/"
            f"a{abs(hash(title)) % 1000}</link><source>{source}</source>{date}</item>")


def _feed(*items: str) -> bytes:
    return (f'<rss version="2.0"><channel>{"".join(items)}</channel></rss>'
            ).encode("utf-8")


class _Transport:
    """`_http_get` stand-in: serves one canned feed per symbol, counts the asks."""

    def __init__(self, pages: dict[str, bytes | Exception]) -> None:
        self.pages = pages
        self.calls: dict[str, list[str]] = {k: [] for k in pages}
        self.unknown: list[str] = []

    def __call__(self, url: str, timeout=None, retries=None) -> bytes:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("q", [""])[0]
        toks = q.split()
        # build_query is always "<name> <KEY> stock …", so the key is the token
        # before the word "stock" — matching on a substring would let ENGRO's
        # page answer for EFERT, whose query also says "Engro".
        key = toks[toks.index("stock") - 1] if "stock" in toks else ""
        for k, payload in self.pages.items():
            if k.upper() != key.upper():
                continue
            self.calls[k].append(url)
            if isinstance(payload, BaseException):
                raise payload
            return payload
        self.unknown.append(url)
        return _feed()


@pytest.fixture
def desk(tmp_path):
    return N.NewsDesk(cache_path=tmp_path / "news_desk.json")


def _patch(monkeypatch, transport: _Transport) -> _Transport:
    monkeypatch.setattr(N, "_http_get", transport)
    return transport


HBL_GOOD = _feed(
    _item("HBL reports record profit and declares dividend"),
    _item("HBL shares surge on better results", 5.0),
    _item("HBL board approves a bonus issue", 20.0),
)


# ---------------------------------------------------------------------------
# 1. the scorer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("title,want", [
    ("HBL reports record profit and declares Rs8 dividend", +1),
    ("Meezan Bank beats estimates and raises payout", +1),
    ("Lucky Cement profit misses estimates", -1),
    ("ENGRO posts a record loss", -1),
    ("MEBL shares plunge as loan defaults rise", -1),
    ("Bank profits grow despite rate cuts", +1),
    ("Output expanded despite the losses", +1),
    ("Gold falls short as demand weakens", -1),
    ("Arbitrum drops after token unlock", -1),
    ("Cosmos gains on staking upgrade", +1),
    ("Systems wins new contract as revenue climbs", +1),
    ("The company did not fail its audit", +1),
    ("Company fails its audit and names a new auditor", -1),
    ("The meeting was held in Lahore", None),
])
def test_polarity_of_one_headline(title, want):
    """Every sentence a reader would call good or bad news carries that sign."""
    score = N.tone_of(title)
    if want is None:
        assert score is None
        return
    assert score is not None, f"{title!r} abstained"
    assert (score > 0) if want > 0 else (score < 0), f"{title!r} -> {score}"


def test_an_unscored_headline_abstains_rather_than_answering_zero():
    total, fired = N._weights("The annual general meeting was announced")
    assert (total, fired) == (0.0, 0)
    assert N.tone_of("The annual general meeting was announced") is None


def test_a_headline_with_no_hedge_is_not_scored_as_a_hedged_one():
    """The regression the sentinel caused.

    `hedge` defaulted to -1, and every token index is greater than -1, so the
    concessive half-weight applied to *every* word of *every* headline that
    contained no "despite" — and because the predicate doubled the last word, a
    "plunge … rise" headline cancelled to almost nothing instead of going down.
    """
    unhedged = N.tone_of("ENGRO shares plunge after the plant shutdown")
    hedged = N.tone_of("ENGRO shares plunge despite the plant upgrade")
    assert unhedged is not None and hedged is not None
    assert unhedged <= -0.6, f"single-word bearish read was {unhedged}"
    assert abs(hedged) < abs(unhedged)


def test_the_hedge_does_not_outvote_the_clause_it_hedges():
    """The documented reason the tail is halved and not flipped."""
    assert N.tone_of("Bank profits grow despite SBP rate cuts") > 0.0


def test_inflected_verbs_reach_the_base_form():
    """A lexicon written in base forms must still read real headlines."""
    for title, want in (("Meezan Bank beats estimates", +1),
                        ("ENGRO expanded output at the plant", +1),
                        ("Gold price plunged on the day", -1),
                        ("MEBL shares fell as defaults mounted", -1)):
        score = N.tone_of(title)
        assert score is not None, title
        assert (score > 0) if want > 0 else (score < 0), f"{title!r} -> {score}"


def test_a_negator_in_front_of_a_scored_word_flips_it():
    assert N.tone_of("The company did not fail its audit") > 0
    assert N.tone_of("The company failed its audit") < 0


# ---------------------------------------------------------------------------
# 2. who a headline is about
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,title,claim", [
    # A sibling named further than the brand is a different company.
    ("ENGRO", "Engro Polymer plans capacity expansion", False),
    ("ENGRO", "EPCL Company Profile - Engro Polymer and Chemicals Ltd", False),
    ("ENGRO", "Engro to sell its stake in a subsidiary", True),
    ("ENGRO", "ENGRO shares slide on selling news", True),
    # …and the veto list must not eat the arm's own name.
    ("EFERT", "Engro Fertilizers profit rises", True),
    ("EFERT", "Engro Polymer slips on weak demand", False),
    ("FFC", "Fauji Fertilizer beats estimates", True),
    ("FFC", "Fauji Cement commissioning is delayed", False),
    ("FCCL", "Fauji Cement starts exports", True),
    ("FCCL", "Fauji Fertilizer raises payout", False),
    # A province is not a bank, and a politician is not a cement company.
    ("BANK", "Punjab to extend lockdown across the province", False),
    ("BANK", "BANK shares top gainers on the PSX", True),
    ("DGKC", "Imran Khan addresses the nation", False),
    ("DGKC", "D.G. Khan Cement exports surge", True),
    # Lowercase ticker-shaped words do not count as tickers.
    ("LUCK", "A stroke of luck for the fund", False),
    ("LUCK", "Lucky Cement profit misses estimates", True),
    # Commodities match the words a wire actually uses.
    ("BTC", "Bitcoin cash registers says", False),
    ("BTC", "Bitcoin rallies as ETF inflows return", True),
    ("GOLD", "Gold rate suffers biggest weekly loss since June", True),
    ("MEBL", "Meezan Bank beats estimates and raises payout", True),
    # A bare word two companies reduce to belongs to neither. "Bank AL Habib"
    # and "Habib Bank Ltd." both shrink to ("habib",) — "bank" is a stop word
    # and "al" is two letters — so the matcher refuses the shortcut for both and
    # each is scored only from its own symbol. Recall traded for a tone that is
    # not somebody else's, the same trade `_GENERIC` makes for BANK and DGKC.
    ("HBL", "Habib Bank shares surge on better results", False),
    ("HBL", "HBL shares top gainers", True),
    ("BAHL", "Habib Bank profit rises", False),
])
def test_a_headline_is_claimed_only_for_the_instrument_it_names(key, title, claim):
    inst = resolve(key)
    assert inst is not None, f"{key} missing from the universe"
    assert N._matches(inst, title) is claim, f"{key} vs {title!r}"


def test_a_reference_page_is_not_news():
    """A vendor's static profile carries the company name and a sentence about
    what the company does, which the lexicon scores as a report."""
    assert N._REFERENCE.search("HBL Company Profile - share price today")
    assert not N._REFERENCE.search("HBL reports record profit")


def test_the_query_asks_for_the_window_and_the_market():
    psx = resolve("HBL")
    glob = resolve("GOLD")
    q = N.build_query(psx, 2)
    assert "when:2d" in q and "Pakistan" in q and psx.key in q
    assert "Pakistan" not in N.build_query(glob, 1)
    assert "when:1d" in N.build_query(glob, 0)      # floored, never when:0d


# ---------------------------------------------------------------------------
# 3. the read path never touches the network
# ---------------------------------------------------------------------------
def test_read_path_never_fetches(desk, monkeypatch):
    t = _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    reading = desk.tone(resolve("HBL"))
    assert reading.available is False
    assert reading.reason == "not yet fetched"
    assert t.calls["HBL"] == []


def test_a_disabled_desk_says_it_is_disabled(desk, monkeypatch):
    monkeypatch.setattr(N.CFG, "section", lambda name: {"enabled": False}
                        if name == "newsdesk" else {})
    reading = desk.tone(resolve("HBL"), allow_fetch=True)
    assert reading.reason == "desk disabled"


# ---------------------------------------------------------------------------
# 4. the write path, the cache and the second process
# ---------------------------------------------------------------------------
def test_refresh_answers_and_writes_the_file(desk, monkeypatch, tmp_path):
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    out = desk.refresh([resolve("HBL")])
    reading = out["HBL"]
    assert reading.available and reading.score > 0
    assert reading.matching == 3 and reading.scored == 3
    assert reading.top, "a tone with no headlines attached is unauditable"
    saved = json.loads((tmp_path / "news_desk.json").read_text(encoding="utf-8"))
    assert saved["readings"]["HBL"][1]["score"] == pytest.approx(reading.score)


def test_a_second_process_reads_without_fetching(tmp_path, monkeypatch):
    """`predict` runs in a process that has never asked the host anything."""
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    warm = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    warm.refresh([resolve("HBL")])

    cold = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    reading = cold.tone(resolve("HBL"))
    assert reading.available and reading.score > 0
    assert reading.age_s < 5.0
    assert "not yet fetched" not in reading.reason


def test_ttl_suppresses_the_next_ask_and_force_does_not(desk, monkeypatch):
    t = _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    desk.refresh([resolve("HBL")])
    again = desk.refresh([resolve("HBL")])
    assert len(t.calls["HBL"]) == 1, "a 300s cycle must not re-ask a 900s answer"
    assert again["HBL"].available
    desk.refresh([resolve("HBL")], force=True)
    assert len(t.calls["HBL"]) == 2


def test_an_adopted_stale_reading_says_it_is_stale(tmp_path, monkeypatch):
    ttl = N.DEFAULT_TTL_S
    _write_cache(tmp_path, "HBL", score=0.4, age_s=ttl + 60)
    t = _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    reading = d.tone(resolve("HBL"))
    assert reading.score == pytest.approx(0.4)
    assert reading.reason.startswith("stale by")
    assert t.calls["HBL"] == []
    # …and the same reading, asked with permission to fetch, is refetched.
    fresh = d.tone(resolve("HBL"), allow_fetch=True)
    assert len(t.calls["HBL"]) == 1 and fresh.score != 0.4


def test_a_cache_past_its_max_age_is_refused_not_believed(tmp_path, monkeypatch):
    _write_cache(tmp_path, "HBL", score=0.4, age_s=N.CACHE_MAX_AGE_S + 60)
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    assert d.tone(resolve("HBL")).reason == "not yet fetched"


def test_an_unreadable_cache_is_recorded_not_swallowed(tmp_path, monkeypatch):
    (tmp_path / "news_desk.json").write_text("{not json at all", encoding="utf-8")
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    assert d.load_error and "JSON" in d.load_error.upper()
    # A broken reader is worse by one answer per symbol, not dead.
    assert d.tone(resolve("HBL")).reason == "not yet fetched"


def _write_cache(tmp_path, key: str, score: float, age_s: float) -> None:
    reading = N.ToneReading(key=key, score=score, matching=3, scored=3,
                            source=N.SOURCE, reason="3/3 scored")
    payload = {"saved": time.time() - age_s,
               "readings": {key: [time.time() - age_s, dict(reading.__dict__)]}}
    (tmp_path / "news_desk.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. failure that can be told apart from quiet
# ---------------------------------------------------------------------------
def test_an_outage_names_the_exception_and_pauses_the_desk(desk, monkeypatch):
    t = _patch(monkeypatch, _Transport({"HBL": TimeoutError("read timed out")}))
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert "TimeoutError" in reading.reason, reading.reason
    assert desk.paused is True
    # While paused the host is not dialled again — for the second symbol of a
    # watchlist either, which is the whole point of the penalty box.
    out = desk.refresh([resolve("HBL"), resolve("MEBL")])
    assert len(t.calls["HBL"]) == 1
    assert "paused" in out["HBL"].reason and "paused" in out["MEBL"].reason


def test_one_refusing_symbol_cannot_void_the_batch(desk, monkeypatch):
    t = _patch(monkeypatch, _Transport({"HBL": HBL_GOOD,
                                       "MEBL": ConnectionError("refused")}))
    out = desk.refresh([resolve("HBL"), resolve("MEBL")])
    assert out["HBL"].available, "one bad symbol discarded the nine that answered"
    assert "ConnectionError" in out["MEBL"].reason
    assert len(t.calls["MEBL"]) == 1


def test_a_host_that_answers_with_reference_pages_only_says_so(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": _feed(
        _item("HBL Company Profile - share price today"),
        _item("HBL stock price live updates"))}))
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert "reference pages and no news" in reading.reason


def test_headlines_that_never_mention_it_are_counted_not_scored(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"ENGRO": _feed(
        _item("Engro Polymer plans capacity expansion"),
        _item("Engro Foods seeks a listing"))}))
    reading = desk.refresh([resolve("ENGRO")])["ENGRO"]
    assert reading.available is False
    assert reading.reason == "2 headlines, none mention it"


def test_one_scored_headline_is_a_story_not_a_tape(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": _feed(
        _item("HBL reports record profit"),
        _item("The meeting was held in Lahore"))}))
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert f"needs {N.MIN_ARTICLES}" in reading.reason


def test_a_score_that_is_a_rounding_of_nothing_declines(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": _feed(
        _item("HBL reports record profit"),
        _item("HBL names a new auditor"),
        _item("HBL announces the date of its annual gathering"))}))
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert "carry a scored word" in reading.reason


def test_undated_headlines_are_dropped_not_read_as_fresh(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": _feed(
        _item("HBL reports record profit", None),
        _item("Habib Bank shares surge", None))}))
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert "undated or outside the window" in reading.reason
    assert N._parse_date("not a date") is None
    assert N._parse_date("") is None


def test_a_fault_inside_the_scoring_is_named_in_the_reason(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))

    def _explode(text):
        raise RuntimeError("lexicon blew up")

    monkeypatch.setattr(N, "tone_of", _explode)
    reading = desk.refresh([resolve("HBL")])["HBL"]
    assert reading.available is False
    assert reading.reason.startswith("desk fault: RuntimeError")


def test_empty_refresh_asks_nothing(desk, monkeypatch):
    t = _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    assert desk.refresh([]) == {}
    assert t.unknown == [] and t.calls["HBL"] == []


# ---------------------------------------------------------------------------
# 6. what the desk holds, for the operator who asks
# ---------------------------------------------------------------------------
def test_holdings_and_status_report_without_asking_the_host(desk, monkeypatch):
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD,
                                   "MEBL": _feed(_item("Meezan Bank shares plunge "
                                                       "on defaults"))}))
    desk.refresh([resolve("HBL"), resolve("MEBL")])
    rows = desk.holdings()
    assert [r["key"] for r in rows] == ["HBL", "MEBL"], "best tone first"
    st = desk.status()
    assert st["held"] == 2 and st["available"] == 1
    assert st["source"] == N.SOURCE and st["ttl_s"] == desk.ttl_s
    assert st["load_error"] is None and st["save_error"] is None


# ---------------------------------------------------------------------------
# 7. the engine's two estimators, and the strategy that reads them
# ---------------------------------------------------------------------------
def _warm_desk(tmp_path, monkeypatch) -> N.NewsDesk:
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    d.refresh([resolve("HBL")])
    return d


def test_the_engine_falls_back_to_the_desk(tmp_path, monkeypatch):
    d = _warm_desk(tmp_path, monkeypatch)
    monkeypatch.setattr(E, "NEWS_DESK", d)
    # The clean test config has no `apis:` section at all, so the LLM is off.
    score, view = E.ENGINE.news_context(resolve("HBL"))
    assert view["source"] == "desk" and view["available"] is True
    assert score is not None and score > 0
    assert view["desk"]["matching"] == 3


def test_the_model_is_not_asked_when_the_desk_can_answer(tmp_path, monkeypatch):
    """Ordering by input quality, and keeping an LLM round trip off the thread.

    Measured 2026-10-23: the wire feed's own relevance filter handed the model,
    for HBL, a Bank of Canada rate note, two ECB notes, a vendor's share-price
    page and the KSE-100 chart — no Habib Bank story at all — and it scored that
    pile 0.0. The desk, which builds its query from the symbol, had seven.
    """
    d = _warm_desk(tmp_path, monkeypatch)
    monkeypatch.setattr(E, "NEWS_DESK", d)
    asked = []

    def _never(inst):
        asked.append(inst.key)
        return 0.0

    monkeypatch.setattr(E.ENGINE, "news_sentiment", _never)
    monkeypatch.setattr(type(CFG), "apis",
                        property(lambda self: {"openrouter": {"enabled": True}}))
    score, view = E.ENGINE.news_context(resolve("HBL"))
    assert view["source"] == "desk" and score is not None and score > 0
    assert asked == [], "the model answered over the desk's reading"


def test_the_model_is_used_when_the_desk_cannot_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "_http_get", _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    monkeypatch.setattr(E, "NEWS_DESK", d)
    monkeypatch.setattr(type(CFG), "apis",
                        property(lambda self: {"openrouter": {"enabled": True}}))
    monkeypatch.setattr(E.ENGINE, "news_sentiment", lambda inst: 0.9)
    score, view = E.ENGINE.news_context(resolve("HBL"))
    assert score == pytest.approx(0.9) and view["source"] == "llm"
    # The desk still travels with it: two estimators, one shown, is unauditable.
    assert view["desk"]["reason"] == "not yet fetched"


def test_an_overdue_desk_tone_is_named_as_overdue(tmp_path, monkeypatch):
    _write_cache(tmp_path, "HBL", score=0.4, age_s=N.DEFAULT_TTL_S + 60)
    _patch(monkeypatch, _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    monkeypatch.setattr(E, "NEWS_DESK", d)
    score, view = E.ENGINE.news_context(resolve("HBL"))
    assert view["source"] == "desk-stale" and score == pytest.approx(0.4)


def test_the_engine_abstains_with_the_reason_attached(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "_http_get", _Transport({"HBL": HBL_GOOD}))
    d = N.NewsDesk(cache_path=tmp_path / "news_desk.json")
    monkeypatch.setattr(E, "NEWS_DESK", d)
    score, view = E.ENGINE.news_context(resolve("MEBL"))
    assert score is None
    assert view["available"] is False
    assert view["desk"]["reason"] == "not yet fetched"


def test_s99_votes_and_says_where_the_tone_came_from():
    s99 = next(s for s in S.STRATEGIES if s[0] == "News Sentiment Tilt")[2]
    vote = s99(None, None, {
        "news_sentiment": 0.36,
        "news_tone": {"source": "desk", "desk": {"matching": 7}}})
    assert vote is not None and vote.direction == 1
    assert "0.36" in vote.rationale and "desk" in vote.rationale
    assert "7 headlines" in vote.rationale


def test_s99_abstains_on_a_weak_tone_and_on_an_absent_one():
    s99 = next(s for s in S.STRATEGIES if s[0] == "News Sentiment Tilt")[2]
    assert s99(None, None, {}) is None
    assert s99(None, None, {"news_sentiment": 0.05}) is None
    bear = s99(None, None, {"news_sentiment": -0.8,
                            "news_tone": {"source": "llm", "desk": {}}})
    assert bear is not None and bear.direction == -1
    assert bear.strength <= 0.6, "one headline must not hand s99 the loudest voice"


def test_s99_votes_from_the_desk_through_the_engine(tmp_path, monkeypatch):
    """The whole path: cached reading → context → vote."""
    d = _warm_desk(tmp_path, monkeypatch)
    monkeypatch.setattr(E, "NEWS_DESK", d)
    score, view = E.ENGINE.news_context(resolve("HBL"))
    s99 = next(s for s in S.STRATEGIES if s[0] == "News Sentiment Tilt")[2]
    vote = s99(None, None, {"news_sentiment": score, "news_tone": view})
    assert vote is not None and vote.direction == 1


# ---------------------------------------------------------------------------
# 8. the two news endpoints must be two endpoints
# ---------------------------------------------------------------------------
def test_no_two_routes_share_a_path_and_method():
    """`/api/news` was registered twice, once for the desk panel's headline list
    and once for the tone. Starlette answers the first registration, so the
    second became unreachable and the desk UI asked for stories and got tones —
    a route that does not exist is worse than one that returns nothing."""
    from kairos.api.server import app

    seen: set[tuple[str, str]] = set()
    dupes = []
    for r in app.routes:
        methods = getattr(r, "methods", None) or set()
        for m in sorted(methods - {"HEAD", "OPTIONS"}):
            key = (m, getattr(r, "path", ""))
            if key in seen:
                dupes.append(key)
            seen.add(key)
    assert not dupes, f"registered twice, first one wins: {dupes}"


def test_the_desk_endpoint_answers_with_tones(tmp_path, monkeypatch):
    """Called as a function, not over HTTP: a TestClient runs the app's startup
    event, which resumes the paper autopilot loop inside the test session."""
    from kairos.api import server as SV

    d = _warm_desk(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "DESK", d)
    body = SV.newsdesk()
    assert body["ok"] is True and body["desk"]["held"] == 1
    assert body["readings"]["HBL"]["score"] > 0
    assert body["desk"]["source"] == N.SOURCE
    # A symbol nobody typed correctly is named, not silently dropped.
    assert SV.newsdesk(symbols="NOTAREALTICKER")["unknown"] == ["NOTAREALTICKER"]
    # And `/api/news` still exists beside it: that is the raw headline list the
    # desk panel reads, and calling it here would fetch the wire.
    paths = [r.path for r in SV.app.routes
             if getattr(r, "path", "").startswith("/api/news")]
    assert sorted(paths) == ["/api/news", "/api/newsdesk"]


# ---------------------------------------------------------------------------
# 9. GDELT, which is blocked from this IP and must cost nothing when it is
# ---------------------------------------------------------------------------
def test_a_rate_limited_gdelt_is_penalised_once():
    calls = []

    def _get(url, timeout=None, retries=None):
        calls.append(url)
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    g = N.GdeltTone(get=_get)
    assert g.tone("Engro Corporation") is None
    assert g.tone("Engro Corporation") is None
    assert len(calls) == 1, "a 429 answered by another request is a self-inflicted DoS"
    assert g.blocked == 1


def test_gdelt_reads_a_real_chart_and_caches_it():
    payload = json.dumps({"timeline": {"chart": [{"tone": 2.0}, {"tone": 6.0}]}})
    calls = []

    def _get(url, timeout=None, retries=None):
        calls.append(url)
        return payload

    g = N.GdeltTone(get=_get)
    assert g.tone("Bitcoin") == pytest.approx(1.0)   # mean 4.0 / GDELT_TONE_SCALE
    g.tone("Bitcoin")
    assert len(calls) == 1


def test_a_gdelt_answer_with_no_chart_is_none_not_zero():
    g = N.GdeltTone(get=lambda url, timeout=None, retries=None: '{"x": 1}')
    assert g.tone("Nothing") is None
