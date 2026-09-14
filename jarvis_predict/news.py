"""
Real-time news + sentiment module.

Sources (no API key required by default):
    - ForexFactory RSS (forex/macro headlines)
    - Investing.com RSS (commodities / equities)
    - FXStreet RSS (FX)
    - MarketWatch RSS (US markets)
    - Yahoo Finance RSS (per-symbol)
    - Google News RSS (keyword search, per-symbol)

Optional (requires free API key):
    - NewsAPI.org  — set listener.news.newsapi_key in config.yaml

Sentiment:
    - VADER compound score in [-1, +1]
    - Topic relevance per symbol via keyword matching against `symbols.py` sector tags
    - Cached per URL for 6 hours (we re-fetch, not re-analyse)
    - Polled every `poll_seconds` seconds; new articles push Dispatcher.NEWS events

All HTTP goes through `requests` with a polite User-Agent and 10s timeout.
Network failures degrade gracefully — Jarvis keeps trading without news.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import feedparser
import requests

from .config import JarvisConfig
from .symbols import get as get_instrument

LOGGER = logging.getLogger("jarvis.news")


# Default RSS feeds — all free, no key required.
DEFAULT_FEEDS: dict[str, str] = {
    "forexfactory": "https://www.forexfactory.com/ffcal_week.xml",
    "investing-fx":   "https://www.investing.com/rss/forex.rss",
    "investing-comm": "https://www.investing.com/rss/commodities.rss",
    "investing-news": "https://www.investing.com/rss/news.rss",
    "fxstreet":       "https://www.fxstreet.com/rss/news",
    "marketwatch":    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "reuters-biz":    "https://feeds.reuters.com/reuters/businessNews",
    "google-gold":    "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD&hl=en-US",
    "google-oil":     "https://news.google.com/rss/search?q=crude+oil+price+OR+OPEC&hl=en-US",
}

# Keyword → sector mapping used to score which article matters to which symbol.
_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fx":      ("dollar", "euro", "yen", "sterling", "fed", "ecb", "boe", "boj", "fx", "currency", "central bank", "rate decision"),
    "metals":  ("gold", "silver", "platinum", "palladium", "copper", "bullion", "xau", "xag"),
    "energy":  ("oil", "opec", "brent", "wti", "natural gas", "crude", "barrel"),
    "grains":  ("wheat", "corn", "soy", "grain", "harvest"),
    "softs":   ("coffee", "sugar", "cotton", "cocoa"),
    "indices": ("s&p", "nasdaq", "dow", "dax", "ftse", "nikkei", "equities", "stocks", "wall street"),
    "crypto":  ("bitcoin", "ethereum", "crypto", "btc", "eth"),
    "rates":   ("treasury", "yield", "10-year", "2-year", "bond", "t-note", "real yield"),
}


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: float                  # unix epoch
    summary: str
    sentiment: float = 0.0            # -1..+1 (VADER compound)
    relevance: dict[str, float] = field(default_factory=dict)  # symbol -> 0..1
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published": self.published,
            "summary": self.summary[:200],
            "sentiment": round(self.sentiment, 3),
            "relevance": {k: round(v, 3) for k, v in self.relevance.items()},
            "keywords": self.keywords[:10],
        }


class NewsService:
    """Polls RSS feeds, scores sentiment + relevance, exposes per-symbol aggregates."""

    def __init__(
        self,
        *,
        config: JarvisConfig,
        enabled: bool = True,
        feeds: dict[str, str] | None = None,
        newsapi_key: str | None = None,
        poll_seconds: float = 180.0,         # 3 min default — RSS rate-limits vary
        cache_ttl_seconds: float = 21600.0,   # 6 h
        timeout: float = 10.0,
        max_age_seconds: float = 7200.0,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self.feeds = dict(feeds or DEFAULT_FEEDS)
        self.newsapi_key = newsapi_key
        self.poll_seconds = max(30.0, float(poll_seconds))
        self.cache_ttl = max(60.0, float(cache_ttl_seconds))
        self.timeout = float(timeout)
        self.max_age_seconds = float(max_age_seconds)

        self._items: dict[str, NewsItem] = {}     # keyed by url hash
        self._lock = threading.RLock()
        self._last_fetch: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Lazy-loaded sentiment analyzer.
        self._sentiment = None

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "NewsService":
        section = config.section("news")
        return cls(
            config=config,
            enabled=bool(section.get("enabled", True)),
            feeds=section.get("feeds"),
            newsapi_key=section.get("newsapi_key") or None,
            poll_seconds=section.get("poll_seconds", 180.0),
            cache_ttl_seconds=section.get("cache_ttl_seconds", 21600.0),
            timeout=section.get("timeout_seconds", 10.0),
            max_age_seconds=section.get("max_age_seconds", 7200.0),
        )

    # --- public --------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("News disabled in config")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-news", daemon=True)
        self._thread.start()
        LOGGER.info("NewsService started (poll=%.0fs, feeds=%d)", self.poll_seconds, len(self.feeds))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def items_for_symbol(self, symbol: str, max_age_seconds: float = 7200.0) -> list[NewsItem]:
        """Return news items relevant to `symbol` within the last `max_age_seconds`."""
        ins = get_instrument(symbol)
        if ins is None:
            return []
        sector = ins.sector
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [
                i for i in self._items.values()
                if i.published >= cutoff
                and (
                    i.relevance.get(symbol, 0.0) > 0.2
                    or _sector_match(i.keywords, sector) > 0.0
                )
            ]

    def sentiment_for_symbol(self, symbol: str, max_age_seconds: float = 7200.0) -> float:
        """Return weighted-average sentiment for `symbol` in [-1, +1]. 0 if no news."""
        items = self.items_for_symbol(symbol, max_age_seconds)
        if not items:
            return 0.0
        ins = get_instrument(symbol)
        sector = ins.sector if ins else ""
        weighted = 0.0
        total = 0.0
        now = time.time()
        for it in items:
            age_h = max(0.0, (now - it.published) / 3600.0)
            # Freshness decay: half-life of 4 hours.
            freshness = 0.5 ** (age_h / 4.0)
            relevance = max(it.relevance.get(symbol, 0.0), _sector_match(it.keywords, sector))
            weight = freshness * (0.5 + relevance)
            weighted += weight * it.sentiment
            total += weight
        if total <= 0:
            return 0.0
        return max(-1.0, min(1.0, weighted / total))

    def snapshot(self) -> list[dict]:
        """All currently-tracked items (used by HTTP /status)."""
        with self._lock:
            return [i.to_dict() for i in sorted(self._items.values(), key=lambda x: x.published, reverse=True)][:100]

    # --- internals -----------------------------------------------------------

    def _run(self) -> None:
        # Prime caches from one full pass on start.
        try:
            self._poll_once()
        except Exception:
            LOGGER.exception("News initial poll failed")
        while not self._stop.is_set():
            self._stop.wait(timeout=self.poll_seconds)
            try:
                self._poll_once()
            except Exception:
                LOGGER.exception("News poll loop error")

    def _poll_once(self) -> None:
        for name, url in self.feeds.items():
            self._fetch_feed(name, url)
        if self.newsapi_key:
            self._fetch_newsapi(self.newsapi_key)

    def _fetch_feed(self, name: str, url: str) -> None:
        # Cache: don't re-fetch if recently done.
        last = self._last_fetch.get(name, 0.0)
        if time.time() - last < self.cache_ttl:
            return
        self._last_fetch[name] = time.time()
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "JarvisPredictor/1.0 (+local; Pakistan)"},
            )
            resp.raise_for_status()
        except Exception as e:
            LOGGER.debug("News fetch %s failed: %s", name, e)
            return

        parsed = feedparser.parse(resp.content)
        for entry in parsed.entries[:50]:
            item = self._parse_entry(entry, source=name)
            if item is None:
                continue
            self._ingest(item)

    def _fetch_newsapi(self, api_key: str) -> None:
        # Free-tier endpoint — only every 6 h to stay under quota.
        last = self._last_fetch.get("newsapi", 0.0)
        if time.time() - last < self.cache_ttl:
            return
        self._last_fetch["newsapi"] = time.time()
        url = "https://newsapi.org/v2/top-headlines"
        params = {"category": "business", "language": "en", "pageSize": 30, "apiKey": api_key}
        try:
            resp = requests.get(url, params=params, timeout=self.timeout,
                                headers={"User-Agent": "JarvisPredictor/1.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            LOGGER.debug("NewsAPI fetch failed: %s", e)
            return
        for a in data.get("articles", []):
            published = _parse_iso(a.get("publishedAt")) or time.time()
            item = NewsItem(
                title=a.get("title", "") or "",
                url=a.get("url", "") or "",
                source="newsapi",
                published=published,
                summary=a.get("description", "") or "",
            )
            self._score_and_ingest(item)

    def _parse_entry(self, entry, *, source: str) -> NewsItem | None:
        title = getattr(entry, "title", "") or ""
        url = getattr(entry, "link", "") or ""
        if not title or not url:
            return None
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        published_parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if published_parsed:
            published = float(datetime(*published_parsed[:6], tzinfo=timezone.utc).timestamp())
        else:
            published = time.time()
        return NewsItem(
            title=title, url=url, source=source,
            published=published, summary=summary,
        )

    def _ingest(self, item: NewsItem) -> None:
        self._score_and_ingest(item)

    def _score_and_ingest(self, item: NewsItem) -> None:
        # Compute URL hash to dedupe.
        key = hashlib.sha1(item.url.encode("utf-8")).hexdigest()[:16]
        # Sentiment via VADER (lazy import — only on first call).
        try:
            if self._sentiment is None:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
                self._sentiment = SentimentIntensityAnalyzer()
            vs = self._sentiment.polarity_scores(f"{item.title}. {item.summary}")
            item.sentiment = float(vs.get("compound", 0.0))
        except ImportError:
            item.sentiment = 0.0
        except Exception:  # pragma: no cover
            item.sentiment = 0.0

        # Keyword extraction + relevance scoring.
        text = f"{item.title}. {item.summary}".lower()
        keywords = _extract_keywords(text)
        item.keywords = keywords
        item.relevance = _score_relevance(keywords, text)
        with self._lock:
            self._items[key] = item
            # Cap cache size to last 500 items.
            if len(self._items) > 500:
                sorted_keys = sorted(self._items, key=lambda k: self._items[k].published)
                for k in sorted_keys[: len(self._items) - 500]:
                    del self._items[k]


# --- pure helpers ------------------------------------------------------------

_KEYWORD_RE = re.compile(r"\b[a-z][a-z0-9\-]{2,}\b")
_BULLISH_WORDS = {"surge", "rally", "soar", "jump", "gain", "rise", "climb", "beat", "strong", "high", "record", "boom", "bull", "up"}
_BEARISH_WORDS = {"fall", "drop", "plunge", "tumble", "sink", "slump", "miss", "weak", "low", "fear", "crash", "bear", "down", "tumble"}


def _extract_keywords(text: str) -> list[str]:
    return [w for w in _KEYWORD_RE.findall(text) if len(w) >= 4][:50]


def _score_relevance(keywords: Sequence[str], text: str) -> dict[str, float]:
    """Per-symbol relevance in [0, 1] using keyword + sector heuristics."""
    out: dict[str, float] = {}
    text_words = set(_KEYWORD_RE.findall(text))
    for sym in _all_symbol_names():
        ins = get_instrument(sym)
        if ins is None:
            continue
        score = 0.0
        # Direct symbol mention.
        if sym.lower() in text:
            score += 0.5
        # Name token match.
        name_words = {w for w in _KEYWORD_RE.findall(ins.name.lower())}
        if name_words & text_words:
            score += 0.3
        # Sector keywords.
        sector = ins.sector
        sector_kws = _SECTOR_KEYWORDS.get(sector, ())
        if any(k in text for k in sector_kws):
            score += 0.4
        # Correlated / inverse instruments get partial weight too.
        for c in ins.correlated_to:
            if c.lower() in text or any(k in text for k in _SECTOR_KEYWORDS.get(get_instrument(c).sector if get_instrument(c) else "", ())):
                score += 0.15
        for inv in ins.inverse_to:
            if inv.lower() in text:
                score += 0.1
        if score > 0:
            out[sym] = min(1.0, score)
    return out


def _sector_match(keywords: Sequence[str], sector: str) -> float:
    kws = _SECTOR_KEYWORDS.get(sector, ())
    if not kws:
        return 0.0
    text = " ".join(keywords).lower()
    return min(1.0, sum(0.2 for k in kws if k in text))


def _all_symbol_names() -> list[str]:
    from .symbols import all_symbols
    return all_symbols()


def _parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _quick_lexicon_sentiment(text: str) -> float:
    """Fallback if VADER missing. Crude lexicon score in [-1, +1]."""
    words = set(_KEYWORD_RE.findall(text.lower()))
    pos = len(words & _BULLISH_WORDS)
    neg = len(words & _BEARISH_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


__all__ = ["NewsService", "NewsItem"]
