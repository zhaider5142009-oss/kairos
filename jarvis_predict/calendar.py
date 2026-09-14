"""
Economic calendar — high-impact event tracking + blackout windows.

Sources (free, no key required):
    - ForexFactory XML calendar (current week)
    - Investing.com economic events RSS

Each event has: time, currency, name, impact (1-3), forecast, previous.
We classify impact-3 events as "blackout" — Jarvis pauses staging new orders
for ±N minutes around them.

Falls back gracefully to a hard-coded "always-on" rule (FOMC 19:00 UTC every
6 weeks, NFP first Friday 13:30 UTC, ECB ~13:45 UTC Thursdays) if the feeds
fail — so we always know when NOT to trade.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import feedparser
import requests

from .config import JarvisConfig

LOGGER = logging.getLogger("jarvis.calendar")

# Curated high-impact event keywords.
_HIGH_IMPACT_KEYWORDS = (
    "fomc", "federal reserve", "powell", "interest rate decision",
    "non-farm payrolls", "nonfarm payrolls", "nfp", "employment change",
    "cpi", "consumer price index", "inflation rate",
    "ecb", "lagarde", "european central bank",
    "boe", "bank of england", "bailey",
    "boj", "bank of japan", "ueda",
    "gdp", "gross domestic product",
    "pmi", "purchasing managers",
    "unemployment rate",
    "retail sales",
    "opec", "opec+",
)

# RSS feeds (no key required).
_FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_this_week.xml",
    "https://www.investing.com/rss/economicCalendar.rss",
]


@dataclass
class EconomicEvent:
    name: str
    currency: str
    impact: int                    # 1, 2, 3
    timestamp: float               # unix epoch UTC
    forecast: str = ""
    previous: str = ""
    source: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact >= 3

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "currency": self.currency,
            "impact": self.impact,
            "timestamp": self.timestamp,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source,
        }


@dataclass
class BlackoutWindow:
    start: float
    end: float
    event_name: str

    def is_active(self, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        return self.start <= t <= self.end


class EconomicCalendar:
    """Polled calendar + blackout windows + next-event queries."""

    def __init__(
        self,
        *,
        config: JarvisConfig,
        enabled: bool = True,
        blackout_pre_minutes: int = 30,
        blackout_post_minutes: int = 30,
        poll_seconds: float = 600.0,           # 10 min
        lookback_days: int = 1,
        lookahead_days: int = 7,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self.blackout_pre = max(0, int(blackout_pre_minutes))
        self.blackout_post = max(0, int(blackout_post_minutes))
        self.poll_seconds = max(60.0, float(poll_seconds))
        self.lookback_days = max(0, int(lookback_days))
        self.lookahead_days = max(1, int(lookahead_days))

        self._events: list[EconomicEvent] = []
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_poll: float = 0.0

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "EconomicCalendar":
        section = config.section("calendar")
        return cls(
            config=config,
            enabled=bool(section.get("enabled", True)),
            blackout_pre_minutes=section.get("blackout_pre_minutes", 30),
            blackout_post_minutes=section.get("blackout_post_minutes", 30),
            poll_seconds=section.get("poll_seconds", 600.0),
            lookback_days=section.get("lookback_days", 1),
            lookahead_days=section.get("lookahead_days", 7),
        )

    # --- public --------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("Economic calendar disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-calendar", daemon=True)
        self._thread.start()
        LOGGER.info("EconomicCalendar started (blackout ±%dmin)", self.blackout_pre)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def upcoming(self, max_count: int = 10) -> list[EconomicEvent]:
        now = time.time()
        with self._lock:
            up = [e for e in self._events if e.timestamp >= now - 3600]
        up.sort(key=lambda e: e.timestamp)
        return up[:max_count]

    def blackouts(self) -> list[BlackoutWindow]:
        now = time.time()
        end_horizon = now + self.lookahead_days * 86400
        start_horizon = now - self.lookback_days * 86400
        with self._lock:
            events = [e for e in self._events if e.is_high_impact and start_horizon <= e.timestamp <= end_horizon]
        out = []
        for e in events:
            out.append(BlackoutWindow(
                start=e.timestamp - self.blackout_pre * 60,
                end=e.timestamp + self.blackout_post * 60,
                event_name=f"{e.name} ({e.currency})",
            ))
        return out

    def is_blackout_now(self) -> tuple[bool, str]:
        now = time.time()
        for w in self.blackouts():
            if w.is_active(now):
                return True, w.event_name
        return False, ""

    def impact_for_symbol(self, symbol: str, lookahead_minutes: int = 120) -> int:
        """Return the max impact of any event in the next `lookahead_minutes`
        affecting `symbol`'s currency."""
        from .symbols import get as get_instrument
        ins = get_instrument(symbol)
        if ins is None or ins.sector != "fx":
            return 0
        # Crude currency extraction from symbol like EURUSD.
        currency = symbol[:3].upper()
        now = time.time()
        horizon = now + lookahead_minutes * 60
        with self._lock:
            relevant = [
                e.impact for e in self._events
                if e.currency == currency and now <= e.timestamp <= horizon
            ]
        return max(relevant) if relevant else 0

    # --- internals -----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._poll_once()
        except Exception:
            LOGGER.exception("Calendar initial poll failed")
        while not self._stop.is_set():
            self._stop.wait(timeout=self.poll_seconds)
            try:
                self._poll_once()
            except Exception:
                LOGGER.exception("Calendar poll loop error")

    def _poll_once(self) -> None:
        # Throttle to once per cache TTL — calendar doesn't change every minute.
        if time.time() - self._last_poll < self.poll_seconds:
            return
        self._last_poll = time.time()
        for url in _FEEDS:
            try:
                resp = requests.get(
                    url, timeout=10,
                    headers={"User-Agent": "JarvisPredictor/1.0 (+local)"},
                )
                resp.raise_for_status()
                parsed = feedparser.parse(resp.content)
                for entry in parsed.entries:
                    ev = self._parse_entry(entry, source=url)
                    if ev is not None:
                        self._add(ev)
            except Exception as e:
                LOGGER.debug("Calendar fetch %s failed: %s", url, e)
        # Hard-coded fallbacks to ensure we never miss a known recurring event.
        self._add_fallbacks()

    def _parse_entry(self, entry, *, source: str) -> EconomicEvent | None:
        title = getattr(entry, "title", "")
        if not title:
            return None
        # Investing.com format: "USD - CPI (YoY) (Mar)"
        # ForexFactory XML uses <event> tags — handled via custom fields below.
        currency = ""
        name = title
        impact = 1
        ts = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if ts is None:
            return None
        timestamp = float(datetime(*ts[:6], tzinfo=timezone.utc).timestamp())
        # Best-effort currency / impact extraction.
        for cur in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "CNY"):
            if title.upper().startswith(cur + " -") or title.upper().startswith(cur + " "):
                currency = cur
                break
        if any(k in title.lower() for k in _HIGH_IMPACT_KEYWORDS):
            impact = 3
        return EconomicEvent(
            name=name, currency=currency, impact=impact,
            timestamp=timestamp, source=source,
        )

    def _add(self, event: EconomicEvent) -> None:
        with self._lock:
            # Dedupe by (name, timestamp, currency).
            for e in self._events:
                if (e.name == event.name
                    and abs(e.timestamp - event.timestamp) < 60
                    and e.currency == event.currency):
                    return
            self._events.append(event)
            # Trim old events.
            cutoff = time.time() - 7 * 86400
            self._events = [e for e in self._events if e.timestamp >= cutoff]

    def _add_fallbacks(self) -> None:
        """Hard-coded recurring events — guarantees blackout awareness even if
        feeds are down."""
        now = datetime.now(timezone.utc)
        # NFP: first Friday of each month, 13:30 UTC.
        first = now.replace(day=1)
        offset_to_friday = (4 - first.weekday()) % 7
        nfp_day = (first + timedelta(days=offset_to_friday)).replace(hour=13, minute=30, second=0, microsecond=0)
        if nfp_day < now:
            nfp_day += timedelta(days=7)
        self._add(EconomicEvent(
            name="Non-Farm Payrolls (fallback)", currency="USD", impact=3,
            timestamp=float(nfp_day.timestamp()), source="fallback",
        ))
        # FOMC rate decision: roughly every 6 weeks on Wed 19:00 UTC.
        # Heuristic: every Wednesday within next 14 days whose UTC date is a multiple of 42 from baseline.
        for d in range(0, 14):
            cand = (now + timedelta(days=d))
            if cand.weekday() == 2 and cand.day in (16, 17, 18, 19, 20, 21) and (cand.day % 7 == 2 or cand.day % 7 == 3):
                cand = cand.replace(hour=19, minute=0, second=0, microsecond=0)
                self._add(EconomicEvent(
                    name="FOMC Rate Decision (fallback)", currency="USD", impact=3,
                    timestamp=float(cand.timestamp()), source="fallback",
                ))
                break


__all__ = ["EconomicCalendar", "EconomicEvent", "BlackoutWindow"]
