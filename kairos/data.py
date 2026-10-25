"""Data layer: multi-source ingestion with failover, caching, and history accumulation.

Source chain (health-probed at runtime):
  GLOBAL instruments -> Yahoo Finance v8 chart API (2y daily OHLCV)
  PSX quotes/factors -> TradingView scanner (484 stocks, live fields)
  PSX history        -> local operator API (localhost:8000) -> accumulated store
  crypto history     -> Yahoo primary, Binance spot klines as the failover
  News               -> RSS feeds (Google News, BRecorder, Dawn, FXStreet, MW)

Nothing is ever fabricated: when no source is reachable the layer reports
honestly and the engine degrades gracefully.
"""

from __future__ import annotations

import io
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from defusedxml.ElementTree import fromstring as _safe_fromstring

from .config import CFG
from .netio import HttpStatus, PayloadTooLarge, close_error, retryable, status_of
from .store import atomic_write
from .universe import Instrument, resolve

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# An RSS/Atom document is measured in tens of kilobytes; Google News' answer is
# ~200 KB. Eight megabytes is far past any real feed and far below what a
# hostile or misconfigured host can hand this box.
FEED_MAX_BYTES = 8 * 1024 * 1024


def _http_get(url: str, timeout: int = 20, headers: dict[str, str] | None = None,
              retries: int = 2, max_bytes: int | None = None) -> bytes:
    """GET a body. `max_bytes` caps how much of it we will hold in memory.

    A feed is somebody else's server answering our socket; without a cap, a
    redirect to a multi-gigabyte file is a memory exhaustion on this box. The
    cap is opt-in because the JSON endpoints need no such bound.
    """
    hdrs = {"User-Agent": _UA, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read() if max_bytes is None else resp.read(max_bytes + 1)
            if max_bytes is not None and len(body) > max_bytes:
                raise PayloadTooLarge(url, max_bytes)
            return body
        except Exception as exc:  # noqa: BLE001 — network layer must never crash the engine
            # The `with` above never sees a failed request: `urlopen` raises
            # the error *as* the response, with its socket still attached.
            close_error(exc)
            last_err = exc
            # a 404 answers 404 again; only a server error or a "come back
            # later" is worth another request on the same call
            if not retryable(exc):
                break
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    if isinstance(last_err, PayloadTooLarge):
        raise last_err
    if isinstance(last_err, urllib.error.HTTPError):
        # Keep the status. "GET failed" cannot be acted on: the news service
        # is built to pause half an hour when a provider answers 429, and it
        # has no way to tell that from a page that does not exist.
        raise HttpStatus(last_err.code, url, str(last_err.reason or ""))
    raise ConnectionError(f"GET failed: {url} ({last_err})")


def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 20,
                    headers: dict[str, str] | None = None) -> Any:
    hdrs = {"User-Agent": _UA, "Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        close_error(exc)
        raise


def _binance():
    """The Binance client, imported on first use.

    `kairos.binance` pulls in the execution layer (audit journal, kill switch,
    order intents), and reading a price must not require any of that. It also
    keeps the signed-order code out of the import graph that every CLI one-off
    and test fixture drags in.
    """
    from .binance import BINANCE
    return BINANCE


# ---------------------------------------------------------------------------
# Store — CSV persistence + genuine history accumulation
# ---------------------------------------------------------------------------
def _same_number(a, b) -> bool:
    """Two prints are the same when both are NaN, or equal.

    The vendor spells a hole as NaN, and `nan == nan` is False, so a strict
    comparison would call two identical holes different and write the bar.
    """
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


class Store:
    """Local OHLCV cache. Every successful fetch appends today's bar so the
    system builds real PSX history over time even from quote-only sources."""

    def __init__(self, cache_dir: Path) -> None:
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key.upper())
        return self.dir / f"{safe}.csv"

    def load(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            if df.empty or "close" not in df.columns:
                return None
            df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        # Nothing lands in the store unless every bar contains its own open and
        # close. `history()` already widens on the way out, which is why an
        # impossible bar never reached a model — but the merge paths persist the
        # provider's rows as printed, so the next fetch put them back and the
        # file, which is the record, went on contradicting itself.
        df = coerce_bar_range(df)
        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        # through the atomic writer, and through a string first: the bar store
        # is now filled by several fetches at once, and a half-written CSV is
        # read back by `load` as "no state yet" — history that looks absent
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        atomic_write(self._path(key), buf.getvalue())

    def upsert_row(self, key: str, date: str, open_: float, high: float, low: float,
                   close: float, volume: float) -> pd.DataFrame | None:
        df = self.load(key)
        row = pd.DataFrame(
            [{"date": pd.to_datetime(date), "open": open_, "high": high,
              "low": low, "close": close, "volume": volume}])
        if df is None:
            df = row
        else:
            # A snapshot refresh restates the same bar for every symbol on the
            # exchange, and rewriting 484 unchanged CSVs cost more than the
            # quote fetch itself — on a disk that is already failing. Only the
            # latest row can match, and a match means the file already says
            # exactly this, so leave it alone.
            last = df.iloc[-1]
            same_day = last["date"] == row["date"].iloc[0]
            if (same_day and _same_number(last["open"], open_)
                    and _same_number(last["close"], close)
                    and _same_number(last["volume"], volume)):
                return df
            # A new date carrying the previous session's close *and* volume is
            # that session served again, not a new one — a weekend, a public
            # holiday, or a pre-open poll all look exactly like this. Writing it
            # anyway invented a flat Saturday and Sunday for every symbol on the
            # exchange, and each invented bar dilutes volatility and pulls every
            # five-bar horizon a day forward. A real flat session prints its own
            # volume, so it differs here and is still written.
            if (not same_day and _same_number(last["close"], close)
                    and _same_number(last["volume"], volume)):
                return df
            df = pd.concat([df[df["date"] != pd.to_datetime(date)], row], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        self.save(key, df)
        return df


STORE = Store(CFG.cache_dir)


# ---------------------------------------------------------------------------
# The one rate two books share
# ---------------------------------------------------------------------------
FX_KEY = "USDPKR"
# the series prints once a session, and a cycle asks per candidate: re-reading
# the CSV each time is pure disk
FX_TTL_S = 900.0
# past this the print is history, not an exchange rate
FX_MAX_AGE_DAYS = 7
_fx_cache: tuple[float, float | None] | None = None


def usd_pkr() -> float | None:
    """The cached rupee per dollar, or None when the series cannot answer.

    Read from the store the rest of the system already fills, never assumed. A
    portfolio brake that silently used 1.0 instead priced a 70,000 dollar
    holding as a 70,000 rupee position — two hundred times too small in front of
    a rupee book — and a None is the answer that makes the caller veto instead.
    """
    global _fx_cache
    now = time.monotonic()
    if _fx_cache is not None and now - _fx_cache[0] < FX_TTL_S:
        return _fx_cache[1]
    rate: float | None = None
    try:
        df = STORE.load(FX_KEY)
        if df is not None and len(df) and {"date", "close"} <= set(df.columns):
            last = df.iloc[-1]
            close = _num(last["close"])
            day = pd.Timestamp(last["date"]).date()
            age = (pd.Timestamp.now().normalize().date() - day).days
            if close is not None and close > 0 and age <= FX_MAX_AGE_DAYS:
                rate = close
    except Exception:  # noqa: BLE001 — no rate is a valid answer, a crash is not
        rate = None
    _fx_cache = (now, rate)
    return rate


def fx_rate(frm: str, to: str) -> float | None:
    """How much of `to` one unit of `frm` buys, or None when it cannot be said.

    This desk holds exactly two currencies, so one rate between them answers
    every mixing the portfolio does. Same-currency is 1.0 without touching the
    tape; any other pair has no print here and is an honest None.
    """
    f, t = str(frm or "").upper(), str(to or "").upper()
    if not f or not t:
        return None
    if f == t:
        return 1.0
    pkr = usd_pkr()
    if pkr is None:
        return None
    if (f, t) == ("USD", "PKR"):
        return pkr
    if (f, t) == ("PKR", "USD"):
        return 1.0 / pkr
    return None


# PSX history accumulation runs here, off the read path. One lock means only a
# single sweep is ever in flight; `concurrent.futures` installs an exit hook
# that joins submitted work, so a fire-and-forget sweep still finishes its
# writes before a normal process exit.
_BAR_WRITER = ThreadPoolExecutor(max_workers=2, thread_name_prefix="psx-bars")
_bar_sweep = threading.Semaphore(1)


def _accumulate_bars(store: Store, df: pd.DataFrame, today: str) -> None:
    # The whole board is the PSX exchange, and PSX does not trade at the
    # weekend: a Saturday snapshot is Friday's session served again. The day is
    # bound when the snapshot is taken, so a sweep that runs across midnight
    # cannot relabel yesterday's close as a new session either.
    if pd.Timestamp(today).weekday() >= 5:
        return
    for _, row in df.iterrows():
        sym = str(row.get("name") or "")
        if not sym:
            continue
        close = row.get("close")
        if close is None or (isinstance(close, float) and np.isnan(close)):
            continue
        try:
            change = float(row.get("change") or 0.0) / 100.0
            c = float(close)
            prev = c / (1 + change) if (1 + change) != 0 else c
            # The board's symbol is not always the registry key (ENGROH/ENGRO,
            # BOP/BANK, NESTLE/NESP). Writing the board spelling would put the
            # bar in a file `history(instrument)` never opens, so a name with no
            # Yahoo series would stay unpriceable however often the board quotes it.
            inst = resolve(sym)
            key = inst.key if inst is not None and inst.market == "PSX" else sym
            # The snapshot carries a close and a percentage change, never an
            # intraday range; the previous close is the only other price in
            # hand, so the bar's range is between those two and nothing more.
            # The old synthesis assumed an up day (high=close, low=previous) and
            # on a down day wrote a high below its own low — an impossible bar
            # that only the read path's widening hid, while the file kept it.
            hi, lo = (c, prev) if c >= prev else (prev, c)
            store.upsert_row(key, today, prev, hi, lo, c,
                             float(row.get("volume") or 0))
        except Exception:  # noqa: BLE001 — one bad row cannot stop the sweep
            continue


def _schedule_bar_accumulation(df: pd.DataFrame) -> None:
    # both the store and the date are bound when the snapshot is taken, not
    # when the worker gets to it: a sweep that runs across midnight must not
    # relabel yesterday's quotes as today's bar, and it must never discover a
    # different cache directory mid-flight.
    store, today = STORE, datetime.now().strftime("%Y-%m-%d")
    if not _bar_sweep.acquire(blocking=False):
        return

    def run() -> None:
        try:
            _accumulate_bars(store, df, today)
        finally:
            _bar_sweep.release()

    try:
        _BAR_WRITER.submit(run)
    except RuntimeError:      # pool already shut down (interpreter exiting)
        _bar_sweep.release()


# ---------------------------------------------------------------------------
# Yahoo Finance client (global instruments)
# ---------------------------------------------------------------------------
def _num(value) -> float | None:
    """A JSON cell as a usable number, or None if it is not one.

    Yahoo spells a hole in the series as a literal ``NaN``, which ``json``
    parses without complaint and which ``is None`` never catches. A bar built
    from one is not a missing bar: it is a bar that poisons every return,
    range and price computed from it, and it backfilled the open, high and low
    of its own row.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# Bar integrity (every provider)
# ---------------------------------------------------------------------------
def out_of_range_bars(df: pd.DataFrame | None) -> int:
    """How many bars contradict their own range: an open or a close outside it."""
    if df is None or df.empty or not {"open", "high", "low", "close"} <= set(df.columns):
        return 0
    above = (df["open"] > df["high"]) | (df["close"] > df["high"])
    below = (df["open"] < df["low"]) | (df["close"] < df["low"])
    return int((above | below).sum())


def coerce_bar_range(df: pd.DataFrame) -> pd.DataFrame:
    """Widen each bar's range until it contains its own open and close.

    A bar is a promise that the price travelled between ``low`` and ``high``,
    so ``low <= min(open, close)`` and ``high >= max(open, close)`` hold for any
    real session. The provider series does not always keep the promise: a daily
    FX print whose close comes from a later fix than its own range lands outside
    it outright — measured on USDPKR, thirteen of the last twenty bars, one of
    them closing three percent above the day's recorded high. Nothing raises,
    because the row still parses as a row.

    Every range-based reading of such a bar is then a statement about a bar that
    cannot exist. Garman-Klass is a per-bar variance estimate, and on a valid bar
    the range term always covers the body term, so it cannot go negative — here
    its window mean did, and the field reported *zero intraday volatility* on a
    currency that had just moved sixteen percent annualized. Stochastics and
    Williams %R place the close outside their own band, and a candlestick
    detector is handed a wick shorter than the body it hangs off.

    Widening only, never narrowing: the printed extremes are prices the market
    reached, and so are the open and the close, so their union is the smallest
    range the row cannot contradict. No price already inside the range is moved,
    so the closes the paper record and the sweep were measured on still match.
    """
    if df is None or df.empty or not {"open", "high", "low", "close"} <= set(df.columns):
        return df
    out = df.copy()
    prices = out[["high", "low", "open", "close"]]
    out["high"] = prices.max(axis=1)
    out["low"] = prices.min(axis=1)
    return out


class YahooClient:
    def __init__(self) -> None:
        cfg = CFG.section("data").get("yahoo", {})
        self.base = cfg.get("base_url", "https://query1.finance.yahoo.com")
        self.lookback = int(cfg.get("lookback_days", 730))
        self.timeout = int(cfg.get("timeout_s", 20))

    def history(self, symbol: str, days: int | None = None) -> pd.DataFrame | None:
        days = days or self.lookback
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        end = int(datetime.now(timezone.utc).timestamp()) + 86400
        url = (f"{self.base}/v8/finance/chart/{urllib.parse.quote(symbol)}"
               f"?period1={start}&period2={end}&interval=1d&events=div%2Csplit")
        try:
            raw = _http_get(url, timeout=self.timeout)
        except ConnectionError:
            return None
        try:
            payload = json.loads(raw)
            result = payload["chart"]["result"][0]
            meta = result["meta"]
            stamps = result.get("timestamp") or []
            quote = result["indicators"]["quote"][0]
            rows = []
            for i, ts in enumerate(stamps):
                close = _num(quote["close"][i])
                if close is None:
                    continue
                o = _num(quote["open"][i])
                h = _num(quote["high"][i])
                lo = _num(quote["low"][i])
                v = _num(quote["volume"][i])
                rows.append({"date": datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                             "open": close if o is None else o,
                             "high": close if h is None else h,
                             "low": close if lo is None else lo,
                             "close": close, "volume": 0.0 if v is None else v})
            df = pd.DataFrame(rows)
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
            df.attrs["currency"] = meta.get("currency", "")
            df.attrs["name"] = meta.get("symbol", symbol)
            return df
        except Exception:
            return None


# ---------------------------------------------------------------------------
# TradingView scanner (PSX live quotes + cross-sectional fields)
# ---------------------------------------------------------------------------
TV_COLUMNS = [
    "name", "description", "close", "change", "change_abs", "volume",
    "change|1W", "change|1M", "Perf.3M", "Perf.6M", "Perf.Y", "Perf.YTD",
    "RSI", "RSI[1]", "SMA50", "SMA200", "EMA20", "EMA50",
    "price_52_week_high", "price_52_week_low", "High.1M", "Low.1M",
    "Volatility.D", "market_cap_basic", "peratio", "price_book_ratio",
    "roe", "net_margin", "debt_to_equity", "current_ratio",
    "average_volume_10d_calc", "average_volume_30d_calc",
    "beta_1_year", "total_revenue", "earnings_per_share_basic_ttm",
    "price_earnings_growth_ratio", "sector", "industry",
    "Recommend.All", "Recommend.MA", "number_of_employees",
    "earnings_release_date", "gap", "premarket_change",
]


class TradingViewClient:
    def __init__(self) -> None:
        cfg = CFG.section("data").get("tradingview", {})
        self.url = cfg.get("scanner_url", "https://scanner.tradingview.com/pakistan/scan")
        self.timeout = int(cfg.get("timeout_s", 25))

    def scan_psx(self, columns: list[str] | None = None,
                 limit: int = 1000) -> pd.DataFrame:
        cols = columns or TV_COLUMNS
        payload = {
            "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
            "options": {"lang": "en"},
            "columns": cols,
            "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
            "range": [0, min(limit, 1000)],
        }
        try:
            data = _http_post_json(self.url, payload, timeout=self.timeout)
        except Exception:
            return pd.DataFrame()
        rows = []
        for item in data.get("data", []):
            # tolerant on purpose: a short vendor row loses only trailing
            # fields, the survivors stay aligned with their names.
            row = dict(zip(cols, item.get("d", []), strict=False))
            row["tv_symbol"] = item.get("s", "")
            rows.append(row)
        return pd.DataFrame(rows)

    def quote(self, tv_symbol: str) -> dict[str, Any] | None:
        cols = ["name", "description", "close", "change", "volume", "RSI",
                "SMA50", "SMA200", "price_52_week_high", "price_52_week_low",
                "Volatility.D", "market_cap_basic", "sector",
                "change|1W", "change|1M", "Perf.3M", "Perf.6M", "Perf.Y",
                "Recommend.All", "peratio", "roe", "net_margin"]
        payload = {
            "filter": [{"left": "name", "operation": "equal",
                        "right": tv_symbol.split(":")[-1]}],
            "options": {"lang": "en"},
            "columns": cols,
            "sort": {"sortBy": "close", "sortOrder": "desc"},
            "range": [0, 5],
        }
        try:
            data = _http_post_json(self.url, payload, timeout=self.timeout)
            items = data.get("data", [])
            if not items:
                return None
            row = dict(zip(cols, items[0].get("d", []), strict=False))
            row["tv_symbol"] = items[0].get("s", "")
            return row
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Local operator PSX API (localhost:8000) — adaptive format parsing
# ---------------------------------------------------------------------------
class LocalPsxClient:
    # An operator-run service on localhost:8000 that is usually not running.
    # Windows does not refuse a closed local port quickly, so each call to a
    # dead host waits out the full socket timeout — and every PSX symbol in a
    # scan pays for it before the cache/TradingView fallback is reached.
    DOWN_PENALTY_S = 900.0

    def __init__(self) -> None:
        cfg = CFG.apis.get("psx_local", {})
        self.base = cfg.get("base_url", "http://localhost:8000/api/stock").rstrip("/")
        self.enabled = bool(cfg.get("enabled", True))
        self._down_until = 0.0

    def alive(self) -> bool:
        if not self.enabled or time.time() < self._down_until:
            return False
        try:
            _http_get(self.base + "/OGDC", timeout=4, retries=0)
            return True
        except Exception:
            self._down_until = time.time() + self.DOWN_PENALTY_S
            return False

    @staticmethod
    def _extract_price(obj: Any) -> float | None:
        price_keys = ("price", "close", "last", "ltp", "value", "rate", "current",
                      "currentprice", "lastprice", "tradedprice", "closeprice", "c")
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return float(obj)
        if isinstance(obj, dict):
            for k in price_keys:
                v = obj.get(k) or obj.get(k.lower()) or obj.get(k.upper())
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                    return float(v)
        return None

    @staticmethod
    def _extract_history(obj: Any) -> pd.DataFrame | None:
        """Look for a list of {date/..., close/price/...} anywhere in the payload."""
        def walk(node: Any) -> list[dict[str, Any]]:
            if isinstance(node, list) and node and isinstance(node[0], dict):
                return node
            if isinstance(node, dict):
                for v in node.values():
                    found = walk(v)
                    if found:
                        return found
            return []

        series = walk(obj)
        if not series:
            return None
        rows = []
        for item in series:
            date = None
            for k in ("date", "d", "time", "tradeDate", "trade_date", "x"):
                if k in item:
                    date = item[k]
                    break
            price = LocalPsxClient._extract_price(item)
            if date is None or price is None:
                continue
            try:
                dt = pd.to_datetime(date)
            except Exception:
                continue
            vol = item.get("volume") or item.get("vol") or item.get("v") or 0
            rows.append({"date": dt, "open": price, "high": price,
                         "low": price, "close": float(price), "volume": float(vol)})
        if len(rows) < 2:
            return None
        df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
        return df.reset_index(drop=True)

    def history(self, ticker: str) -> pd.DataFrame | None:
        if not self.enabled or time.time() < self._down_until:
            return None
        try:
            raw = _http_get(f"{self.base}/{urllib.parse.quote(ticker)}",
                            timeout=3, retries=0)
            payload = json.loads(raw)
        except Exception:
            self._down_until = time.time() + self.DOWN_PENALTY_S
            return None
        return self._extract_history(payload)


# ---------------------------------------------------------------------------
# News ingestion (RSS)
# ---------------------------------------------------------------------------
@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    link: str = ""
    summary: str = ""


def news_timestamp(published: str) -> float | None:
    """Epoch seconds for a feed timestamp, or None when it cannot be read.

    RSS dates are RFC-822 ("Tue, 21 Oct 2026 00:00:00 GMT") and Atom dates are
    ISO-8601; both arrive from the feeds this desk reads. None instead of 0.0
    keeps "no date" distinguishable from 1970, so a recency sort can put
    undated rows last instead of ranking them as the oldest news ever written.
    """
    text = (published or "").strip()
    if not text:
        return None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            when = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def _feed_xml(raw: bytes) -> ET.Element | None:
    """Parse a third-party feed, with entity declarations refused.

    `xml.etree` expands internal entities, so a hundred-line document can
    declare ten nested entities and ask the parser for a billion `&a;`. This
    is a feed from somebody else's server, not a trusted config file, so the
    parser is the one that refuses entities (`defusedxml`), while a plain
    `<!DOCTYPE rss …>` — which real feeds still carry — is allowed through.
    """
    try:
        return _safe_fromstring(raw)
    except Exception:  # noqa: BLE001 — a malformed feed costs that feed, nothing else
        return None


class NewsService:
    def __init__(self) -> None:
        self.feeds = CFG.section("news").get("feeds", [])
        self.max_per_feed = int(CFG.section("news").get("max_items_per_feed", 25))
        self._last: dict[str, Any] = {
            "checked_at": None, "feeds": len(self.feeds), "answered": 0,
            "failed": [], "items": 0,
        }

    def health(self) -> dict[str, Any]:
        """Which feeds answered the last fetch, and which did not.

        An empty headline list has two very different causes — a quiet tape and
        a dead source list — and a reader who cannot tell them apart will read
        an outage as "nothing happened today". The last fetch is described here
        without fetching again, so a health panel is free.
        """
        return dict(self._last)

    def fetch(self, limit_per_feed: int | None = None) -> list[NewsItem]:
        from concurrent.futures import ThreadPoolExecutor

        limit = limit_per_feed or self.max_per_feed

        def get(feed: dict) -> tuple[bytes | None, str]:
            try:
                return _http_get(feed["url"], timeout=12, retries=1,
                                 max_bytes=FEED_MAX_BYTES), ""
            except Exception as exc:  # noqa: BLE001 — one feed's fault is one feed's
                code = status_of(exc)
                return None, f"HTTP {code}" if code else type(exc).__name__

        # every feed is a separate host waiting on a separate socket; the
        # downloads are parallel but the assembly below walks the feeds in
        # declared order, so the item list (and which headlines tie in
        # `relevant`) is byte-for-byte what the sequential version produced.
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(self.feeds))),
                                thread_name_prefix="news") as ex:
            answers = list(ex.map(get, self.feeds))

        items: list[NewsItem] = []
        failed: list[dict[str, str]] = []
        # strict: `answers` is `ex.map` over `self.feeds`, so the two lists are
        # the same length by construction. If that ever stops being true, a
        # silent truncation would label every later item with the wrong
        # source, and a headline attributed to the wrong feed is worse than a
        # loud failure.
        for feed, (raw, error) in zip(self.feeds, answers, strict=True):
            if raw is None:
                failed.append({"feed": feed["name"], "error": error or "no answer"})
                continue
            root = _feed_xml(raw)
            if root is None:
                failed.append({"feed": feed["name"],
                               "error": "not parseable as a feed"})
                continue
            count = 0
            for el in root.iter("item"):
                if count >= limit:
                    break
                title = (el.findtext("title") or "").strip()
                if not title:
                    continue
                desc_raw = (el.findtext("description") or "").strip()
                summary = re.sub(r"<[^>]+>", " ", desc_raw)
                summary = re.sub(r"\s+", " ", summary).strip()[:400]
                items.append(NewsItem(
                    title=title, source=feed["name"],
                    # kept whole: cutting at 25 characters chopped the timezone
                    # off every RFC-822 date ("… 07:00:0"), and a timestamp
                    # missing its zone cannot be ordered against another feed's.
                    published=(el.findtext("pubDate") or "").strip(),
                    link=(el.findtext("link") or "").strip(),
                    summary=summary,
                ))
                count += 1
        self._last = {
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feeds": len(self.feeds),
            "answered": len(self.feeds) - len(failed),
            "failed": failed,
            "items": len(items),
        }
        return items

    def relevant(self, instrument: Instrument, items: list[NewsItem],
                 limit: int = 8) -> list[NewsItem]:
        terms = {instrument.key.lower(), instrument.name.lower()}
        for tag in instrument.tags:
            terms.add(tag.lower())
        sector_words = {w for w in re.split(r"\W+", instrument.sector.lower()) if len(w) > 3}
        scored: list[tuple[float, NewsItem]] = []
        for it in items:
            text = f"{it.title} {it.summary}".lower()
            score = 0.0
            for t in terms:
                if t in text:
                    score += 2.0
            for w in sector_words:
                if w in text:
                    score += 1.0
            if instrument.market == "PSX" and ("kse" in text or "psx" in text or
                                               "pakistan" in text or "karachi" in text):
                score += 1.5
            if score > 0:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:limit]]


# ---------------------------------------------------------------------------
# Facade — unified market data access with failover and accumulation
# ---------------------------------------------------------------------------
@dataclass
class SourceHealth:
    yahoo: bool = False
    tradingview: bool = False
    psx_local: bool = False
    binance: bool = False
    checked_at: str = ""

    @property
    def any_live(self) -> bool:
        return (self.yahoo or self.tradingview or self.psx_local
                or self.binance)


class MarketData:
    # equal-weight basket approximating KSE-100 (all Yahoo-covered heavyweights)
    KSE100_BASKET = ("OGDC", "PPL", "MARI", "LUCK", "HBL", "UBL", "MCB", "MEBL",
                     "FFC", "EFERT", "SYS", "PTC", "DGKC", "PSO", "POL", "FCCL")

    def __init__(self) -> None:
        self.yahoo = YahooClient()
        self.tv = TradingViewClient()
        self.local_psx = LocalPsxClient()
        self.news = NewsService()
        self._health = SourceHealth()
        self._tv_universe: pd.DataFrame | None = None
        self._tv_fetched_at: float = 0.0
        self._hist_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._kse100_cache: tuple[float, pd.DataFrame | None] = (0.0, None)

    # -- health -------------------------------------------------------------
    def health(self, force: bool = False) -> SourceHealth:
        if force or not self._health.checked_at:
            self._health = SourceHealth(
                yahoo=self.yahoo.history("GC=F", days=5) is not None,
                tradingview=not self.tv.scan_psx(columns=["name", "close"], limit=5).empty,
                psx_local=self.local_psx.alive(),
                binance=_binance().alive(),
                checked_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        return self._health

    # -- primary: OHLCV history ----------------------------------------------
    def history(self, instrument: Instrument, days: int | None = None,
                use_cache: bool = True) -> pd.DataFrame | None:
        # One door for every provider, so a bar that contradicts its own range is
        # repaired before any feature, regime, risk limit or backtest sees it —
        # `history()` is called by all of them, and each one would otherwise have
        # to remember to ask.
        return coerce_bar_range(self._history(instrument, days, use_cache))

    def _history(self, instrument: Instrument, days: int | None = None,
                 use_cache: bool = True) -> pd.DataFrame | None:
        # KSE-100: Yahoo does not carry the index — build an equal-weight
        # composite of index heavyweights (genuine, reproducible proxy)
        if instrument.key == "KSE100":
            df = self._kse100_composite(days)
            if df is not None and len(df) >= 60:
                return df
            cached = STORE.load("KSE100")
            if cached is not None and len(cached) >= 60:
                return cached

        if instrument.yahoo:
            ts, cached_df = self._hist_cache.get(instrument.yahoo, (0.0, None))
            df = cached_df if (cached_df is not None and time.time() - ts < 900.0) \
                else self.yahoo.history(instrument.yahoo, days=days)
            if df is not None and len(df) >= 10:
                if df is not cached_df:
                    self._hist_cache[instrument.yahoo] = (time.time(), df)
                if use_cache and instrument.market == "PSX":
                    # PSX: grow the accumulated store with this fetch and hand back
                    # the union — Yahoo has the history, the store has the bars
                    # TradingView caught on the days Yahoo was missing them.
                    return self._merge_with_store(instrument.key, df)
                if use_cache:
                    STORE.save(instrument.key, df)
                return df
            # Yahoo is down. For a crypto name the venue's own tape answers, but
            # it answers *without* writing the store: the published paper record
            # was measured on Yahoo closes, and a failover series must not
            # silently rewrite the history the record was built from.
            if instrument.binance:
                df = self._binance_history(instrument, days, persist=False)
                if df is not None:
                    return df
            cached = STORE.load(instrument.key)
            if cached is not None and len(cached) >= 10:
                return cached
            return None

        # PSX without Yahoo coverage: local API history -> cached store
        if instrument.market == "PSX":
            df = self.local_psx.history(instrument.psx_ticker or instrument.key)
            if df is not None and len(df) >= 10:
                if use_cache:
                    STORE.save(instrument.key, df)
                return df
            cached = STORE.load(instrument.key)
            if cached is not None and len(cached) >= 10:
                return cached
            # bootstrap: today's bar from TradingView (genuine accumulation)
            self._accumulate_psx_bar(instrument)
            return STORE.load(instrument.key)

        # A Binance-listed name Yahoo does not carry: the venue tape is primary
        # here, so the store accumulates the same series the order desk prices on.
        if instrument.binance:
            df = self._binance_history(instrument, days,
                                       persist=use_cache)
            if df is not None:
                return df
        return STORE.load(instrument.key)

    def _binance_history(self, instrument: Instrument, days: int | None,
                         persist: bool) -> pd.DataFrame | None:
        """Daily bars from the Binance spot tape, cached like every other source."""
        pair = (instrument.binance or "").upper()
        if not pair:
            return None
        ts, cached_df = self._hist_cache.get(pair, (0.0, None))
        if cached_df is not None and time.time() - ts < 900.0:
            return cached_df
        # Binance takes a bar count, not a lookback: 1000 daily bars is ~2.7 years
        # and is the API's own ceiling per request.
        limit = min(1000, max(60, int(days or 730)))
        df = _binance().klines(pair, "1d", limit)
        if df is None or len(df) < 10:
            return None
        self._hist_cache[pair] = (time.time(), df)
        if persist:
            STORE.save(instrument.key, df)
        return df

    @staticmethod
    def _merge_with_store(key: str, df: pd.DataFrame) -> pd.DataFrame:
        cached = STORE.load(key)
        if cached is None or cached.empty:
            STORE.save(key, df)
            return df
        live = df.copy()
        # The store is day-grained no matter what a vendor answers with:
        # `Store.save` stamps every row "%Y-%m-%d" and `load` reads those back at
        # midnight. Deduplicating at a finer grain than that is how one trading
        # day ends up with two bars — a source that answers with a 15:30
        # timestamp keeps it beside the store's 00:00 row for the same date,
        # `drop_duplicates` sees two different values, and the strategy is handed
        # a series whose rolling windows each contain that day twice. Worse, the
        # next `load` dedupes to the first row, which is the stored one, so the
        # live print is the bar that disappears — the opposite of the rule two
        # lines below. Normalize first, and copy: the caller's frame is also the
        # one sitting in `_hist_cache`, and restamping it in place would rewrite
        # what every other reader of that cache sees.
        live["date"] = pd.to_datetime(live["date"]).dt.normalize()
        merged = pd.concat([cached, live], ignore_index=True)
        # The live fetch wins on a duplicate date (`keep="last"` over
        # [store, live]), so today's bar follows the vendor instead of freezing at
        # the intraday price the store happened to catch. The comment here used to
        # claim the store's bars won, which is the opposite of this line.
        merged = (merged.drop_duplicates(subset="date", keep="last")
                  .sort_values("date").reset_index(drop=True))
        STORE.save(key, merged)
        return merged

    def _kse100_composite(self, days: int | None = None) -> pd.DataFrame | None:
        from concurrent.futures import ThreadPoolExecutor

        ts, cached = self._kse100_cache
        if cached is not None and time.time() - ts < 3600.0:
            return cached
        # 16 heavyweight histories, and every one of them is a socket wait. The
        # assembly below walks the declared basket order rather than the order
        # the fetches finished in, because the mean/slope math is float-summed
        # column-by-column: a different column order could move the third
        # decimal of the index, and the index gates every PSX position size.
        wanted = [k for k in self.KSE100_BASKET
                  if (resolve(k) is not None and resolve(k).yahoo)]

        def get(key: str) -> tuple[str, pd.DataFrame | None]:
            return key, self.history(resolve(key), days=days)

        with ThreadPoolExecutor(max_workers=min(5, max(1, len(wanted))),
                                thread_name_prefix="kse100") as ex:
            frames = dict(ex.map(get, wanted))
        closes, highs, lows, vols = {}, {}, {}, {}
        for key in wanted:
            df = frames[key]
            if df is None or len(df) < 80:
                continue
            s = df.set_index("date")
            closes[key] = s["close"]
            highs[key] = s["high"] / s["close"].shift(1)
            lows[key] = s["low"] / s["close"].shift(1)
            vols[key] = s["volume"]
        if len(closes) < 6:
            return None
        c = pd.DataFrame(closes).dropna()
        h = pd.DataFrame(highs).reindex(c.index).fillna(1.0)
        lo = pd.DataFrame(lows).reindex(c.index).fillna(1.0)
        v = pd.DataFrame(vols).reindex(c.index).fillna(0.0)
        ret = c.pct_change().dropna()
        if len(ret) < 60:
            return None
        close = 100.0 * (1.0 + ret.mean(axis=1)).cumprod()
        open_ = close.shift(1).fillna(100.0)
        high = (open_ * h.loc[ret.index].mean(axis=1)).clip(lower=close)
        low = (open_ * lo.loc[ret.index].mean(axis=1)).clip(upper=close)
        out = pd.DataFrame({
            "date": close.index,
            "open": open_.values,
            "high": np.maximum(high.values, close.values),
            "low": np.minimum(low.values, close.values),
            "close": close.values,
            "volume": v.loc[ret.index].sum(axis=1).values,
        }).reset_index(drop=True)
        out["close"] = out["close"].astype(float)
        out.attrs["name"] = "KSE100 (equal-weight composite of index heavyweights)"
        self._kse100_cache = (time.time(), out)
        return out

    def _accumulate_psx_bar(self, instrument: Instrument) -> None:
        q = self.tv.quote(instrument.tv or f"PSX:{instrument.key}")
        if not q or q.get("close") in (None, 0):
            return
        close = float(q["close"])
        change = float(q.get("change") or 0.0) / 100.0
        prev = close / (1 + change) if (1 + change) != 0 else close
        now = datetime.now()
        if now.weekday() >= 5:      # PSX is closed; this is the last session again
            return
        today = now.strftime("%Y-%m-%d")
        hi, lo = (close, prev) if close >= prev else (prev, close)
        STORE.upsert_row(instrument.key, today, prev, hi, lo, close,
                         float(q.get("volume") or 0))

    # -- cross-sectional PSX snapshot ----------------------------------------
    def psx_snapshot(self, refresh_after_s: float = 300.0) -> pd.DataFrame:
        if (self._tv_universe is not None and
                time.time() - self._tv_fetched_at < refresh_after_s):
            return self._tv_universe
        df = self.tv.scan_psx()
        if not df.empty:
            self._tv_universe = df
            self._tv_fetched_at = time.time()
            # accumulate today's bar for the whole exchange, but not on this
            # thread: the caller wants the snapshot, and 484 read/merge/write
            # round-trips used to be paid by every factor row and desk refresh
            _schedule_bar_accumulation(df)
        return self._tv_universe if self._tv_universe is not None else pd.DataFrame()

    # -- convenience ----------------------------------------------------------
    def series(self, symbol: str, days: int | None = None) -> pd.DataFrame | None:
        inst = resolve(symbol)
        if inst is None:
            return None
        return self.history(inst, days=days)
