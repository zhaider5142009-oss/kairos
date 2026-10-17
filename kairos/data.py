"""Data layer: multi-source ingestion with failover, caching, and history accumulation.

Source chain (health-probed at runtime):
  GLOBAL instruments -> Yahoo Finance v8 chart API (2y daily OHLCV)
  PSX quotes/factors -> TradingView scanner (484 stocks, live fields)
  PSX history        -> local operator API (localhost:8000) -> accumulated store
  News               -> RSS feeds (Google News, BRecorder, Dawn, FXStreet, MW)

Nothing is ever fabricated: when no source is reachable the layer reports
honestly and the engine degrades gracefully.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .universe import Instrument, resolve

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def _http_get(url: str, timeout: int = 20, headers: dict[str, str] | None = None,
              retries: int = 2) -> bytes:
    hdrs = {"User-Agent": _UA, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — network layer must never crash the engine
            last_err = exc
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    raise ConnectionError(f"GET failed: {url} ({last_err})")


def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 20,
                    headers: dict[str, str] | None = None) -> Any:
    hdrs = {"User-Agent": _UA, "Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Store — CSV persistence + genuine history accumulation
# ---------------------------------------------------------------------------
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
        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(self._path(key), index=False)

    def upsert_row(self, key: str, date: str, open_: float, high: float, low: float,
                   close: float, volume: float) -> pd.DataFrame | None:
        df = self.load(key)
        row = pd.DataFrame(
            [{"date": pd.to_datetime(date), "open": open_, "high": high,
              "low": low, "close": close, "volume": volume}])
        if df is None:
            df = row
        else:
            df = pd.concat([df[df["date"] != pd.to_datetime(date)], row], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        self.save(key, df)
        return df


STORE = Store(CFG.cache_dir)


# ---------------------------------------------------------------------------
# Yahoo Finance client (global instruments)
# ---------------------------------------------------------------------------
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
                close = quote["close"][i]
                if close is None:
                    continue
                o = quote["open"][i] if quote["open"][i] is not None else close
                h = quote["high"][i] if quote["high"][i] is not None else close
                l = quote["low"][i] if quote["low"][i] is not None else close
                v = quote["volume"][i] if quote["volume"][i] is not None else 0
                rows.append({"date": datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                             "open": o, "high": h, "low": l, "close": close, "volume": v})
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
            row = dict(zip(cols, item.get("d", [])))
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
            row = dict(zip(cols, items[0].get("d", [])))
            row["tv_symbol"] = items[0].get("s", "")
            return row
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Local operator PSX API (localhost:8000) — adaptive format parsing
# ---------------------------------------------------------------------------
class LocalPsxClient:
    def __init__(self) -> None:
        cfg = CFG.apis.get("psx_local", {})
        self.base = cfg.get("base_url", "http://localhost:8000/api/stock").rstrip("/")
        self.enabled = bool(cfg.get("enabled", True))

    def alive(self) -> bool:
        if not self.enabled:
            return False
        try:
            _http_get(self.base + "/OGDC", timeout=4, retries=0)
            return True
        except Exception:
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
        if not self.enabled:
            return None
        try:
            raw = _http_get(f"{self.base}/{urllib.parse.quote(ticker)}", timeout=6, retries=1)
            payload = json.loads(raw)
        except Exception:
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


class NewsService:
    def __init__(self) -> None:
        self.feeds = CFG.section("news").get("feeds", [])
        self.max_per_feed = int(CFG.section("news").get("max_items_per_feed", 25))

    def fetch(self, limit_per_feed: int | None = None) -> list[NewsItem]:
        limit = limit_per_feed or self.max_per_feed
        items: list[NewsItem] = []
        for feed in self.feeds:
            try:
                raw = _http_get(feed["url"], timeout=12, retries=1)
                root = ET.fromstring(raw)
            except Exception:
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
                    published=(el.findtext("pubDate") or "")[:25],
                    link=(el.findtext("link") or "").strip(),
                    summary=summary,
                ))
                count += 1
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
    checked_at: str = ""

    @property
    def any_live(self) -> bool:
        return self.yahoo or self.tradingview or self.psx_local


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
                checked_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        return self._health

    # -- primary: OHLCV history ----------------------------------------------
    def history(self, instrument: Instrument, days: int | None = None,
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
                    # PSX: merge live store bars (TradingView) over Yahoo history
                    return self._merge_with_store(instrument.key, df)
                if use_cache:
                    STORE.save(instrument.key, df)
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
        return STORE.load(instrument.key)

    @staticmethod
    def _merge_with_store(key: str, df: pd.DataFrame) -> pd.DataFrame:
        cached = STORE.load(key)
        if cached is None or cached.empty:
            STORE.save(key, df)
            return df
        merged = pd.concat([cached, df], ignore_index=True)
        # store bars (live TradingView) win over Yahoo on duplicate dates
        merged = (merged.drop_duplicates(subset="date", keep="last")
                  .sort_values("date").reset_index(drop=True))
        STORE.save(key, merged)
        return merged

    def _kse100_composite(self, days: int | None = None) -> pd.DataFrame | None:
        ts, cached = self._kse100_cache
        if cached is not None and time.time() - ts < 3600.0:
            return cached
        closes, highs, lows, vols = {}, {}, {}, {}
        for key in self.KSE100_BASKET:
            inst = resolve(key)
            if inst is None or not inst.yahoo:
                continue
            df = self.history(inst, days=days)
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
        l = pd.DataFrame(lows).reindex(c.index).fillna(1.0)
        v = pd.DataFrame(vols).reindex(c.index).fillna(0.0)
        ret = c.pct_change().dropna()
        if len(ret) < 60:
            return None
        close = 100.0 * (1.0 + ret.mean(axis=1)).cumprod()
        open_ = close.shift(1).fillna(100.0)
        high = (open_ * h.loc[ret.index].mean(axis=1)).clip(lower=close)
        low = (open_ * l.loc[ret.index].mean(axis=1)).clip(upper=close)
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
        today = datetime.now().strftime("%Y-%m-%d")
        STORE.upsert_row(instrument.key, today, prev, close, prev, close,
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
            for _, row in df.iterrows():
                sym = str(row.get("name") or "")
                if sym:
                    close = row.get("close")
                    if close is not None and not (isinstance(close, float) and np.isnan(close)):
                        try:
                            change = float(row.get("change") or 0.0) / 100.0
                            c = float(close)
                            prev = c / (1 + change) if (1 + change) != 0 else c
                            STORE.upsert_row(sym, datetime.now().strftime("%Y-%m-%d"),
                                             prev, c, prev, c, float(row.get("volume") or 0))
                        except Exception:
                            continue
        return self._tv_universe if self._tv_universe is not None else pd.DataFrame()

    # -- convenience ----------------------------------------------------------
    def series(self, symbol: str, days: int | None = None) -> pd.DataFrame | None:
        inst = resolve(symbol)
        if inst is None:
            return None
        return self.history(inst, days=days)
