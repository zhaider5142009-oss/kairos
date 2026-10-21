"""The network fan-out must stay a fan-out, and must stay *identical*.

`engine.intermarket()` and `NewsService.fetch()` both walk a list of independent
HTTP endpoints. They were sequential, which meant ~17s of waiting for sockets
inside every prediction; they are now pooled. Parallelism is the easy half — the
contract that matters is that a caller cannot tell the difference: same result
order, same dropped-feed behaviour, same exception surfacing.

Nothing here touches the network.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import pandas as pd
import pytest

from kairos import data as D
from kairos import engine as E

# --------------------------------------------------------------------- helpers
def _frame(n: int, last: float) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": days, "open": last, "high": last, "low": last,
                         "close": [last] * n, "volume": [1.0] * n})


class Overlap:
    """Counts how many calls were in flight at once — the proof of pooling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.max_live = 0

    def __enter__(self):
        with self._lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.live -= 1

    def wait(self, seconds: float) -> None:
        with self:
            time.sleep(seconds)


class _FakeMd:
    def __init__(self, frames: dict[str, pd.DataFrame], overlap: Overlap,
                 raises: tuple[str, ...] = ()) -> None:
        self.frames, self.overlap, self.raises = frames, overlap, raises
        self.calls: list[str] = []

    def history(self, instrument):
        key = instrument.key
        self.calls.append(key)
        self.overlap.wait(0.02)
        if key in self.raises:
            raise OSError(f"{key} is down")
        return self.frames.get(key)


# ------------------------------------------------------------------ intermarket
def _engine(overlap, raises=(), keys=None):
    frames = {k: _frame(60, 100.0 + i) for i, k in enumerate(keys or E.INTERMARKET_KEYS)}
    md = _FakeMd(frames, overlap, raises)
    eng = E.PredictionEngine(md=md)
    return eng, frames


def test_intermarket_returns_every_symbol_in_declared_order():
    overlap = Overlap()
    eng, frames = _engine(overlap)
    ctx = eng.intermarket()
    assert list(ctx) == list(E.INTERMARKET_KEYS)
    for key in E.INTERMARKET_KEYS:
        assert ctx[key] is frames[key]


def test_intermarket_drops_a_symbol_that_is_too_short():
    overlap = Overlap()
    keys = list(E.INTERMARKET_KEYS)
    eng, _ = _engine(overlap, keys=keys)
    # a half-empty series is not context, it is noise — same as before
    eng.md.frames["VIX"] = _frame(10, 13.0)
    ctx = eng.intermarket()
    assert "VIX" not in ctx
    assert list(ctx) == [k for k in keys if k != "VIX"]


def test_intermarket_runs_the_fetches_concurrently():
    overlap = Overlap()
    eng, _ = _engine(overlap)
    eng.intermarket()
    assert overlap.max_live >= 2, (
        f"9 independent endpoints fetched with peak concurrency "
        f"{overlap.max_live} — the fan-out collapsed back to sequential")


def test_intermarket_still_refuses_to_swallow_a_dead_symbol():
    # the sequential loop let an OSError out of `history()` reach the caller;
    # a pooled version that quietly returned a partial dict would turn a data
    # outage into a silently degraded strategy context.
    overlap = Overlap()
    eng, _ = _engine(overlap, raises=("DXY",))
    with pytest.raises(OSError):
        eng.intermarket()
    assert eng._intermarket is None


def test_intermarket_is_cached_and_force_refetches():
    overlap = Overlap()
    eng, _ = _engine(overlap)
    eng.intermarket()
    first = len(eng.md.calls)
    eng.intermarket()
    assert len(eng.md.calls) == first
    eng.intermarket(force=True)
    assert len(eng.md.calls) == first * 2


# ------------------------------------------------------------------- news feeds
def _rss(title: str, n: int) -> bytes:
    items = "".join(f"<item><title>{title} {i}</title>"
                    f"<description>&lt;b&gt;body {i}&lt;/b&gt;</description>"
                    f"<pubDate>Tue, 21 Oct 2026 0{i}:00:00 GMT</pubDate>"
                    f"<link>https://x/{i}</link></item>" for i in range(n))
    return (f'<?xml version="1.0"?><rss version="2.0"><channel>{items}'
            f"</channel></rss>").encode()


def _news_service(monkeypatch, bodies: dict[str, object], overlap: Overlap):
    feeds = [{"name": f"f{i}", "url": f"https://f{i}"} for i in range(len(bodies))]
    got: list[str] = []

    def fake_get(url, timeout=20, headers=None, retries=2):
        got.append(url)
        with overlap:
            time.sleep(0.02)
        body = bodies[url]
        if isinstance(body, Exception):
            raise body
        return body

    monkeypatch.setattr(D, "_http_get", fake_get)
    ns = D.NewsService()
    ns.feeds = feeds
    ns.max_per_feed = 25
    return ns, got


def test_news_fetch_keeps_feed_order_and_content(monkeypatch):
    overlap = Overlap()
    bodies = {f"https://f{i}": _rss(f"s{i}", 3) for i in range(5)}
    ns, _ = _news_service(monkeypatch, bodies, overlap)
    items = ns.fetch(limit_per_feed=10)
    assert [i.source for i in items] == sum(([f"f{i}"] * 3 for i in range(5)), [])
    assert [i.title for i in items] == sum(([f"s{i} {j}" for j in range(3)]
                                           for i in range(5)), [])
    assert items[0].summary == "body 0"
    assert items[0].published == "Tue, 21 Oct 2026 00:00:00"


def test_news_fetch_runs_the_feeds_concurrently(monkeypatch):
    overlap = Overlap()
    bodies = {f"https://f{i}": _rss(f"s{i}", 2) for i in range(6)}
    ns, _ = _news_service(monkeypatch, bodies, overlap)
    ns.fetch()
    assert overlap.max_live >= 2, "7 RSS hosts fetched one at a time again"


def test_news_fetch_a_dead_feed_costs_only_that_feed(monkeypatch):
    overlap = Overlap()
    bodies = {f"https://f{i}": _rss(f"s{i}", 2) for i in range(5)}
    bodies["https://f2"] = ConnectionError("GET failed: https://f2")
    bodies["https://f3"] = b"<rss><broken"
    ns, _ = _news_service(monkeypatch, bodies, overlap)
    items = ns.fetch(limit_per_feed=10)
    assert sorted({i.source for i in items}) == ["f0", "f1", "f4"]
    assert len(items) == 6


def test_news_fetch_honours_the_per_feed_limit(monkeypatch):
    overlap = Overlap()
    bodies = {f"https://f{i}": _rss(f"s{i}", 9) for i in range(3)}
    ns, _ = _news_service(monkeypatch, bodies, overlap)
    assert len(ns.fetch(limit_per_feed=4)) == 12


def test_the_fetch_clients_do_not_mutate_themselves():
    # the pooling only holds if a fetch needs no lock. Reading `self.timeout`
    # from five threads is fine; writing `self._last_response` from five threads
    # is a race that shows up as wrong prices, not as an error.
    import inspect
    import re

    for fn in (D.YahooClient.history, D.TradingViewClient.quote):
        src = inspect.getsource(fn)
        writes = re.findall(r"^\s*self\.[A-Za-z_]+ *[:+]?=(?!=)", src, re.M)
        assert not writes, f"{fn.__qualname__} writes instance state: {writes}"


# -------------------------------------------------- predict: fits over context
class _CtxMd:
    """Answers every history call with one frame, and counts itself as in flight."""

    def __init__(self, frame: pd.DataFrame, overlap: Overlap) -> None:
        self.frame, self.overlap = frame, overlap
        self.news = None

    def history(self, instrument) -> pd.DataFrame:
        self.overlap.wait(0.02)
        return self.frame


def _ohlcv(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    spread = np.abs(rng.normal(0.0, 0.006, n))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close * (1 - spread), "high": close * (1 + spread),
        "low": close * (1 - spread), "close": close,
        "volume": rng.integers(10_000, 1_000_000, n).astype(float)})


@pytest.fixture
def overlapped(monkeypatch):
    from kairos.kse_regime import KseRegime

    overlap = Overlap()
    eng = E.PredictionEngine(md=_CtxMd(_ohlcv(), overlap))
    calls = {"battery": 0, "intermarket": 0, "factors": 0, "kse": 0}
    seen: dict = {}

    def battery(inst, df, horizon):
        calls["battery"] += 1
        overlap.wait(0.3)
        return []

    def intermarket(force=False):
        calls["intermarket"] += 1
        overlap.wait(0.3)
        return {"STUB": "context"}

    def factor_row(inst):
        calls["factors"] += 1
        overlap.wait(0.3)
        return pd.Series({"momentum_1m": 0.4, "momentum_3m": -0.2, "value": 1.1,
                          "quality": 0.3, "rank": 7})

    def kse(md):
        calls["kse"] += 1
        overlap.wait(0.3)
        return KseRegime("volatile", 0.6, 0.7, 0.0, 0.3, -0.05, "stub")

    monkeypatch.setattr(eng, "_battery", battery)
    monkeypatch.setattr(eng, "intermarket", intermarket)
    monkeypatch.setattr(eng, "factor_row", factor_row)
    monkeypatch.setattr(E, "kse_regime", kse)
    monkeypatch.setattr(E, "run_strategy_library",
                        lambda df, fe, ctx: seen.update(ctx) or [])
    eng._news_cache = (time.time(), [])
    return eng, overlap, calls, seen


def test_the_market_context_is_fetched_while_the_models_fit(overlapped):
    eng, overlap, calls, _ = overlapped
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"]
    assert calls == {"battery": 1, "intermarket": 1, "factors": 1, "kse": 0}
    assert overlap.max_live >= 2, (
        f"the fits and the context ran one after another (peak overlap "
        f"{overlap.max_live}) — the prediction is sequential again")


def test_the_overlapped_context_is_the_context_the_strategies_get(overlapped):
    eng, _, _, seen = overlapped
    eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert seen["intermarket"] == {"STUB": "context"}
    assert float(seen["factor_row"]["rank"]) == 7.0
    assert isinstance(seen["regime"], str) and seen["regime"]


def test_the_index_regime_scales_only_a_psx_position(overlapped):
    eng, _, calls, _ = overlapped
    res = eng._predict_uncached("LUCK", 5, use_wsl=False, include_news=False)
    assert res["ok"] and res["market"] == "PSX"
    assert calls["kse"] == 1
    assert res["kse_regime_multiplier"] == 0.7


def test_a_global_position_is_not_scaled_by_the_karachi_index(overlapped):
    eng, _, calls, _ = overlapped
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert calls["kse"] == 0
    assert res["kse_regime_multiplier"] == 1.0


def test_the_dossier_reports_nothing_it_did_not_compute(overlapped):
    # a field that is always empty is worse than no field: it reads like a
    # feature and gets trusted like one
    eng, *_ = overlapped
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert "features_snapshot" not in res
    assert res["computation_ms"] >= 0
