"""A price must be a usable number before it is a price.

Three links of the same chain were unguarded. Yahoo's chart payload spells a
missing close as a literal ``NaN``, and the loader only dropped ``None``, so the
NaN rode through as a real bar. On the desk, ``predict`` read the last bar
straight into ``price`` and returned ``ok: true`` with it — and that number is
what sizes a live order and stamps the paper book the published track record is
read from. And the PSX microstructure screen, whose whole job is to reject
unusable scrips, asked ``px < min_px`` of it: a NaN answers no, so it passed.

Nothing here touches the network.
"""

from __future__ import annotations

import json

import numpy as np

import pandas as pd

from kairos import data as D
from kairos import engine as E


# ------------------------------------------------------------------ the loader
class _Chart:
    """Stands in for query1.finance.yahoo.com with a hole in the closes."""

    def __init__(self, closes) -> None:
        self.closes = closes

    def __call__(self, url, timeout=None, retries=None):
        n = len(self.closes)
        return json.dumps({"chart": {"result": [{
            "meta": {"currency": "USD", "symbol": "XAUUSD"},
            "timestamp": [1_700_000_000 + 86_400 * i for i in range(n)],
            "indicators": {"quote": [{
                "close": self.closes,
                "open": [c if c is not None else 1.0 for c in self.closes],
                "high": [c if c is not None else 1.0 for c in self.closes],
                "low": [c if c is not None else 1.0 for c in self.closes],
                "volume": [1000.0] * n}]}}]}})


def _yahoo(monkeypatch, closes):
    c = D.YahooClient()
    monkeypatch.setattr(D, "_http_get", _Chart(closes))
    return c


def test_a_nan_close_is_not_a_bar(monkeypatch):
    df = _yahoo(monkeypatch, [100.0, 101.0, float("nan"), np.nan, 103.0]).history("XAUUSD")
    assert df is not None
    assert list(df["close"]) == [100.0, 101.0, 103.0]


def test_a_nan_close_never_becomes_the_open_of_the_next_bar(monkeypatch):
    # the loader backfills a missing open/high/low from that row's close, so a
    # NaN close that survived would have quietly become a valid-looking open.
    df = _yahoo(monkeypatch, [100.0, float("nan"), 104.0]).history("XAUUSD")
    assert df is not None
    assert np.isfinite(df[["open", "high", "low", "close"]].to_numpy()).all()


def test_a_chart_of_only_nan_is_no_history_at_all(monkeypatch):
    assert _yahoo(monkeypatch, [float("nan"), np.nan]).history("XAUUSD") is None


# ------------------------------------------------------------------ the desk
class _StubMd:
    def __init__(self, frame) -> None:
        self.frame = frame
        self.calls = 0

    def history(self, instrument) -> pd.DataFrame:
        self.calls += 1
        return self.frame


def _frame(closes) -> pd.DataFrame:
    closes = list(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000.0] * len(closes)})


def _engine(monkeypatch, frame):
    """A prediction whose model battery is a tripwire: it must never run for a
    frame the desk cannot price."""
    md = _StubMd(frame)
    eng = E.PredictionEngine(md=md)
    ran: list[str] = []

    def battery(inst, df, horizon):
        ran.append(inst.key)
        return []

    monkeypatch.setattr(eng, "_battery", battery)
    monkeypatch.setattr(eng, "intermarket", lambda force=False: {})
    monkeypatch.setattr(eng, "factor_row", lambda inst: pd.Series(dtype=float))
    monkeypatch.setattr(E, "kse_regime", lambda md: None)
    monkeypatch.setattr(E, "run_strategy_library", lambda df, fe, ctx: [])
    eng._news_cache = (0.0, [])
    return eng, ran


def test_a_prediction_refuses_a_nan_last_close(monkeypatch):
    eng, ran = _engine(monkeypatch, _frame([100.0 + i for i in range(60)]
                                           + [float("nan")]))
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is False
    assert "price" in res["error"].lower()
    assert ran == [], "the models fitted a series the desk cannot price"


def test_a_prediction_refuses_an_infinite_last_close(monkeypatch):
    eng, _ = _engine(monkeypatch, _frame([100.0 + i for i in range(60)] + [np.inf]))
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is False


def test_a_prediction_refuses_a_zero_last_close(monkeypatch):
    # A quote of exactly zero is never a print for a stock, future, FX pair or
    # CFD; it is a hole. And every number in the sizing block is a quotient by
    # this one, so a zero that passed as "finite" divided by it.
    eng, ran = _engine(monkeypatch, _frame([100.0 + i for i in range(60)] + [0.0]))
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is False
    assert ran == []


def test_a_prediction_refuses_a_negative_last_close(monkeypatch):
    eng, _ = _engine(monkeypatch, _frame([100.0 + i for i in range(60)] + [-4.0]))
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is False


def test_a_usable_close_still_prices(monkeypatch):
    eng, ran = _engine(monkeypatch, _frame([100.0 + i for i in range(61)]))
    res = eng._predict_uncached("GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is True
    assert res["price"] == 160.0
    assert ran, "the guard must not take the models offline"


# ------------------------------------------------------ the liquidity screen
class _BoomMd:
    def __init__(self, frame=None, boom=False) -> None:
        self.frame, self.boom = frame, boom

    def history(self, instrument) -> pd.DataFrame:
        if self.boom:
            raise OSError("psx store is down")
        return self.frame


def _screen(monkeypatch, md):
    from kairos.autopilot import Autopilot

    monkeypatch.setattr(E.ENGINE, "md", md)
    return Autopilot()._psx_liquid(E.resolve("OGDC"))


def test_the_penny_screen_refuses_an_unreadable_price(monkeypatch):
    # `px < min_px` is False for a NaN, so an unusable close used to clear the
    # screen that exists to reject unusable scrips.
    reason = _screen(monkeypatch, _BoomMd(_frame([45.0] * 25 + [float("nan")])))
    assert reason and "usable number" in reason


def test_the_penny_screen_refuses_a_history_it_cannot_read(monkeypatch):
    reason = _screen(monkeypatch, _BoomMd(boom=True))
    assert reason and "unreadable" in reason


def test_the_penny_screen_still_passes_a_liquid_scrip(monkeypatch):
    frame = _frame([450.0] * 30)
    frame["volume"] = [500_000.0] * 30
    assert _screen(monkeypatch, _BoomMd(frame)) is None
