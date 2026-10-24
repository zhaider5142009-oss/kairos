"""A correlated cluster is one position, not five.

`rules.max_correlation` was declared as a portfolio cap and read by nothing.
Every entry was sized on its own conviction, so a cycle could put eight PSX
scrips on the book whose five-day windows overlap and whose returns move
together: one macro shock, eight stops, all inside the gross-exposure breaker
because the breaker only ever looked at the total. The cap now has a reader —
the cluster budget below — and these tests pin what "correlated" means, what an
unknown pair is assumed to be, and that no unreadable row ever buys a free pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos import risk as RK
from kairos import rules as R
from kairos.autopilot import Autopilot
from kairos.universe import Instrument

_DATES = pd.date_range("2026-01-01", periods=80, freq="D")


def _rets(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, len(_DATES)), index=_DATES)


@pytest.fixture(scope="module")
def correlated() -> tuple[pd.Series, pd.Series, pd.Series]:
    """A loud pair and an independent one, from fixed seeds."""
    a = _rets(7)
    b = a * 0.9 + _rets(11) * 0.25
    c = _rets(23)
    return a, b, c


def _book(symbol, side, notional, returns, market="PSX"):
    return {"symbol": symbol, "market": market, "side": side,
            "notional": notional, "returns": returns}


# ------------------------------------------------------------- what is rho
def test_correlation_reads_only_the_dates_both_series_share(correlated):
    a, b, _c = correlated
    assert RK.correlation(a, b) is not None
    assert RK.correlation(a, b) > 0.8
    # a series that starts after the other is still answered on the overlap
    late = b.iloc[40:]
    assert RK.correlation(a, late) is not None


def test_a_pair_with_no_common_history_is_unknown_not_zero(correlated):
    a, _b, _c = correlated
    other = pd.Series(_rets(31).to_numpy(),
                      index=pd.date_range("2027-01-01", periods=80, freq="D"))
    assert RK.correlation(a, other) is None


def test_a_flat_series_has_no_defined_correlation(correlated):
    a, _b, _c = correlated
    flat = pd.Series(0.0, index=_DATES)
    assert RK.correlation(a, flat) is None


def test_too_few_returns_is_an_answer_of_none():
    short = pd.Series([100.0, 101.0, 100.5, 102.0], index=_DATES[:4])
    assert RK.daily_returns(short) is None


def test_the_bar_reader_deduplicates_a_repeated_session():
    """Two rows for one session would align the pair into a cartesian product."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    closes = [100.0 + i for i in range(30)]
    df = pd.DataFrame({"date": list(dates) + [dates[5]],
                       "close": closes + [closes[5]]})
    r = RK.returns_from_bars(df)
    assert r is not None and len(r) == 29


def test_a_frame_the_reader_cannot_parse_is_none_not_an_exception():
    assert RK.returns_from_bars(None) is None
    assert RK.returns_from_bars(pd.DataFrame({"last": [1.0, 2.0]})) is None


# ------------------------------------------------------------ the cluster cap
def test_a_full_cluster_takes_no_more_and_names_what_filled_it(correlated):
    a, b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    book = [_book("OGDC", "BUY", 300_000.0, b)]
    out = RK.cluster_budget(cand, book, 1_000_000.0,
                            rho_cap=0.7, cluster_cap_pct=30.0)
    assert out["allowed_notional"] == 0.0
    assert out["members"] == ["OGDC"]
    assert "OGDC" in out["reason"] and "30% cap" in out["reason"]


def test_a_partly_used_cluster_scales_the_entry_to_fit(correlated):
    a, b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    book = [_book("OGDC", "BUY", 200_000.0, b)]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 100_000.0
    assert out["scaled"] is True
    assert out["cluster_pct"] == 30.0
    assert out["rho_max"] > 0.8


def test_an_independent_name_is_not_charged_to_the_cluster(correlated):
    a, _b, c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    book = [_book("GOLD", "BUY", 300_000.0, c, market="GLOBAL")]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 150_000.0
    assert out["members"] == []


def test_opposite_sides_offset_rather_than_compound(correlated):
    a, b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    book = [_book("OGDC", "SELL", 300_000.0, b)]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 150_000.0
    assert out["members"] == []


def test_missing_history_is_assumed_correlated_within_one_market():
    """A short store is not evidence of independence."""
    cand = _book("MLCF", "BUY", 150_000.0, None)
    book = [_book("OGDC", "BUY", 300_000.0, None)]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 0.0
    assert out["members"] == ["OGDC"]
    assert out["rho_max"] == RK.UNKNOWN_RHO_SAME_MARKET


def test_missing_history_across_markets_is_not_clustered():
    cand = _book("MLCF", "BUY", 150_000.0, None)
    book = [_book("BTC", "BUY", 300_000.0, None, market="CRYPTO")]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 150_000.0
    assert out["rho_max"] == RK.UNKNOWN_RHO_CROSS_MARKET


def test_an_unreadable_notional_spends_the_whole_cluster(correlated):
    """Unreadable is the conservative answer, and not the free one."""
    a, b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    book = [{"symbol": "OGDC", "market": "PSX", "side": "BUY",
             "notional": "unreadable", "returns": b}]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 0.0
    assert out["members"] == ["OGDC"]


@pytest.mark.parametrize("equity", [0, -1, None, "n/a"])
def test_no_usable_equity_is_a_refusal_not_a_free_pass(equity, correlated):
    a, b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    out = RK.cluster_budget(cand, [_book("OGDC", "BUY", 1.0, b)], equity)
    assert out["allowed_notional"] == 0.0
    assert out["reason"]


def test_an_unreadable_cluster_cap_refuses(correlated):
    a, _b, _c = correlated
    cand = _book("MLCF", "BUY", 150_000.0, a)
    out = RK.cluster_budget(cand, [], 1_000_000.0, cluster_cap_pct="wide")
    assert out["allowed_notional"] == 0.0
    assert "cap" in out["reason"]


# ------------------------------------------------------- the cycle really asks
def test_the_autopilot_reads_the_book_and_skips_the_candidate(monkeypatch):
    """A cluster of two full-size PSX names leaves no room for a third."""
    from kairos import data as D
    monkeypatch.setattr(D.STORE, "load", lambda key: None, raising=False)
    a = Autopilot()
    inst = Instrument(key="MLCF", name="Maple Leaf", asset_class="equity",
                      sector="materials", currency="PKR", market="PSX")
    book = [{"symbol": "OGDC", "side": "BUY", "qty": 1_000, "last": 150.0},
            {"symbol": "HBL", "side": "BUY", "qty": 1_000, "last": 150.0},
            {"symbol": "MLCF", "side": "BUY", "qty": 1_000, "last": 150.0}]
    out = a._cluster_guard(inst, "BUY", 150_000.0, 1_000_000.0, book)
    assert out["allowed_notional"] == 0.0
    assert set(out["members"]) == {"OGDC", "HBL"}
    assert "MLCF" not in out["members"]


def test_the_cycle_reads_each_history_once(monkeypatch):
    from kairos import data as D
    calls: list[str] = []

    def load(key):
        calls.append(key)
        return None

    monkeypatch.setattr(D.STORE, "load", load, raising=False)
    a = Autopilot()
    inst = Instrument(key="OGDC", name="OGDC", asset_class="equity",
                      sector="energy", currency="PKR", market="PSX")
    book = [{"symbol": "HBL", "side": "BUY", "qty": 1.0, "last": 100.0}]
    for _ in range(3):
        a._cluster_guard(inst, "BUY", 10_000.0, 1_000_000.0, book)
    assert calls.count("OGDC") == 1 and calls.count("HBL") == 1


def test_the_declared_cap_is_no_longer_an_admission():
    assert "max_correlation" in R.ENFORCED
    assert "max_correlation" not in R.NOT_ENFORCED
    assert "max_cluster_pct" in R.ENFORCED
    assert "cluster cap" in R.describe()
