"""A cluster budget that adds dollars to rupees is not a budget.

`risk.cluster_budget` compares the money a correlated cluster already holds with
the cap the candidate's own venue allows it, and it did that arithmetic on
whatever number each row carried. Three books, two currencies: a 70,000 dollar
MT5 holding in front of a 1,000,000 rupee JS Global book was charged as 70,000
rupees — 7% of equity — while the same position is 2,100% of it. One correlated
cluster could therefore be filled again and again, and the brake meant to stop
"one shock, eight stops" never saw the shock. The reverse reading blocked valid
entries: rupee holdings counted as dollars against a dollar book.

The other half of the hole was silence. A holding with no readable price produced
`qty * 0.0`, and `notional or 0.0` swallowed a None, so both reached the budget as
zero — free room in the cluster that the code elsewhere calls blindness. Now a row
is either stated in the candidate's currency or it is `None`, and `None` spends
the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos.data as D
from kairos import risk as RK
from kairos.autopilot import Autopilot
from kairos.universe import Instrument

RATE = 300.0
_DATES = pd.date_range("2026-01-01", periods=80, freq="D")
_TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")


def _rets(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, len(_DATES)), index=_DATES)


# a pair that moves together, and one that does not
LOUD = _rets(7)
ALSO_LOUD = LOUD * 0.9 + _rets(11) * 0.25
INDEPENDENT = _rets(23)


def _inst(key: str, ccy: str, market: str) -> Instrument:
    return Instrument(key=key, name=key, asset_class="equity", sector="test",
                      currency=ccy, market=market)


@pytest.fixture
def ap(monkeypatch) -> Autopilot:
    """An autopilot whose history reads are the seeded pair above.

    The cluster's membership question is correlation, and these tests are about
    what the cluster is *charged*. Fixing the series here keeps the join decided
    by the fixture rather than by whatever the real cache happens to hold.
    """
    a = Autopilot()
    series = {"GOLD": LOUD, "SILVER": ALSO_LOUD, "OGDC": LOUD,
              "MLCF": ALSO_LOUD, "WHEAT": INDEPENDENT, "TESTX": None}
    monkeypatch.setattr(Autopilot, "_returns_for",
                        lambda self, key: series.get(key))
    return a


@pytest.fixture
def rate(monkeypatch) -> float:
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    monkeypatch.setattr(D, "_fx_cache", None)
    return RATE


def _cached(monkeypatch, tmp_path, close, date: str) -> None:
    """Put one USDPKR bar in the store the rate reader actually opens."""
    monkeypatch.setattr(D.STORE, "dir", tmp_path)
    monkeypatch.setattr(D, "_fx_cache", None)
    pd.DataFrame([{"date": date, "open": close, "high": close, "low": close,
                   "close": close, "volume": 0.0}]).to_csv(
        tmp_path / "USDPKR.csv", index=False)


# ------------------------------------------------------------------ the rate
def test_a_series_in_the_cache_answers_the_rate(monkeypatch, tmp_path):
    _cached(monkeypatch, tmp_path, 276.75, _TODAY)
    assert D.fx_rate("USD", "PKR") == 276.75
    assert D.fx_rate("PKR", "USD") == pytest.approx(1 / 276.75)


def test_the_same_currency_never_asks_the_tape(monkeypatch, tmp_path):
    """A rupee-to-rupee budget is the common case and must cost no disk."""
    reads: list[str] = []
    monkeypatch.setattr(D.STORE, "load",
                        lambda key: reads.append(key) or None, raising=False)
    monkeypatch.setattr(D, "_fx_cache", None)
    assert D.fx_rate("PKR", "PKR") == 1.0
    assert reads == []


def test_no_series_is_no_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(D.STORE, "dir", tmp_path)
    monkeypatch.setattr(D, "_fx_cache", None)
    assert D.fx_rate("USD", "PKR") is None
    assert D.fx_rate("PKR", "USD") is None


def test_a_rate_weeks_old_is_history_not_a_rate(monkeypatch, tmp_path):
    """A managed peg devalues in steps; a stale print would understate it."""
    stale = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    _cached(monkeypatch, tmp_path, 240.0, stale)
    assert D.fx_rate("USD", "PKR") is None


@pytest.mark.parametrize("close", [0.0, -5.0, float("nan")])
def test_an_unusable_print_is_no_rate(monkeypatch, tmp_path, close):
    _cached(monkeypatch, tmp_path, close, _TODAY)
    assert D.fx_rate("USD", "PKR") is None


def test_a_currency_this_desk_does_not_hold_is_no_rate(monkeypatch, rate):
    assert D.fx_rate("EUR", "PKR") is None
    assert D.fx_rate("", "PKR") is None


def test_the_rate_is_read_once_not_per_row(monkeypatch, tmp_path):
    """A cycle asks this for every candidate against every holding."""
    reads: list[str] = []

    def load(key):
        reads.append(key)
        return pd.DataFrame([{"date": pd.Timestamp(_TODAY), "close": 276.0}])

    monkeypatch.setattr(D.STORE, "load", load, raising=False)
    monkeypatch.setattr(D, "_fx_cache", None)
    for _ in range(9):
        assert D.fx_rate("USD", "PKR") == 276.0
    assert reads == ["USDPKR"]


# ------------------------------------------------------- the book, one currency
def test_a_dollar_holding_is_priced_for_the_rupee_book(ap, rate):
    book = ap._cluster_book([{"symbol": "GOLD", "side": "BUY", "qty": 70.0,
                              "last": 1000.0, "venue": "MT5"}], "MLCF", "PKR")
    assert book[0]["notional"] == pytest.approx(70_000.0 * RATE)


def test_a_rupee_holding_is_priced_for_the_dollar_book(ap, rate):
    book = ap._cluster_book([{"symbol": "OGDC", "side": "BUY", "qty": 1000.0,
                              "last": 150.0, "venue": "JS_GLOBAL"}],
                            "GOLD", "USD")
    assert book[0]["notional"] == pytest.approx(150_000.0 / RATE)


def test_one_currency_needs_no_conversion_at_all(ap, monkeypatch):
    monkeypatch.setattr(D, "usd_pkr", lambda: pytest.fail("read the tape"))
    book = ap._cluster_book([{"symbol": "OGDC", "side": "BUY", "qty": 1000.0,
                              "last": 150.0, "venue": "JS_GLOBAL"}],
                            "MLCF", "PKR")
    assert book[0]["notional"] == 150_000.0


def test_a_name_the_registry_does_not_know_is_still_sited(ap, rate):
    """A venue settles in one currency, and that is a fact about the row."""
    book = ap._cluster_book([{"symbol": "GOLD.a", "side": "BUY", "qty": 70.0,
                              "last": 1000.0, "venue": "MT5"}], "MLCF", "PKR")
    assert book[0]["notional"] == pytest.approx(70_000.0 * RATE)


def test_a_holding_with_no_price_is_blind_not_flat(ap, rate):
    """Spot balances report no price; a held row that cannot be valued is dark."""
    book = ap._cluster_book([{"symbol": "BTCUSDT", "side": "BUY", "qty": 0.5,
                              "last": None, "entry": None}], "MLCF", "PKR")
    assert book[0]["notional"] is None


@pytest.mark.parametrize("qty, last, entry", [
    # a quantity printed as a word — the old path read that as flat
    ("n/a", 1000.0, 900.0),
    (None, 1000.0, 900.0),
    # no price anywhere, including the entry
    (70.0, "n/a", "n/a"),
    (70.0, None, None),
    (70.0, 0.0, 0.0),
])
def test_an_unmeasurable_holding_says_so(ap, rate, qty, last, entry):
    book = ap._cluster_book([{"symbol": "GOLD", "side": "BUY", "qty": qty,
                              "last": last, "entry": entry}], "MLCF", "PKR")
    assert book[0]["notional"] is None


def test_a_flat_row_charges_nothing(ap, rate):
    """Nothing held is not the same fact as something unreadable."""
    book = ap._cluster_book([{"symbol": "GOLD", "side": "BUY", "qty": 0.0,
                              "last": None, "entry": None}], "MLCF", "PKR")
    assert book[0]["notional"] == 0.0


def test_a_holding_with_no_rate_is_not_counted_as_free(ap, monkeypatch):
    """A missing rate is a veto, not a zero and not a guessed 1.0."""
    monkeypatch.setattr(D, "usd_pkr", lambda: None)
    book = ap._cluster_book([{"symbol": "GOLD", "side": "BUY", "qty": 70.0,
                              "last": 1000.0, "venue": "MT5"}], "MLCF", "PKR")
    assert book[0]["notional"] is None


# ------------------------------------------------------------- the budget reads
def test_a_dollar_cluster_fills_a_rupee_cap(ap, rate):
    """The fail-open: HEAD charged 7% and let a second full-size entry through."""
    cand = _inst("MLCF", "PKR", "PSX")
    book = [{"symbol": "GOLD", "side": "BUY", "qty": 70.0, "last": 1000.0,
             "venue": "MT5"}]
    out = ap._cluster_guard(cand, "BUY", 150_000.0, 1_000_000.0, book)
    assert out["members"] == ["GOLD"]
    assert out["allowed_notional"] == 0.0
    assert "2100% of equity" in out["reason"]


def test_a_rupee_cluster_does_not_fill_a_dollar_cap(ap, rate):
    """The same mistake read the other way refused an entry that fitted."""
    cand = _inst("GOLD", "USD", "GLOBAL")
    book = [{"symbol": "OGDC", "side": "BUY", "qty": 1000.0, "last": 150.0,
             "venue": "JS_GLOBAL"}]
    out = ap._cluster_guard(cand, "BUY", 15_000.0, 100_000.0, book)
    assert out["allowed_notional"] == 15_000.0
    assert out["scaled"] is False
    # 500 dollars of rupee holdings plus the entry, against a 100,000 dollar book
    assert out["cluster_pct"] == pytest.approx(15.5, abs=0.1)


def test_a_cluster_with_no_rate_available_is_charged_the_cap(ap, monkeypatch):
    cand = _inst("MLCF", "PKR", "PSX")
    book = [{"symbol": "GOLD", "side": "BUY", "qty": 70.0, "last": 1000.0,
             "venue": "MT5"}]
    monkeypatch.setattr(D, "usd_pkr", lambda: None)
    out = ap._cluster_guard(cand, "BUY", 150_000.0, 1_000_000.0, book)
    assert out["allowed_notional"] == 0.0
    assert "cannot be measured" in out["reason"]


def test_an_independent_name_costs_the_cluster_nothing_whatever_its_currency(
        ap, rate):
    cand = _inst("MLCF", "PKR", "PSX")
    book = [{"symbol": "WHEAT", "side": "BUY", "qty": 70.0, "last": 1000.0,
             "venue": "MT5"}]
    out = ap._cluster_guard(cand, "BUY", 150_000.0, 1_000_000.0, book)
    assert out["members"] == []
    assert out["allowed_notional"] == 150_000.0


# ------------------------------------------------------------ the budget alone
def test_a_notional_of_none_spends_the_cap():
    a, b = LOUD, ALSO_LOUD
    cand = {"symbol": "MLCF", "market": "PSX", "side": "BUY",
            "notional": 150_000.0, "returns": a}
    book = [{"symbol": "OGDC", "market": "PSX", "side": "BUY",
             "notional": None, "returns": b}]
    out = RK.cluster_budget(cand, book, 1_000_000.0)
    assert out["allowed_notional"] == 0.0
    assert "cannot be measured" in out["reason"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "-1"])
def test_a_notional_that_is_not_a_size_spends_the_cap(value):
    cand = {"symbol": "MLCF", "market": "PSX", "side": "BUY",
            "notional": 150_000.0, "returns": LOUD}
    book = [{"symbol": "OGDC", "market": "PSX", "side": "BUY",
             "notional": value, "returns": ALSO_LOUD}]
    assert RK.cluster_budget(cand, book, 1_000_000.0)["allowed_notional"] == 0.0


def test_a_row_that_states_no_notional_is_flat_not_blind():
    """Callers that hand over a shape without the key mean an empty position."""
    cand = {"symbol": "MLCF", "market": "PSX", "side": "BUY",
            "notional": 150_000.0, "returns": LOUD}
    out = RK.cluster_budget(cand, [{"symbol": "OGDC", "market": "PSX",
                                    "side": "BUY", "returns": ALSO_LOUD}],
                            1_000_000.0)
    assert out["allowed_notional"] == 150_000.0
    assert out["cluster_pct"] == 15.0
