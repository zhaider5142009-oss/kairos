"""A restart must not refit models the bars have not changed under.

The battery memo lived in the process, so every daemon restart paid the whole
stat + ML + neural refit again — about 153 s for the eleven-name watchlist on
this class of CPU, which is the first cycle after every deploy and every reboot.
Over a weekend the bars are byte-identical, so the fits are too.

The store is allowed to be wrong in exactly one direction: if it cannot be read,
is corrupt, was written by other model code, or was written with a different set
of batteries enabled, the engine refits. It may be slower; it may never serve a
forecast the current code would not have produced. These tests pin that line.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from kairos import engine as E
from kairos.models_stat import ModelForecast
from kairos.universe import resolve


def _frame(n: int, last: float) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": days, "open": [last] * n, "high": [last] * n,
                         "low": [last] * n, "close": [last] * n,
                         "volume": [1.0] * n})


class _Md:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def history(self, instrument):
        return self._df


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private cache directory and a switchable set of battery flags."""
    flags = {"use_stat": True, "use_ml": True, "use_neural": True}
    monkeypatch.setattr(E, "CFG", SimpleNamespace(cache_dir=tmp_path / "cache",
                                                  engine=flags))
    monkeypatch.setattr(E, "_battery_fp", None)
    return tmp_path / "cache"


@pytest.fixture
def fits(monkeypatch):
    """Three cheap batteries that record every call they are asked to make."""
    calls: list[str] = []

    def fake(name):
        def run(*args, **kwargs):
            calls.append(name)
            horizon = args[1] if len(args) > 1 else kwargs.get("horizon", 5)
            return [ModelForecast(model=name, direction=1, expected_return=0.01,
                                  horizon=int(horizon), confidence=0.5)]
        return run

    monkeypatch.setattr(E, "run_statistical_battery", fake("stat"))
    monkeypatch.setattr(E, "tree_stack_forecast", fake("ml"))
    monkeypatch.setattr(E, "neural_battery", fake("neural"))
    return calls


def _engine(df: pd.DataFrame) -> E.PredictionEngine:
    return E.PredictionEngine(md=_Md(df))


def test_a_restart_reuses_fits_for_bars_it_already_fitted(store, fits):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    first = _engine(df)._battery(inst, df, 5)
    assert len(fits) == 3

    # a new engine is a restart: the process memo is empty, the disk store is not
    second = _engine(df)._battery(inst, df, 5)
    assert len(fits) == 3, "the restart refitted bars that had not moved"
    assert second == first


def test_a_new_close_refits(store, fits):
    inst, horizon = resolve("OGDC"), 5
    _engine(_frame(60, 100.0))._battery(inst, _frame(60, 100.0), horizon)
    _engine(_frame(60, 101.0))._battery(inst, _frame(60, 101.0), horizon)
    assert len(fits) == 6


def test_a_new_bar_refits_even_when_the_close_repeats(store, fits):
    inst, horizon = resolve("OGDC"), 5
    _engine(_frame(60, 100.0))._battery(inst, _frame(60, 100.0), horizon)
    longer = _frame(61, 100.0)
    _engine(longer)._battery(inst, longer, horizon)
    assert len(fits) == 6


def test_fits_from_other_model_code_are_not_served(store, fits, monkeypatch):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    monkeypatch.setattr(E, "_battery_fingerprint", lambda: "aaaa0000")
    _engine(df)._battery(inst, df, 5)
    monkeypatch.setattr(E, "_battery_fingerprint", lambda: "bbbb1111")
    _engine(df)._battery(inst, df, 5)
    assert len(fits) == 6, "a deploy changed the models but kept their old fits"


def test_fits_made_with_another_battery_set_are_not_served(store, fits):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    _engine(df)._battery(inst, df, 5)
    E.CFG.engine["use_neural"] = False
    _engine(df)._battery(inst, df, 5)
    assert len(fits) == 5, "the engine served a fit from a battery it no longer runs"


def test_a_corrupt_store_refits_instead_of_raising(store, fits):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    _engine(df)._battery(inst, df, 5)
    path = next(store.glob("battery/*.json"))
    path.write_text("{not json", encoding="utf-8")
    predicted = _engine(df)._battery(inst, df, 5)
    assert len(fits) == 6
    assert predicted and predicted[0].model == "stat"


def test_an_unwritable_store_costs_time_not_a_prediction(store, fits):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("not a directory", encoding="utf-8")
    predicted = _engine(df)._battery(inst, df, 5)
    assert len(fits) == 3
    assert len(predicted) == 3


def test_each_horizon_has_its_own_file(store, fits):
    df, inst = _frame(60, 100.0), resolve("OGDC")
    engine = _engine(df)
    engine._battery(inst, df, 5)
    engine._battery(inst, df, 10)
    assert {p.name for p in store.glob("battery/*.json")} == {"OGDC-5.json",
                                                             "OGDC-10.json"}


def test_the_memory_memo_stays_bounded(store, fits):
    """The disk store is bounded by symbols × horizons; the dict is not, unless
    it evicts — and an always-on daemon walked a new bar set per cycle."""
    inst, engine = resolve("OGDC"), _engine(_frame(60, 100.0))
    for i in range(80):
        engine._battery(inst, _frame(60, 100.0 + i), 5)
    assert len(engine._battery_cache) <= 64


def test_the_fingerprint_covers_the_named_sources(store, monkeypatch):
    real = E._battery_fingerprint()
    assert len(real) == 16
    monkeypatch.setattr(E, "_battery_fp", None)
    monkeypatch.setattr(E, "_BATTERY_SOURCES", ("__init__.py",))
    assert E._battery_fingerprint() != real
