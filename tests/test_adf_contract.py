"""The ADF test that chooses ARIMA's differencing order, and its return shape.

`arima_forecast` asks statsmodels whether the log-price is stationary and uses
the answer to pick `d` — the order every one of the nine AICs in the grid below
that choice is then compared across. It used to read that answer out of position
1 of a tuple, which is a shape the library has announced it will stop returning
by default ("in release 0.16 or after July 2027, whichever is later"), and it
said so by writing a FutureWarning into the daemon's log on every forecast — the
only notice anyone would get, in the one file nobody opens.

These tests hold three separate things shut: the leg is silent, the number that
drives the choice is the one the code names, and a statsmodels too old to know
the option still works rather than falling through to the default.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ST

_ORDER = re.compile(r"order=\((\d+),\s*(\d+),\s*(\d+)\)")


def _close(n: int = 240, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))) * 100.0)


def _d(forecast) -> int:
    m = _ORDER.search(forecast.detail or "")
    assert m, f"the ARIMA row no longer prints the order it fitted: {forecast.detail!r}"
    return int(m.group(2))


class _Result:
    """The newer return: a named result whose `pvalue` is not the tuple's."""

    def __init__(self, pvalue: float) -> None:
        self.pvalue = pvalue


def test_a_forecast_emits_no_library_warning():
    """The warning was per forecast, not per process, so a scan loop writes
    thousands of identical lines a day into the operator's log — and a log that
    full is a log nobody reads.

    `record=True` with "always", rather than turning the warning into an error:
    `Warning` subclasses `Exception`, so the leg's own `except Exception: d = 1`
    absorbs an errored warning and the test would pass while the noise stayed.
    Counting what the interpreter actually emitted is the only assertion that
    can see the defect it is written for.
    """
    import warnings

    from kairos.models_stat import arima_forecast

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        f = arima_forecast(_close(), horizon=3)
    assert f is not None and f.model == "ARIMA"
    adf = [str(x.message) for x in seen if "adfuller" in str(x.message)]
    assert not adf, f"the leg warned again: {adf[0][:120]}"


def test_the_differencing_order_is_read_from_the_field_the_call_asks_for(monkeypatch):
    """The named result and the tuple are made to disagree on purpose: only then
    does the fitted order say which of them the code actually consulted."""
    from kairos.models_stat import arima_forecast

    def adfuller(x, autolag=None, result_object=None, **kw):
        assert result_object is True, "the call no longer names the shape it unpacks"
        return _Result(0.001)          # stationary by the named field…

    monkeypatch.setattr(ST, "adfuller", adfuller)
    f = arima_forecast(_close(), horizon=3)
    assert f is not None
    assert _d(f) == 0, "the stationarity answer the code asked for did not drive the order"


def test_a_statsmodels_without_the_option_is_still_used(monkeypatch):
    """`requirements` allows a statsmodels that has never heard of
    `result_object`. A TypeError from the keyword must fall back to that
    version's tuple — not slide past the outer guard into the `d = 1` default,
    which would difference every series and leave the grid comparing AICs across
    the wrong order with nothing said about it."""
    from kairos.models_stat import arima_forecast

    seen: list[str] = []

    def old_adfuller(x, autolag=None):     # the pre-option signature
        seen.append("tuple")
        return (1.0, 0.93, 3, 200, {"5%": -2.9}, {})   # …non-stationary

    def adfuller(x, autolag=None, **kw):
        if "result_object" in kw:
            seen.append("rejected")
            raise TypeError("adfuller() got an unexpected keyword argument "
                            "'result_object'")
        return old_adfuller(x, autolag=autolag)

    monkeypatch.setattr(ST, "adfuller", adfuller)
    f = arima_forecast(_close(), horizon=3)
    assert seen == ["rejected", "tuple"], seen
    assert f is not None and _d(f) == 1


def test_position_one_and_the_named_field_are_the_same_number():
    """The fallback reads `[1]`, so the one thing that has to be true is that
    position 1 *is* the p-value — checked against the real library, because the
    whole point of the fix is that nobody should have to take that on faith."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    y = np.asarray(np.log(_close(300)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = adfuller(y, autolag="AIC")
    named = adfuller(y, autolag="AIC", result_object=True)
    assert float(named.pvalue) == float(legacy[1])
