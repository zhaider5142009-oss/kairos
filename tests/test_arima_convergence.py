"""An ARIMA order is chosen by AIC, so a fit that never converged must not compete.

Two defects, one of them measured:

1. `arima_forecast` fitted nine orders with `maxiter=60` — a deliberately short
   budget — and kept whichever reported the lowest `aic`. statsmodels says so
   when a fit stops early: it raises a `ConvergenceWarning` whose own text is
   "Check mle_retvals", because the flag is in the result. The grid wrapped
   itself in `warnings.simplefilter("ignore")`, so nothing read the notice, and
   the flag was never read either. Across 14 symbols and 126 real fits, 8
   admitted non-convergence and 2 of those held the lowest AIC in their grid
   (BANK -1603.19 against -1599.5 for the best finished fit; BTC -1873.43
   against -1871.5). An unconverged fit is not merely untidy: its `aic`, its
   `predicted_mean` and its `se_mean` are all read at parameters the optimizer
   rejected, and `se_mean` is the denominator of the signal-to-noise ratio that
   becomes this model's confidence — which the ensemble weights it by and the
   size channel reads.
2. `models_stat` and `models_ml` each ran `warnings.filterwarnings("ignore")`
   at import — no category, no message. Both modules are imported by the engine,
   so the daemon, the desk and the API server ran with every warning class in
   the process muted. `RuntimeWarning: invalid value encountered in divide` is
   the notice that explains a NaN before it reaches the journal, and it could
   not be seen. Measured after removing them, the ML battery emits nothing at
   all on real data: the filter was silencing no noise, only the operator.

The fix is to read what the library already reports rather than to print louder:
`_converged()` takes the flag, the grid compares only finished fits, and the
notices are silenced by message inside the block that raises them.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.tsa.arima.model as AM
import statsmodels.tsa.stattools as ST

from kairos import models_stat

_PKG = pathlib.Path(__file__).resolve().parents[1] / "kairos"
_ORDER = re.compile(r"order=\((\d+),\s*(\d+),\s*(\d+)\)")
_FINISHED = re.compile(r"finished=(\d+)/(\d+)")
_MISS = object()


def _close(n: int = 240, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))) * 100.0)


def _detail_order(forecast) -> tuple[int, int, int]:
    m = _ORDER.search(forecast.detail or "")
    assert m, f"the ARIMA row no longer prints the order it fitted: {forecast.detail!r}"
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


class _FakeFit:
    """A result with an `aic`, a `mle_retvals` and a forecast, nothing else."""

    def __init__(self, aic: float, converged: bool, y: np.ndarray):
        self.aic = aic
        self.mle_retvals = {"converged": converged}
        self._y = y

    def get_forecast(self, steps: int = 5):
        last = float(self._y[-1])
        return SimpleNamespace(predicted_mean=np.linspace(last, last + 0.004, steps),
                               se_mean=np.full(steps, 0.02))


def _fake_arima(monkeypatch, table: dict[tuple[int, int], tuple[float, bool]]):
    """Install a stand-in ARIMA whose (p, q) → (aic, converged) is `table`.

    The real optimizer decides which fits finish, so these tests have to say so
    themselves: patching the class the leg imports is the only way to put the
    lowest AIC on an admitted failure on purpose.
    """
    calls: list[tuple[int, int, int]] = []

    class Fake:
        def __init__(self, y, order=(1, 1, 1), **kw):
            self._y = np.asarray(y)
            self._order = tuple(order)
            calls.append(self._order)

        def fit(self, method_kwargs=None, **kw):
            aic, converged = table.get((self._order[0], self._order[2]), _MISS)
            if aic is _MISS:
                raise ValueError("no such order in this experiment")
            return _FakeFit(aic, converged, self._y)

    monkeypatch.setattr(AM, "ARIMA", Fake)
    return calls


@pytest.fixture
def d_one(monkeypatch):
    """Pin the differencing order so the fake's (p, q) keys mean what they say."""
    monkeypatch.setattr(ST, "adfuller",
                        lambda x, autolag=None, **kw: SimpleNamespace(pvalue=0.9))


# ------------------------------------------------------------- the AIC contest
def test_a_fit_that_did_not_converge_does_not_win_the_contest(monkeypatch, d_one):
    """The exact shape measured on BANK: the lowest number in the grid belongs to
    an unfinished fit, and the best finished fit loses by a few AIC points."""
    _fake_arima(monkeypatch, {(0, 0): (-1603.19, False),      # lowest AIC, unfinished
                              (1, 1): (-1599.55, True),       # the best finished fit
                              (2, 2): (-1590.00, True)})
    f = models_stat.arima_forecast(_close(), horizon=3)
    assert f is not None
    assert _detail_order(f) == (1, 1, 1), (
        "an order the optimizer rejected was chosen because its number was lowest")
    assert "-1599." in f.detail and "-1603" not in f.detail


def test_no_forecast_is_offered_when_nothing_finished(monkeypatch, d_one):
    """`best_res is None` already means "no usable fit" in this module, and a grid
    in which the optimizer admits it finished nothing is exactly that."""
    table = {(p, q): (-1000.0 + p + q, False) for p in (0, 1, 2) for q in (0, 1, 2)}
    _fake_arima(monkeypatch, table)
    assert models_stat.arima_forecast(_close(), horizon=3) is None


def test_the_row_says_how_many_fits_actually_finished(monkeypatch, d_one):
    """The operator reading the desk cannot tell an order that beat nine finished
    fits from one that beat a single one, and the count says how much of the grid
    was usable at all."""
    _fake_arima(monkeypatch, {(1, 1): (-1200.0, True), (0, 0): (-1300.0, False)})
    f = models_stat.arima_forecast(_close(), horizon=3)
    assert f is not None
    m = _FINISHED.search(f.detail or "")
    assert m, f"the ARIMA row does not report its grid: {f.detail!r}"
    assert (int(m.group(1)), int(m.group(2))) == (1, 2)


def test_a_result_that_reports_no_flag_is_still_used(monkeypatch, d_one):
    """No flag is not a refusal. A statsmodels that stops filling `mle_retvals`,
    or a fit method that has no such dict, must not quietly delete a model class
    from the battery — that is the phantom seat again, counted from the other end."""

    class Bare:
        def __init__(self, y, order=(1, 1, 1), **kw):
            self._y = np.asarray(y)

        def fit(self, method_kwargs=None, **kw):
            last = float(self._y[-1])
            return SimpleNamespace(
                aic=-500.0,
                get_forecast=lambda steps=5: SimpleNamespace(
                    predicted_mean=np.full(steps, last + 0.004),
                    se_mean=np.full(steps, 0.02)))

    monkeypatch.setattr(AM, "ARIMA", Bare)
    f = models_stat.arima_forecast(_close(), horizon=3)
    assert f is not None
    assert _FINISHED.search(f.detail).groups() == ("9", "9")


@pytest.mark.parametrize(("res", "expected"), [
    (SimpleNamespace(mle_retvals={"converged": False}), False),
    (SimpleNamespace(mle_retvals={"converged": True}), True),
    (SimpleNamespace(mle_retvals={}), True),               # a key nobody filled in
    (SimpleNamespace(), True),                             # no attribute at all
    (SimpleNamespace(mle_retvals=None), True),             # reports nothing
    (SimpleNamespace(mle_retvals={"converged": np.float64(0.0)}), False),
])
def test_the_flag_is_read_as_the_library_writes_it(res, expected):
    assert models_stat._converged(res) is expected


# ------------------------------------------------- against the real library
def _grid(y: np.ndarray, d: int) -> list[tuple[tuple[int, int, int], float, bool]]:
    """Run the same nine fits the leg runs, and keep what each one reported."""
    fits = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*starting (autoregressive|MA) parameters.*")
        warnings.filterwarnings("ignore", message=r".*failed to converge.*")
        for p in (0, 1, 2):
            for q in (0, 1, 2):
                try:
                    res = AM.ARIMA(y, order=(p, d, q)).fit(method_kwargs={"maxiter": 60})
                except Exception:
                    continue
                fits.append(((p, d, q), float(res.aic), models_stat._converged(res)))
    return fits


def test_the_real_grid_picks_the_lowest_aic_among_finished_fits():
    """The rule has to hold against statsmodels itself, not only under a fake —
    so the grid is re-run here, which is the only way to see for a real series
    which fits finished and what the winner among those would be."""
    close = _close(180, seed=7)
    y = np.log(close.values.astype(float))[-400:]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*adfuller.*")
        pval = float(ST.adfuller(y, autolag="AIC", result_object=True).pvalue)
    d = 0 if pval < 0.05 else 1
    fits = _grid(y, d)
    finished = [f for f in fits if f[2]]
    assert finished, "nothing in this grid converged; the experiment needs a usable series"
    f = models_stat.arima_forecast(close, horizon=3)
    assert f is not None
    assert _detail_order(f) == min(finished, key=lambda x: x[1])[0], (
        f"the grid reported {fits} but the leg chose another order")
    got = _FINISHED.search(f.detail).groups()
    assert got == (str(len(finished)), str(len(fits)))


def test_the_battery_still_offers_an_arima_vote():
    """Guard against the fix over-reaching: a rule that rejects bad fits and then
    loses ARIMA from the ensemble leaves the battery quieter, not better."""
    out = models_stat.run_statistical_battery(_close(), horizon=3)
    assert "ARIMA" in [f.model for f in out]


# ------------------------------------------- the process-wide warning policy
def _warn_after_import(module: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c",
         f"import warnings, {module}\nwarnings.warn('marker-visible', RuntimeWarning)\n"],
        capture_output=True, text=True, cwd=str(_PKG.parent), timeout=300)
    assert done.returncode == 0, done.stderr[-400:]
    return done.stderr


@pytest.mark.parametrize("module", ["kairos.models_stat", "kairos.models_ml"])
def test_importing_the_battery_does_not_mute_the_process(module):
    """A bare `filterwarnings("ignore")` at import is a policy for the whole
    interpreter, chosen by whichever module happened to be imported first. A
    warning raised afterwards must still reach the log — in a subprocess, so the
    test runner's own filters cannot hide what the package installed."""
    assert "marker-visible" in _warn_after_import(module), (
        f"{module} installed a blanket warning filter at import")


# ------------------------------------------------------------- the static rule
def _filter_calls(tree: ast.AST) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("filterwarnings", "simplefilter", "resetwarnings")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "warnings"):
            out.append(node)
    return out


def _mutes_everything(call: ast.Call) -> bool:
    if call.func.attr == "resetwarnings":
        return True                       # deletes the policy the caller set, test or not
    action = call.args[0].value if call.args else None
    if action not in ("ignore", "once"):
        return False
    if {k.arg for k in call.keywords} & {"category", "message", "module", "lineno"}:
        return False                      # it says what it suppresses
    return len(call.args) < 2             # …or would, if it said so positionally


def test_nothing_in_the_package_mutes_every_warning():
    """A suppressor that names no message and no category hides whatever nobody
    has read yet — which is how a fit that reported its own failure reached the
    desk as a confidence number. Silence a notice where it is raised, by name."""
    offenders = []
    for path in sorted(_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _filter_calls(tree):
            if _mutes_everything(call):
                offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, f"blanket warning suppressors at {offenders}"


def test_the_guard_itself_can_see_the_defect():
    """`_mutes_everything` is the rule; a rule that matches nothing would let the
    next module keep the line this one removed. Both shapes below are the two
    ways the battery had it."""
    def _offender(src: str) -> bool:
        calls = _filter_calls(ast.parse(src))
        assert len(calls) == 1, "the sample no longer holds one filter call"
        return _mutes_everything(calls[0])

    assert _offender("import warnings; warnings.filterwarnings('ignore')")
    assert _offender("import warnings; warnings.simplefilter('ignore')")
    assert _offender("import warnings; warnings.resetwarnings()")
    assert not _offender("import warnings; "
                         "warnings.filterwarnings('ignore', category=FutureWarning)")
    assert not _offender("import warnings; "
                         "warnings.filterwarnings('ignore', 'a message')")
    assert not _offender("import warnings; "
                         "warnings.simplefilter('ignore', message='noisy')")
