"""The ML battery's up-probability, and which column of the fit it comes from.

Two defects, one line each, in `tree_stack_forecast`:

1. `y_sign = np.sign(y)` gave a bar whose forward return is exactly `0.0` a
   class of its own. `predict_proba` columns follow `model.classes_`, so on such
   a fit column 1 is the probability of *no move* — and the code read it as the
   probability of a rise. Measured on the cached series the engine trades: 33 of
   97. On USDPKR the fit said 90.2% up and the desk was handed 0.67%, which
   `exp_ret = (2*p_up - 1) * magnitude` turns into a confident short against the
   model's own opinion. On the fixture below, every seed flips the sign.

2. `float(model.decision_function(...)[0])` is only a number while the fit is
   binary. With the third class it returns one score per class, `float()` of a
   length-3 array raises, and the battery's own `except Exception: continue`
   dropped ML-Ridge. The two sets are identical: 33 series with a flat bar, 33
   series with no ML-Ridge vote. Nothing said so.

Both are silent because a positional index into a library-shaped array is not an
error, and a swallowed exception is not either.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

import kairos.models_ml as ML

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "kairos"
HORIZON = 5


def _frame(seed: int = 0, n: int = 340) -> pd.DataFrame:
    """A rising, occasionally-falling series with two runs of identical prices.

    The flat runs are the whole point: a forward return of exactly zero is what
    a tick-constrained exchange prints, and `np.sign` turns it into a class.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0035, 0.006, n))
    for start in (120, 250):
        close[start:start + HORIZON + 1] = close[start - 1]
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.004,
        "low": close * 0.996,
        "close": close,
        "volume": np.full(n, 1e6),
    })


def _three_class_fit(frame: pd.DataFrame):
    """A real classifier fitted the way the battery used to fit: on `np.sign`."""
    from sklearn.ensemble import GradientBoostingClassifier

    X, y, _ = ML._prepare_xy(frame, HORIZON)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06,
                                       subsample=0.9, min_samples_leaf=10, random_state=7)
    model.fit(X[:-HORIZON], np.sign(y[:-HORIZON]))
    return model, X[-1]


class _Spy:
    """Records the labels it is trained on; predicts like a sane binary model."""

    seen: list = []

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        _Spy.seen.append(sorted(np.unique(np.asarray(y)).tolist()))
        self.classes_ = np.array([-1.0, 1.0])
        return self

    def predict(self, X):
        return np.full(len(X), 1.0)

    def decision_function(self, X):
        return np.full(len(X), 0.8)


class _NoUp:
    """A fit whose classes carry no rising label: classes_ == [-1, 0]."""

    classes_ = np.array([-1.0, 0.0])

    def predict_proba(self, X):
        return np.array([[0.62, 0.38]])


class _MultiScore:
    """A linear fit that came back with one score per class, as three labels do."""

    def decision_function(self, X):
        return np.array([[1.4, -0.2, 0.7]])


# ---------------------------------------------------------------------------
# 1. The vote must point where the model says.
# ---------------------------------------------------------------------------
def test_the_flat_bar_does_not_take_the_models_apart():
    """All four members answer on a series with six zero-return bars."""
    out = ML.tree_stack_forecast(_frame(), horizon=HORIZON)
    assert {f.model for f in out} == {"ML-GBM", "ML-RandomForest",
                                      "ML-ExtraTrees", "ML-Ridge"}, \
        f"the battery lost a member on a series with a flat bar: {[f.model for f in out]}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_direction_is_the_sign_the_model_actually_holds(seed):
    """The model puts 95%+ on rising; the forecast must not be short.

    The old read took column 1 — the flat class — which sits near zero on these
    fixtures, so `(2*p_up - 1)` came out at about -1 and every ML member voted
    down while holding an opinion of up.
    """
    frame = _frame(seed)
    model, row = _three_class_fit(frame)
    proba = model.predict_proba(row.reshape(1, -1))[0]
    assert list(model.classes_) == [-1.0, 0.0, 1.0], "fixture stopped producing a third class"
    assert proba[2] > 0.9, f"fixture stopped producing a confident long: {proba}"
    # The precondition for the flip, stated in the test rather than assumed.
    assert proba[1] < 0.5

    out = {f.model: f for f in ML.tree_stack_forecast(frame, horizon=HORIZON)}
    assert out["ML-GBM"].direction == 1, (
        f"the fit holds P(up)={proba[2]:.3f} but the vote is "
        f"{out['ML-GBM'].direction}: {out['ML-GBM'].detail}")
    assert out["ML-GBM"].expected_return > 0.0


def test_the_up_probability_is_the_rising_class_not_column_one():
    """`_p_up` agrees with the named column and disagrees with the position."""
    model, row = _three_class_fit(_frame())
    proba = model.predict_proba(row.reshape(1, -1))[0]
    got = ML._p_up(model, row)
    assert got == pytest.approx(float(proba[2]))
    assert abs(got - float(proba[1])) > 0.5, (
        "the two reads agree, so this test can no longer see the defect it exists for")


# ---------------------------------------------------------------------------
# 2. Refusal, when a fit has no up-probability to give.
# ---------------------------------------------------------------------------
def test_a_fit_with_no_rising_class_is_refused_not_invented():
    assert ML._p_up(_NoUp(), np.zeros((1, 4))) is None


def test_a_score_that_is_not_one_number_is_refused():
    assert ML._p_up(_MultiScore(), np.zeros((1, 4))) is None


def test_a_refused_fit_does_not_become_a_vote():
    """The battery must drop the model, not hand the ensemble a position it read
    off a column of a different shape."""
    _Spy.seen = []
    monkey = pytest.MonkeyPatch()
    for path in ("sklearn.ensemble.GradientBoostingClassifier",
                 "sklearn.ensemble.RandomForestClassifier",
                 "sklearn.ensemble.ExtraTreesClassifier",
                 "sklearn.linear_model.RidgeClassifier"):
        monkey.setattr(path, _Spy, raising=True)
    out = ML.tree_stack_forecast(_frame(), horizon=HORIZON)
    monkey.undo()
    assert len(out) == 4
    assert all(f.direction == 1 for f in out)


# ---------------------------------------------------------------------------
# 3. The label itself.
# ---------------------------------------------------------------------------
def test_no_classifier_is_handed_a_third_label():
    """Every fit sees {-1, +1} and nothing else, however the series prints."""
    _Spy.seen = []
    monkey = pytest.MonkeyPatch()
    for path in ("sklearn.ensemble.GradientBoostingClassifier",
                 "sklearn.ensemble.RandomForestClassifier",
                 "sklearn.ensemble.ExtraTreesClassifier",
                 "sklearn.linear_model.RidgeClassifier"):
        monkey.setattr(path, _Spy, raising=True)
    ML.tree_stack_forecast(_frame(), horizon=HORIZON)
    monkey.undo()
    assert _Spy.seen, "the battery never fitted anything"
    for labels in _Spy.seen:
        assert set(labels) <= {-1.0, 1.0}, f"a three-way label reached a fit: {labels}"


# ---------------------------------------------------------------------------
# 4. The rule: a probability is never taken by position.
# ---------------------------------------------------------------------------
def _literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int)


def _predict_proba(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "predict_proba")


def _proba_positions(src: str) -> list[int]:
    """Lines where a `predict_proba` result's *class* axis is a literal index.

    A result is indexed on two axes and only one of them carries meaning: row 0
    is always the one row asked for, but the second index is whichever class the
    fit happened to put there. So the rule flags a hard integer on the class axis
    — `pred[0][1]` (the second subscript) or `pred[0, 1]` (the second element of
    the tuple) — and lets `pred[0]` alone through. `_p_up` passes because its
    class index is `up[0]`, a number it derived from `classes_`.
    """
    bad: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Subscript):
            continue
        depth, cur = 0, node
        while isinstance(cur, ast.Subscript):
            depth += 1
            cur = cur.value
        if not _predict_proba(cur):
            continue
        if depth >= 2 and _literal(node.slice):
            bad.add(node.lineno)
        elif isinstance(node.slice, ast.Tuple) and any(_literal(e) for e in node.slice.elts[1:]):
            bad.add(node.lineno)
    return sorted(bad)


@pytest.mark.parametrize("name", ["models_ml.py", "models_stat.py", "ensemble.py"])
def test_no_probability_is_read_by_position(name):
    src = (MODELS_DIR / name).read_text(encoding="utf-8")
    bad = _proba_positions(src)
    assert not bad, f"{name} reads a probability by class position on line(s) {bad}"


def test_the_positional_rule_can_fire():
    """A guard that cannot fail is not a guard.

    Both spellings the battery used to run have to be caught by the rule that now
    forbids them, and the same rule has to pass the file it stands in.
    """
    src = (MODELS_DIR / "models_ml.py").read_text(encoding="utf-8")
    assert _proba_positions(src) == []
    for spelling in ("model.predict_proba(row.reshape(1, -1))[0][1]",
                     "model.predict_proba(row.reshape(1, -1))[0, 1]"):
        broken = src.replace("model.predict_proba(row.reshape(1, -1))[0][up[0]]",
                             spelling, 1)
        assert broken != src, "the line this rule exists to catch has changed shape"
        assert len(_proba_positions(broken)) == 1, f"the rule missed {spelling!r}"
