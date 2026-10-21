"""Strategy votes that mean what they say.

A strategy cannot be reviewed by reading its output alone: a permanently bearish
vote is indistinguishable from a bearish market unless someone checks the
arithmetic underneath it. `Tenkan/Kijun (Ichimoku core)` shipped for its whole
life on `score = (px > b) + (px > k) + (t > k)`, and because those operands are
numpy scalars, `+` on them is xor, not addition — so `score` was True or False,
`score >= 2` was unreachable, and the library carried a guaranteed short vote of
strength 1.0 on every symbol with 52 bars or more, straight into the consensus
that gates live sizing.

These tests pin the direction semantics and then ban the arithmetic pattern
across the package, because the pattern never raises: it only votes.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from kairos import strategies


def _frame(kind: str, n: int = 300) -> pd.DataFrame:
    """A textbook stack: strictly rising or strictly falling, with fixed-width
    bars so rolling highs and lows move predictably instead of by luck."""
    if kind == "up":
        close = np.linspace(50.0, 150.0, n)
    elif kind == "down":
        close = np.linspace(150.0, 50.0, n)
    else:  # flat: tenkan and kijun cross constantly, so no stack holds
        close = np.full(n, 100.0)
    high, low = close * 1.01, close * 0.99
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="D"),
                         "open": (high + low) / 2, "high": high, "low": low,
                         "close": close, "volume": np.linspace(1e4, 5e4, n)})


def _by_name(name: str):
    return next(s for s in strategies.STRATEGIES if s[0] == name)[2]


def _vote(name: str, df: pd.DataFrame):
    ctx = {"regime": "sideways", "intermarket": {}, "date_today": df["date"].iloc[-1],
           "factor_row": None, "patterns": [], "tags": []}
    return _by_name(name)(df, {}, ctx)


ICHIMOKU = "Tenkan/Kijun (Ichimoku core)"


def test_a_rising_stack_votes_long():
    v = _vote(ICHIMOKU, _frame("up"))
    assert v is not None and v.direction == 1, (
        f"a clean uptrend scored {v.direction} ({v.rationale}) — the three "
        "stack conditions were not counted")
    assert v.strength == pytest.approx(1.0)


def test_a_falling_stack_votes_short():
    v = _vote(ICHIMOKU, _frame("down"))
    assert v is not None and v.direction == -1
    assert v.strength == pytest.approx(1.0)


def test_the_long_side_is_reachable_at_all():
    # before the fix no input to this strategy could ever produce direction=1,
    # which is the whole failure: the branch existed and was dead.
    frames = [_vote(ICHIMOKU, _frame(k)) for k in ("up", "down", "flat")]
    assert any(f is not None and f.direction == 1 for f in frames)


def test_a_short_history_still_declines_to_vote():
    assert _vote(ICHIMOKU, _frame("up", n=40)) is None


# ------------------------------------------------------------------ pattern ban
def _add_chains(tree: ast.AST):
    """Yield (lineno, comparison-count) for `+` chains of two or more comparisons."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
            continue
        leaves, stack, seen = [], [node], set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
                stack.extend([cur.left, cur.right])
            else:
                leaves.append(cur)
        n_cmp = sum(1 for L in leaves if isinstance(L, ast.Compare))
        if n_cmp >= 2:
            yield node.lineno, n_cmp


def test_no_boolean_sum_is_built_from_numpy_scalars():
    root = pathlib.Path(strategies.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, n_cmp in _add_chains(tree):
            offenders.append(f"{path.name}:{lineno} sums {n_cmp} comparisons")
    assert not offenders, (
        "adding comparison results is safe for Python bools and wrong for numpy "
        f"bools, where + is xor: {offenders}")
