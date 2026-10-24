"""A neural member's confidence was the absolute value of its validation IC.

`_neural_forecast` scored itself with `clip(0.28 + abs(val_ic) * 1.5, 0.10,
0.90)` and the blend weights every member by exactly that confidence. IC is
signed, so the absolute value says a model whose predictions ran *opposite* to
the outcomes it was validated on is as trustworthy as one that tracked them.

Measured on the cached universe — 13 instruments that have neural history, two
kinds each, KSE100 absent — **13 of the 26 members had negative validation
IC**, and the strongest of them was handed the most confidence: OGDC's GRU at
-0.215 earned 0.603 and LUCK's GRU at -0.205 earned 0.587, against 0.634 for
the best positive member (LUCK's LSTM at +0.236). Anti-skill was outbidding
most of the battery with an opinion its own validation says is backwards.

The helper also answered 0.0 for a window too short to score, a target that
never moved and a prediction that never moved, so an unmeasurable member
printed `val_IC=+0.000` as though measured and carried the 0.28 a
measured-at-zero member is worth.
"""

from __future__ import annotations

import numpy as np
import pytest

from kairos import models_ml as ML


def test_a_window_too_short_to_score_is_unmeasured():
    for n in (0, 1, 7):
        assert ML._ic(np.arange(n, dtype=float), np.arange(n, dtype=float)) is None


def test_eight_bars_is_the_first_window_that_can_be_scored():
    y = np.arange(8, dtype=float)
    assert ML._ic(y, y) == pytest.approx(1.0)


def test_a_target_that_never_moved_is_unmeasured():
    assert ML._ic(np.zeros(20), np.arange(20, dtype=float)) is None


def test_a_prediction_that_never_moved_is_unmeasured():
    assert ML._ic(np.arange(20, dtype=float), np.zeros(20)) is None


def test_a_negative_correlation_keeps_its_sign():
    y = np.arange(30, dtype=float)
    assert ML._ic(y, -y) == pytest.approx(-1.0)
    assert ML._ic(y, 2 * y + 3) == pytest.approx(1.0)


@pytest.mark.parametrize(("ic", "want"),
                         [(-0.215, 0.28), (-0.05, 0.28), (0.0, 0.28), (0.236, 0.634)])
def test_confidence_reads_the_sign(ic, want):
    assert ML._neural_confidence(ic) == pytest.approx(want, abs=5e-4)


def test_anti_skill_cannot_outbid_skill():
    assert ML._neural_confidence(-0.215) < ML._neural_confidence(0.215)
    assert ML._neural_confidence(-10.0) == pytest.approx(ML._neural_confidence(0.0))


def test_the_ceiling_the_floor_and_the_unmeasured_case():
    assert ML._neural_confidence(10.0) == pytest.approx(0.90)
    assert ML._neural_confidence(-10.0) == pytest.approx(0.28)
    assert ML._neural_confidence(None) == pytest.approx(0.10)


def test_an_unmeasured_member_says_so():
    assert ML._ic_text(None) == "unmeasured"
    assert ML._ic_text(0.041) == "+0.041"
    assert ML._ic_text(-0.215) == "-0.215 inverse"


def test_the_rule_this_replaced_paid_the_strongest_inverse_member_best():
    def old(ic):
        return float(np.clip(0.28 + abs(ic) * 1.5, 0.10, 0.90))

    assert old(-0.215) == pytest.approx(0.6025)
    assert ML._neural_confidence(-0.215) == pytest.approx(0.28)


@pytest.mark.parametrize(("ic", "want_conf", "want_text"),
                         [(-0.5, 0.28, "val_IC=-0.500 inverse"),
                          (None, 0.10, "val_IC=unmeasured")])
def test_the_forecast_uses_the_rule_it_is_given(monkeypatch, ic, want_conf, want_text):
    """The wiring, not just the helper: a monkeypatched IC must reach the row."""
    pytest.importorskip("torch")
    import pandas as pd

    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({"close": close})
    monkeypatch.setattr(ML, "_ic", lambda y_true, y_pred: ic)
    f = ML._neural_forecast(df, 5, "gru")
    assert f is not None
    assert f.confidence == pytest.approx(want_conf)
    assert want_text in f.detail
