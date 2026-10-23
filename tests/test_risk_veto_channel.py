"""The strategy poll has two channels, and one of them governed nothing.

Six members of the library vote `direction == 0` and put the whole of their
message into the rationale: "cut size, tail risk", "stand aside", "illiquid —
avoid full size", "price impact 40x this instrument's own median — reduce size",
"no edge", "wait for resolution". They are not abstentions from the argument;
they are a different argument, about executability rather than direction.

Until now a vote like that entered the poll as a bare seat in the denominator of
a *directional* average, and nothing else. Its stated strength was discarded, the
attribution multiplier computed for it was discarded, and the only effect it had
was to shrink the consensus by one-part-in-N — where N is how many members
happened to speak that bar, an unrelated quantity. A 0.6 "stand aside" and a
0.25 "mixed momentum" dampened identically, and the same warning cost 16.7% of
the poll on a five-vote tape and 2.4% on a forty-vote one. Meanwhile the words
"reduce size" reached no sizing code at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kairos.engine as E
from kairos.ensemble import RISK_VETO_SIZE_CUT, StrategyVote, fuse, veto_size_multiplier


def _vote(direction: int, strength: float, name: str = "Member") -> StrategyVote:
    v = StrategyVote(name, "Volume", direction, strength, f"test {name}")
    return v


# --------------------------------------------------------------------- the poll
def test_a_size_warning_no_longer_moves_the_directional_poll():
    """A member that refuses to name a side must not get a vote on which side."""
    votes = [_vote(1, 0.2, "Bull A"), _vote(1, 0.2, "Bull B")]
    without = fuse("X", [], votes, "Range", 5)
    with_veto = fuse("X", [], votes + [_vote(0, 0.6, "Stand aside")], "Range", 5)
    # The old denominator counted every vote cast, so adding a non-directional
    # warning divided the same 0.4 by 3 x 0.35 instead of 2 x 0.35.
    assert without.strategy_consensus == with_veto.strategy_consensus, (
        f"a warning about size moved the directional poll from "
        f"{without.strategy_consensus} to {with_veto.strategy_consensus}")


def test_the_poll_normalizes_over_the_members_that_took_a_side():
    """Mirrors `model_agreement`, which has always ignored a zero-direction model."""
    votes = [_vote(1, 0.9, "Bull")] + [_vote(0, 0.5, f"Veto {i}") for i in range(3)]
    sig = fuse("X", [], votes, "Range", 5)
    assert sig.strategy_consensus == 0.9
    assert sig.details["strategy_votes"] == 1


def test_a_warning_is_only_as_loud_as_its_stated_strength():
    """Strength was thrown away before: both cases below were one bare seat."""
    loud = fuse("X", [], [_vote(1, 0.5, "Bull"), _vote(0, 0.6, "Stand aside")],
                "Range", 5)
    soft = fuse("X", [], [_vote(1, 0.5, "Bull"), _vote(0, 0.25, "Mixed momentum")],
                "Range", 5)
    assert loud.risk_veto == 0.6 and soft.risk_veto == 0.25
    assert loud.risk_veto > soft.risk_veto


def test_the_loudest_warning_brakes_and_several_do_not_stack():
    """Six members noticing one thin book is one fact, not six."""
    one = fuse("X", [], [_vote(0, 0.6, "Kurtosis")], "Range", 5)
    many = fuse("X", [], [_vote(0, 0.6, f"Member {i}") for i in range(6)], "Range", 5)
    assert one.risk_veto == many.risk_veto == 0.6
    assert many.details["risk_veto_name"]


def test_a_demoted_member_keeps_half_its_brake():
    """The attribution multiplier is the market's opinion of the warning, and it
    has to survive into the channel the warning owns."""
    class _Halved:
        def weight_tag(self, name, horizon, condition):
            return (0.5, "half voice")

        def status(self):
            return {"applied": True, "demoted": 1}

    full = fuse("X", [], [_vote(0, 0.6, "Amihud")], "Range", 5)
    demoted = fuse("X", [], [_vote(0, 0.6, "Amihud")], "Range", 5,
                   attribution=_Halved())
    assert full.risk_veto == 0.6
    assert demoted.risk_veto == 0.3


# ----------------------------------------------------------------- the arithmetic
def test_the_brake_is_bounded_and_never_empties_the_book():
    assert veto_size_multiplier(0.0) == 1.0
    assert veto_size_multiplier(0.6) == round(1.0 - RISK_VETO_SIZE_CUT * 0.6, 12)
    assert veto_size_multiplier(1.0) == 1.0 - RISK_VETO_SIZE_CUT
    # A number cannot exceed its own range, and a brake that reaches zero is a
    # veto the desk cannot see coming.
    assert veto_size_multiplier(5.0) == veto_size_multiplier(1.0)
    assert veto_size_multiplier(-1.0) == 1.0
    assert min(veto_size_multiplier(v) for v in (0.0, 0.5, 1.0, 4.0)) > 0.0


# --------------------------------------------------------------------- the wiring
class _StubMd:
    def __init__(self, frame):
        self.frame = frame

    def history(self, instrument, days=None, use_cache=True):
        return self.frame


def _frame(n=120):
    # Deliberately not a smooth curve, and deliberately a good one. A perfectly
    # compounding tape has zero realized volatility, so the VaR budget sizes it
    # at zero; a noisy tape with no drift fails the Kelly floor (`p*(1+b) > 1`),
    # which also sizes at zero. Either way the assertions below would compare
    # two zeros and pass. This is a tape the risk engine is willing to hold.
    import random

    rng = random.Random(20261024)
    closes, price = [], 100.0
    for _ in range(n):
        price *= 1.0 + rng.gauss(0.0025, 0.012)
        closes.append(round(price, 4))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": closes,
        "high": [c * 1.012 for c in closes],
        "low": [c * 0.988 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n})


def _engine(monkeypatch, votes):
    eng = E.PredictionEngine(md=_StubMd(_frame()))
    monkeypatch.setattr(eng, "_battery", lambda inst, df, horizon: [])
    monkeypatch.setattr(eng, "intermarket", lambda force=False: {})
    monkeypatch.setattr(eng, "factor_row", lambda inst: pd.Series(dtype=float))
    monkeypatch.setattr(eng, "_news_cache", (0.0, []), raising=False)
    monkeypatch.setattr(E, "kse_regime", lambda md: None)
    monkeypatch.setattr(E, "run_strategy_library", lambda df, fe, ctx: list(votes))
    monkeypatch.setattr(E.attribution, "current", lambda: (None, "test"))
    return eng


def _weight_and_notes(monkeypatch, votes):
    res = _engine(monkeypatch, votes)._predict_uncached(
        "GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is True, res.get("error")
    return res["risk"]["suggested_weight"], res["risk"]["notes"], res


def test_reduce_size_actually_reduces_size(monkeypatch):
    """The rationale says "reduce size". Nothing in the sizing chain used to read
    it, so the only way a thin book could be smaller was as a side effect of a
    diluted consensus — which arrives at the book as a lower *win probability*,
    a claim about being right rather than about a tape that cannot absorb the
    order (`compute_risk` spends `signal.confidence` as Kelly's p)."""
    plain, notes_plain, res_plain = _weight_and_notes(
        monkeypatch, [_vote(1, 0.9, "Bull")])
    vetoed, notes_vetoed, res_vetoed = _weight_and_notes(
        monkeypatch, [_vote(1, 0.9, "Bull"), _vote(0, 0.6, "Amihud Illiquidity Cost")])
    assert plain > 0.001, "the book was already empty, so this proved nothing"
    assert vetoed < plain
    assert vetoed == pytest.approx(plain * veto_size_multiplier(0.6), abs=2e-4)
    assert res_plain["signal"]["risk_veto"] == 0.0
    assert res_vetoed["signal"]["risk_veto"] == 0.6
    # And a smaller position has to name the member that asked for it.
    assert "Amihud Illiquidity Cost" in notes_vetoed
    assert "Amihud" not in notes_plain
