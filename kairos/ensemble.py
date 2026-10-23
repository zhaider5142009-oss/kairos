"""Ensemble fusion: regime-adaptive blending of all model forecasts and
strategy votes into a single actionable signal.

Fusion design (first principles):
  1. Each model's vote is weighted by confidence × regime-affinity.
     Trend models earn weight in trending regimes; mean-reversion models
     earn it in range-bound/mean-reverting regimes.
  2. Strategy votes enter as a bounded opinion poll with veto power when
     consensus is overwhelming.
  3. Expected return = confidence-weighted model blend, shrunk toward zero
     by disagreement (a JARVIS-original "disagreement shrinkage" that
     mirrors how real PMs size into contested views).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models_stat import ModelForecast

TREND_MODELS = {"ARIMA", "Kalman", "Fourier-Momentum", "Holt-Winters",
                "ML-GBM", "ML-RandomForest", "ML-ExtraTrees", "ML-Ridge",
                "Neural-LSTM", "Neural-GRU"}
REVERSION_MODELS = {"OU-MeanRev", "GARCH(1,1)"}

REGIME_AFFINITY: dict[str, dict[str, float]] = {
    "Bull-Quiet": {"trend": 1.25, "reversion": 0.75},
    "Bull-Volatile": {"trend": 1.05, "reversion": 0.85},
    "Bear-Quiet": {"trend": 1.15, "reversion": 0.80},
    "Bear-Volatile": {"trend": 0.85, "reversion": 1.20},
    "Range": {"trend": 0.75, "reversion": 1.30},
    "MeanRev-Quiet": {"trend": 0.65, "reversion": 1.45},
    "MeanRev-Volatile": {"trend": 0.60, "reversion": 1.40},
}

# The most a single standing warning may take off a position. It is a brake, not
# a kill switch: a member that says "stand aside" at full strength halves the
# book rather than emptying it, because the warning is an opinion about
# executability and the desk still has a right to act on its thesis. One
# warning is the whole of it — `risk_veto` takes the loudest, not the sum, since
# six members noticing the same thin book is one fact, not six.
RISK_VETO_SIZE_CUT = 0.5


def veto_size_multiplier(risk_veto: float) -> float:
    """How much of a position a standing warning leaves.

    Lives here rather than inline at the one call site because it multiplies
    real order size on three venues: the number a desk sees as a suggested
    weight has to have one definition, and a guard on this path must be able to
    test the arithmetic and not only the wiring.
    """
    return float(1.0 - RISK_VETO_SIZE_CUT * min(1.0, max(0.0, risk_veto)))


@dataclass
class StrategyVote:
    name: str
    category: str
    direction: int
    strength: float                     # 0..1
    rationale: str = ""


@dataclass
class FusedSignal:
    symbol: str
    score: float                        # -1..+1 composite conviction
    direction: str                      # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    expected_return: float              # blended over horizon
    horizon: int
    confidence: float                   # 0..1
    model_agreement: float              # fraction of models agreeing
    strategy_consensus: float           # net strategy vote -1..+1
    regime: str
    # The loudest standing warning from a strategy that refused to name a
    # direction — 0..1, strength scaled by measured skill. It sizes, it does not
    # argue; see `risk_veto` in `fuse`.
    risk_veto: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def _model_weight(fc: ModelForecast, regime: str) -> float:
    aff = REGIME_AFFINITY.get(regime, {"trend": 1.0, "reversion": 1.0})
    if fc.model in TREND_MODELS:
        multiplier = aff["trend"]
    elif fc.model in REVERSION_MODELS:
        multiplier = aff["reversion"]
    else:
        multiplier = 1.0
    return float(np.clip(fc.confidence * multiplier, 0.0, 1.0))


def fuse(symbol: str, forecasts: list[ModelForecast], votes: list[StrategyVote],
         regime: str, horizon: int, attribution=None,
         condition: str | None = None) -> FusedSignal:
    """Blend the battery and the strategy poll into one actionable signal.

    `attribution` is an optional `attribution.Attribution` view: when given,
    each strategy vote enters the poll scaled by the skill its own votes have
    realized out-of-sample in this horizon and regime. Without it the poll is
    exactly what it always was — every vote at face value — which is the state
    the system must fall back to whenever the artifact cannot be read.

    `regime` and `condition` are two different keys. The model affinity table is
    written against the instrument's own micro regime ("Bull-Quiet"), while the
    attribution buckets are addressed by the full condition — micro regime plus
    macro tag. Collapsing them into one string would have REGIME_AFFINITY miss
    on every prediction and silently flatten the trend/reversion tilt that has
    been here since v1.
    """
    if not forecasts and not votes:
        return FusedSignal(symbol, 0.0, "HOLD", 0.0, horizon, 0.0, 0.0, 0.0, regime)

    # -- model blend ---------------------------------------------------------
    weights = [_model_weight(f, regime) for f in forecasts]
    total_w = sum(weights)
    exp_ret = 0.0
    if total_w > 0:
        exp_ret = float(sum(w * f.expected_return
                            for w, f in zip(weights, forecasts, strict=True)) / total_w)

    directions = [f.direction for f in forecasts if f.direction != 0]
    model_agreement = 0.0
    if directions:
        model_agreement = float(abs(sum(directions))) / len(directions)

    # -- strategy poll -------------------------------------------------------
    # A vote's own strength is its conviction; the attribution multiplier is the
    # market's opinion of that conviction, measured out-of-sample on this
    # strategy in this horizon and this regime. Nothing is dropped or added — a
    # demoted strategy still speaks, at half voice — and the poll writes both
    # numbers out, so a changed consensus can be traced to the votes that
    # changed it.
    #
    # The library contains two kinds of vote and they must not share a
    # denominator. Six members speak with `direction == 0` about *size* — "cut
    # size, tail risk", "stand aside", "reduce size" — which is a different
    # argument from which way the price goes. Counting such a warning as a seat
    # in a directional poll made it a hidden dampener on the side it never
    # disputed, and its stated strength was thrown away with the count: a 0.6
    # "stand aside" and a 0.25 "mixed momentum" suppressed consensus identically,
    # by an amount set by how many *unrelated* members voted that bar (one veto
    # costs 16.7% of a five-vote poll and 2.4% of a forty-vote one). So the
    # directional average is normalized over the votes that took a side, exactly
    # as `model_agreement` above has always done, and the warnings leave as
    # `risk_veto` — strength and measured skill included — which `engine` spends
    # on position size, the only thing they ever spoke about.
    cond = condition or regime
    scaled: list[tuple[StrategyVote, float]] = [
        (v, 1.0 if attribution is None else attribution.weight_tag(
            v.name, horizon, cond)[0])
        for v in votes]
    cast = [(v, w) for v, w in scaled if v.direction != 0]
    bull = sum(v.strength * w for v, w in cast if v.direction > 0)
    bear = sum(v.strength * w for v, w in cast if v.direction < 0)
    if cast:
        strategy_consensus = float(np.clip((bull - bear) / max(len(cast) * 0.35, 1.0),
                                           -1.0, 1.0))
    else:
        strategy_consensus = 0.0

    # One loud warning is enough to cut size; five of them do not multiply, so
    # the channel takes the strongest rather than the sum. `risk_veto_name` is
    # journaled because a smaller position has to be traceable to the member
    # that asked for it.
    vetoes = [(v, v.strength * w) for v, w in scaled if v.direction == 0]
    risk_veto = 0.0
    risk_veto_name = ""
    if vetoes:
        loudest = max(vetoes, key=lambda x: x[1])
        risk_veto = float(np.clip(loudest[1], 0.0, 1.0))
        risk_veto_name = loudest[0].name

    # -- disagreement shrinkage ----------------------------------------------
    if forecasts:
        rets = np.array([f.expected_return for f in forecasts])
        if np.std(rets) > 1e-9:
            noise_ratio = float(np.std(rets) / (abs(np.mean(rets)) + np.std(rets)))
        else:
            noise_ratio = 0.0
        shrink = float(np.clip(1.0 - 0.55 * noise_ratio, 0.30, 1.0))
    else:
        shrink = 0.6

    # -- composite score -----------------------------------------------------
    model_score = float(np.clip(exp_ret / 0.045, -1.0, 1.0)) if forecasts else 0.0
    conf_avg = float(np.mean([f.confidence for f in forecasts])) if forecasts else 0.0
    score = 0.62 * model_score + 0.38 * strategy_consensus
    score *= shrink
    # confidence bonus for aligned models
    if forecasts and model_agreement >= 0.75:
        score *= 1.15
    score = float(np.clip(score, -1.0, 1.0))

    confidence = float(np.clip(
        0.45 * conf_avg + 0.30 * model_agreement + 0.25 * min(1.0, abs(strategy_consensus)),
        0.05, 0.95))

    if score > 0.55:
        direction = "STRONG BUY"
    elif score > 0.18:
        direction = "BUY"
    elif score < -0.55:
        direction = "STRONG SELL"
    elif score < -0.18:
        direction = "SELL"
    else:
        direction = "HOLD"

    details: dict[str, Any] = {
        "shrinkage": round(shrink, 3),
        "condition": cond,
        "models": {f.model: {"dir": f.direction, "er": round(f.expected_return, 5),
                             "conf": round(f.confidence, 3),
                             "detail": f.detail} for f in forecasts},
        "strategy_bull": round(bull, 2),
        "strategy_bear": round(bear, 2),
        "strategy_votes": len(cast),
        "risk_veto": round(risk_veto, 3),
        "risk_veto_name": risk_veto_name,
        "strategies": {v.name: {"dir": v.direction, "s": round(v.strength, 2),
                                **({"w": round(w, 3)} if w != 1.0 else {})}
                       for v, w in sorted(scaled, key=lambda x: -x[0].strength)[:15]}}
    if attribution is not None:
        details["attribution"] = attribution.status()
    return FusedSignal(
        symbol=symbol, score=round(score, 4), direction=direction,
        expected_return=round(exp_ret * shrink, 5), horizon=horizon,
        confidence=round(confidence, 4),
        model_agreement=round(model_agreement, 3),
        strategy_consensus=round(strategy_consensus, 3), regime=regime,
        risk_veto=round(risk_veto, 3), details=details)
