"""
StrategyAdapter — a library of classic systematic strategies from the
bank/CTA playbook, exposed as first-class fleet models.

Each strategy is a rule over the *shared* FastFeatures bundle computed once
per symbol per cycle (see strategy_features.py).  No per-model indicator
math → every strategy vote costs microseconds, which is what lets the fleet
weigh ~30 independent inductive biases on every poll without hurting the
millisecond decision path.

Rule contract:  fn(params, f, prices) -> quality ∈ [-1, 1]
    sign(quality) = direction, |quality| ~ signal strength used for
    confidence.  Rules are normalised by ATR / vol so they trade across
    commodities (gold vs oil) without re-tuning.

Strategy catalogue (22 rules across 5 personalities):
    trend:      ema_cross, macd_cross, adx_trend, ema_ribbon, macd_hist
    momentum:   tsmom, dual_momentum, adx_quiet, breakout_ma
    reversion:  rsi_fade_r2, bb_reversion, cci_reversion, zscore_reversion,
                percentile_reversion, stoch_fade
    breakout:   donchian_20, donchian_40, keltner_breakout,
                volatility_expansion, macd_hist_reversion
    flow:       vol_spike_fade, range_expansion_break, momentum_stack
"""

from __future__ import annotations

import numpy as np
import time

from .base import PredictiveModel, ModelForecast
from .strategy_features import StrategyContext

_str = np.tanh


def _q(direction: int, strength: float) -> float:
    """Clamp a raw strength into [-1, 1] scaled by direction."""
    return float(direction) * max(-1.0, min(1.0, strength))


def _confluence(votes: list[float]) -> float:
    """Average a list of quality votes (each ∈ [-1,1]) → [-1,1]."""
    return float(np.mean(votes)) if votes else 0.0


# ---------------------------------------------------------------------- rules
# Each: (params, f, prices) -> quality ∈ [-1, 1]


def r_ema_cross(p, f, pr):
    a = f["ema14"] - f["ema30"]
    b = f["ema30"] - f["ema60"]
    trend = 0.5 * a + 0.5 * b
    scale = np.abs(f["atr14"][-1]) / (np.abs(pr["close"][-1]) + 1e-9) + 1e-9
    return _q(int(np.sign(trend[-1])), abs(trend[-1]) / (pr["close"][-1] * scale) / 40.0)


def r_macd_cross(p, f, pr):
    h = f["macd_hist"]
    strength = np.clip(abs(h[-1]) / (f["atr14"][-1] + 1e-9) * 0.5, 0, 1)
    return _q(int(np.sign(h[-1])), strength)


def r_adx_trend(p, f, pr):
    adx = f["adx14"][-1]
    if adx < float(p.get("adx_min", 20.0)):
        return 0.0
    d = f["di_plus"][-1] - f["di_minus"][-1]
    q = _q(int(np.sign(d)), (adx - float(p.get("adx_min", 20.0))) / 40.0 + 0.15)
    return q


def r_ema_ribbon(p, f, pr):
    s7 = (f["ema7"][-1] > f["ema14"][-1]) + (f["ema14"][-1] > f["ema30"][-1])
    s7 += (f["ema30"][-1] > f["ema60"][-1]) + (f["ema60"][-1] > f["ema100"][-1])
    score = (s7 - 2.0) / 2.0  # -1 all bearish .. +1 all bullish
    slope = (f["ema60"][-1] - f["ema60"][-5]) / (f["ema60"][-5] + 1e-9) * 100.0
    return _q(int(np.sign(score)), abs(score) * (0.6 + 0.4 * min(1.0, abs(slope) / 0.4)))


def r_macd_hist(p, f, pr):
    h = f["macd_hist"]
    w1, w2 = h[-3], h[-2]
    # expanding histogram in one direction = strong flow; fading = tentative
    growing = abs(h[-1]) > abs(h[-2])
    q = _q(int(np.sign(h[-1])), (abs(h[-1]) / (f["atr14"][-1] + 1e-9)) * (0.6 if growing else 0.35))
    return q


def r_tsmom(p, f, pr):
    ret = f[f"mom{int(p.get('period', 20))}"][-1]
    return _q(int(np.sign(ret)), min(1.0, abs(ret) / float(p.get("scale", 2.0))))


def r_dual_momentum(p, f, pr):
    m10 = f["mom10"][-1]
    m20 = f["mom20"][-1]
    if np.sign(m10) != np.sign(m20):
        return 0.0
    return _q(int(np.sign(m20)), min(1.0, (abs(m10) + abs(m20)) / 3.0))


def r_adx_quiet(p, f, pr):
    """Low-ADX chop: trade the short momentum inside the range."""
    if f["adx14"][-1] > 16.0:
        return 0.0
    m = f["mom5"][-1]
    return _q(int(np.sign(m)), min(1.0, abs(m) / 0.6))


def r_breakout_ma(p, f, pr):
    above = pr["close"][-1] > f["ema30"][-1]
    rr = f["range_ratio"][-1]
    if rr < 1.4:
        return 0.0
    return _q(1 if above else -1, 0.4 + 0.25 * min(1.0, rr / 3.0))


def r_rsi_fade(p, f, pr):
    r = f["rsi2"][-1]
    if r > float(p.get("hi", 96.0)):
        return _q(-1, (r - 96.0) / 4.0)
    if r < float(p.get("lo", 4.0)):
        return _q(1, (4.0 - r) / 4.0)
    return 0.0


def r_bb_reversion(p, f, pr):
    z = f["bb_z"][-1]
    thr = float(p.get("width", 2.0))
    if abs(z) < thr:
        return 0.0
    return _q(-int(np.sign(z)), (abs(z) - thr) / 2.0)


def r_cci_reversion(p, f, pr):
    c = f["cci20"][-1]
    if abs(c) < 100.0:
        return 0.0
    return _q(-int(np.sign(c)), (abs(c) - 100.0) / 200.0)


def r_zscore_reversion(p, f, pr):
    z = f["z60"][-1]
    if abs(z) < 1.5:
        return 0.0
    return _q(-int(np.sign(z)), (abs(z) - 1.5) / 2.5)


def r_percentile_reversion(p, f, pr):
    pc = f["pct20"][-1]
    if pc < 10.0:
        return _q(1, (10.0 - pc) / 10.0)
    if pc > 90.0:
        return _q(-1, (pc - 90.0) / 10.0)
    return 0.0


def r_stoch_fade(p, f, pr):
    k = f["stoch_k"][-1]
    if k < 15.0:
        return _q(1, (15.0 - k) / 15.0)
    if k > 85.0:
        return _q(-1, (k - 85.0) / 15.0)
    return 0.0


def r_donchian(p, f, pr):
    span = int(p.get("span", 20))
    hi = f[f"dch{span}_hi"]
    lo = f[f"dch{span}_lo"]
    c = pr["close"]
    above = c[-1] > hi[-2] if len(hi) > 1 else False
    below = c[-1] < lo[-2] if len(lo) > 1 else False
    if above:
        return _q(1, 0.45 + 0.2 * min(1.0, f["range_ratio"][-1] / 2.5))
    if below:
        return _q(-1, 0.45 + 0.2 * min(1.0, f["range_ratio"][-1] / 2.5))
    return 0.0


def r_keltner_breakout(p, f, pr):
    c = pr["close"][-1]
    rr = f["range_ratio"][-1]
    if rr < 1.5:
        return 0.0
    if c > f["kelt_up"][-1]:
        return _q(1, 0.5 + 0.2 * min(1.0, rr / 3.0))
    if c < f["kelt_lo"][-1]:
        return _q(-1, 0.5 + 0.2 * min(1.0, rr / 3.0))
    return 0.0


def r_volatility_expansion(p, f, pr):
    rr = f["range_ratio"][-1]
    if rr < float(p.get("min_rr", 2.0)):
        return 0.0
    up = 0.5 - f["pct20"][-1] / 100.0  # negative => upper range
    if abs(up) < 0.25:
        return 0.0
    return _q(1 if up < 0 else -1, 0.5 + 0.3 * min(1.0, rr / 4.0))


def r_macd_hist_reversion(p, f, pr):
    """Extreme histogram reading = exhaustion → fade back to signal."""
    h = f["macd_hist"]
    z = h[-1] / (f["atr14"][-1] + 1e-9)
    if abs(z) < 0.8:
        return 0.0
    return _q(-int(np.sign(z)), (abs(z) - 0.8) / 2.0)


def r_vol_spike_fade(p, f, pr):
    v20, v60 = f["vol20"][-1], f["vol60"][-1]
    if np.isnan(v60) or v60 <= 0 or v20 < v60 * 1.4:
        return 0.0
    z = f["z60"][-1]
    if abs(z) < 1.0:
        return 0.0
    return _q(-int(np.sign(z)), min(1.0, v20 / v60 * 0.7))


def r_range_expansion_break(p, f, pr):
    """Range bound then expansion: keep the break direction."""
    recent_range = f["pct20"][-1]
    if f["range_ratio"][-1] < 2.0:
        return 0.0
    c = pr["close"][-1]
    mid = (f["dch20_hi"][-1] + f["dch20_lo"][-1]) / 2.0
    if abs(recent_range - 50.0) > 30.0:
        return 0.0
    return _q(1 if c > mid else -1, 0.4 + 0.2 * min(1.0, f["range_ratio"][-1] / 3.0))


def r_momentum_stack(p, f, pr):
    """Stacked short-term momentum + medium momentum + trend filter."""
    votes = []
    for m in ("mom5", "mom10", "mom20"):
        v = f[m][-1]
        votes.append(min(1.0, abs(v) / 1.0) * int(np.sign(v)))
    above = 1.0 if pr["close"][-1] > f["ema30"][-1] else -1.0
    return _q(int(np.sign(_confluence(votes))), min(1.0, abs(_confluence(votes)) + 0.15 * above))


def r_rsi_fade_soft(p, f, pr):
    r = f["rsi7"][-1]
    if r > 70.0:
        return _q(-1, (r - 70.0) / 60.0)
    if r < 30.0:
        return _q(1, (30.0 - r) / 60.0)
    return 0.0


def r_stoch_momentum(p, f, pr):
    k = f["stoch_k"][-1]
    if k > 80.0 and f["mom10"][-1] > 0:
        return _q(1, (k - 80.0) / 40.0)
    if k < 20.0 and f["mom10"][-1] < 0:
        return _q(-1, (20.0 - k) / 40.0)
    return 0.0


# ------------------------------------------------------------------ catalogue

RULES: dict[str, tuple[str, object]] = {
    "ema_cross":            ("trend", r_ema_cross),
    "macd_cross":           ("trend", r_macd_cross),
    "adx_trend":            ("trend", r_adx_trend),
    "ema_ribbon":           ("trend", r_ema_ribbon),
    "macd_hist":            ("trend", r_macd_hist),
    "tsmom20":              ("momentum", r_tsmom),
    "dual_momentum":        ("momentum", r_dual_momentum),
    "adx_quiet":            ("momentum", r_adx_quiet),
    "breakout_ma":          ("momentum", r_breakout_ma),
    "momentum_stack":       ("momentum", r_momentum_stack),
    "rsi_fade_r2":          ("reversion", r_rsi_fade),
    "rsi_fade_soft":        ("reversion", r_rsi_fade_soft),
    "bb_reversion":         ("reversion", r_bb_reversion),
    "cci_reversion":        ("reversion", r_cci_reversion),
    "zscore_reversion":     ("reversion", r_zscore_reversion),
    "percentile_reversion": ("reversion", r_percentile_reversion),
    "stoch_fade":           ("reversion", r_stoch_fade),
    "donchian_20":          ("breakout", r_donchian),
    "donchian_40":          ("breakout", r_donchian),
    "keltner_breakout":     ("breakout", r_keltner_breakout),
    "volatility_expansion": ("breakout", r_volatility_expansion),
    "macd_hist_reversion":  ("reversion", r_macd_hist_reversion),
    "vol_spike_fade":       ("flow", r_vol_spike_fade),
    "range_expansion_break":("breakout", r_range_expansion_break),
    "stoch_momentum":       ("momentum", r_stoch_momentum),
}

PERSONALITIES = sorted({rule[0] for rule in RULES.values()})


class StrategyAdapter(PredictiveModel):
    """One config-line fleet model wrapping a single systematic rule."""

    family = "strategy"
    description = "classic systematic rule (bank/CTA playbook)"

    def __init__(self, strategy: str, **params) -> None:
        super().__init__()
        if strategy not in RULES:
            raise ValueError(f"unknown strategy rule: {strategy}")
        self.personality, self._rule = RULES[strategy]
        self.strategy = strategy
        self.name = params.pop("name", strategy)
        # params: rule-specific (period, adx_min, span, width, ...) +
        # personality card: steps = holder horizon, strength_floor.
        self._params = dict(params)
        self._steps = int(self._params.pop("steps", 4))
        self._floor = float(self._params.pop("strength_floor", 0.18))
        self._base_conf = float(self._params.pop("base_conf", 0.42))
        self._status = "ready"

    def _load_impl(self) -> None:
        self._status = "ready"

    @property
    def is_rule(self) -> bool:
        return True

    def _forecast_impl(self, values: np.ndarray, horizon: int,
                       symbol: str | None = None,
                       context: StrategyContext | None = None) -> ModelForecast:
        t0 = time.perf_counter()
        close = np.asarray(values, dtype=np.float64)
        if context is not None:
            f, prices = context.feat, context.prices
        else:
            # Degraded no-context path: synthesize OHLC from closes, compute
            # the shared bundle on the fly (slow path, correctness first).
            pad = close.copy()
            ctx = StrategyContext.compute(pad, pad, pad, close, None)
            f, prices = ctx.feat, ctx.prices
        if len(close) < 40:
            return ModelForecast(
                model_name=self.name, family=self.family,
                point=np.zeros(horizon), quantiles=None, horizon=horizon,
                direction=0, expected_return_pct=0.0, confidence=0.0,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                status="skipped: <40 bars",
            )
        try:
            quality = float(self._rule(self._params, f, prices))
            if not np.isfinite(quality):
                quality = 0.0
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            return ModelForecast(
                model_name=self.name, family=self.family,
                point=np.zeros(horizon), quantiles=None, horizon=horizon,
                direction=0, expected_return_pct=0.0, confidence=0.0,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                status=f"error: {self._last_error[:60]}",
            )
        q = max(-1.0, min(1.0, quality))
        direction = 0 if abs(q) < 1e-9 else int(np.sign(q))
        if abs(q) < self._floor:
            direction = 0
        conf = min(0.98, self._base_conf + 0.5 * abs(q))
        if direction == 0:
            conf = min(conf, 0.3)
        last = float(close[-1])
        # Expected move over the horizon, scaled by ATR-normalised torque.
        atr = float(np.nan_to_num(f["atr_norm"][-1]))
        expected_pct = abs(q) * atr * 100.0 * min(horizon / max(self._steps, 1), 3.0)
        point = np.full(horizon, last, dtype=np.float64) + direction * last * expected_pct / 100.0
        return ModelForecast(
            model_name=self.name, family=self.family,
            point=point, quantiles=None, horizon=horizon,
            direction=direction,
            expected_return_pct=direction * expected_pct,
            confidence=float(conf),
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            status="ok",
            extras={
                "personality": self.personality,
                "strategy": self.strategy,
                "quality": round(q, 4),
                "mode": "rule",
            },
        )


def make_strategy_models(names: list[str] | None = None) -> list[StrategyAdapter]:
    """Build one adapter per configured catalogue rule (test/backtest use)."""
    names = names or sorted(RULES.keys())
    return [StrategyAdapter(strategy=n) for n in names if n in RULES]


__all__ = ["StrategyAdapter", "make_strategy_models", "RULES", "PERSONALITIES"]