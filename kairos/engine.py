"""PredictionEngine — the orchestration heart.

Pipeline per instrument:
  data -> features -> regime -> [stat | ML | neural | WSL foundation models]
       -> 100-strategy votes (with intermarket, factor, news context)
       -> regime-adaptive fusion -> risk overlay -> optional LLM council
       -> personalized dossier with targets, probabilities, and sizing.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .data import MarketData
from .calendar import risk_calendar
from .ensemble import FusedSignal, fuse
from .feedback import confidence_adjustment, record
from .features import (compute_features, cross_sectional_factors, detect_patterns,
                       detect_regime)
from .kse_regime import kse_regime
from .llm import ROUTER
from .models_ml import neural_battery, tree_stack_forecast
from .models_stat import ModelForecast, run_statistical_battery
from .risk import compute_risk, portfolio_metrics, position_size
from .strategies import run_strategy_library
from .universe import Instrument, resolve
from .wsl_bridge import WSL

INTERMARKET_KEYS = ("DXY", "US10Y", "VIX", "SPX", "WTI", "GOLD", "BTC", "USDPKR", "WHEAT")


class PredictionEngine:
    def __init__(self, md: MarketData | None = None) -> None:
        self.md = md or MarketData()
        self._intermarket: dict[str, pd.DataFrame] | None = None
        self._factors: pd.DataFrame | None = None
        self._news_cache: tuple[float, list] = (0.0, [])
        self._pred_cache: dict[tuple, tuple[float, dict]] = {}

    # ------------------------------------------------------------- context
    def intermarket(self, force: bool = False) -> dict[str, pd.DataFrame]:
        if self._intermarket is None or force:
            ctx: dict[str, pd.DataFrame] = {}
            for key in INTERMARKET_KEYS:
                inst = resolve(key)
                if inst is None:
                    continue
                df = self.md.history(inst)
                if df is not None and len(df) >= 40:
                    ctx[key] = df
            self._intermarket = ctx
        return self._intermarket

    def factor_row(self, instrument: Instrument) -> pd.Series | None:
        if instrument.market != "PSX":
            return None
        if self._factors is None:
            snap = self.md.psx_snapshot()
            self._factors = cross_sectional_factors(snap)
        if self._factors is None or self._factors.empty:
            return None
        rows = self._factors[self._factors["name"] == instrument.key]
        return rows.iloc[0] if len(rows) else None

    def news_items(self, max_age_s: float = 900.0) -> list:
        ts, items = self._news_cache
        if time.time() - ts > max_age_s:
            try:
                items = self.md.news.fetch(limit_per_feed=10)
            except Exception:
                items = []
            self._news_cache = (time.time(), items)
        return items

    def news_sentiment(self, instrument: Instrument) -> float | None:
        """LLM-scored news tone for the instrument; None when no LLM/news."""
        items = self.news_items()
        rel = self.md.news.relevant(instrument, items, limit=6)
        if not rel:
            return None
        cfg_on = CFG.apis.get("openrouter", {}).get("enabled", False)
        if not cfg_on:
            return None
        headlines = "\n".join(f"- {i.title} ({i.source})" for i in rel)
        prompt = (f"Score the aggregate news tone for {instrument.name} "
                  f"({instrument.key}, {instrument.market}) on a scale from -1 "
                  f"(very bearish) to +1 (very bullish). Reply with ONLY a JSON "
                  f'object: {{"sentiment": <float>, "confidence": <float 0-1>, '
                  f'"drivers": "<max 20 words>"}}\n\nHeadlines:\n{headlines}')
        resp = ROUTER.chat(prompt, max_tokens=200, temperature=0.2)
        if not resp.ok:
            return None
        from .llm import _extract_json
        data = _extract_json(resp.text)
        if isinstance(data, dict) and isinstance(data.get("sentiment"), (int, float)):
            return float(np.clip(data["sentiment"], -1, 1))
        return None

    # ------------------------------------------------------------- predict
    def predict(self, symbol: str, horizon: int = 5, use_llm: bool = False,
                use_wsl: bool = True, include_news: bool = True) -> dict[str, Any]:
        # research memoization: daily-bar process + 10-min TTL (news cache is
        # 15-min) keeps the watchlist/web instant without stale views
        key = (symbol.upper(), int(horizon), bool(use_llm), bool(use_wsl),
               bool(include_news))
        ts, cached = self._pred_cache.get(key, (0.0, {}))
        if cached and time.time() - ts < 600.0:
            out = dict(cached)
            out["cached"] = True
            return out
        t0 = time.perf_counter()
        res = self._predict_uncached(symbol, horizon, use_llm, use_wsl, include_news)
        if res.get("ok"):
            self._pred_cache[key] = (time.time(), res)
            record(res)
        return res

    def _predict_uncached(self, symbol: str, horizon: int = 5, use_llm: bool = False,
                          use_wsl: bool = True, include_news: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        instrument = resolve(symbol)
        if instrument is None:
            return {"ok": False, "symbol": symbol,
                    "error": f"unknown symbol '{symbol}' — see `kairos symbols`"}
        df = self.md.history(instrument)
        min_hist = int(CFG.engine.get("min_history", 40))
        if df is None or len(df) < min_hist:
            return {"ok": False, "symbol": instrument.key,
                    "error": (f"insufficient price history ({0 if df is None else len(df)} "
                              f"bars, need {min_hist}). Start your local PSX API or "
                              "let the store accumulate daily bars.")}

        close = df["close"]
        price = float(close.iloc[-1])
        features = compute_features(df)
        regime_info = detect_regime(df)
        patterns = detect_patterns(df)

        # --- model battery ---------------------------------------------------
        forecasts: list[ModelForecast] = []
        if CFG.engine.get("use_stat", True):
            forecasts += run_statistical_battery(close, horizon)
        if CFG.engine.get("use_ml", True):
            forecasts += tree_stack_forecast(df, horizon)
        if CFG.engine.get("use_neural", True):
            forecasts += neural_battery(df, horizon)
        if use_wsl:
            series = [float(x) for x in close.tail(400).values]
            try:
                fc = WSL.chronos_forecast(series, horizon)
                if fc:
                    forecasts.append(fc)
                fc = WSL.timesfm_forecast(series, horizon)
                if fc:
                    forecasts.append(fc)
            except Exception:
                pass

        # --- strategy battery --------------------------------------------------
        ctx: dict[str, Any] = {
            "regime": regime_info["regime"],
            "intermarket": self.intermarket(),
            "date_today": df["date"].iloc[-1],
            "factor_row": self.factor_row(instrument),
            "patterns": patterns,
            "tags": instrument.tags,
        }
        if include_news and CFG.apis.get("openrouter", {}).get("enabled", False):
            sent = self.news_sentiment(instrument)
            if sent is not None:
                ctx["news_sentiment"] = sent
        votes = run_strategy_library(df, features, ctx) if CFG.engine.get(
            "use_strategies", True) else []

        # --- fusion -------------------------------------------------------------
        signal = fuse(instrument.key, forecasts, votes, regime_info["regime"], horizon)

        # --- realized track-record calibration ---------------------------------
        adj = confidence_adjustment()
        if adj != 1.0:
            signal.confidence = round(min(0.95, signal.confidence * adj), 3)
            signal.details["track_record_adjustment"] = adj

        # --- risk overlay ---------------------------------------------------------
        risk = compute_risk(instrument.key, close, signal.confidence, horizon)
        # signal-strength scale: neutral signals allocate near-zero even when
        # the vol budget alone would allow the position cap
        risk.suggested_weight *= min(1.0, abs(signal.score) / 0.35)
        cal = risk_calendar()
        if cal["position_multiplier"] < 1.0:
            risk.suggested_weight *= cal["position_multiplier"]
        kse_mult = 1.0
        if instrument.market == "PSX":
            kse = kse_regime(self.md)
            kse_mult = kse.position_multiplier
            risk.suggested_weight *= kse_mult
        vol = risk.annual_vol if risk.annual_vol > 0 else 0.2
        sigma_h = vol / math.sqrt(252) * math.sqrt(horizon)
        targets = {
            "1d": price * (1 + signal.expected_return * (1 / max(horizon, 1))),
            f"{horizon}d": price * (1 + signal.expected_return),
            "21d": price * (1 + signal.expected_return * 21 / max(horizon, 1)),
        }
        bands = {
            "p10": price * (1 + signal.expected_return - 1.282 * sigma_h),
            "p90": price * (1 + signal.expected_return + 1.282 * sigma_h),
            "sigma_h": sigma_h,
        }
        p_up = float(0.5 + 0.5 * math.erf(signal.expected_return /
                                          (sigma_h * math.sqrt(2)))) if sigma_h > 0 else 0.5
        # empirical calibration: blend the theoretical P(up) with the realized
        # hit-rate ladder of historically similar composite-signal scores
        calib = None
        try:
            from .backtest import empirical_calibration
            calib = empirical_calibration(df, horizon)
        except Exception:
            calib = None
        if calib and calib.get("p_up_ladder"):
            ladder = calib["p_up_ladder"]
            bucket = str(int(np.clip(round(signal.score * 2), -2, 2)))
            emp = ladder.get(bucket) or ladder.get("1") or ladder.get("-1")
            if emp is not None:
                p_up = 0.5 * p_up + 0.5 * float(emp)
                p_up = float(np.clip(p_up, 0.05, 0.95))

        result: dict[str, Any] = {
            "ok": True,
            "symbol": instrument.key,
            "name": instrument.name,
            "asset_class": instrument.asset_class,
            "market": instrument.market,
            "sector": instrument.sector,
            "currency": instrument.currency,
            "as_of": str(df["date"].iloc[-1].date()),
            "price": round(price, 4),
            "history_bars": int(len(df)),
            "regime": regime_info,
            "signal": {
                "direction": signal.direction, "score": signal.score,
                "confidence": signal.confidence,
                "expected_return": signal.expected_return,
                "horizon_days": horizon,
                "model_agreement": signal.model_agreement,
                "strategy_consensus": signal.strategy_consensus,
                "shrinkage": signal.details.get("shrinkage"),
            },
            "targets": {k: round(v, 4) for k, v in targets.items()},
            "probability_up": round(min(0.95, max(0.05, p_up)), 4),
            "probability_calibration": calib,
            "confidence_band": {k: round(v, 4) for k, v in bands.items()},
            "models": [{"model": f.model, "direction": f.direction,
                        "expected_return": round(f.expected_return, 5),
                        "confidence": round(f.confidence, 3), "detail": f.detail}
                       for f in forecasts],
            "strategies_voted": len(votes),
            "top_strategies": [{"name": v.name, "category": v.category,
                                 "direction": v.direction, "strength": round(v.strength, 2),
                                 "rationale": v.rationale}
                                for v in sorted(votes, key=lambda x: -x.strength)[:12]],
            "patterns": patterns,
            "risk": {
                "var_95_daily": round(risk.var_95_daily, 5),
                "cvar_95_daily": round(risk.cvar_95_daily, 5),
                "annual_vol": round(risk.annual_vol, 4),
                "sharpe": round(risk.sharpe, 3),
                "sortino": round(risk.sortino, 3),
                "max_drawdown": round(risk.max_drawdown, 4),
                "kelly_fraction": round(risk.kelly_fraction, 4),
                "suggested_weight": round(risk.suggested_weight, 4),
                "notes": risk.position_notes,
            },
            "event_risk": cal,
            "kse_regime_multiplier": round(kse_mult, 2),
            "features_snapshot": {k: (round(v, 5) if isinstance(v, (int, float)) and
                                      np.isfinite(v) else v)
                                  for k, v in list(features.items())[:0]},  # omitted for size
            "computation_ms": int((time.perf_counter() - t0) * 1000),
        }

        # --- PSX factor context ---------------------------------------------------
        frow = ctx.get("factor_row")
        if frow is not None:
            result["factor_context"] = {
                "momentum_1m_z": _r(frow.get("momentum_1m")),
                "momentum_3m_z": _r(frow.get("momentum_3m")),
                "value_z": _r(frow.get("value")),
                "quality_z": _r(frow.get("quality")),
                "composite_rank": int(frow.get("rank")) if frow.get("rank") is not None else None,
            }

        # --- LLM council ----------------------------------------------------------
        if use_llm:
            result["council"] = self.llm_dossier(instrument, result)
        return result

    # ------------------------------------------------------------- council
    def llm_dossier(self, instrument: Instrument, pred: dict[str, Any]) -> dict[str, Any]:
        sig = pred["signal"]
        models_txt = "\n".join(
            f"- {m['model']}: dir={m['direction']:+d}, "
            f"expected_return={m['expected_return']:+.2%}, conf={m['confidence']:.2f}"
            for m in pred["models"][:12]) or "- (no models)"
        strat_txt = "\n".join(
            f"- {s['name']} [{s['category']}]: dir={s['direction']:+d}, "
            f"strength={s['strength']:.2f} — {s['rationale']}"
            for s in pred.get("top_strategies", [])[:10]) or "- (no votes)"
        prompt = f"""You are the KAIROS Quant Council reviewing {instrument.name} ({instrument.key}), listed on {instrument.market} ({instrument.sector}).

CURRENT STATE
- Price: {pred['price']} {instrument.currency} (as of {pred['as_of']})
- Regime: {pred['regime']['regime']} (Hurst={pred['regime']['hurst']}, ADX={pred['regime']['adx']}, vol={pred['regime']['vol_annual']:.1%})
- System signal: {sig['direction']} score={sig['score']} confidence={sig['confidence']} expected {sig['expected_return']:+.2%} over {sig['horizon_days']}d
- P(up)={pred['probability_up']}, VaR95(1d)={pred['risk']['var_95_daily']:.2%}, annual vol={pred['risk']['annual_vol']:.1%}

MODEL BATTERY
{models_txt}

STRATEGY VOTES (top of 100-strategy library)
{strat_txt}

CANDLESTICK PATTERNS: {', '.join(pred.get('patterns', [])) or 'none'}

Respond ONLY with a JSON object:
{{"view": "bullish|bearish|neutral", "probability_up": <0-1>, "key_drivers": ["<3-5 drivers>"], "risks": ["<2-4 risks>"], "entry": <price>, "stop": <price>, "target": <price>, "thesis": "<= 60 words"}}"""

        return ROUTER.council(
            prompt,
            system="You are a BlackRock-grade asset-allocation committee. Be rigorous, "
                   "probability-calibrated, and skeptical of overconfidence. JSON only.")

    # ------------------------------------------------------------- scan
    def scan(self, market: str = "PSX", top: int = 20) -> dict[str, Any]:
        if market.upper() != "PSX":
            return {"ok": False, "error": "scan currently supports the PSX cross-section"}
        snap = self.md.psx_snapshot()
        if snap.empty:
            return {"ok": False, "error": "PSX snapshot unavailable (TradingView down?)"}
        factors = self._factors = cross_sectional_factors(snap)
        cols = [c for c in ("name", "close", "volume", "sector", "momentum_1m",
                            "momentum_3m",
                            "value", "quality", "low_vol", "composite_score", "rank")
                if c in factors.columns]
        top_df = factors[cols].head(top)
        return {"ok": True, "universe": int(len(factors)),
                "as_of": datetime.now().isoformat(timespec="seconds"),
                "top": top_df.to_dict(orient="records")}

    # ------------------------------------------------------------- portfolio
    def portfolio(self, symbols: list[str], capital: float = 1_000_000.0,
                  k: int | None = None) -> dict[str, Any]:
        from .quantum import optimize_portfolio, signal_coherence_check
        series: dict[str, pd.Series] = {}
        for s in symbols:
            inst = resolve(s)
            if inst is None:
                continue
            df = self.md.history(inst)
            if df is not None and len(df) >= 60:
                series[inst.key] = df.set_index("date")["close"]
        if len(series) < 2:
            return {"ok": False, "error": "need >= 2 instruments with history"}
        prices = pd.DataFrame(series).dropna()
        rets = prices.pct_change().dropna()
        if len(rets) < 40:
            return {"ok": False, "error": "insufficient overlapping history"}
        sol = optimize_portfolio(rets, k=k or max(4, len(rets.columns) // 2))
        if sol is None:
            return {"ok": False, "error": "optimizer failed"}
        allocations = {sym: round(w * capital, 0) for sym, w in sol.weights.items()}
        coherence = signal_coherence_check(rets, sol.weights)
        return {"ok": True, "capital": capital, "weights": sol.weights,
                "allocations": allocations,
                "expected_return": round(sol.expected_return, 4),
                "expected_vol": round(sol.expected_vol, 4),
                "sharpe": round(sol.sharpe, 3), "cardinality": sol.cardinality,
                "solver": sol.solver, "quantum_check": coherence,
                "per_instrument_metrics": {
                    sym: {k2: round(v, 4) for k2, v in portfolio_metrics(
                        prices[sym]).items()}
                    for sym in prices.columns}}


def _r(v) -> float | None:
    try:
        f = float(v)
        return round(f, 3) if np.isfinite(f) else None
    except Exception:
        return None


ENGINE = PredictionEngine()
