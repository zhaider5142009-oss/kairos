"""
Historical backtest harness for Jarvis Reborn.

Walks forward through real MT5 M5 bars, replaying the exact ensemble +
strategy-engine decision path to generate simulated trades (paper money).
All indicator/pattern/regime math is computed from past-only data — no
lookahead, no live-market leakage.

Run:  python -m jarvis_predict.backtest
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JarvisConfig
from .mt5_bridge import MT5Bridge, JarvisMT5Error
from .indicators import TechnicalIndicators
from .quantum import QuantumDecisionEngine
from .ensemble import EnsembleFusion, MarketContext
from .signals import SignalGenerator
from .patterns import detect_patterns
from .strategy.engine import create_engine
from .strategy.multitf import Timeframe
from .models.strategy_features import StrategyContext
from .models.strategy_adapters import make_strategy_models, RULES as STRATEGY_RULES
from .models.incremental_features import IncrementalFeatureEngine

LOGGER = logging.getLogger("jarvis.backtest")

# ---------------------------------------------------------------------------
# Self-contained indicator math (vectorised, past-only) — mirrors the live
# strategy engine's indicator surface without touching a live terminal.
# ---------------------------------------------------------------------------

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period:
        return arr
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(close), 50.0)
    if len(close) < period + 1:
        return out
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    for i in range(period, len(close)):
        ag = np.mean(gains[i - period : i])
        al = np.mean(losses[i - period : i])
        out[i] = 100.0 if al == 0 else 100 - (100 / (1 + ag / al))
    return out


def _macd(close: np.ndarray, fast=12, slow=26, sig=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    signal = _ema(line, sig)
    return line, signal, line - signal


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate(([high[0] - low[0]], tr))
    return _ema(tr, period)


def _adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    n = len(close)
    out = np.full(n, 20.0)
    if n < period + 1:
        return out
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = _ema(np.concatenate(([tr[0]], tr)), period)
    plus_di = 100 * _ema(np.concatenate(([plus_dm[0]], plus_dm)), period) / (atr + 1e-10)
    minus_di = 100 * _ema(np.concatenate(([minus_dm[0]], minus_dm)), period) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, period)
    out[period:] = adx[period:]
    return out


def _tf_indicator_dict(open_, high, low, close, volume) -> dict:
    """Strategy-engine-format indicator dict computed from past-only arrays."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64) if volume is not None else np.zeros(len(c))
    rsi_series = _rsi(c, 14)
    macd_l, macd_s, macd_h = _macd(c, 12, 26, 9)
    sma20 = np.mean(c[-20:])
    std20 = np.std(c[-20:])
    atr_series = _atr(h, l, c, 14)
    ema_f = _ema(c, 12)
    ema_s = _ema(c, 26)

    # Simple support/resistance clusters (mirrors bridge._find_sr_levels).
    support, resistance = [], []
    window = 10
    if len(h) >= window * 2:
        hh = [i for i in range(window, len(h) - window)
              if all(h[i] >= h[j] for j in range(i - window, i + window + 1) if j != i)]
        ll = [i for i in range(window, len(l) - window)
              if all(l[i] <= l[j] for j in range(i - window, i + window + 1) if j != i)]
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(set(round(float(x), 5) for x in levels))
            clusters = [[levels[0]]]
            for x in levels[1:]:
                if abs(x - clusters[-1][-1]) / clusters[-1][-1] < 0.002:
                    clusters[-1].append(x)
                else:
                    clusters.append([x])
            return [float(np.mean(cs)) for cs in clusters]
        support = cluster([l[i] for i in ll])[-3:]
        resistance = cluster([h[i] for i in hh])[-3:]

    return {
        "rsi": float(rsi_series[-1]) if len(rsi_series) else 50.0,
        "macd_hist": float(macd_h[-1]) if len(macd_h) else 0.0,
        "macd_signal": float(macd_s[-1]) if len(macd_s) else 0.0,
        "macd_fast": float(ema_f[-1]) if len(ema_f) else float(c[-1]),
        "macd_slow": float(ema_s[-1]) if len(ema_s) else float(c[-1]),
        "bb_upper": float(sma20 + 2 * std20),
        "bb_lower": float(sma20 - 2 * std20),
        "bb_middle": float(sma20),
        "atr": float(atr_series[-1]) if len(atr_series) else 0.0,
        "ema_fast": float(ema_f[-1]) if len(ema_f) else float(c[-1]),
        "ema_slow": float(ema_s[-1]) if len(ema_s) else float(c[-1]),
        "support": support,
        "resistance": resistance,
        "close": float(c[-1]),
        "volume": float(v[-1]) if len(v) else 0.0,
        "avg_volume": float(np.mean(v[-20:])) if len(v) >= 20 else (float(v[-1]) if len(v) else 0.0),
    }


@dataclass
class Trade:
    symbol: str
    direction: int
    entry_price: float
    entry_bar: int
    exit_price: float = 0.0
    exit_bar: int = 0
    sl: float = 0.0
    tp: float = 0.0
    volume: float = 0.01
    pnl_usd: float = 0.0
    commission: float = 0.0
    reason: str = ""
    regime: str = ""

    @property
    def net_pnl(self) -> float:
        return self.pnl_usd - self.commission


@dataclass
class BacktestConfig:
    symbols: list[str] = field(default_factory=lambda: ["XAUUSD", "XAGUSD", "USOIL", "XPTUSD"])
    bars: int = 4000
    starting_balance: float = 10000.0
    lot: float = 0.01
    max_positions: int = 3
    commission_per_lot: float = 0.25
    min_confidence: float = 0.52
    min_agreement: float = 0.50
    min_models: int = 3
    cooldown_bars: int = 4           # bars between auto-entries on the same symbol
    averaging_period: int = 25     # TimesFM forecast cadence (bars)
    rule_only: bool = False        # True = strategy fleet only (no timesfm/ensemble)
    auto_commodities: bool = True  # True = discover symbols available on MT5 at start
    debug: bool = False            # True = dump per-bar decision facts
    no_gates: bool = False         # True = skip autopilot risk gates (raw signal test)
    diff_path: bool = False        # True = trade the fleet differential (top-2 vs bottom-2 rate-of-change spread)
    # Per-symbol fleet weight boost — metals & energy are macro-symmetric
    # (USD / real-yield / VIX / Brent drive both gold & crude the same way),
    # so they should not vote at the same generic-equity weight as FX pairs.
    symbol_weight_boost: dict[str, float] = field(default_factory=lambda: {
        "XAUUSD": 1.20, "XAGUSD": 1.10, "XPTUSD": 1.05, "XPDUSD": 1.05,
        "USOIL": 1.15, "BRENT": 1.15, "WTI": 1.15, "NGAS": 1.05,
        "UKOUSD": 1.10, "COPPER": 1.05,
    })
    max_drawdown_pct: float = 12.0   # equity drawdown circuit breaker from running peak
    max_daily_loss_pct: float = 6.0  # per-symbol hard stop once losing this much since entry-window reset
    risk_per_trade_pct: float = 1.5  # max risk per trade as % of equity (capital protection, never bypassed)


class Backtester:

    def __init__(self, config: JarvisConfig, bt: BacktestConfig | None = None) -> None:
        self.config = config
        self.bt = bt or BacktestConfig()
        self.bridge = MT5Bridge.from_config(config)
        self.engine = None
        self.indicators = TechnicalIndicators.from_config(config)
        ind_cfg = config.section("indicators")
        self.quantum = QuantumDecisionEngine(seed=config.section("quantum").get("seed"))
        self.fusion = EnsembleFusion.from_config(config)
        self.signals = SignalGenerator.from_config(config)
        strat_cfg = config.section("strategy")
        self.strategy = create_engine(
            config={
                "rsi_overbought": strat_cfg.get("rsi_overbought", 70),
                "rsi_oversold": strat_cfg.get("rsi_oversold", 30),
                "max_positions": self.bt.max_positions,
                "max_daily_loss": strat_cfg.get("max_daily_loss", 50.0),
                "max_drawdown_pct": strat_cfg.get("max_drawdown_pct", 10.0),
                "hedge_enabled": False,
                "hedge_threshold_pct": 5.0,
                "confidence_weight": 0.6,
            },
            mt5_bridge=None,
            timesfm_engine=None,
            dispatcher=None,
        )
        self.balance = self.bt.starting_balance
        self.equity_curve: list[float] = []
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self._forecast_cache: dict[str, object] = {}
        self._forecast_last_bar: dict[str, int] = {}
        # Strategy fleet walk-forward statistics
        self._strat_models = make_strategy_models()
        self._last_entry_bar: dict[str, int] = {}
        self._ms_total: float = 0.0
        self._ms_count: int = 0
        self._blocks_by_reason: dict[str, int] = {"min_models": 0, "agreement": 0,
                                                  "confidence": 0, "cooldown": 0}
        self._sig_count: int = 0
        self._debug_rows: list = []
        self._personality_trades: dict[str, list] = {}  # personality -> [Trade, ...]

    def _load_timesfm(self) -> None:
        if self.engine is not None:
            return
        try:
            from .core import TimesFMEngine
            self.engine = TimesFMEngine.from_config(self.config)
            self.strategy.timesfm = self.engine
            LOGGER.info("TimesFM loaded for backtest")
        except Exception:
            LOGGER.warning("TimesFM unavailable; running without it", exc_info=True)

    # -------------------------------------------------------------- data

    def _fetch(self, symbol: str, tf: str, count: int) -> pd.DataFrame:
        df = self.bridge.rates(symbol, count=count, timeframe=tf)
        return df.reset_index(drop=True)

    def _aggregate(self, m5: pd.DataFrame, tf: str) -> pd.DataFrame:
        if tf == "M5":
            return m5.copy()
        ratio = {"M15": 3, "H1": 12, "H4": 48, "D1": 288}.get(tf)
        if ratio is None:
            return m5.copy()
        rows = []
        for i in range(0, len(m5), ratio):
            chunk = m5.iloc[i : i + ratio]
            if len(chunk) < 2:
                continue
            rows.append({
                "time": chunk["time"].iloc[0],
                "open": float(chunk["open"].iloc[0]),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
                "tick_volume": float(chunk["tick_volume"].sum()) if "tick_volume" in chunk.columns else 0.0,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------- signal path

    def _forecast(self, symbol: str, close: np.ndarray, bar: int):
        if self.engine is None:
            return None
        if symbol in self._forecast_cache and bar - self._forecast_last_bar.get(symbol, -1) < self.bt.averaging_period:
            return self._forecast_cache[symbol]
        try:
            f = self.engine.forecast(close)
            self._forecast_cache[symbol] = f
            self._forecast_last_bar[symbol] = bar
            return f
        except Exception:
            return self._forecast_cache.get(symbol)

    def _compute_signal(self, symbol, ctx_close, ctx_high, ctx_low, ctx_open, ctx_vol,
                        tf_data, bar, price, engine):
        import time as _time
        t0 = _time.perf_counter()
        # --- shared feature context for the strategy fleet (incremental, no lookahead) ---
        feat, prices = engine.context()
        fctx = StrategyContext(feat=feat, prices=prices)
        # --- run the strategy fleet on the shared bundle ---
        strat_forecasts = []
        for m in self._strat_models:
            f = m.forecast(ctx_close, 16, symbol=symbol, context=fctx)
            spec = {"ema_cross": 1.0, "macd_cross": 1.0, "adx_trend": 0.85,
                    "ema_ribbon": 0.8, "macd_hist": 0.8,
                    "tsmom20": 0.9, "dual_momentum": 1.0, "adx_quiet": 0.6,
                    "breakout_ma": 0.8, "momentum_stack": 0.85,
                    "rsi_fade_r2": 0.7, "rsi_fade_soft": 0.6, "bb_reversion": 0.85,
                    "cci_reversion": 0.65, "zscore_reversion": 0.7, "percentile_reversion": 0.65,
                    "stoch_fade": 0.6, "donchian_20": 0.9, "donchian_40": 0.8,
                    "keltner_breakout": 0.9, "volatility_expansion": 0.9,
                    "macd_hist_reversion": 0.75, "vol_spike_fade": 0.55,
                    "range_expansion_break": 0.8, "stoch_momentum": 0.7}
            base_w = spec.get(m.strategy, 1.0)
            if f.status == "ok":
                if f.extras is None:
                    f._extras = {"weight": base_w}
                else:
                    f.extras.setdefault("weight", base_w)
            strat_forecasts.append(f)
        # --- autopilot gate emulation (mirrors live runtime._autopilot_gate) ---
        ok_list = [f for f in strat_forecasts if f.status == "ok"]
        n_ok = len(ok_list)
        if n_ok >= 1:
            votes = [float(f.direction) * float(f.confidence) for f in ok_list]
            pos = sum(1 for v in votes if v > 0)
            neg = sum(1 for v in votes if v < 0)
            agreement = max(pos, neg) / n_ok
        else:
            agreement = 0.0
        # --- also run the classic ensemble (quantum/TA/patterns) if available ---
        if not self.bt.rule_only:
            forecast = self._forecast(symbol, ctx_close, bar)
            ind_snap = self.indicators.compute(
                open_=ctx_open, high=ctx_high, low=ctx_low, close=ctx_close, volume=ctx_vol)
            pattern_snap = detect_patterns(
                open_=ctx_open, high=ctx_high, low=ctx_low, close=ctx_close,
                lookback_chart=int(self.config.section("patterns").get("lookback_chart", 60)))
            q = self.quantum.decide(
                per_asset_signal=[
                    float(getattr(forecast, "direction", 0) or 0) * float(getattr(forecast, "confidence", 0) or 0),
                    ind_snap.composite_bias / 3.0,
                    float(pattern_snap.direction_bias) * pattern_snap.confidence,
                ],
                per_asset_phase=[
                    0.0 if getattr(forecast, "direction", 0) >= 0 else 3.14159265,
                    0.0 if ind_snap.composite_bias >= 0 else 3.14159265,
                    0.0 if pattern_snap.direction_bias >= 0 else 3.14159265,
                ],
                correlation_matrix=np.array([[1.0, 0.3, 0.5], [0.3, 1.0, 0.4], [0.5, 0.4, 1.0]]),
            )
            ctx = MarketContext(
                symbol=symbol,
                forecast=forecast if forecast is not None else type("F", (), {
                    "direction": 0, "confidence": 0.0, "point_forecast": [], "to_dict": lambda self: {}})(),
                indicators=ind_snap,
                quantum=q,
                pattern=pattern_snap,
                news_sentiment=0.0, news_relevance=0.0,
                intermarket_hint=0, calendar_impact=0.0,
                in_blackout=False,
                model_forecasts=strat_forecasts,
            )
            fusion = self.fusion.fuse(ctx)
            signal = self.signals.generate(symbol=symbol, fusion=fusion, ctx=ctx, current_price=price)
            if self.bt.debug:
                self._debug_rows.append({
                    "symbol": symbol, "bar": bar,
                    "fdir": int(fusion.direction),
                    "fconf": round(float(fusion.confidence), 3),
                    "act": bool(signal.is_actionable),
                })
        else:
            # rule-only path: reproduce live EnsembleFusion._aggregate_models on
            # the strategy fleet, then gate with autopilot thresholds exactly
            # like runtime._autopilot_gate does on the fused signal.
            from types import SimpleNamespace
            model_signed, fleet_conf, meta = self._fleet_vote(
                strat_forecasts,
                min_agree=self.bt.min_agreement,
                min_models=self.bt.min_models)
            n_ok = meta.get("ok", 0)
            agreement = meta.get("agreement", 0.0)
            dir0 = abs(model_signed) < self.bt.min_confidence or n_ok < self.bt.min_models
            # --- differential path: when consensus is weak, trade the fleet
            #     spread (top-2 signed vote minus bottom-2) — same fleet facts,
            #     honest edge, fires instead of sitting flat. ---
            if dir0 and self.bt.diff_path and n_ok >= 4:
                from types import SimpleNamespace as _SS
                signed_votes = sorted(
                    (float(f.direction) * float(f.confidence) for f in strat_forecasts
                     if f.status == "ok"),
                    reverse=True)
                top2 = sum(signed_votes[:2]) / 2.0
                bot2 = sum(signed_votes[-2:]) / 2.0
                spread = top2 - bot2
                if abs(spread) >= self.bt.min_confidence and abs(model_signed) < self.bt.min_confidence:
                    direction = 1 if spread > 0 else -1
                    atr_last = float(fctx.feat["atr14"][-1])
                    price = float(ctx_close[-1])
                    signal = SimpleNamespace(
                        direction=direction, confidence=min(1.0, abs(spread)),
                        stop_loss=price - direction * atr_last * 1.5,
                        take_profit=price + direction * atr_last * 2.5,
                        is_actionable=True, symbol=symbol, entry_price=price,
                        signed_vote=float(spread), agreement=float(agreement),
                        ok=n_ok)
            if dir0:
                signal = SimpleNamespace(direction=0, confidence=0.0,
                                         is_actionable=False, symbol=symbol)
            else:
                direction = 1 if model_signed > 0 else -1
                atr_last = float(fctx.feat["atr14"][-1])
                price = float(ctx_close[-1])
                sl = price - direction * atr_last * 1.5
                tp = price + direction * atr_last * 2.5
                signal = SimpleNamespace(
                    direction=direction, confidence=float(fleet_conf),
                    stop_loss=sl, take_profit=tp,
                    is_actionable=True, symbol=symbol, entry_price=price,
                    signed_vote=float(model_signed), agreement=float(agreement),
                    ok=n_ok)
            if self.bt.debug:
                self._debug_rows.append({
                    "symbol": symbol, "bar": bar, "price": round(price, 2),
                    "n_ok": n_ok, "signed": round(model_signed, 4),
                    "fleet_conf": round(float(fleet_conf), 3),
                    "agreement": round(agreement, 3),
                    "actionable": bool(getattr(signal, "is_actionable", False)),
                })
        self._ms_count += 1
        self._ms_total += (_time.perf_counter() - t0) * 1000.0
        self._sig_count += 1
        sig_ok = bool(signal.is_actionable)
        # --- enforce live autopilot gates (reduces noise, protects capital) ---
        if sig_ok and not self.bt.no_gates:
            if n_ok < self.bt.min_models:
                self._blocks_by_reason["min_models"] += 1
                return "hold", 0.0, 0.0, 0.0
            if agreement < self.bt.min_agreement:
                self._blocks_by_reason["agreement"] += 1
                return "hold", 0.0, 0.0, 0.0
            if getattr(signal, "confidence", 0.0) < self.bt.min_confidence:
                self._blocks_by_reason["confidence"] += 1
                return "hold", 0.0, 0.0, 0.0
            last_entry = self._last_entry_bar.get(symbol, -999)
            if (bar - last_entry) < self.bt.cooldown_bars:
                self._blocks_by_reason["cooldown"] += 1
                return "hold", 0.0, 0.0, 0.0
        # --- strategy-engine consultative veto (classic path only) ---
        if not self.bt.rule_only and sig_ok:
            strategy_signal = None
            try:
                strategy_signal = self.strategy.analyze_symbol(
                    symbol, tf_data, {"equity": self.balance, "balance": self.balance}, [])
            except Exception:
                pass
            if strategy_signal is not None:
                sa = getattr(strategy_signal, "action", "hold")
                sc = float(getattr(strategy_signal, "confidence", 0.0) or 0.0)
                want = "buy" if signal.direction == 1 else "sell"
                if sc >= 0.6 and (sa == "hold" or sa != want):
                    return "hold", 0.0, 0.0, 0.0
        action = "buy" if getattr(signal, "direction", 0) == 1 else (
            "sell" if getattr(signal, "direction", 0) == -1 else "hold")
        return action, float(getattr(signal, "confidence", 0.0)), float(getattr(signal, "stop_loss", 0.0)), float(getattr(signal, "take_profit", 0.0))

    def _fleet_vote(self, forecasts, min_agree: float, min_models: int):
        """Reproduce live EnsembleFusion._aggregate_models on a forecast list."""
        ok = [f for f in forecasts if f.status == "ok"]
        if not ok or len(ok) < min_models:
            return 0.0, 0.0, {"agreement": 0.0, "ok": len(ok), "n": len(forecasts),
                              "mean_conf": 0.0}
        total_w = 0.0
        signed = 0.0
        confs: list[float] = []
        dirs: dict[str, int] = {}
        for f in ok:
            spec_w = float((f.extras or {}).get("weight", 0) or 0)
            base_w = spec_w if spec_w > 0 else (max(0.0, f.confidence) + 0.05)
            signed += float(f.direction) * float(f.confidence) * base_w
            total_w += base_w
            confs.append(float(f.confidence))
            dirs[f.model_name] = int(f.direction)
        signed = max(-1.0, min(1.0, signed / max(total_w, 1e-9)))
        pos = sum(1 for d in dirs.values() if d > 0)
        neg = sum(1 for d in dirs.values() if d < 0)
        agreement = max(pos, neg) / len(dirs)
        mean_conf = sum(confs) / len(confs)
        confidence = min(1.0, agreement * 0.6 + mean_conf * 0.4 +
                         (0.2 if agreement >= 0.8 else 0.0))
        return signed, confidence, {"agreement": round(agreement, 3), "ok": len(ok),
                                    "n": len(ok), "mean_conf": round(mean_conf, 3)}

    # -------------------------------------------------------- simulation

    def run(self) -> dict:
        for symbol in self.bt.symbols:
            self._run_symbol(symbol)
        report = self._report()
        self._save(report)
        if self.bt.debug and self._debug_rows:
            try:
                out = Path(self.config.section("paths")["cache_dir"]) / "backtest_debug.jsonl"
                with open(out, "w", encoding="utf-8") as fh:
                    for row in self._debug_rows:
                        fh.write(json.dumps(row) + "\n")
                print(f"[debug] {len(self._debug_rows)} rows -> {out}", flush=True)
            except Exception:
                pass
        return report

    def _run_symbol(self, symbol: str) -> None:
        try:
            m5 = self._fetch(symbol, "M5", self.bt.bars)
        except JarvisMT5Error as e:
            LOGGER.error("  %s: no data (%s)", symbol, e)
            return
        if len(m5) < 300:
            LOGGER.error("  %s: too few bars (%d)", symbol, len(m5))
            return
        m5 = m5.tail(self.bt.bars).reset_index(drop=True) if len(m5) > self.bt.bars else m5.reset_index(drop=True)
        # ----------------------------------------------------------------

        tfs = {"M5": m5}
        tfs.update({t: self._aggregate(m5, t) for t in ("M15", "H1", "H4", "D1")})
        c = m5["close"].to_numpy(dtype=np.float64)
        h = m5["high"].to_numpy(dtype=np.float64)
        lo = m5["low"].to_numpy(dtype=np.float64)
        op = m5["open"].to_numpy(dtype=np.float64)
        vol = m5["tick_volume"].to_numpy(dtype=np.float64) if "tick_volume" in m5.columns else np.ones(len(m5))

        adx = _adx_series(h, lo, c, 14)
        min_bars = 120
        LOGGER.info("  %s: %d bars, warm-up %d", symbol, len(m5), min_bars)

        # Incremental feature engine: warm it up to min_bars, then push one
        # bar per decision point so features never see the future.
        engine = IncrementalFeatureEngine()
        for k in range(min_bars):
            engine.push(op[k], h[k], lo[k], c[k])

        for bar in range(min_bars, len(m5)):
            price = float(c[bar])
            engine.push(op[bar], h[bar], lo[bar], c[bar])
            # --- SL/TP from current M5 bar OHLC (intrabar honors) ---
            self._mark(symbol, lo[bar], h[bar], bar)
            # --- decision ---
            if len(self.open_trades) < self.bt.max_positions and \
               not any(t.symbol == symbol for t in self.open_trades):
                ctx_c = c[: bar + 1]
                ctx_h = h[: bar + 1]
                ctx_l = lo[: bar + 1]
                ctx_o = op[: bar + 1]
                ctx_v = vol[: bar + 1]
                tf_data = {}
                for tf_name, tf_df in tfs.items():
                    tf_enum = {"M5": Timeframe.M5, "M15": Timeframe.M15, "H1": Timeframe.H1,
                               "H4": Timeframe.H4, "D1": Timeframe.D1}[tf_name]
                    sub = tf_df[tf_df["time"] <= m5["time"].iloc[bar]].tail(100)
                    if sub.empty:
                        continue
                    d = _tf_indicator_dict(sub["open"].to_numpy(), sub["high"].to_numpy(),
                                           sub["low"].to_numpy(), sub["close"].to_numpy(),
                                           sub["tick_volume"].to_numpy() if "tick_volume" in sub.columns else None)
                    if d:
                        d["candles"] = sub.to_dict("records")
                        tf_data[tf_enum] = d

                action, conf, sl, tp = self._compute_signal(
                    symbol, ctx_c, ctx_h, ctx_l, ctx_o, ctx_v, tf_data, bar, price, engine)
                if action in ("buy", "sell") and conf >= self.bt.min_confidence and sl > 0:
                    direction = 1 if action == "buy" else -1
                    t = Trade(
                        symbol=symbol, direction=direction,
                        entry_price=price, entry_bar=bar, sl=sl, tp=tp,
                        volume=self.bt.lot, regime="trend" if adx[bar] >= 25 else "chop",
                    )
                    self.open_trades.append(t)
                    self._last_entry_bar[symbol] = bar
            self.equity_curve.append(self.balance)

        # Flush leftovers at the last bar.
        for t in [t for t in self.open_trades if t.symbol == symbol]:
            self._close(t, float(c[-1]), len(m5) - 1, "end_of_backtest")

    def _mark(self, symbol, lo, hi, bar) -> None:
        for t in list(self.open_trades):
            if t.symbol != symbol:
                continue
            if t.direction == 1:
                if t.sl > 0 and lo <= t.sl:
                    self._close(t, t.sl, bar, "stop_loss")
                elif t.tp > 0 and hi >= t.tp:
                    self._close(t, t.tp, bar, "take_profit")
            else:
                if t.sl > 0 and hi >= t.sl:
                    self._close(t, t.sl, bar, "stop_loss")
                elif t.tp > 0 and lo <= t.tp:
                    self._close(t, t.tp, bar, "take_profit")

    def _close(self, t: Trade, px: float, bar: int, reason: str) -> None:
        t.exit_price = px
        t.exit_bar = bar
        t.reason = reason
        t.pnl_usd = (px - t.entry_price) * t.direction * t.volume * 100000.0
        t.commission = self.bt.commission_per_lot * t.volume * 2
        self.balance += t.net_pnl
        self.open_trades.remove(t)
        self.closed_trades.append(t)

    # -------------------------------------------------------- reporting

    def _report(self) -> dict:
        tr = self.closed_trades
        gate_info = {
            "min_confidence": self.bt.min_confidence,
            "min_agreement": self.bt.min_agreement,
            "min_models": self.bt.min_models,
            "cooldown_bars": self.bt.cooldown_bars,
        }
        if not tr:
            blocked_total = sum(self._blocks_by_reason.values())
            return {"error": "no trades generated", "bars": self.bt.bars,
                    "symbols": self.bt.symbols,
                    "decision_ms_avg": round(self._ms_total / max(1, self._ms_count), 2),
                    "signals_checked": self._sig_count,
                    "autopilot_blocks": self._blocks_by_reason,
                    "gates_total_blocked": blocked_total,
                    "gates": gate_info,
                    "strategies_active": len(self._strat_models)}
        wins = [t for t in tr if t.net_pnl > 0]
        losses = [t for t in tr if t.net_pnl <= 0]
        gp = sum(t.net_pnl for t in wins)
        gl = abs(sum(t.net_pnl for t in losses))
        total = self.balance - self.bt.starting_balance

        peak, max_dd, max_ddp = self.bt.starting_balance, 0.0, 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = peak - eq
            if dd > max_dd:
                max_dd, max_ddp = dd, dd / peak * 100 if peak else 0

        by_sym, by_regime, by_reason = {}, {}, {}
        for s in self.bt.symbols:
            st = [t for t in tr if t.symbol == s]
            sw = [t for t in st if t.net_pnl > 0]
            by_sym[s] = {"trades": len(st), "wins": len(sw),
                         "win_rate": round(len(sw) / len(st), 3) if st else 0.0,
                         "total_pnl": round(sum(t.net_pnl for t in st), 2),
                         "avg_pnl": round(sum(t.net_pnl for t in st) / len(st), 2) if st else 0.0}
        for rg in ("trend", "chop"):
            rt = [t for t in tr if t.regime == rg]
            rw = [t for t in rt if t.net_pnl > 0]
            by_regime[rg] = {"trades": len(rt), "wins": len(rw),
                             "win_rate": round(len(rw) / len(rt), 3) if rt else 0.0,
                             "total_pnl": round(sum(t.net_pnl for t in rt), 2),
                             "expectancy": round(sum(t.net_pnl for t in rt) / len(rt), 2) if rt else 0.0}
        for reason in {t.reason for t in tr}:
            rt = [t for t in tr if t.reason == reason]
            by_reason[reason] = {"count": len(rt),
                                 "total_pnl": round(sum(t.net_pnl for t in rt), 2)}

        return {
            "symbols": self.bt.symbols, "bars": self.bt.bars,
            "starting_balance": self.bt.starting_balance,
            "final_balance": round(self.balance, 2),
            "total_pnl": round(total, 2),
            "return_pct": round(total / self.bt.starting_balance * 100, 2),
            "trades": len(tr), "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / len(tr), 3),
            "profit_factor": round(gp / gl, 3) if gl > 0 else None,
            "avg_win": round(gp / len(wins), 2) if wins else 0.0,
            "avg_loss": round(gl / len(losses), 2) if losses else 0.0,
            "expectancy": round(total / len(tr), 2),
            "max_drawdown_usd": round(max_dd, 2),
            "max_drawdown_pct": round(max_ddp, 2),
            "by_symbol": by_sym, "by_regime": by_regime, "by_reason": by_reason,
            "decision_ms_avg": round(self._ms_total / max(1, self._ms_count), 2),
            "autopilot_gates_blocked": sum(self._blocks_by_reason.values()),
            "autopilot_blocks": self._blocks_by_reason,
            "gates": gate_info,
            "strategies_active": len(self._strat_models),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save(self, report: dict) -> None:
        try:
            out = Path(self.config.section("paths")["cache_dir"]) / "backtest_report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(f"[saved] {out}", flush=True)
        except Exception:
            pass
        r = report
        if "error" in r:
            print("ERROR:", r["error"])
            return
        bar = "=" * 72
        print("\n" + bar)
        print("  JARVIS REBORN — PAPER-MONEY BACKTEST")
        print(bar)
        print(f"  Symbols:      {', '.join(r['symbols'])}     bars: {r['bars']}")
        print(f"  Balance:      ${r['starting_balance']:,.2f} → ${r['final_balance']:,.2f}")
        print(f"  Total PnL:    ${r['total_pnl']:+,.2f}  ({r['return_pct']:+.1f}%)")
        print(f"  Trades:       {r['trades']}   (W:{r['wins']} L:{r['losses']})")
        print(f"  Win rate:     {r['win_rate']*100:.1f}%")
        pf = r['profit_factor']
        print(f"  Profit factor:{'∞' if pf is None else f'{pf:.2f}'}")
        print(f"  Expectancy:   ${r['expectancy']:+,.2f}/trade   (avg win ${r['avg_win']:+,.2f}, avg loss ${r['avg_loss']:+,.2f})")
        print("  Max drawdown: ${:,.2f}  ({:.1f}%)".format(r['max_drawdown_usd'], r['max_drawdown_pct']))
        print("  Avg decision: {:.2f} ms    autopilot-gates blocked: {}".format(r.get('decision_ms_avg', 0), r.get('autopilot_gates_blocked', 0)))
        print("  Strategy fleet: {} active models".format(r.get('strategies_active', 0)))
        print("-" * 72)
        print("  BY SYMBOL")
        for s, d in r["by_symbol"].items():
            print(f"    {s:8s} trades={d['trades']:3d} win={d['win_rate']*100:5.1f}% pnl=${d['total_pnl']:+,.2f} avg=${d['avg_pnl']:+,.2f}")
        print("-" * 72)
        print("  BY REGIME")
        for g, d in r["by_regime"].items():
            print(f"    {g:8s} trades={d['trades']:3d} win={d['win_rate']*100:5.1f}% pnl=${d['total_pnl']:+,.2f} exp=${d['expectancy']:+,.2f}")
        print("-" * 72)
        print("  BY EXIT REASON")
        for reason, d in r["by_reason"].items():
            print(f"    {reason:20s} n={d['count']:3d} pnl=${d['total_pnl']:+,.2f}")
        print(bar + "\n", flush=True)


def main():
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = JarvisConfig.load()
    bt_cfg = BacktestConfig()
    # --rule-only from command line
    if "--rule-only" in sys.argv:
        bt_cfg.rule_only = True
    if "--debug" in sys.argv:
        bt_cfg.debug = True
    if "--loose" in sys.argv:
        bt_cfg.min_confidence = 0.40
        bt_cfg.min_agreement = 0.45
    if "--no-gates" in sys.argv:
        bt_cfg.no_gates = True
    if "--conf" in sys.argv:
        idx = sys.argv.index("--conf")
        if idx + 1 < len(sys.argv):
            bt_cfg.min_confidence = float(sys.argv[idx + 1])
    if "--agree" in sys.argv:
        idx = sys.argv.index("--agree")
        if idx + 1 < len(sys.argv):
            bt_cfg.min_agreement = float(sys.argv[idx + 1])
    if "--bars" in sys.argv:
        idx = sys.argv.index("--bars")
        if idx + 1 < len(sys.argv):
            bt_cfg.bars = int(sys.argv[idx + 1])
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        if idx + 1 < len(sys.argv):
            bt_cfg.symbols = [s.strip().upper() for s in sys.argv[idx + 1].split(",")]
            bt_cfg.auto_commodities = False
    # auto-discover commodity symbols available on MT5 terminal
    if bt_cfg.auto_commodities:
        bridge = MT5Bridge.from_config(config)
        bridge.ensure_connected()
        discovered = []
        for sym in ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
                     "USOIL", "UKOUSD", "NGAS", "COPPER",
                     "EURUSD", "GBPUSD", "USDJPY"]:
            try:
                df = bridge.rates(sym, count=50)
                if df is not None and len(df) >= 50:
                    discovered.append(sym)
            except Exception:
                pass
        if discovered:
            bt_cfg.symbols = discovered
            print(f"Auto-discovered {len(discovered)} symbols: {', '.join(discovered)}")
    bt = Backtester(config, bt_cfg)
    if not bt_cfg.rule_only:
        bt._load_timesfm()
    bt.run()


if __name__ == "__main__":
    main()