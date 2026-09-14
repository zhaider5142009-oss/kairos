"""
JARVIS Reborn Strategy Engine - Main orchestration.
Combines pattern recognition, multi-timeframe analysis, and position management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import threading

from .patterns import PatternRecognizer, Candle
from .multitf import MultiTimeframeAnalyzer, Timeframe, ConfluenceResult, create_default_analyzer
from .position_mgr import PositionManager, create_default_manager, Position, RiskMetrics, _position_to_dict
from . import assess_market, StrategyConfig, should_enter, should_exit


@dataclass
class StrategySignal:
    """Complete trading signal with all analysis."""
    symbol: str
    timestamp: float
    action: str  # "buy", "sell", "close", "hedge", "hold"
    confidence: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.0
    risk_reward: float = 0.0
    entry_grade: str = "D"
    pattern_summary: dict = field(default_factory=dict)
    confluence: ConfluenceResult | None = None
    reasoning: str = ""
    meta: dict = field(default_factory=dict)


class StrategyEngine:
    """
    Ultra-sophisticated strategy engine for JARVIS Reborn.

    Combines:
    - Pattern recognition (candlestick, chart, indicator)
    - Multi-timeframe confluence analysis
    - Position management with hedging
    - Risk management
    - TimesFM forecast integration
    """

    def __init__(
        self,
        config: dict | None = None,
        mt5_bridge = None,
        timesfm_engine = None,
        dispatcher = None
    ):
        self.config = config or {}
        self.mt5 = mt5_bridge
        self.timesfm = timesfm_engine
        self.dispatcher = dispatcher

        # Sub-components
        self.pattern_recognizer = PatternRecognizer(self.config.get("patterns", {}))
        self.multitf_analyzer = create_default_analyzer()
        self.position_manager = create_default_manager(mt5_bridge)

        # Strategy config
        self.strategy_config = StrategyConfig(
            rsi_overbought=self.config.get("rsi_overbought", 70),
            rsi_oversold=self.config.get("rsi_oversold", 30),
            max_open_positions=self.config.get("max_positions", 5),
            max_daily_loss=self.config.get("max_daily_loss", 50.0),
            max_drawdown_pct=self.config.get("max_drawdown_pct", 10.0),
            hedge_enabled=self.config.get("hedge_enabled", True),
            hedge_threshold_pct=self.config.get("hedge_threshold_pct", 5.0),
            confidence_weight=self.config.get("confidence_weight", 0.6)
        )

        # State
        self.last_analysis: dict[str, StrategySignal] = {}
        self.last_metrics: RiskMetrics | None = None
        self.last_confluence: ConfluenceResult | None = None
        self.candle_cache: dict[str, dict[Timeframe, list[Candle]]] = {}
        self._lock = threading.RLock()
        self.enabled = True
        self.last_run = 0.0

        # Callbacks
        self.on_signal: Callable[[StrategySignal], None] | None = None
        self.on_position_update: Callable[[list[Position]], None] | None = None
        self.on_risk_alert: Callable[[str, dict], None] | None = None

    # -------------------------------------------------------------------------
    # Main Analysis Loop
    # -------------------------------------------------------------------------

    def analyze_symbol(
        self,
        symbol: str,
        tf_data: dict[Timeframe, dict],
        account_info: dict,
        mt5_positions: list[dict]
    ) -> StrategySignal:
        """Run complete analysis for a symbol."""
        with self._lock:
            current_price = tf_data.get(Timeframe.M15, {}).get("close", 0)
            if current_price == 0:
                current_price = tf_data.get(Timeframe.H1, {}).get("close", 0)

            # Update position manager
            metrics = self.position_manager.update_positions(mt5_positions, account_info)
            self.last_metrics = metrics

            # Get TimesFM forecast
            forecast = self._get_timesfm_forecast(symbol)

            # Pattern recognition on each timeframe
            patterns_by_tf = {}
            for tf, data in tf_data.items():
                candles = self._dict_to_candles(data.get("candles", []))
                if candles:
                    patterns = self.pattern_recognizer.detect_candlestick_patterns(candles)
                    patterns.extend(self.pattern_recognizer.detect_chart_patterns(candles))
                    patterns_by_tf[tf.value] = patterns

            # Indicator patterns (using M15 as primary)
            m15_data = tf_data.get(Timeframe.M15, {})
            indicator_patterns = self.pattern_recognizer.detect_indicator_patterns(
                rsi=m15_data.get("rsi", 50),
                macd_hist=m15_data.get("macd_hist", 0),
                macd_signal=m15_data.get("macd_signal", 0),
                bb_upper=m15_data.get("bb_upper", 0),
                bb_lower=m15_data.get("bb_lower", 0),
                bb_middle=m15_data.get("bb_middle", 0),
                price=current_price,
                volume=m15_data.get("volume", 0),
                avg_volume=m15_data.get("avg_volume", 0)
            )
            if Timeframe.M15.value not in patterns_by_tf:
                patterns_by_tf[Timeframe.M15.value] = []
            patterns_by_tf[Timeframe.M15.value].extend(indicator_patterns)

            # Multi-timeframe confluence
            confluence = self.multitf_analyzer.analyze(
                tf_data, current_price, account_info.get("equity", 0), mt5_positions
            )
            self.last_confluence = confluence

            # Base market assessment
            assessment = assess_market(
                forecast=forecast,
                rsi=m15_data.get("rsi", 50),
                macd_fast=m15_data.get("macd_fast", m15_data.get("ema_fast", current_price)),
                macd_slow=m15_data.get("macd_slow", m15_data.get("ema_slow", current_price)),
                macd_signal=m15_data.get("macd_signal", 0),
                account_equity=account_info.get("equity", 0),
                account_drawdown_pct=metrics.current_drawdown,
                cfg=self.strategy_config
            )

            # Generate signal
            signal = self._generate_signal(
                symbol=symbol,
                current_price=current_price,
                assessment=assessment,
                confluence=confluence,
                patterns_by_tf=patterns_by_tf,
                metrics=metrics,
                account_info=account_info,
                mt5_positions=mt5_positions
            )

            self.last_analysis[symbol] = signal
            return signal

    def _generate_signal(
        self,
        symbol: str,
        current_price: float,
        assessment: dict,
        confluence: ConfluenceResult,
        patterns_by_tf: dict,
        metrics: RiskMetrics,
        account_info: dict,
        mt5_positions: list
    ) -> StrategySignal:
        """Generate final trading signal from all analysis."""

        # Normalize MT5 position objects/dicts into plain dicts (defensive).
        mt5_positions = [_position_to_dict(p) for p in (mt5_positions or [])]

        # Count existing positions for this symbol
        symbol_positions = [p for p in mt5_positions if p.get("symbol") == symbol]
        has_long = any(p.get("type") == 0 for p in symbol_positions)
        has_short = any(p.get("type") == 1 for p in symbol_positions)

        # Pattern summary
        pattern_summary = {}
        total_patterns = 0
        bullish_patterns = 0
        bearish_patterns = 0
        for tf, patterns in patterns_by_tf.items():
            for p in patterns:
                total_patterns += 1
                if p.direction.value == "bullish":
                    bullish_patterns += 1
                elif p.direction.value == "bearish":
                    bearish_patterns += 1
        pattern_summary = {
            "total": total_patterns,
            "bullish": bullish_patterns,
            "bearish": bearish_patterns,
            "by_timeframe": {tf: len(p) for tf, p in patterns_by_tf.items()}
        }

        # Determine action
        action = "hold"
        confidence = 0.0
        entry_price = current_price
        stop_loss = 0.0
        take_profit = 0.0
        volume = 0.0
        reasoning_parts = []

        # Check if we should hedge
        if assessment.get("hedge", False) and not any(
            p.side.value.startswith("hedge") for p in self.position_manager.positions.values()
            if p.symbol == symbol
        ):
            action = "hedge"
            confidence = 0.8
            reasoning_parts.append(f"Drawdown hedge triggered ({assessment.get('trend', 'neutral')})")

        # Check confluence recommendation
        elif confluence.recommendation in ("long", "short"):
            if confluence.recommendation == "long" and not has_long:
                action = "buy"
                confidence = min(0.9, confluence.overall_strength * 0.8 + 0.2)
                reasoning_parts.append(f"MTF Confluence: {confluence.overall_trend} ({confluence.entry_quality})")
            elif confluence.recommendation == "short" and not has_short:
                action = "sell"
                confidence = min(0.9, confluence.overall_strength * 0.8 + 0.2)
                reasoning_parts.append(f"MTF Confluence: {confluence.overall_trend} ({confluence.entry_quality})")

        # Check assessment decision
        elif assessment["decision"] in ("buy", "sell"):
            if assessment["decision"] == "buy" and not has_long:
                action = "buy"
                confidence = abs(assessment.get("score", 0))
                reasoning_parts.append(f"Assessment: {assessment.get('trend', 'neutral')} (score: {assessment.get('score', 0)})")
            elif assessment["decision"] == "sell" and not has_short:
                action = "sell"
                confidence = abs(assessment.get("score", 0))
                reasoning_parts.append(f"Assessment: {assessment.get('trend', 'neutral')} (score: {assessment.get('score', 0)})")

        # Check exit conditions for existing positions
        if action == "hold" and symbol_positions:
            exit_actions = self.position_manager.check_exit_conditions(metrics, confluence)
            for exit_action in exit_actions:
                if any(str(p.get("ticket")) == str(exit_action.get("ticket")) for p in symbol_positions):
                    action = exit_action["action"]
                    confidence = 0.9
                    reasoning_parts.append(f"Exit: {exit_action.get('reason', 'unknown')}")
                    break

        # Calculate SL/TP and volume for entry signals
        if action in ("buy", "sell"):
            volume = self._calculate_volume(symbol, current_price, account_info)
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, current_price, action, confluence, metrics
            )

        # Build reasoning
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No clear signal"

        return StrategySignal(
            symbol=symbol,
            timestamp=time.time(),
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=volume,
            risk_reward=confluence.risk_reward_ratio,
            entry_grade=confluence.entry_quality,
            pattern_summary=pattern_summary,
            confluence=confluence,
            reasoning=reasoning,
            meta={
                "assessment": assessment,
                "metrics": metrics.__dict__,
                "account_equity": account_info.get("equity", 0)
            }
        )

    def _calculate_volume(
        self,
        symbol: str,
        entry_price: float,
        account_info: dict
    ) -> float:
        """Calculate position size using position manager."""
        # Need a dummy stop loss for sizing calculation
        # Use 2% as fallback
        stop_loss = entry_price * 0.98  # Will be refined
        return self.position_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            account_equity=account_info.get("equity", 0),
            symbol_info={"trade_tick_value": 1.0, "trade_tick_size": 0.00001}
        )

    def _calculate_sl_tp(
        self,
        symbol: str,
        price: float,
        side: str,
        confluence: ConfluenceResult,
        metrics: RiskMetrics
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit."""
        # Use ATR from confluence
        avg_atr = sum(a.atr for a in confluence.details.get("analyses", {}).values()) / max(len(confluence.details.get("analyses", {})), 1)

        if avg_atr <= 0:
            # Fallback to key levels
            support = confluence.key_levels.get("support", [])
            resistance = confluence.key_levels.get("resistance", [])

            if side == "buy":
                sl = max(support) if support else price * 0.98
                tp = min(resistance) if resistance else price * 1.04
            else:
                sl = min(resistance) if resistance else price * 1.02
                tp = max(support) if support else price * 0.96
        else:
            # ATR-based
            if side == "buy":
                sl = price - avg_atr * 2
                tp = price + avg_atr * 3  # 1:1.5 risk/reward
            else:
                sl = price + avg_atr * 2
                tp = price - avg_atr * 3

        return sl, tp

    def _get_timesfm_forecast(self, symbol: str) -> float:
        """Get TimesFM forecast for symbol."""
        if self.timesfm and hasattr(self.timesfm, "forecast"):
            try:
                result = self.timesfm.forecast(symbol, horizon=12)
                if result and len(result) > 0:
                    # Return directional bias: >0 bullish, <0 bearish
                    return float(result[-1] - result[0])  # Simplified
            except Exception:
                pass
        return 0.0

    def _dict_to_candles(self, candle_dicts: list[dict]) -> list[Candle]:
        """Convert dict candles to Candle objects."""
        candles = []
        for c in candle_dicts:
            candles.append(Candle(
                open=c.get("open", 0),
                high=c.get("high", 0),
                low=c.get("low", 0),
                close=c.get("close", 0),
                volume=c.get("volume", 0),
                timestamp=c.get("time", 0)
            ))
        return candles

    # -------------------------------------------------------------------------
    # Portfolio-level methods
    # -------------------------------------------------------------------------

    def run_full_analysis(
        self,
        symbols: list[str],
        market_data: dict[str, dict],
        account_info: dict,
        mt5_positions: list[dict]
    ) -> dict[str, StrategySignal]:
        """Run analysis for all symbols."""
        signals = {}
        for symbol in symbols:
            if symbol in market_data:
                signals[symbol] = self.analyze_symbol(
                    symbol, market_data[symbol], account_info, mt5_positions
                )

                # Emit signal callback
                if self.on_signal:
                    try:
                        self.on_signal(signals[symbol])
                    except Exception:
                        pass

        return signals

    def get_portfolio_risk(self, account_info: dict, mt5_positions: list[dict]) -> RiskMetrics:
        """Get current portfolio risk metrics."""
        return self.position_manager.update_positions(mt5_positions, account_info)

    def execute_signal(self, signal: StrategySignal) -> bool:
        """Execute a trading signal via MT5 bridge."""
        if not self.mt5 or not hasattr(self.mt5, "execute_order"):
            return False

        if signal.action in ("buy", "sell") and signal.volume > 0:
            order_type = 0 if signal.action == "buy" else 1
            try:
                result = self.mt5.execute_order(
                    symbol=signal.symbol,
                    order_type=order_type,
                    volume=signal.volume,
                    price=signal.entry_price,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                    magic=123456,
                    comment=f"JARVIS-{signal.entry_grade}"
                )
                if self.dispatcher:
                    self.dispatcher.emit("signal_executed", {
                        "signal": signal.__dict__,
                        "result": result
                    })
                return result.get("success", False)
            except Exception as e:
                if self.dispatcher:
                    self.dispatcher.emit("signal_error", {"signal": signal.__dict__, "error": str(e)})
                return False

        elif signal.action == "hedge":
            # Execute hedge order
            pass

        return False

    def apply_management(self) -> list[dict]:
        """Apply exit/trailing/hedge actions from the last analysis through MT5.

        Returns the raw action list (for telemetry). Uses the last analysis's
        metrics + confluence; safe no-op if no analysis has run.
        """
        if not self.mt5 or not hasattr(self.mt5, "close_position"):
            return []
        if not self.last_metrics or self.last_confluence is None:
            return []

        actions = self.position_manager.check_exit_conditions(
            self.last_metrics, self.last_confluence
        )
        results = []
        for act in actions:
            kind = act.get("action")
            try:
                if kind == "close" and act.get("ticket"):
                    r = self.mt5.close_position(int(act["ticket"]))
                    results.append({**act, "executed": bool(r.success), "comment": r.comment})
                elif kind == "modify" and act.get("ticket"):
                    r = self.mt5.modify_position(
                        int(act["ticket"]),
                        sl=act.get("sl"),
                        tp=act.get("tp"),
                    )
                    results.append({**act, "executed": bool(r.success), "comment": r.comment})
                else:
                    results.append({**act, "executed": False, "comment": "unsupported action"})
            except Exception as e:
                results.append({**act, "executed": False, "comment": str(e)})
        return results

    def get_status(self) -> dict:
        """Get engine status for UI/monitoring."""
        hc = getattr(self.position_manager.hedge_config, "enabled", None)
        sc = self.position_manager.sizing_config
        return {
            "enabled": self.enabled,
            "last_run": self.last_run,
            "symbols_analyzed": len(self.last_analysis),
            "signals": {s: sig.__dict__ for s, sig in self.last_analysis.items()},
            "positions": len(self.position_manager.positions),
            "config": {
                "hedge_enabled": hc if hc is not None else self.strategy_config.hedge_enabled,
                "hedge_threshold_pct": self.strategy_config.hedge_threshold_pct,
                "max_positions": sc.max_positions,
                "risk_per_trade": sc.risk_per_trade_pct,
                "max_daily_loss": self.strategy_config.max_daily_loss,
                "max_drawdown_pct": self.strategy_config.max_drawdown_pct,
                "confidence_weight": self.strategy_config.confidence_weight,
            }
        }


def create_engine(config: dict, mt5_bridge=None, timesfm_engine=None, dispatcher=None) -> StrategyEngine:
    """Factory function to create configured strategy engine."""
    return StrategyEngine(
        config=config,
        mt5_bridge=mt5_bridge,
        timesfm_engine=timesfm_engine,
        dispatcher=dispatcher
    )