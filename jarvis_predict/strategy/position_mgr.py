"""
Advanced position management with hedging for JARVIS Reborn.
Handles position sizing, hedging, trailing stops, and portfolio risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math
import time


def _position_to_dict(p) -> dict:
    """Normalize an MT5 position (namedtuple/object) or dict into a plain dict."""
    if isinstance(p, dict):
        return dict(p)
    if p is None:
        return {}
    keys = ("ticket", "symbol", "type", "volume", "price_open", "price_current",
            "sl", "tp", "time", "profit", "swap", "commission", "magic", "comment")
    out: dict = {}
    for k in keys:
        try:
            v = getattr(p, k, None)
        except Exception:
            v = None
        if v is not None:
            out[k] = v
    return out


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    HEDGE_LONG = "hedge_long"   # Hedge for short position
    HEDGE_SHORT = "hedge_short" # Hedge for long position


class PositionStatus(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    HEDGED = "hedged"


@dataclass
class Position:
    ticket: int
    symbol: str
    side: PositionSide
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    opened_at: float
    status: PositionStatus = PositionStatus.OPEN
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    hedge_ticket: int | None = None
    magic: int = 0
    comment: str = ""


@dataclass
class RiskMetrics:
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    margin_used: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    open_positions: int = 0
    hedge_positions: int = 0


@dataclass
class HedgeConfig:
    enabled: bool = True
    trigger_drawdown_pct: float = 5.0      # Start hedging at 5% drawdown
    max_hedge_ratio: float = 0.5           # Max hedge size as fraction of position
    min_hedge_distance_pips: int = 50      # Min distance from main position
    close_hedge_at_be: bool = True         # Close hedge at breakeven
    hedge_symbol_suffix: str = ""          # For correlated pairs hedge


@dataclass
class PositionSizingConfig:
    risk_per_trade_pct: float = 1.0        # Risk per trade as % of equity
    max_risk_per_symbol_pct: float = 3.0   # Max risk per symbol
    max_total_risk_pct: float = 5.0        # Max portfolio risk
    max_positions: int = 10
    max_positions_per_symbol: int = 3
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 100.0


class PositionManager:
    """Advanced position management with hedging, trailing stops, and risk control."""

    def __init__(
        self,
        hedge_config: HedgeConfig | None = None,
        sizing_config: PositionSizingConfig | None = None,
        mt5_bridge = None
    ):
        self.hedge_config = hedge_config or HedgeConfig()
        self.sizing_config = sizing_config or PositionSizingConfig()
        self.mt5 = mt5_bridge
        self.positions: dict[int, Position] = {}
        self.closed_positions: list[Position] = []
        self.equity_high_water: float = 0.0
        self.daily_start_equity: float = 0.0
        self.last_update: float = 0.0

    # -------------------------------------------------------------------------
    # Position Tracking
    # -------------------------------------------------------------------------

    def update_positions(self, mt5_positions: list[dict], account_info: dict) -> RiskMetrics:
        """Update internal position state from MT5 data."""
        self.positions.clear()

        for p in mt5_positions:
            norm = _position_to_dict(p)
            side = self._mt5_to_side(norm.get("type", 0))
            pos = Position(
                ticket=norm.get("ticket", 0),
                symbol=norm.get("symbol", ""),
                side=side,
                volume=norm.get("volume", 0),
                entry_price=norm.get("price_open", 0),
                current_price=norm.get("price_current", 0),
                stop_loss=norm.get("sl", 0),
                take_profit=norm.get("tp", 0),
                opened_at=norm.get("time", time.time()),
                profit=norm.get("profit", 0),
                swap=norm.get("swap", 0),
                commission=norm.get("commission", 0),
                magic=norm.get("magic", 0),
                comment=norm.get("comment", "")
            )
            self.positions[pos.ticket] = pos

        return self.calculate_risk_metrics(account_info)

    def _mt5_to_side(self, mt5_type: int) -> PositionSide:
        # MT5: 0=buy, 1=sell
        if mt5_type == 0:
            return PositionSide.LONG
        return PositionSide.SHORT

    def calculate_risk_metrics(self, account_info: dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        equity = account_info.get("equity", 0)
        balance = account_info.get("balance", 0)
        margin = account_info.get("margin", 0)
        free_margin = account_info.get("free_margin", 0)

        # Update high water mark
        if equity > self.equity_high_water:
            self.equity_high_water = equity

        # Daily start equity (reset at midnight)
        # Simplified: use balance as proxy
        if self.daily_start_equity == 0:
            self.daily_start_equity = balance

        total_exposure = 0
        long_exp = 0
        short_exp = 0
        unrealized = 0
        hedge_count = 0

        for pos in self.positions.values():
            exposure = pos.volume * pos.current_price
            total_exposure += exposure
            unrealized += pos.profit

            if pos.side in (PositionSide.LONG, PositionSide.HEDGE_LONG):
                long_exp += exposure
            else:
                short_exp += exposure

            if pos.side in (PositionSide.HEDGE_LONG, PositionSide.HEDGE_SHORT):
                hedge_count += 1

        net_exposure = long_exp - short_exp
        current_dd = (self.equity_high_water - equity) / self.equity_high_water * 100 if self.equity_high_water else 0
        daily_pnl = equity - self.daily_start_equity

        return RiskMetrics(
            total_exposure=total_exposure,
            long_exposure=long_exp,
            short_exposure=short_exp,
            net_exposure=net_exposure,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            max_drawdown=max(self.equity_high_water - equity, 0) / self.equity_high_water * 100 if self.equity_high_water else 0,
            current_drawdown=current_dd,
            margin_used=margin,
            free_margin=free_margin,
            margin_level=equity / margin * 100 if margin > 0 else 0,
            open_positions=len(self.positions),
            hedge_positions=hedge_count
        )

    # -------------------------------------------------------------------------
    # Position Sizing
    # -------------------------------------------------------------------------

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        account_equity: float,
        symbol_info: dict
    ) -> float:
        """Calculate optimal position size based on risk parameters."""
        if stop_loss <= 0 or entry_price <= 0:
            return self.sizing_config.min_lot

        # Risk amount in account currency
        risk_amount = account_equity * (self.sizing_config.risk_per_trade_pct / 100)

        # Price distance to stop loss
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            return self.sizing_config.min_lot

        # Pip value calculation (simplified)
        tick_value = symbol_info.get("trade_tick_value", 1.0)
        tick_size = symbol_info.get("trade_tick_size", 0.00001)
        pip_value = tick_value / tick_size * 0.0001  # Approximate pip value

        # Position size in lots
        raw_lots = risk_amount / (price_diff / tick_size * tick_value)

        # Apply constraints
        lots = max(self.sizing_config.min_lot, min(self.sizing_config.max_lot, raw_lots))
        lots = round(lots / self.sizing_config.lot_step) * self.sizing_config.lot_step

        # Check symbol-specific limits
        symbol_positions = [p for p in self.positions.values() if p.symbol == symbol]
        if len(symbol_positions) >= self.sizing_config.max_positions_per_symbol:
            return 0

        # Check total risk
        current_risk = sum(
            abs(p.entry_price - p.stop_loss) * p.volume * pip_value
            for p in self.positions.values()
        )
        new_risk = price_diff / tick_size * tick_value * lots
        max_risk = account_equity * (self.sizing_config.max_total_risk_pct / 100)

        if current_risk + new_risk > max_risk:
            # Reduce size to fit within risk limits
            available_risk = max(0, max_risk - current_risk)
            lots = max(self.sizing_config.min_lot, available_risk / (price_diff / tick_size * tick_value))
            lots = round(lots / self.sizing_config.lot_step) * self.sizing_config.lot_step

        return lots

    # -------------------------------------------------------------------------
    # Hedging Logic
    # -------------------------------------------------------------------------

    def should_hedge(self, metrics: RiskMetrics, account_equity: float) -> list[dict]:
        """Determine if hedging is needed and return hedge orders."""
        if not self.hedge_config.enabled:
            return []

        hedges = []

        # Portfolio-level hedging
        if metrics.current_drawdown >= self.hedge_config.trigger_drawdown_pct:
            # Find largest losing position to hedge
            losing_positions = [
                p for p in self.positions.values()
                if p.profit < 0 and p.side not in (PositionSide.HEDGE_LONG, PositionSide.HEDGE_SHORT)
            ]

            if losing_positions:
                # Sort by loss amount
                losing_positions.sort(key=lambda p: p.profit)
                main_pos = losing_positions[0]

                hedge_volume = main_pos.volume * self.hedge_config.max_hedge_ratio
                hedge_volume = round(hedge_volume / self.sizing_config.lot_step) * self.sizing_config.lot_step
                hedge_volume = max(self.sizing_config.min_lot, min(hedge_volume, self.sizing_config.max_lot))

                if hedge_volume > 0:
                    # Determine hedge direction (opposite)
                    hedge_side = PositionSide.HEDGE_SHORT if main_pos.side == PositionSide.LONG else PositionSide.HEDGE_LONG
                    hedge_entry = main_pos.current_price
                    hedge_sl = self._calculate_hedge_sl(main_pos, hedge_side, hedge_entry)

                    hedges.append({
                        "symbol": main_pos.symbol,
                        "side": hedge_side,
                        "volume": hedge_volume,
                        "entry_price": hedge_entry,
                        "stop_loss": hedge_sl,
                        "take_profit": 0,  # No TP for hedges, managed manually
                        "hedge_for_ticket": main_pos.ticket,
                        "type": "drawdown_hedge"
                    })

        # Symbol-level correlation hedging (if enabled)
        if self.hedge_config.hedge_symbol_suffix:
            hedges.extend(self._correlation_hedges(metrics))

        return hedges

    def _calculate_hedge_sl(self, main_pos: Position, hedge_side: PositionSide, entry: float) -> float:
        """Calculate stop loss for hedge position."""
        distance_pips = self.hedge_config.min_hedge_distance_pips
        # Simplified: use ATR or fixed pips
        if hedge_side == PositionSide.HEDGE_LONG:
            return entry - distance_pips * 0.0001  # Below entry
        return entry + distance_pips * 0.0001  # Above entry

    def _correlation_hedges(self, metrics: RiskMetrics) -> list[dict]:
        """Create hedges on correlated symbols."""
        # Placeholder for correlated pair hedging
        # Would need correlation matrix and open positions on correlated pairs
        return []

    def manage_hedges(self, metrics: RiskMetrics) -> list[dict]:
        """Manage existing hedge positions - close if conditions met."""
        actions = []

        for pos in list(self.positions.values()):
            if pos.side not in (PositionSide.HEDGE_LONG, PositionSide.HEDGE_SHORT):
                continue

            main_pos = self.positions.get(pos.hedge_ticket) if pos.hedge_ticket else None
            if not main_pos:
                # Orphaned hedge - close it
                actions.append({"action": "close", "ticket": pos.ticket, "reason": "orphaned_hedge"})
                continue

            # Close hedge if main position is profitable and hedge at breakeven
            if self.hedge_config.close_hedge_at_be and main_pos.profit > 0:
                if (pos.side == PositionSide.HEDGE_SHORT and pos.current_price <= pos.entry_price) or \
                   (pos.side == PositionSide.HEDGE_LONG and pos.current_price >= pos.entry_price):
                    actions.append({"action": "close", "ticket": pos.ticket, "reason": "hedge_be"})

            # Close hedge if main position closed
            if main_pos.status == PositionStatus.CLOSED:
                actions.append({"action": "close", "ticket": pos.ticket, "reason": "main_closed"})

        return actions

    # -------------------------------------------------------------------------
    # Trailing Stops & Exit Management
    # -------------------------------------------------------------------------

    def update_trailing_stops(self, atr_multiplier: float = 2.0) -> list[dict]:
        """Update trailing stops for open positions."""
        actions = []

        for pos in self.positions.values():
            if pos.status != PositionStatus.OPEN:
                continue

            # Skip hedges - they have fixed stops
            if pos.side in (PositionSide.HEDGE_LONG, PositionSide.HEDGE_SHORT):
                continue

            # Calculate new trailing stop based on ATR
            # This would need ATR data per symbol
            # Simplified: trail by fixed % of profit
            if pos.profit > 0:
                trail_distance = pos.profit * 0.3  # Trail 30% of profit
                new_sl = pos.entry_price + (trail_distance if pos.side == PositionSide.LONG else -trail_distance)

                # Only move SL in favorable direction
                if pos.side == PositionSide.LONG and new_sl > pos.stop_loss:
                    actions.append({"action": "modify", "ticket": pos.ticket, "sl": new_sl})
                elif pos.side == PositionSide.SHORT and new_sl < pos.stop_loss:
                    actions.append({"action": "modify", "ticket": pos.ticket, "sl": new_sl})

        return actions

    def check_exit_conditions(
        self,
        metrics: RiskMetrics,
        confluence_result=None  # MultiTimeframeAnalyzer.ConfluenceResult
    ) -> list[dict]:
        """Check all exit conditions and return close/modify actions."""
        actions = []

        for pos in self.positions.values():
            if pos.status != PositionStatus.OPEN:
                continue

            # Hard stop loss / take profit (handled by MT5, but double-check)
            if pos.side in (PositionSide.LONG, PositionSide.HEDGE_LONG):
                if pos.current_price <= pos.stop_loss and pos.stop_loss > 0:
                    actions.append({"action": "close", "ticket": pos.ticket, "reason": "stop_loss"})
                elif pos.current_price >= pos.take_profit and pos.take_profit > 0:
                    actions.append({"action": "close", "ticket": pos.ticket, "reason": "take_profit"})
            else:
                if pos.current_price >= pos.stop_loss and pos.stop_loss > 0:
                    actions.append({"action": "close", "ticket": pos.ticket, "reason": "stop_loss"})
                elif pos.current_price <= pos.take_profit and pos.take_profit > 0:
                    actions.append({"action": "close", "ticket": pos.ticket, "reason": "take_profit"})

            # Portfolio risk exit
            if metrics.current_drawdown > self.sizing_config.max_total_risk_pct * 2:
                actions.append({"action": "close", "ticket": pos.ticket, "reason": "portfolio_risk"})

            # Daily loss limit
            if metrics.daily_pnl < -abs(self.daily_start_equity * 0.05):  # 5% daily loss
                actions.append({"action": "close", "ticket": pos.ticket, "reason": "daily_loss_limit"})

            # Confluence-based exit
            if confluence_result and confluence_result.recommendation == "hedge":
                if pos.side in (PositionSide.LONG, PositionSide.SHORT):
                    actions.append({"action": "hedge", "ticket": pos.ticket, "reason": "confluence_hedge"})

        # Add hedge management actions
        actions.extend(self.manage_hedges(metrics))

        return actions


def create_default_manager(mt5_bridge=None) -> PositionManager:
    """Create position manager with JARVIS Reborn defaults."""
    return PositionManager(
        hedge_config=HedgeConfig(
            enabled=True,
            trigger_drawdown_pct=5.0,
            max_hedge_ratio=0.5,
            min_hedge_distance_pips=50,
            close_hedge_at_be=True
        ),
        sizing_config=PositionSizingConfig(
            risk_per_trade_pct=1.0,
            max_risk_per_symbol_pct=3.0,
            max_total_risk_pct=5.0,
            max_positions=10,
            max_positions_per_symbol=3,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0
        ),
        mt5_bridge=mt5_bridge
    )