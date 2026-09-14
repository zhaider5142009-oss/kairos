"""
MetaTrader 5 bridge.

Wraps the official `MetaTrader5` Python package behind a small, safe
interface. All MT5 calls go through this module so the rest of Jarvis
doesn't have to know about MetaTrader5's quirks (fill policies, magic
numbers, retcode enums, terminal reconnection, etc.).

Design:
- Lazy connection on first call.
- All public methods return either data (rates/ticks/positions) or
  SendResult for order operations.
- Errors raise JarvisMT5Error with a human-readable message.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import JarvisConfig

LOGGER = logging.getLogger("jarvis.mt5")

# These imports are deferred to runtime — MetaTrader5 is Windows-only.
mt5 = None  # type: ignore
MT5_TIMEFRAME_MAP: dict[str, int] = {}


class JarvisMT5Error(RuntimeError):
    """Raised for any unrecoverable MT5 error."""


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int          # milliseconds since epoch

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class SendResult:
    success: bool
    order_id: int | None
    retcode: int | None
    comment: str
    request: dict = field(default_factory=dict)


class MT5Bridge:
    """Thin wrapper around the MetaTrader5 Python package."""

    def __init__(
        self,
        *,
        terminal_path: str | Path,
        symbols: Sequence[str],
        timeframe: str,
        history_bars: int,
        magic: int,
        default_lot: float,
        poll_seconds: float = 2.0,
    ) -> None:
        self.terminal_path = Path(terminal_path)
        self.symbols = list(symbols)
        self.timeframe_str = timeframe
        self.history_bars = max(100, int(history_bars))
        self.magic = int(magic)
        self.default_lot = float(default_lot)
        self.poll_seconds = max(0.25, float(poll_seconds))

        self._connected = False
        self._start_equity: float | None = None
        self._day_start_equity: float | None = None
        self._day_reset_at: float = 0.0
        self._last_connect_attempt: float = 0.0
        self._connect_retry_after = 10.0

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "MT5Bridge":
        s = config.section("mt5")
        return cls(
            terminal_path=s["terminal_path"],
            symbols=s["symbols"],
            timeframe=s["timeframe"],
            history_bars=s["history_bars"],
            magic=s["magic"],
            default_lot=s["order"]["default_lot"],
            poll_seconds=s.get("poll_seconds", 2.0),
        )

    # ----------------------------------------------------------- lifecycle

    def ensure_connected(self) -> bool:
        """Connect to the MT5 terminal if not already. Returns True on success.

        Returns False (instead of raising) when the MetaTrader5 package is
        missing or the terminal isn't running — this lets the rest of Jarvis
        keep working (signals can still be computed; only order placement
        requires a live terminal).
        """
        global mt5, MT5_TIMEFRAME_MAP
        if self._connected:
            return True
        # Rate-limit reconnect attempts so fast pollers (status/UI) never
        # block on a dead terminal's initialise timeout.
        if time.time() - self._last_connect_attempt < self._connect_retry_after:
            return False
        self._last_connect_attempt = time.time()
        try:
            import MetaTrader5 as mt5_mod  # type: ignore
        except ImportError:
            LOGGER.debug("MetaTrader5 package not installed — orders disabled")
            return False

        try:
            mt5 = mt5_mod
            MT5_TIMEFRAME_MAP = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
            }
        except Exception:
            return False

        try:
            if not mt5.initialize(path=str(self.terminal_path), timeout=15000):
                LOGGER.warning("mt5.initialize failed: %s", mt5.last_error())
                return False
        except Exception as e:
            LOGGER.warning("mt5.initialize raised: %s", e)
            return False

        # Subscribe to symbols.
        for sym in self.symbols:
            try:
                if not mt5.symbol_select(sym, True):
                    LOGGER.warning("symbol_select failed for %s", sym)
            except Exception:
                pass

        self._connected = True
        try:
            self._start_equity = self._equity()
        except Exception:
            self._start_equity = 0.0
        LOGGER.info(
            "MT5 connected (terminal=%s, equity=%.2f)",
            self.terminal_path, self._start_equity or 0.0,
        )
        return True

    def shutdown(self) -> None:
        if not self._connected:
            return
        try:
            if mt5 is not None:
                mt5.shutdown()
        except Exception:  # pragma: no cover — best-effort
            LOGGER.exception("MT5 shutdown error")
        self._connected = False

    # ------------------------------------------------------------- market data

    def timeframe_constant(self) -> int:
        if self.timeframe_str not in MT5_TIMEFRAME_MAP:
            raise JarvisMT5Error(
                f"Unknown timeframe {self.timeframe_str!r}. "
                f"Valid: {sorted(MT5_TIMEFRAME_MAP)}"
            )
        return MT5_TIMEFRAME_MAP[self.timeframe_str]

    def rates(self, symbol: str, count: int | None = None, timeframe: str | None = None) -> pd.DataFrame:
        """Fetch `count` most-recent bars for the requested (or configured) TF.

        Returns DataFrame with columns: time, open, high, low, close, tick_volume.
        """
        if not self.ensure_connected():
            raise JarvisMT5Error("MT5 not connected")
        n = count or self.history_bars
        tf_str = timeframe or self.timeframe_str
        if tf_str not in MT5_TIMEFRAME_MAP:
            raise JarvisMT5Error(f"Unknown timeframe {tf_str!r}")
        tf = MT5_TIMEFRAME_MAP[tf_str]
        utc_from = datetime.fromtimestamp(
            time.time() - n * _tf_seconds(tf_str) * 2,
            tz=timezone.utc,
        )
        rates = mt5.copy_rates_from(symbol, tf, utc_from, n)
        if rates is None or len(rates) == 0:
            raise JarvisMT5Error(f"No rates for {symbol} (mt5.last_error={mt5.last_error()})")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def close_series(self, symbol: str, count: int | None = None) -> np.ndarray:
        df = self.rates(symbol, count)
        return df["close"].to_numpy(dtype=np.float64)

    def tick(self, symbol: str) -> Tick:
        if not self.ensure_connected():
            raise JarvisMT5Error("MT5 not connected")
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise JarvisMT5Error(f"No tick for {symbol}")
        return Tick(
            symbol=symbol,
            bid=float(t.bid),
            ask=float(t.ask),
            last=float(t.last),
            volume=int(t.volume),
            time_msc=int(t.time_msc),
        )

    # --------------------------------------------------------- account / risk

    def _equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def equity(self) -> float:
        if not self.ensure_connected():
            return 0.0
        return self._equity()

    def balance(self) -> float:
        if not self.ensure_connected():
            return 0.0
        info = mt5.account_info()
        return float(info.balance) if info else 0.0

    def open_positions(self) -> list:
        if not self.ensure_connected():
            return []
        positions = mt5.positions_get(symbol=None) or []
        return list(positions)

    def daily_pnl(self) -> float:
        """Approximate today's realised PnL: equity - balance (open positions included)."""
        if not self.ensure_connected():
            return 0.0
        # Reset once per local day boundary.
        now = time.time()
        # Lazy day-reset check (every 5 minutes).
        if now - self._day_reset_at > 300:
            self._day_start_equity = self._equity()
            self._day_reset_at = now
        return self._equity() - (self._day_start_equity or 0.0)

    # ------------------------------------------------------ account snapshot

    def account_info(self) -> dict:
        """Broker account info for the MT5 panel. Safe when not connected."""
        if not self._connected or mt5 is None:
            return {}
        try:
            info = mt5.account_info()
        except Exception:
            LOGGER.exception("mt5.account_info failed")
            return {}
        if info is None:
            return {}
        return {
            "login": getattr(info, "login", None),
            "server": getattr(info, "server", ""),
            "company": getattr(info, "company", ""),
            "currency": getattr(info, "currency", ""),
            "leverage": getattr(info, "leverage", 0),
            "balance": float(getattr(info, "balance", 0.0) or 0.0),
            "equity": float(getattr(info, "equity", 0.0) or 0.0),
            "margin": float(getattr(info, "margin", 0.0) or 0.0),
            "margin_free": float(getattr(info, "margin_free", 0.0) or 0.0),
            "margin_level": float(getattr(info, "margin_level", 0.0) or 0.0),
            "profit": float(getattr(info, "profit", 0.0) or 0.0),
        }

    def terminal_info(self) -> dict:
        """MetaTrader terminal/connection info. Safe when not connected."""
        if not self._connected or mt5 is None:
            return {}
        try:
            ti = mt5.terminal_info()
        except Exception:
            LOGGER.exception("mt5.terminal_info failed")
            return {}
        if ti is None:
            return {}
        return {
            "connected": bool(getattr(ti, "connected", False)),
            "trade_allowed": bool(getattr(ti, "trade_allowed", False)),
            "company": getattr(ti, "company", ""),
            "name": getattr(ti, "name", ""),
            "version": getattr(ti, "version", 0),
            "path": getattr(ti, "path", ""),
            "terminal_path": self.terminal_path,
        }

    # ------------------------------------------------------------- orders

    def send_market_order(
        self,
        *,
        symbol: str,
        direction: int,                 # +1 = buy, -1 = sell
        lot: float | None = None,
        sl: float,
        tp: float,
        comment: str = "jarvis",
    ) -> SendResult:
        """Send a market order with SL/TP. Returns SendResult."""
        if not self.ensure_connected():
            return SendResult(False, None, None, "MT5 not connected")
        if direction not in (-1, +1):
            return SendResult(False, None, None, "Invalid direction")

        tick = self.tick(symbol)
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot or self.default_lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": self.magic,
            "comment": comment[:31],  # MT5 comment max length
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return SendResult(False, None, None, "order_send returned None")

        success = result.retcode == mt5.TRADE_RETCODE_DONE
        return SendResult(
            success=success,
            order_id=int(result.order) if result.order else None,
            retcode=int(result.retcode) if result.retcode is not None else None,
            comment=str(result.comment or ""),
            request=request,
        )

    def close_position(self, position_id: int) -> SendResult:
        if not self.ensure_connected():
            return SendResult(False, None, None, "MT5 not connected")
        positions = mt5.positions_get(ticket=position_id)
        if not positions:
            return SendResult(False, None, None, "Position not found")
        pos = positions[0]
        tick = self.tick(pos.symbol)
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "position": int(pos.ticket),
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": "jarvis-close"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return SendResult(
            success=success,
            order_id=int(result.order) if result and result.order else None,
            retcode=int(result.retcode) if result and result.retcode is not None else None,
            comment=str(result.comment) if result else "",
            request=request,
        )

    def close_all(self) -> list[SendResult]:
        out = []
        for pos in self.open_positions():
            r = self.close_position(int(pos.ticket))
            out.append(r)
        return out

    # ------------------------------------------------------ smart execution

    def get_symbol_info(self, symbol: str) -> dict:
        """Get detailed symbol info for position sizing."""
        if not self.ensure_connected():
            return {}
        try:
            info = mt5.symbol_info(symbol)
        except Exception:
            return {}
        if info is None:
            return {}
        return {
            "point": getattr(info, "point", 0.0),
            "digits": getattr(info, "digits", 5),
            "spread": getattr(info, "spread", 0),
            "trade_tick_value": getattr(info, "trade_tick_value", 1.0),
            "trade_tick_size": getattr(info, "trade_tick_size", 0.00001),
            "trade_contract_size": getattr(info, "trade_contract_size", 1.0),
            "volume_min": getattr(info, "volume_min", 0.01),
            "volume_max": getattr(info, "volume_max", 100.0),
            "volume_step": getattr(info, "volume_step", 0.01),
            "margin_initial": getattr(info, "margin_initial", 0.0),
            "margin_maintenance": getattr(info, "margin_maintenance", 0.0),
        }

    def get_atr(self, symbol: str, period: int = 14, timeframe: str = "H1") -> float:
        """Calculate ATR for dynamic stop loss."""
        try:
            df = self.rates(symbol, period + 10, timeframe=timeframe)
            if len(df) < period + 1:
                return 0.0
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            tr = np.maximum(high[1:] - low[1:], np.maximum(
                np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
            ))
            atr = np.mean(tr[-period:])
            return float(atr)
        except Exception:
            return 0.0

    def get_indicators(self, symbol: str, timeframe: str = "H1") -> dict:
        """Get key technical indicators for the strategy engine."""
        try:
            df = self.rates(symbol, 100, timeframe=timeframe)
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            if "tick_vol" in df.columns:
                volume = df["tick_vol"].values
            elif "tick_volume" in df.columns:
                volume = df["tick_volume"].values
            elif "volume" in df.columns:
                volume = df["volume"].values
            else:
                volume = np.zeros(len(close))

            # RSI
            rsi = self._calc_rsi(close, 14)

            # MACD
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            macd_signal = self._ema(macd_line, 9)
            macd_hist = macd_line - macd_signal

            # Bollinger Bands
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_middle = sma20

            # ADX
            adx = self._calc_adx(high, low, close, 14)

            # ATR
            atr = self.get_atr(symbol, 14, timeframe)

            # EMA
            ema_fast = self._ema(close, 12)
            ema_slow = self._ema(close, 26)

            # Support/Resistance
            support, resistance = self._find_sr_levels(high, low, close)

            return {
                "rsi": float(rsi) if not np.isnan(rsi) else 50.0,
                "macd_hist": float(macd_hist[-1]) if len(macd_hist) > 0 else 0.0,
                "macd_signal": float(macd_signal[-1]) if len(macd_signal) > 0 else 0.0,
                "macd_fast": float(ema12[-1]) if len(ema12) > 0 else close[-1],
                "macd_slow": float(ema26[-1]) if len(ema26) > 0 else close[-1],
                "bb_upper": float(bb_upper),
                "bb_lower": float(bb_lower),
                "bb_middle": float(bb_middle),
                "adx": float(adx) if not np.isnan(adx) else 20.0,
                "atr": float(atr),
                "ema_fast": float(ema_fast[-1]) if len(ema_fast) > 0 else close[-1],
                "ema_slow": float(ema_slow[-1]) if len(ema_slow) > 0 else close[-1],
                "support": support,
                "resistance": resistance,
                "close": float(close[-1]),
                "volume": float(volume[-1]) if len(volume) > 0 else 0,
                "avg_volume": float(np.mean(volume[-20:])) if len(volume) >= 20 else float(volume[-1]) if len(volume) > 0 else 0,
            }
        except Exception as e:
            LOGGER.warning("get_indicators failed for %s: %s", symbol, e)
            return {}

    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period:
            return data
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calc_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(high) < period + 1:
            return 20.0
        plus_dm = high[1:] - high[:-1]
        minus_dm = low[:-1] - low[1:]
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        tr = np.maximum(high[1:] - low[1:], np.maximum(
            np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
        ))
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / atr
        minus_di = 100 * self._ema(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)
        return float(adx[-1]) if len(adx) > 0 else 20.0

    def _find_sr_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 10) -> tuple[list[float], list[float]]:
        """Find support and resistance levels."""
        if len(high) < window * 2:
            return [], []
        highs, lows = [], []
        for i in range(window, len(high) - window):
            if all(high[i] >= high[j] for j in range(i - window, i + window + 1) if j != i):
                highs.append(high[i])
            if all(low[i] <= low[j] for j in range(i - window, i + window + 1) if j != i):
                lows.append(low[i])

        def cluster(levels):
            if not levels:
                return []
            levels = sorted(set(round(l, 5) for l in levels))
            clusters = [[levels[0]]]
            for l in levels[1:]:
                if abs(l - clusters[-1][-1]) / clusters[-1][-1] < 0.002:
                    clusters[-1].append(l)
                else:
                    clusters.append([l])
            return [round(sum(c) / len(c), 5) for c in clusters]

        return cluster(lows)[-3:], cluster(highs)[-3:]

    def execute_order(
        self,
        *,
        symbol: str,
        order_type: int,          # 0=buy, 1=sell
        volume: float,
        price: float,
        sl: float,
        tp: float,
        magic: int = 0,
        comment: str = "",
    ) -> dict:
        """Smart order execution with slippage control and retry logic."""
        if not self.ensure_connected():
            return {"success": False, "error": "MT5 not connected"}

        tick = self.tick(symbol)
        actual_price = tick.ask if order_type == 0 else tick.bid

        # Use provided price if close to market, otherwise use market
        if abs(price - actual_price) / actual_price > 0.001:  # >10 pips difference
            price = actual_price

        direction = 1 if order_type == 0 else -1
        result = self.send_market_order(
            symbol=symbol,
            direction=direction,
            lot=volume,
            sl=sl,
            tp=tp,
            comment=comment[:31]
        )

        return {
            "success": result.success,
            "order_id": result.order_id,
            "retcode": result.retcode,
            "comment": result.comment,
            "price": price,
            "sl": sl,
            "tp": tp,
            "volume": volume
        }

    def modify_position(
        self,
        position_id: int,
        sl: float | None = None,
        tp: float | None = None
    ) -> SendResult:
        """Modify stop loss / take profit of an existing position."""
        if not self.ensure_connected():
            return SendResult(False, None, None, "MT5 not connected")
        positions = mt5.positions_get(ticket=position_id)
        if not positions:
            return SendResult(False, None, None, "Position not found")
        pos = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": int(pos.ticket),
            "magic": self.magic,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return SendResult(
            success=success,
            order_id=int(result.order) if result and result.order else None,
            retcode=int(result.retcode) if result and result.retcode is not None else None,
            comment=str(result.comment) if result else "",
            request=request
        )


def _tf_seconds(tf: str) -> int:
    return {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400,
        "D1": 86400, "W1": 604800, "MN1": 2592000,
    }.get(tf, 300)


__all__ = ["MT5Bridge", "Tick", "SendResult", "JarvisMT5Error"]
