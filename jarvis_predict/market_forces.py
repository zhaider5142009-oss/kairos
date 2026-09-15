"""
Intermarket forces — captures the macro variables that drive almost every
asset: USD index, US 10Y yield, VIX, equity indices, gold/oil ratio.

Data source: yfinance (free, no API key, 15-min delayed quotes — fine for
non-HFT directional trading). All calls are cached and fail gracefully.

Each `Force` exposes:
    - name, symbol (yahoo ticker)
    - current value (latest close)
    - 5d / 20d percent change
    - regime classification (risk-on / risk-off / neutral, etc.)

The forces feed into the ensemble as auxiliary directional votes.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import JarvisConfig
from .symbols import get as get_instrument

LOGGER = logging.getLogger("jarvis.market_forces")

# Curated macro forces that drive asset prices.
DEFAULT_FORCES: list[dict] = [
    {"name": "DXY",        "yahoo": "DX-Y.NYB",  "kind": "dollar"},
    {"name": "US10Y",      "yahoo": "^TNX",      "kind": "yield"},
    {"name": "VIX",        "yahoo": "^VIX",      "kind": "volatility"},
    {"name": "SPX",        "yahoo": "^GSPC",     "kind": "equity"},
    {"name": "GOLD",       "yahoo": "GC=F",      "kind": "metals"},
    {"name": "WTI",        "yahoo": "CL=F",      "kind": "energy"},
    {"name": "COPPER",     "yahoo": "HG=F",      "kind": "metals"},
    {"name": "BTC",        "yahoo": "BTC-USD",   "kind": "crypto"},
]


@dataclass
class ForceSnapshot:
    name: str
    kind: str
    last: float
    change_pct_5d: float
    change_pct_20d: float
    regime: str                  # one-word descriptor: "risk_on" / "risk_off" / "hawkish" / "dovish" / ...
    direction_hint: int          # -1, 0, +1 — hint for bullish/bearish on equities/risk
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "last": round(self.last, 4),
            "change_5d_pct": round(self.change_pct_5d, 3),
            "change_20d_pct": round(self.change_pct_20d, 3),
            "regime": self.regime,
            "direction_hint": self.direction_hint,
            "timestamp": self.timestamp,
        }


class MarketForcesService:
    """Polls macro tickers via yfinance, classifies regime, exposes per-symbol
    direction adjustments."""

    def __init__(
        self,
        *,
        config: JarvisConfig,
        enabled: bool = True,
        forces: list[dict] | None = None,
        poll_seconds: float = 300.0,            # 5 min
        history_days: int = 90,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self.forces = list(forces or DEFAULT_FORCES)
        self.poll_seconds = max(60.0, float(poll_seconds))
        self.history_days = max(30, int(history_days))

        self._snapshots: dict[str, ForceSnapshot] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._yf = None  # lazy import
        self._last_poll: float = 0.0

    @classmethod
    def from_config(cls, config: JarvisConfig) -> "MarketForcesService":
        section = config.section("market_forces")
        return cls(
            config=config,
            enabled=bool(section.get("enabled", True)),
            forces=section.get("forces"),
            poll_seconds=section.get("poll_seconds", 300.0),
            history_days=section.get("history_days", 90),
        )

    # --- public --------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("Market forces disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-forces", daemon=True)
        self._thread.start()
        LOGGER.info("MarketForcesService started (poll=%.0fs, %d forces)",
                    self.poll_seconds, len(self.forces))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, ForceSnapshot]:
        with self._lock:
            return dict(self._snapshots)

    def correlation_matrix(self, symbols: Sequence[str]) -> np.ndarray:
        """Estimate N×N correlation of recent returns across `symbols` using
        whatever data we have in the cache plus the force tickers as proxies.

        Returns a square numpy array of shape (len(symbols), len(symbols)).
        Falls back to identity + sector-derived structure if data is sparse.
        """
        n = len(symbols)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        M = np.eye(n, dtype=np.float64)
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if i == j:
                    continue
                M[i, j] = _static_correlation(a, b)
        return M

    def force_direction_for(self, symbol: str) -> int:
        """Aggregate macro forces into a -1/0/+1 hint for `symbol`.

        Logic:
            - DXY up + (symbol inverse_to DXY) => bearish hint for symbol
            - VIX spike => risk-off => bearish hint for risk assets
            - US10Y up => bearish for gold, bullish for USDJPY
            - SPX up => bullish hint for correlated indices
        """
        ins = get_instrument(symbol)
        if ins is None:
            return 0
        score = 0
        count = 0

        dxy = self._snapshots.get("DXY")
        if dxy and "DXY" in ins.inverse_to or "USDX" in ins.inverse_to:
            score += -dxy.direction_hint
            count += 1
        if dxy and (ins.sector == "fx" and "USD" in symbol and not symbol.startswith("USD")):
            # USD-positive pair (EURUSD, GBPUSD...) — DXY up is bearish.
            score += -dxy.direction_hint
            count += 1

        us10y = self._snapshots.get("US10Y")
        if us10y:
            if ins.sector == "metals":
                score += -us10y.direction_hint  # yields up = gold down
                count += 1
            if symbol == "USDJPY":
                score += us10y.direction_hint
                count += 1

        vix = self._snapshots.get("VIX")
        if vix:
            if ins.sector in ("fx", "metals", "indices", "energy", "crypto"):
                if vix.regime == "risk_off":
                    if ins.sector == "metals" and symbol == "XAUUSD":
                        score += 1                # gold is the safe haven
                    elif ins.sector in ("indices", "crypto"):
                        score += -1               # risk-off hurts these
                    count += 1

        spx = self._snapshots.get("SPX")
        if spx and ins.sector == "indices":
            score += spx.direction_hint
            count += 1

        if count == 0:
            return 0
        avg = score / count
        if avg > 0.3:
            return 1
        if avg < -0.3:
            return -1
        return 0

    def regime(self) -> str:
        """Single-word overall regime string."""
        vix = self._snapshots.get("VIX")
        if vix and vix.regime == "risk_off":
            return "risk_off"
        spx = self._snapshots.get("SPX")
        if spx and spx.regime == "bull":
            return "risk_on"
        return "neutral"

    # --- internals -----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._poll_once()
        except Exception:
            LOGGER.exception("Forces initial poll failed")
        while not self._stop.is_set():
            self._stop.wait(timeout=self.poll_seconds)
            try:
                self._poll_once()
            except Exception:
                LOGGER.exception("Forces poll loop error")

    def _poll_once(self) -> None:
        if not self._ensure_yfinance():
            return
        import concurrent.futures
        results: dict[str, ForceSnapshot] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._fetch_one, f["name"], f["yahoo"], f["kind"]): f["name"]
                    for f in self.forces}
            for fut in concurrent.futures.as_completed(futs, timeout=45):
                try:
                    snap = fut.result()
                except Exception as e:
                    LOGGER.debug("Force fetch failed: %s", e)
                    continue
                if snap is not None:
                    results[snap.name] = snap
        if results:
            with self._lock:
                self._snapshots.update(results)
        self._last_poll = time.time()
        LOGGER.debug("forces poll complete: %d/%d tickers", len(results), len(self.forces))

    def _ensure_yfinance(self) -> bool:
        if self._yf is not None:
            return True
        try:
            import yfinance as yf  # type: ignore
            self._yf = yf
            return True
        except ImportError:
            LOGGER.warning("yfinance not installed; market forces disabled")
            self.enabled = False
            return False

    def _fetch_one(self, name: str, yahoo: str, kind: str) -> ForceSnapshot | None:
        assert self._yf is not None
        ticker = self._yf.Ticker(yahoo)
        df = ticker.history(period=f"{self.history_days}d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 6:
            return None
        closes = df["Close"].astype(float)
        last = float(closes.iloc[-1])
        prev_5 = float(closes.iloc[-6]) if len(closes) >= 6 else last
        prev_20 = float(closes.iloc[-21]) if len(closes) >= 21 else last
        chg5 = (last / prev_5 - 1.0) * 100.0 if prev_5 else 0.0
        chg20 = (last / prev_20 - 1.0) * 100.0 if prev_20 else 0.0

        regime, direction_hint = _classify_regime(kind, chg5, chg20, last)
        return ForceSnapshot(
            name=name, kind=kind, last=last,
            change_pct_5d=chg5, change_pct_20d=chg20,
            regime=regime, direction_hint=direction_hint,
        )


# --- pure helpers ------------------------------------------------------------

def _classify_regime(kind: str, chg5: float, chg20: float, last: float) -> tuple[str, int]:
    if kind == "dollar":
        if chg5 > 1.5:
            return ("hawkish_dollar", 1)
        if chg5 < -1.5:
            return ("dovish_dollar", -1)
        return ("neutral", 0)
    if kind == "yield":
        if chg20 > 5:
            return ("hawkish_fed", 1)
        if chg20 < -5:
            return ("dovish_fed", -1)
        return ("neutral", 0)
    if kind == "volatility":
        if last > 30:
            return ("risk_off", -1)
        if last > 20:
            return ("elevated", -1)
        if last < 13:
            return ("complacent", 1)
        return ("neutral", 0)
    if kind == "equity":
        if chg5 > 2:
            return ("bull", 1)
        if chg5 < -2:
            return ("bear", -1)
        return ("neutral", 0)
    # Metals / energy / crypto — generic trend.
    if chg20 > 5:
        return ("uptrend", 1)
    if chg20 < -5:
        return ("downtrend", -1)
    return ("neutral", 0)


# Hard-coded intermarket correlation priors (in [-1, +1]). Used when we don't
# have enough joint price history to estimate a correlation from data.
_STATIC_PRIORS: dict[tuple[str, str], float] = {
    ("XAUUSD", "DXY"):   -0.75,
    ("XAUUSD", "US10Y"): -0.55,
    ("XAUUSD", "VIX"):    0.30,
    ("XAUUSD", "SPX"):   -0.10,
    ("XAGUSD", "XAUUSD"): 0.85,
    ("USOIL", "USDCAD"): -0.65,
    ("USOIL", "DXY"):    -0.55,
    ("EURUSD", "DXY"):   -0.90,
    ("GBPUSD", "DXY"):   -0.85,
    ("USDJPY", "US10Y"):  0.55,
    ("AUDUSD", "XAUUSD"): 0.55,
    ("NAS100", "SPX"):    0.92,
    ("US500", "SPX"):     0.98,
    ("BTCUSD", "NAS100"): 0.45,
    ("HG", "XAUUSD"):     0.65,
}


def _static_correlation(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return _STATIC_PRIORS.get((a, b), _STATIC_PRIORS.get((b, a), 0.0))


__all__ = ["MarketForcesService", "ForceSnapshot", "DEFAULT_FORCES"]
