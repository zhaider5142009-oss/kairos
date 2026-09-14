"""
Instrument universe — central registry of every symbol Jarvis trades or watches.

Each entry carries sector, exchange, typical session hours (UTC),
average daily volatility, and a list of cross-asset correlations that the
intermarket layer uses for entanglement-inspired coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    sector: str           # fx | metals | energy | grains | softs | indices | crypto | rates
    exchange: str         # MT5 broker-specific (e.g. JSGCL)
    point: float          # smallest price increment
    contract_size: float  # lots × contract_size = units of underlying
    avg_daily_range_pct: float  # rough daily volatility for sizing
    sessions: tuple[str, ...] = ("24/5",)  # "asia" | "london" | "ny"
    correlated_to: tuple[str, ...] = ()    # symbols it tends to move with (or against — encoded in correlation matrix)
    inverse_to: tuple[str, ...] = ()       # symbols it tends to move against
    notes: str = ""


# ---- FX majors + crosses (TradeCast/JS Global Capital symbols) ----
_FX_MAJORS = [
    Instrument("EURUSD", "Euro vs US Dollar", "fx", "MT5", 0.00001, 100000, 0.6,
               sessions=("asia", "london", "ny"),
               inverse_to=("USDX",),
               notes="Most liquid pair; tightest spreads."),
    Instrument("GBPUSD", "British Pound vs US Dollar", "fx", "MT5", 0.00001, 100000, 0.8,
               sessions=("london", "ny"),
               inverse_to=("USDX",)),
    Instrument("USDJPY", "US Dollar vs Japanese Yen", "fx", "MT5", 0.001, 100000, 0.6,
               sessions=("asia", "london", "ny"),
               correlated_to=("US10Y",)),
    Instrument("AUDUSD", "Australian Dollar vs US Dollar", "fx", "MT5", 0.00001, 100000, 0.8,
               sessions=("asia", "london"),
               correlated_to=("XAUUSD", "HG", "IRON"),  # commodity currencies
               inverse_to=("USDX",)),
    Instrument("USDCAD", "US Dollar vs Canadian Dollar", "fx", "MT5", 0.00001, 100000, 0.6,
               sessions=("london", "ny"),
               correlated_to=("USOIL",),
               inverse_to=("USDX",)),
    Instrument("USDCHF", "US Dollar vs Swiss Franc", "fx", "MT5", 0.00001, 100000, 0.6,
               sessions=("london", "ny"),
               inverse_to=("USDX",)),
    Instrument("NZDUSD", "New Zealand Dollar vs US Dollar", "fx", "MT5", 0.00001, 100000, 0.8,
               sessions=("asia", "london"),
               correlated_to=("AUDUSD",)),
]

# ---- Metals ----
_METALS = [
    Instrument("XAUUSD", "Gold vs US Dollar", "metals", "MT5", 0.01, 100, 1.2,
               sessions=("asia", "london", "ny"),
               inverse_to=("USDX", "US10Y", "DXY"),
               correlated_to=("XAGUSD",),
               notes="Safe-haven; inverse to real yields."),
    Instrument("XAGUSD", "Silver vs US Dollar", "metals", "MT5", 0.01, 5000, 2.2,
               sessions=("asia", "london", "ny"),
               correlated_to=("XAUUSD", "HG"),
               inverse_to=("USDX",)),
    Instrument("XPTUSD", "Platinum vs US Dollar", "metals", "MT5", 0.01, 100, 1.6,
               correlated_to=("XAUUSD", "XPDUSD")),
    Instrument("XPDUSD", "Palladium vs US Dollar", "metals", "MT5", 0.01, 100, 2.5,
               correlated_to=("XPTUSD")),
]

# ---- Energy ----
_ENERGY = [
    Instrument("USOIL", "WTI Crude Oil", "energy", "MT5", 0.01, 1000, 2.5,
               sessions=("london", "ny"),
               correlated_to=("USDCAD", "CAD"),
               inverse_to=("USDX",),
               notes="OPEC+ headlines dominate; inventories matter weekly."),
    Instrument("UKOIL", "Brent Crude Oil", "energy", "MT5", 0.01, 1000, 2.4,
               correlated_to=("USOIL",)),
    Instrument("NGAS", "Natural Gas", "energy", "MT5", 0.001, 10000, 3.5,
               correlated_to=("USOIL",),
               notes="Weather-driven; highly seasonal."),
]

# ---- Grains & softs ----
_AGRICULTURE = [
    Instrument("ZC", "Corn futures", "grains", "MT5", 0.25, 5000, 1.6,
               sessions=("ny",)),
    Instrument("ZW", "Wheat futures", "grains", "MT5", 0.25, 5000, 1.9,
               sessions=("ny",)),
    Instrument("ZS", "Soybean futures", "grains", "MT5", 0.25, 5000, 1.5,
               sessions=("ny",)),
    Instrument("KC", "Coffee futures", "softs", "MT5", 0.05, 37500, 2.4,
               sessions=("ny",)),
    Instrument("SB", "Sugar futures", "softs", "MT5", 0.01, 112000, 2.0,
               sessions=("ny",)),
    Instrument("CT", "Cotton futures", "softs", "MT5", 0.01, 50000, 1.9,
               sessions=("ny",)),
    Instrument("HG", "Copper futures", "metals", "MT5", 0.0005, 25000, 1.5,
               sessions=("london", "ny"),
               correlated_to=("XAUUSD", "XAGUSD"),
               notes="Dr. Copper — growth proxy."),
]

# ---- Indices (cash CFDs on MT5) ----
_INDICES = [
    Instrument("US500", "S&P 500", "indices", "MT5", 0.01, 1, 1.2,
               sessions=("ny",),
               correlated_to=("US30", "NAS100")),
    Instrument("US30", "Dow Jones 30", "indices", "MT5", 0.01, 1, 1.2,
               sessions=("ny",)),
    Instrument("NAS100", "Nasdaq 100", "indices", "MT5", 0.01, 1, 1.7,
               sessions=("ny",),
               correlated_to=("US500",)),
    Instrument("GER40", "DAX 40", "indices", "MT5", 0.01, 1, 1.3,
               sessions=("london", "ny")),
    Instrument("UK100", "FTSE 100", "indices", "MT5", 0.01, 1, 1.0,
               sessions=("london",)),
    Instrument("JP225", "Nikkei 225", "indices", "MT5", 0.01, 1, 1.3,
               sessions=("asia", "ny")),
]

# ---- Crypto (if broker offers) ----
_CRYPTO = [
    Instrument("BTCUSD", "Bitcoin vs US Dollar", "crypto", "MT5", 0.01, 1, 4.5,
               sessions=("24/7",)),
    Instrument("ETHUSD", "Ethereum vs US Dollar", "crypto", "MT5", 0.01, 1, 5.0,
               correlated_to=("BTCUSD",)),
]

# ---- Macro/rates (used by intermarket layer via Yahoo, not MT5) ----
_MACRO = [
    Instrument("US10Y", "US 10-Year Treasury Yield", "rates", "yahoo", 0.001, 1, 0.0,
               sessions=("ny",),
               notes="Real-yield proxy. Inverse to gold."),
    Instrument("DXY", "US Dollar Index", "rates", "yahoo", 0.001, 1, 0.0,
               notes="Inverse to most FX majors and metals."),
    Instrument("VIX", "CBOE Volatility Index", "rates", "yahoo", 0.01, 1, 0.0,
               notes="Risk-off gauge; >30 = stress."),
    Instrument("SPX", "S&P 500 Index", "rates", "yahoo", 0.01, 1, 0.0,
               correlated_to=("US500", "NAS100")),
]


# ---- Registry ----

_REGISTRY: dict[str, Instrument] = {}


def _add(*groups: Sequence[Instrument]) -> None:
    for grp in groups:
        for ins in grp:
            _REGISTRY[ins.symbol] = ins


_add(_FX_MAJORS, _METALS, _ENERGY, _AGRICULTURE, _INDICES, _CRYPTO, _MACRO)


def get(symbol: str) -> Instrument | None:
    return _REGISTRY.get(symbol)


def all_symbols() -> list[str]:
    return sorted(_REGISTRY.keys())


def by_sector(sector: str) -> list[Instrument]:
    return [i for i in _REGISTRY.values() if i.sector == sector]


def tradable() -> list[Instrument]:
    """Instruments that can actually be traded via MT5 (excludes macro proxies)."""
    return [i for i in _REGISTRY.values() if i.exchange == "MT5"]


def universe() -> dict[str, dict]:
    """Full universe as plain dicts (for HTTP responses / logs)."""
    return {
        sym: {
            "name": i.name,
            "sector": i.sector,
            "exchange": i.exchange,
            "point": i.point,
            "contract_size": i.contract_size,
            "avg_daily_range_pct": i.avg_daily_range_pct,
            "sessions": list(i.sessions),
            "correlated_to": list(i.correlated_to),
            "inverse_to": list(i.inverse_to),
        }
        for sym, i in sorted(_REGISTRY.items())
    }


__all__ = [
    "Instrument",
    "get",
    "all_symbols",
    "by_sector",
    "tradable",
    "universe",
]
