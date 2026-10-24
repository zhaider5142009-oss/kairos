"""Instrument universe registry.

Every instrument carries resolution rules per data source plus fundamental
metadata used by the factor engine (sector, currency, market).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Instrument:
    key: str                          # canonical KAIROS key, e.g. "OGDC", "GOLD"
    name: str
    asset_class: str                  # equity | commodity | index | fx | rate | crypto
    sector: str
    currency: str
    market: str                       # PSX | GLOBAL
    yahoo: str | None = None          # Yahoo Finance symbol
    tv: str | None = None             # TradingView symbol
    psx_ticker: str | None = None     # ticker on the local PSX API
    binance: str | None = None        # Binance spot pair, e.g. "BTCUSDT"
    tags: tuple[str, ...] = field(default_factory=tuple)


def _psx(key: str, name: str, sector: str, tags: tuple[str, ...] = (),
         yahoo: str | None = None, ticker: str | None = None) -> Instrument:
    # `key` is KAIROS' stable handle (regime baskets, news attribution, saved
    # state); `ticker` is what the exchange and the feeds actually answer to.
    # They are not always the same — PSX renamed ENGRO to ENGROH and the board
    # spells Bank of Punjab BOP and Nestle NESTLE. Deriving the venue symbols
    # from the key silently pointed every fetch at a symbol that does not exist.
    board = ticker or key
    # Yahoo carries most PSX names as "<TICKER>.KA" (probed live 2026-10-16);
    # names without Yahoo coverage fall back to local API / store accumulation
    return Instrument(
        key=key, name=name, asset_class="equity", sector=sector,
        currency="PKR", market="PSX", tv=f"PSX:{board}", psx_ticker=board,
        yahoo=yahoo or f"{board}.KA", tags=tags,
    )


def _psx_no_yahoo(key: str, name: str, sector: str, tags: tuple[str, ...] = (),
                  ticker: str | None = None) -> Instrument:
    board = ticker or key
    return Instrument(
        key=key, name=name, asset_class="equity", sector=sector,
        currency="PKR", market="PSX", tv=f"PSX:{board}", psx_ticker=board,
        tags=tags,
    )


def _yf(key: str, name: str, asset_class: str, sector: str, yahoo: str,
        currency: str = "USD", tags: tuple[str, ...] = ()) -> Instrument:
    return Instrument(
        key=key, name=name, asset_class=asset_class, sector=sector,
        currency=currency, market="GLOBAL", yahoo=yahoo, tags=tags,
    )


PSX_BLUE_CHIPS: list[Instrument] = [
    _psx("OGDC", "Oil & Gas Development Co.", "Energy", ("index-heavy", "soe")),
    _psx("PPL", "Pakistan Petroleum Ltd.", "Energy", ("index-heavy", "soe")),
    _psx("POL", "Pakistan Oilfields Ltd.", "Energy", ("oil-proxy",)),
    _psx("MARI", "Mari Energies Ltd.", "Energy", ("index-heavy",)),
    _psx("PSO", "Pakistan State Oil", "Energy", ("oil-proxy", "soe")),
    _psx("ATRL", "Attock Refinery Ltd.", "Energy", ("oil-proxy",)),
    _psx("SNGP", "Sui Northern Gas", "Utilities", ("gas-proxy", "soe")),
    _psx("SSGC", "Sui Southern Gas", "Utilities", ("gas-proxy", "soe")),
    _psx_no_yahoo("HUBC", "Hub Power Co.", "Utilities", ("index-heavy",)),
    _psx("KAPCO", "Kot Addu Power", "Utilities", ()),
    _psx("MLCF", "MLCF Global (Mustafaabad Cotton)", "Textile", ()),
    _psx("LUCK", "Lucky Cement", "Cement", ("index-heavy", "cyclical")),
    _psx("DGKC", "D.G. Khan Cement", "Cement", ("cyclical",)),
    _psx("FCCL", "Fauji Cement", "Cement", ("cyclical",)),
    _psx("CHCC", "Cherat Cement", "Cement", ("cyclical",)),
    _psx("PIOC", "Pioneer Cement", "Cement", ("cyclical",)),
    _psx_no_yahoo("ENGRO", "Engro Holdings Ltd.", "Conglomerate",
                  ("index-heavy",), ticker="ENGROH"),
    _psx("EFERT", "Engro Fertilizers", "Fertilizer", ("agri-proxy",)),
    _psx("FFC", "Fauji Fertilizer Co.", "Fertilizer", ("index-heavy", "agri-proxy")),
    _psx("FATIMA", "Fatima Fertilizer", "Fertilizer", ("agri-proxy",)),
    _psx_no_yahoo("FFBL", "Fauji Fertilizer Bin Qasim", "Fertilizer", ("agri-proxy",)),
    _psx("HBL", "Habib Bank Ltd.", "Banking", ("index-heavy", "bank")),
    _psx("UBL", "United Bank Ltd.", "Banking", ("index-heavy", "bank")),
    _psx("MCB", "MCB Bank", "Banking", ("index-heavy", "bank")),
    _psx("ABL", "Allied Bank Ltd.", "Banking", ("bank",)),
    _psx("MEBL", "Meezan Bank", "Banking", ("index-heavy", "bank", "islamic")),
    _psx("BAHL", "Bank AL Habib", "Banking", ("bank",)),
    _psx("NBP", "National Bank of Pakistan", "Banking", ("bank", "soe")),
    _psx("BANK", "Bank of Punjab", "Banking", ("bank", "soe"), ticker="BOP"),
    _psx("AKBL", "Askari Bank", "Banking", ("bank",)),
    _psx("SYS", "Systems Ltd.", "Technology", ("tech",)),
    _psx("TRG", "TRG Pakistan", "Technology", ("tech",)),
    _psx("NETSOL", "NetSol Technologies", "Technology", ("tech",)),
    _psx("PTC", "Pakistan Telecommunication", "Telecom", ("soe",)),
    _psx("INDU", "Indus Motor Co.", "Auto", ("cyclical",)),
    _psx("HCAR", "Honda Atlas Cars", "Auto", ("cyclical",)),
    _psx("MUREB", "Murree Brewery", "Consumer", ()),
    _psx("SEARL", "Searle Co.", "Pharma", ()),
    _psx("ABOT", "Abbott Pakistan", "Pharma", ()),
    _psx("GLAXO", "GSK Pakistan", "Pharma", ()),
    _psx("NML", "Nishat Mills", "Textile", ("cyclical", "export")),
    _psx("ILP", "Interloop Ltd.", "Textile", ("export",)),
    _psx("COLG", "Colgate-Palmolive Pakistan", "Consumer", ()),
    _psx("NESP", "Nestle Pakistan", "Consumer", (), ticker="NESTLE"),
    _psx("MFL", "Millat Tractors", "Auto", ("agri-proxy",)),
]

PSX_INDEX: Instrument = Instrument(
    key="KSE100", name="KSE-100 Index", asset_class="index", sector="Index",
    currency="PKR", market="PSX", tv="PSX:KSE100", psx_ticker="KSE100",
    tags=("index-heavy",),
)

COMMODITIES: list[Instrument] = [
    _yf("GOLD", "Gold Futures", "commodity", "Precious Metals", "GC=F", tags=("metal", "safe-haven")),
    _yf("SILVER", "Silver Futures", "commodity", "Precious Metals", "SI=F", tags=("metal",)),
    _yf("PLATINUM", "Platinum Futures", "commodity", "Precious Metals", "PL=F", tags=("metal",)),
    _yf("PALLADIUM", "Palladium Futures", "commodity", "Precious Metals", "PA=F", tags=("metal",)),
    _yf("WTI", "WTI Crude Oil", "commodity", "Energy", "CL=F", tags=("oil",)),
    _yf("BRENT", "Brent Crude Oil", "commodity", "Energy", "BZ=F", tags=("oil",)),
    _yf("NATGAS", "Natural Gas (Henry Hub)", "commodity", "Energy", "NG=F", tags=("gas",)),
    _yf("COPPER", "Copper Futures", "commodity", "Industrial Metals", "HG=F", tags=("metal", "growth-proxy")),
    _yf("WHEAT", "Wheat Futures", "commodity", "Agriculture", "ZW=F", tags=("agri",)),
    _yf("CORN", "Corn Futures", "commodity", "Agriculture", "ZC=F", tags=("agri",)),
    _yf("SOYBEAN", "Soybean Futures", "commodity", "Agriculture", "ZS=F", tags=("agri",)),
    _yf("COFFEE", "Coffee Futures", "commodity", "Agriculture", "KC=F", tags=("soft",)),
    _yf("SUGAR", "Sugar Futures", "commodity", "Agriculture", "SB=F", tags=("soft", "agri-proxy")),
    _yf("COCOA", "Cocoa Futures", "commodity", "Agriculture", "CC=F", tags=("soft",)),
    _yf("COTTON", "Cotton Futures", "commodity", "Agriculture", "CT=F", tags=("soft", "textile-proxy")),
]

GLOBAL_INDICES: list[Instrument] = [
    _yf("SPX", "S&P 500", "index", "US Equity", "^GSPC"),
    _yf("NDX", "Nasdaq Composite", "index", "US Equity", "^IXIC", tags=("tech",)),
    _yf("DJI", "Dow Jones Industrial", "index", "US Equity", "^DJI"),
    _yf("VIX", "CBOE Volatility Index", "index", "Volatility", "^VIX", tags=("risk-gauge",)),
    _yf("STOXX50", "Euro Stoxx 50", "index", "EU Equity", "^STOXX50E"),
    _yf("DAX", "DAX 40", "index", "EU Equity", "^GDAXI"),
    _yf("NIKKEI", "Nikkei 225", "index", "Japan Equity", "^N225"),
    _yf("HANGSENG", "Hang Seng", "index", "HK/China Equity", "^HSI"),
    _yf("SENSEX", "BSE Sensex", "index", "India Equity", "^BSESN", tags=("regional",)),
]

FX: list[Instrument] = [
    _yf("EURUSD", "Euro / US Dollar", "fx", "Currency", "EURUSD=X"),
    _yf("GBPUSD", "Pound / US Dollar", "fx", "Currency", "GBPUSD=X"),
    _yf("USDJPY", "US Dollar / Yen", "fx", "Currency", "USDJPY=X"),
    _yf("USDCHF", "US Dollar / Franc", "fx", "Currency", "USDCHF=X"),
    _yf("AUDUSD", "Aussie / US Dollar", "fx", "Currency", "AUDUSD=X"),
    _yf("USDCAD", "US Dollar / Loonie", "fx", "Currency", "USDCAD=X"),
    _yf("USDPKR", "US Dollar / Pakistani Rupee", "fx", "Currency", "PKR=X", tags=("psx-macro",)),
]

RATES: list[Instrument] = [
    _yf("US10Y", "US 10-Year Treasury Yield", "rate", "Rates", "^TNX", tags=("macro",)),
    _yf("DXY", "US Dollar Index", "rate", "Macro", "DX-Y.NYB", tags=("macro", "risk-gauge")),
]

def _cr(key: str, name: str, yahoo: str, tags: tuple[str, ...] = ()) -> Instrument:
    # The Binance pair is derived from the Yahoo quote ("BTC-USD" -> "BTCUSDT").
    # It is not decoration: Binance is one of the three venues KAIROS is built to
    # serve, so a crypto name carries the tape its orders would actually be
    # priced on. Yahoo stays the *primary history* for these names because the
    # published paper record was measured on Yahoo closes — see `MarketData.history`.
    return Instrument(
        key=key, name=name, asset_class="crypto", sector="Crypto",
        currency="USD", market="GLOBAL", yahoo=yahoo,
        binance=yahoo.split("-")[0].upper() + "USDT",
        tags=("crypto", "risk-appetite", *tags),
    )


CRYPTO: list[Instrument] = [
    _cr("BTC", "Bitcoin", "BTC-USD", ("safe-haven", "store-of-value")),
    _cr("ETH", "Ethereum", "ETH-USD", ("l1", "defi")),
    _cr("SOL", "Solana", "SOL-USD", ("l1", "high-beta")),
    _cr("BNB", "BNB", "BNB-USD", ("exchange")),
    _cr("XRP", "XRP", "XRP-USD", ("payments")),
    _cr("ADA", "Cardano", "ADA-USD", ("l1")),
    _cr("DOGE", "Dogecoin", "DOGE-USD", ("meme", "high-beta")),
    _cr("AVAX", "Avalanche", "AVAX-USD", ("l1")),
    _cr("TRX", "TRON", "TRX-USD", ("payments")),
    _cr("LINK", "Chainlink", "LINK-USD", ("oracle", "defi")),
    _cr("DOT", "Polkadot", "DOT-USD", ("l1")),
    _cr("LTC", "Litecoin", "LTC-USD", ("payments")),
    _cr("BCH", "Bitcoin Cash", "BCH-USD", ("payments")),
    _cr("NEAR", "NEAR Protocol", "NEAR-USD", ("l1", "ai")),
    _cr("APT", "Aptos", "APT-USD", ("l1")),
    _cr("UNI", "Uniswap", "UNI-USD", ("defi")),
    _cr("ATOM", "Cosmos", "ATOM-USD", ("l1", "interop")),
    _cr("FIL", "Filecoin", "FIL-USD", ("storage")),
    _cr("ARB", "Arbitrum", "ARB-USD", ("l2")),
    _cr("OP", "Optimism", "OP-USD", ("l2")),
    _cr("SUI", "Sui", "SUI-USD", ("l1", "high-beta")),
    _cr("INJ", "Injective", "INJ-USD", ("defi", "high-beta")),
]

ALL: list[Instrument] = [PSX_INDEX, *PSX_BLUE_CHIPS, *COMMODITIES,
                         *GLOBAL_INDICES, *FX, *RATES, *CRYPTO]

_BY_KEY = {i.key.upper(): i for i in ALL}
_BY_YAHOO = {i.yahoo.upper(): i for i in ALL if i.yahoo}
_BY_TV = {(i.tv or "").upper(): i for i in ALL if i.tv}
_BY_BINANCE = {(i.binance or "").upper(): i for i in ALL if i.binance}
# the static list above is the floor; scan-admitted tickers live beside it and
# must survive a daemon restart (see _registered_path)
_SEED_KEYS = frozenset(i.key.upper() for i in ALL)


def resolve(symbol: str) -> Instrument | None:
    """Resolve any symbol form (KAIROS key, Yahoo, TV, Binance pair, or PSX ticker)."""
    s = symbol.strip().upper()
    # the full TV form ("PSX:OGDC") is tried as well as the bare ticker: the
    # docstring promised it, and every caller that held a `tv` symbol missed
    return (_BY_KEY.get(s) or _BY_YAHOO.get(s) or _BY_BINANCE.get(s)
            or _BY_TV.get(s) or _BY_TV.get(f"PSX:{s}"))


_MISSING = ("", "nan", "none", "null", "n/a", "-", "?")


def _clean(raw: Any) -> str:
    """A missing cell in a fetched table arrives as float NaN, and `str()` of it
    is the four-letter word "nan" — which is truthy, so `name or key` happily
    made "nan" a company name and published it on the desk. Nothing that reads
    like a missing value is allowed to become a label."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    s = str(raw).strip()
    return "" if s.lower() in _MISSING else s


def register_psx(symbol: str, name: str | None = None,
                 sector: str = "Unregistered", _persist: bool = True) -> Instrument | None:
    """Admit a live scan-feed PSX ticker into the tradable registry at
    runtime. Returns None if the key collides with a non-PSX instrument."""
    key = symbol.strip().upper()
    if not key:
        return None
    existing = resolve(key)
    if existing is not None:
        return existing if existing.market == "PSX" else None
    inst = _psx(key, _clean(name) or key, _clean(sector) or "Unregistered")
    ALL.append(inst)
    _BY_KEY[key] = inst
    if inst.yahoo:
        _BY_YAHOO[inst.yahoo.upper()] = inst
    if inst.tv:
        _BY_TV[inst.tv.upper()] = inst
    if _persist:
        _save_registered()
    return inst


def _registered_path():
    try:
        from .config import CFG
        return CFG.data_dir / "psx_registered.json"
    except Exception:
        return None


def _save_registered() -> None:
    path = _registered_path()
    if path is None:
        return
    try:
        rows = [{"symbol": i.key, "name": i.name, "sector": i.sector}
                for i in ALL if i.key.upper() not in _SEED_KEYS]
        # bounded: daily rotation would otherwise admit forever
        path.write_text(json.dumps(rows[-40:], indent=1), encoding="utf-8")
    except OSError:
        pass  # losing the cache costs re-admission, never a trade


def _restore_registered() -> None:
    """Re-admit yesterday's scan leaders so a restarted daemon can still
    analyse and exit the positions the watchlist points at."""
    path = _registered_path()
    if path is None or not path.exists():
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(rows, list):
        return
    n = 0
    for row in rows:
        try:
            if register_psx(str(row["symbol"]), str(row.get("name") or ""),
                            str(row.get("sector") or "Unregistered"),
                            _persist=False) is not None:
                n += 1
        except (KeyError, TypeError, ValueError):
            continue
    if n:
        _save_registered()


_restore_registered()


def keys() -> list[str]:
    return list(_BY_KEY.keys())


def universe() -> list[Instrument]:
    return list(ALL)


def psx_universe() -> list[Instrument]:
    return [PSX_INDEX, *PSX_BLUE_CHIPS]


def global_universe() -> list[Instrument]:
    return [*COMMODITIES, *GLOBAL_INDICES, *FX, *RATES, *CRYPTO]
