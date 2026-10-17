"""KSE-100 regime engine — ported from the VoiceAI jarvis brain (2026-09) and
adapted to the unified data layer.

Classification from daily closes:
  bull      SMA20 > SMA50 AND 30d OLS slope > 0 (drawdown > -8% else demoted)
  bear      SMA20 < SMA50 AND slope < 0
  volatile  annualized realized vol > 25%
  sideways  otherwise
Position multipliers: bull 1.2 / sideways 1.0 / volatile 0.7 / bear 0.5.

History source chain: KSE100 store -> synthetic 8-ticker blue-chip basket
(equal-weight OGDC/HBL/UBL/PPL/ENGRO/HUBC/LUCK/MCB) -> honest "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BASKET = ("OGDC", "HBL", "UBL", "PPL", "ENGRO", "HUBC", "LUCK", "MCB")
MULTIPLIERS = {"bull": 1.2, "sideways": 1.0, "volatile": 0.7, "bear": 0.5,
               "unknown": 0.8}


@dataclass
class KseRegime:
    label: str
    confidence: float
    position_multiplier: float
    slope: float
    vol_annual: float
    drawdown: float
    source: str


def _classify(closes: np.ndarray, source: str) -> KseRegime:
    c = closes.astype(float)
    if len(c) < 60:
        return KseRegime("unknown", 0.3, MULTIPLIERS["unknown"], 0.0, 0.0, 0.0, source)
    sma20 = float(np.mean(c[-20:]))
    sma50 = float(np.mean(c[-50:]))
    y = np.log(c[-30:])
    slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
    rets = np.diff(np.log(c))
    vol = float(np.std(rets) * np.sqrt(252))
    peak = float(np.max(c[-60:]))
    dd = float(c[-1] / peak - 1)

    if vol > 0.25:
        conf = float(np.clip((vol - 0.25) / 0.25 + 0.4, 0.4, 0.95))
        return KseRegime("volatile", conf, MULTIPLIERS["volatile"], slope, vol, dd, source)
    if sma20 > sma50 and slope > 0:
        if dd < -0.08:
            return KseRegime("sideways", 0.55, MULTIPLIERS["sideways"], slope, vol, dd, source)
        conf = float(np.clip(0.5 + 50 * slope / (sma20 / c[-1]) + 5 * (sma20 - sma50) / sma50,
                             0.4, 0.95))
        return KseRegime("bull", conf, MULTIPLIERS["bull"], slope, vol, dd, source)
    if sma20 < sma50 and slope < 0:
        conf = float(np.clip(0.5 + 50 * -slope / (sma50 / c[-1]) + 5 * (sma50 - sma20) / sma20,
                             0.4, 0.95))
        return KseRegime("bear", conf, MULTIPLIERS["bear"], slope, vol, dd, source)
    return KseRegime("sideways", 0.5, MULTIPLIERS["sideways"], slope, vol, dd, source)


def kse_regime(md) -> KseRegime:
    """md: MarketData instance."""
    from .universe import resolve
    kse = resolve("KSE100")
    df = md.history(kse)
    if df is not None and len(df) >= 60:
        return _classify(df["close"].values, "kse100-store")

    # synthetic equal-weight blue-chip basket from the accumulated store
    from .config import CFG
    series: dict[str, pd.Series] = {}
    for sym in BASKET:
        p = CFG.cache_dir / f"{sym}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, parse_dates=["date"])
            if len(d) >= 60:
                series[sym] = d.set_index("date")["close"]
        except Exception:
            continue
    if len(series) >= 5:
        basket = pd.DataFrame(series).dropna()
        if len(basket) >= 60:
            idx = basket.mean(axis=1).pct_change().add(1).cumprod()
            return _classify(idx.values, f"basket-{len(series)}tickers")
    return KseRegime("unknown", 0.3, MULTIPLIERS["unknown"], 0.0, 0.0, 0.0, "none")


def regime_summary(md) -> dict[str, Any]:
    r = kse_regime(md)
    return {"label": r.label, "confidence": round(r.confidence, 3),
            "position_multiplier": r.position_multiplier,
            "slope_30d_annualized": round(r.slope * 252, 4),
            "vol_annual": round(r.vol_annual, 4),
            "drawdown_60d": round(r.drawdown, 4), "source": r.source}
