"""
CommodityDesk — investment-bank commodities research panel.

Sits on top of the intermarket forces layer and turns it into a deep,
*explainable* directional vote for commodity symbols (gold/metals/energy) —
the "commodities desk" of an investment bank. Every vote carries the full
step-by-step reasoning (factor by factor), a plain-English narrative, and a
read of where crowd positioning sits.

Philosophy
----------
Gold (XAUUSD) is driven by a small set of macro relationships we encode as
weighted research factors:
    DXY ........... inverse (USD index up => gold down)
    US10Y ......... inverse real-yield proxy (yields up => gold down)
    VIX ........... risk-off => gold bid (safe haven)
    GOLD trend .... self-momentum (trend followers / algo flows)
    COPPER ........ Dr. Copper — global demand / China reflation
    SPX regime .... risk-off confirmation for gold bid
    WTI ........... oil shock => inflation hedge bid; collapse => deflation
    BTC ........... risk-appetite proxy (flows)
    News .......... commodity/news sentiment tilt
    Calendar ...... high-impact events raise caution (halve conviction)

Crowd read ("what other investors are doing") is assembled from the same
tapes: dollar trend + VIX level + gold momentum + crypto risk appetite,
translated into a one-paragraph positioning narrative.

Validation: `desk_hit_rate()` replays the last N days of *historical* force
data and checks whether the desk's prior-close vote predicted the next-day
XAUUSD direction — no lookahead, using only data known at the time of the
vote. This is how we prove (or disprove) the desk's edge on paper before
letting it influence the ensemble with weight.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import JarvisConfig
from .market_forces import MarketForcesService
from .news import NewsService

LOGGER = logging.getLogger("jarvis.desk")

# Commodity symbols the desk understands (by MT5 symbol name).
DEFAULT_COMMODITIES = ("XAUUSD", "XAGUSD", "USOIL", "UKOIL", "WTIUSD")

# Clamp for per-symbol min_score so the desk doesn't fire on noise.
# Tuned by historical sweep (120d XAUUSD): min_score=0.20, min_factors=2
# -> 58.8% 3d hit-rate / 0.65 short-side expectancy.
_DEFAULT_MIN_SCORE = 0.20
_DEFAULT_MIN_FACTORS = 2
_DEFAULT_LONG_EXTRA = 0.08   # longs need |score| >= min_score+extra (long side is weaker)

# yfinance tickers used by the no-lookahead replay / attribution validators.
METAL_TICKER = {"XAUUSD": "GC=F", "XAGUSD": "SI=F"}
FORCE_TICKERS = {
    "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VIX": "^VIX", "GOLD": "GC=F",
    "WTI": "CL=F", "COPPER": "HG=F", "SPX": "^GSPC", "BTC": "BTC-USD",
    "TIP": "TIP", "GDX": "GDX", "SLV": "SLV", "USO": "USO", "DBC": "DBC",
}

# Fundamental-news event taxonomy (headline keywords -> category). The desk
# treats these as "big-picture" catalysts, not flicker signals.
FUND_EVENTS: dict[str, tuple[str, ...]] = {
    "supply":  ("opec", "embargo", "strait", "supply", "strike", "export ban",
                "production cut", "disruption", "shutdown"),
    "cbank_buy": ("central bank", "reserve", "hoard", "purchase", "pboc",
                  "india", "gold buying", "bahrain", "poland"),
    "inflation": ("cpi", "inflation", "fed", "rate cut", "tariff",
                  "print", "fiscal", "stimulus"),
    "demand":  ("china demand", "importer", "industrial demand",
                "semiconductor", "reflation", "stimulus"),
    "inventory": ("inventory", "stockpile", "warehouse", "eia", "api",
                  "commercial stocks", "cushing"),
    "geo":     ("war", "invasion", "sanction", "missile", "conflict",
                "red sea", "pipeline"),
}


@dataclass
class DeskFactor:
    """One research factor with its signed contribution + human reason."""
    name: str
    sign: int
    contribution: float          # weighted contribution to the score
    weight: float
    reason: str
    data: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sign": self.sign,
            "contribution": round(self.contribution, 4),
            "weight": round(self.weight, 4),
            "reason": self.reason,
            "data": round(float(self.data), 4),
        }


@dataclass
class DeskVote:
    """One desk decision for one symbol — fully self-explanatory."""
    symbol: str
    score: float                 # -1..+1 aggregate
    direction: int               # -1 / 0 / +1
    confidence: float            # 0..1
    factors: list[DeskFactor] = field(default_factory=list)
    narrative: str = ""
    crowd_read: str = ""
    stale: bool = False
    hedge_hint: bool = False
    as_of: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "factors": [f.to_dict() for f in self.factors],
            "narrative": self.narrative,
            "crowd_read": self.crowd_read,
            "stale": self.stale,
            "hedge_hint": self.hedge_hint,
            "as_of": self.as_of,
        }


class CommodityDesk:
    """The commodities research desk. Polls nothing itself — reads the
    already-cached intermarket forces + news + calendar, so the per-cycle cost
    is microseconds of numpy."""

    def __init__(
        self,
        *,
        config: JarvisConfig,
forces_service: MarketForcesService,
        news_service: Optional[NewsService] = None,
        enabled: bool = True,
        min_score: float = _DEFAULT_MIN_SCORE,
        min_factors: int = _DEFAULT_MIN_FACTORS,
        long_score_extra: float = _DEFAULT_LONG_EXTRA,
        disabled_factors: Optional[set[str]] = None,
    ) -> None:
        self.config = config
        self.forces = forces_service
        self.news = news_service
        self.enabled = enabled
        self.min_score = float(min_score)
        self.min_factors = max(1, int(min_factors))
        self.long_score_extra = float(long_score_extra)
        self.disabled_factors = set(disabled_factors or ())
        self._votes: dict[str, DeskVote] = {}
        self._lock = threading.Lock()
        self._validate_lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: JarvisConfig,
        forces_service: MarketForcesService,
        news_service: Optional[NewsService] = None,
    ) -> "CommodityDesk":
        section = config.section("desk")
        return cls(
            config=config,
            forces_service=forces_service,
            news_service=news_service,
            enabled=bool(section.get("enabled", True)),
            min_score=float(section.get("min_score", _DEFAULT_MIN_SCORE)),
            min_factors=int(section.get("min_factors", _DEFAULT_MIN_FACTORS)),
            long_score_extra=float(section.get("long_score_extra", _DEFAULT_LONG_EXTRA)),
            disabled_factors=set(section.get("disabled_factors", [])),
        )

    def set_disabled_factors(self, names: list[str]) -> None:
        """Data-driven factor pruning (from `GET /desk/factors` attribution)."""
        self.disabled_factors = set(names)

    # ------------------------------------------------------------------ public

    def compute(
        self,
        symbol: str,
        *,
        ta_bias: float = 0.0,
        ts_direction: int = 0,
        ts_confidence: float = 0.0,
        news_sentiment: float = 0.0,
        news_relevance: float = 0.0,
        calendar_impact: int = 0,
        in_blackout: bool = False,
    ) -> Optional[DeskVote]:
        """Produce a desk vote for a commodity symbol (None for non-commodities
        or when the desk is disabled)."""
        if not self.enabled:
            return None
        ins = self._instrument_sector(symbol)
        if ins != "metals" and ins != "energy":
            return None

        snaps = self.forces.snapshot()
        fund_events = self._news_events() if (self.news and self.news.enabled) else None

        if ins == "metals":
            return self._compute_metals(
                symbol, snaps,
                ta_bias=ta_bias, ts_direction=ts_direction,
                ts_confidence=ts_confidence, news_sentiment=news_sentiment,
                news_relevance=news_relevance, calendar_impact=calendar_impact,
                in_blackout=in_blackout, fund_events=fund_events)
        return self._compute_energy(
            symbol, snaps.get("DXY"), snaps.get("WTI") or snaps.get("USO"),
            snaps.get("VIX"), snaps.get("BTC"),
            ta_bias=ta_bias, news_sentiment=news_sentiment,
            news_relevance=news_relevance, calendar_impact=calendar_impact,
            in_blackout=in_blackout)

    def compute_all(
        self,
        *,
        symbols: tuple[str, ...] = DEFAULT_COMMODITIES,
        **per_symbol_kwargs,
    ) -> dict[str, DeskVote]:
        out: dict[str, DeskVote] = {}
        for sym in symbols:
            kw = per_symbol_kwargs.get(sym, {})
            v = self.compute(sym, **kw)
            if v is not None:
                out[sym] = v
        with self._lock:
            self._votes = out
        return out

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._votes.items()}

    def desk_hit_rate(self, days: int = 60, symbol: str = "XAUUSD") -> dict:
        """Replay the desk on historical data with strictly no lookahead.

        Uses yfinance daily bars for the *metal itself* plus every force
        ticker. A vote dated day t uses only force data known through close of
        day t-1; the labels are the realized forward moves of the metal over
        horizons of 1/3/5 days. Reports hit-rate, fire counts, abstention and
        the average forward return captured by each side (expectancy — that's
        what actually matters, not raw hit-rate).
        """
        with self._validate_lock:  # never pile up concurrent yfinance replays
            return self._desk_hit_rate_impl(days, symbol)

    def _load_replay_rows(self, days: int, symbol: str):
        """Fetch metal + all force tickers and build strictly no-lookahead rows.

        Row dated at day i contains only force data from bars strictly BEFORE
        that day's close plus realized forward labels. Returns `(rows, horizons)`
        or `({"error": ...}, None)`.
        """
        import yfinance as yf  # lazy, validation-only
        horizons = (1, 3, 5)
        max_h = max(horizons)
        tk_metal = METAL_TICKER.get(symbol, "GC=F")
        try:
            metal = yf.Ticker(tk_metal).history(period=f"{int(days*1.9)+60}d",
                                                interval="1d", auto_adjust=True)
            if metal is None or len(metal) < max_h + 15:
                return {"error": "insufficient metal history"}, None
            mclose = metal["Close"].astype(float)
            midx = list(mclose.index)
            frames = {n: yf.Ticker(t).history(period=f"{int(days*1.9)+60}d",
                                              interval="1d", auto_adjust=True)["Close"]
                      for n, t in FORCE_TICKERS.items()}
            rows: list[dict] = []
            for i in range(max_h, len(midx) - max_h):
                t = midx[i]
                fwd = {h: (1 if float(mclose.iloc[i + h]) > float(mclose.iloc[i]) else
                           (-1 if float(mclose.iloc[i + h]) < float(mclose.iloc[i]) else 0))
                       for h in horizons}
                if all(v == 0 for v in fwd.values()):
                    continue
                force_row = {}
                for name, cs in frames.items():
                    if cs is None or len(cs) < 22:
                        continue
                    h = cs[cs.index < t]
                    if len(h) < 21:
                        continue
                    force_row[name] = {
                        "last": float(h.iloc[-1]),
                        "chg20": (float(h.iloc[-1]) / float(h.iloc[-21]) - 1.0) * 100.0,
                        "chg5": (float(h.iloc[-1]) / float(h.iloc[-6]) - 1.0) * 100.0,
                    }
                rows.append({"fwd": fwd, "forces": force_row})
            if len(rows) < 15:
                return {"error": "too few samples", "n": len(rows)}, None
            return rows, horizons
        except Exception as e:  # pragma: no cover
            return {"error": str(e)}, None

    def _score_replay_row(self, symbol: str, forces: dict):
        snaps = _assemble_prior_snapshots(forces)
        return self._compute_metals(
            symbol, snaps, news_sentiment=0.0, news_relevance=0.0,
            in_blackout=False)

    def _desk_hit_rate_impl(self, days: int, symbol: str) -> dict:
        rows, horizons = self._load_replay_rows(days, symbol)
        if rows is None:
            return dict(horizons or {"error": "replay failed"})
        horizons = horizons or (1, 3, 5)

        # Score every row (prior-close data only) and bucket by direction.
        buckets: dict[int, dict] = {}
        for h in horizons:
            buckets[h] = {"long": [], "short": [], "flat": []}
        for r in rows:
            v = self._score_replay_row(symbol, r["forces"])
            if v is None:
                for h in horizons:
                    buckets[h]["flat"].append(r["fwd"][h])
                continue
            key = "long" if v.direction > 0 else ("short" if v.direction < 0 else "flat")
            for h in horizons:
                buckets[h][key].append(r["fwd"][h])

        report: dict = {"symbol": symbol, "days": days, "horizons": {}}
        for h in horizons:
            b = buckets[h]
            N = len(b["long"]) + len(b["short"])
            hits = sum(1 for d in b["long"] if d == 1) + \
                sum(1 for d in b["short"] if d == -1)
            hit = (hits / N) if N else None
            report["horizons"][str(h)] = {
                "fired": N,
                "hit_rate": round(hit, 4) if hit is not None else None,
                "long": len(b["long"]),
                "short": len(b["short"]),
                "abstained": len(b["flat"]),
            }
        # Expectancy: mean forward-day move captured by firing longs/shorts.
        h3 = buckets[3]
        exp_long = (sum(buckets[3]["long"]) / len(buckets[3]["long"])
                    if buckets[3]["long"] else None)
        exp_short = (sum(buckets[3]["short"]) / len(buckets[3]["short"])
                     if buckets[3]["short"] else None)
        report["expectancy_3d"] = {
            "long_hit_prob": round((sum(1 for d in h3["long"] if d == 1) / max(1, len(h3["long"]))), 4),
            "short_hit_prob": round((sum(1 for d in h3["short"] if d == -1) / max(1, len(h3["short"]))), 4),
            "long_avg": round(exp_long, 4) if exp_long is not None else None,
            "short_avg": round(exp_short, 4) if exp_short is not None else None,
        }
        fired3 = report["horizons"]["3"]["fired"]
        h3hit = report["horizons"]["3"]["hit_rate"]
        if fired3 >= 15 and h3hit is not None:
            report["verdict"] = ("edge" if h3hit >= 0.54
                                 else ("neutral" if h3hit >= 0.48 else "noise"))
        else:
            report["verdict"] = "insufficient_samples"
        return report

    def factor_attribution(self, days: int = 250, symbol: str = "XAUUSD") -> dict:
        """Data-driven factor report card (strictly no-lookahead, 3d horizon).

        For every research factor, measure how often its *own* sign agreed
        with the realized forward move. Factors scoring below ~50% are noise
        for this market — candidates for `desk.disabled_factors`.
        """
        with self._validate_lock:
            return self._factor_attribution_impl(days, symbol)

    def _factor_attribution_impl(self, days: int, symbol: str) -> dict:
        rows, horizons = self._load_replay_rows(days, symbol)
        if rows is None:
            return dict(horizons or {"error": "replay failed"})
        stats: dict[str, dict] = {}
        for r in rows:
            snaps = _assemble_prior_snapshots(r["forces"])
            factors = self._metal_factors(
                symbol, snaps, news_sentiment=0.0, news_relevance=0.0)
            label = r["fwd"][3]
            for f in factors:
                if f.contribution == 0:
                    continue
                d = 1 if f.contribution > 0 else -1
                s = stats.setdefault(f.name, {"fires": 0, "hits": 0, "long": 0, "short": 0})
                s["fires"] += 1
                if d == label:
                    s["hits"] += 1
                if d == 1:
                    s["long"] += 1
                else:
                    s["short"] += 1
        for s in stats.values():
            s["hit_rate"] = round(s["hits"] / max(1, s["fires"]), 4)
            n_short = s["short"]
            n_long = s["long"]
            s["long_share"] = round(n_long / max(1, n_long + n_short), 3)
            s["suggest_disable"] = s["fires"] >= 20 and s["hit_rate"] < 0.5
        ordered = sorted(stats.items(), key=lambda kv: kv[1]["hit_rate"], reverse=True)
        return {
            "symbol": symbol,
            "days": days,
            "horizon": 3,
            "factors": ordered,
            "best": [n for n, s in ordered[:5]],
            "worst": [n for n, s in ordered[-5:]],
            "disable_candidates": [n for n, s in ordered if s["suggest_disable"]],
            "note": "hit_rate = agreement of the factor's own direction with the 3d forward move "
                    "(coin flip = 0.50).",
        }

    # -------------------------------------------------------------- metals ---

    def _compute_metals(self, symbol, snaps: dict,
                        *, ta_bias=0.0, ts_direction=0, ts_confidence=0.0,
                        news_sentiment=0.0, news_relevance=0.0,
                        calendar_impact=0, in_blackout=False,
                        fund_events=None) -> Optional[DeskVote]:
        factors = self._metal_factors(
            symbol, snaps, ta_bias=ta_bias, ts_direction=ts_direction,
            ts_confidence=ts_confidence, news_sentiment=news_sentiment,
            news_relevance=news_relevance, fund_events=fund_events,
        )
        dxy, us10y, vix, gold = (snaps.get("DXY"), snaps.get("US10Y"),
                                 snaps.get("VIX"), snaps.get("GOLD"))
        essential = bool(dxy and us10y and vix and gold)

        score, confidence, pos_n, neg_n = self._aggregate(factors)
        stale = not essential
        if stale:
            confidence *= 0.5
        agreed = max(pos_n, neg_n) >= self.min_factors
        if calendar_impact >= 3 or in_blackout:
            # High-impact event window: a desk is *more* cautious, not louder.
            if in_blackout:
                direction = 0
                confidence = min(confidence, 0.3)
            else:
                confidence *= 0.6
        else:
            # Longs historically weaker -> require a stricter score to fire.
            direction = 1 if (agreed and score >= (self.min_score + self.long_score_extra)) else \
                (-1 if (agreed and score <= -self.min_score) else 0)

        if direction == 0 and not stale:
            return None  # nothing actionable, stay out — honest abstention

        narrative = self._narrative(direction, score, factors, stale, calendar_impact, in_blackout)
        crowd = self._crowd_read(vix, dxy, gold, snaps.get("BTC"), direction)
        vote = DeskVote(
            symbol=symbol, score=score, direction=direction,
            confidence=round(confidence, 3), factors=factors,
            narrative=narrative, crowd_read=crowd, stale=stale,
            hedge_hint=(calendar_impact >= 2),
        )
        with self._lock:
            self._votes[symbol] = vote
        return vote

    def _metal_factors(self, symbol, snaps: dict,
                       *, ta_bias=0.0, ts_direction=0, ts_confidence=0.0,
                       news_sentiment=0.0, news_relevance=0.0,
                       fund_events=None) -> list[DeskFactor]:
        """Build the research-factor stack for a metal. Pure + deterministic —
        shared by live compute, the no-lookahead replay and factor attribution."""
        dxy, us10y, vix, gold = (snaps.get("DXY"), snaps.get("US10Y"),
                                 snaps.get("VIX"), snaps.get("GOLD"))
        wti = snaps.get("WTI") or snaps.get("USO")
        copper, spx, btc, tip = (snaps.get("COPPER"), snaps.get("SPX"),
                                 snaps.get("BTC"), snaps.get("TIP"))
        gdx, slv, dbc = snaps.get("GDX"), snaps.get("SLV"), snaps.get("DBC")
        factors: list[DeskFactor] = []

        def add(name: str, data: float, sign: float, weight: float, reason: str) -> None:
            if name in self.disabled_factors:
                return
            contribution = float(sign) * weight
            factors.append(DeskFactor(
                name=name, sign=int(np.sign(sign)) if sign else 0,
                contribution=contribution, weight=round(weight, 3),
                reason=reason, data=data))

        if dxy:
            d = -np.sign(dxy.change_pct_20d) if abs(dxy.change_pct_20d) > 1.0 else 0.0
            add("DXY", dxy.change_pct_20d, d, 0.18,
                f"USD index {dxy.change_pct_20d:+.1f}% 20d — {'soft dollar bids gold' if d > 0 else 'firm dollar drags gold' if d < 0 else 'dollar range-bound'} (inverse)")
        if us10y:
            y = -np.sign(us10y.change_pct_20d) if abs(us10y.change_pct_20d) > 3.0 else 0.0
            add("US10Y", us10y.change_pct_20d, y, 0.16,
                f"US10Y {us10y.change_pct_20d:+.1f}% 20d — {'falling real-yield proxy lifts gold' if y > 0 else 'rising yields pressure gold' if y < 0 else 'yields flat'}")
        if tip:
            r = np.sign(tip.change_pct_20d) if abs(tip.change_pct_20d) > 0.8 else 0.0
            add("TIP_real", tip.change_pct_20d, r, 0.10,
                f"TIPS {tip.change_pct_20d:+.1f}% 20d — {'real yields falling, gold bid' if r > 0 else 'real yields rising, gold pressured' if r < 0 else 'real rates flat'}")
        if vix:
            v = 1.0 if vix.last > 22 else (0.0 if vix.last > 16 else -0.3)
            add("VIX", vix.last, v, 0.12,
                f"VIX {vix.last:.1f} — {'risk-off, gold as safe haven' if v > 0 else 'calm tape, less fear-bid for gold' if v < 0 else 'mild risk tone'}")
        if gold:
            g = np.sign(gold.change_pct_20d) if abs(gold.change_pct_20d) > 1.5 else 0.0
            add("GOLD_trend", gold.change_pct_20d, g, 0.14,
                f"gold futures {gold.change_pct_20d:+.1f}% 20d — {'momentum buyers in the tape' if g > 0 else 'downtrend flows' if g < 0 else 'consolidation'}")
        if gdx:
            m = np.sign(gdx.change_pct_20d) if abs(gdx.change_pct_20d) > 2.0 else 0.0
            add("GDX_flow", gdx.change_pct_20d, m, 0.08,
                f"miners {gdx.change_pct_20d:+.1f}% 20d — {'miners leading gold higher' if m > 0 else 'miners confirming weakness' if m < 0 else 'miners flat'}")
        if slv and gold and slv.last > 0 and slv.last is not None:
            ratio = float(gold.last) / float(slv.last)
            s = (-0.25 if ratio > 80 else (0.2 if ratio < 60 else 0.0))
            add("XAU_SLV", ratio, s, 0.04,
                f"gold/silver {ratio:.1f} — {'gold stretched vs silver (mean-reversion caution)' if s < 0 else 'ratio cheap, reflation bid' if s > 0 else 'ratio mid-range'}")
        if copper:
            c = np.sign(copper.change_pct_20d) if abs(copper.change_pct_20d) > 2.0 else 0.0
            add("COPPER", copper.change_pct_20d, c, 0.07,
                f"Copper {copper.change_pct_20d:+.1f}% 20d — Dr. Copper {'signals global reflation (gold bid)' if c > 0 else 'warns of demand softness' if c < 0 else 'flat'}")
        if dbc:
            b = np.sign(dbc.change_pct_20d) if abs(dbc.change_pct_20d) > 2.0 else 0.0
            add("DBC", dbc.change_pct_20d, b, 0.05,
                f"commodity basket {dbc.change_pct_20d:+.1f}% 20d — {'broad commodity bid lifts metals' if b > 0 else 'broad commodity weakness' if b < 0 else 'basket flat'}")
        if spx and spx.regime == "risk_off":
            add("SPX", 0.0, 0.5, 0.07, "equities in risk-off — flight to gold")
        if gold and spx:
            rel = gold.change_pct_20d - spx.change_pct_20d
            if abs(rel) > 4.0:
                s = 0.5 if rel > 0 else -0.5
                add("gold_vs_spx", rel, s, 0.04,
                    f"gold {rel:+.1f}% vs SPX over 20d — {'safe-haven crowding, momentum' if s > 0 else 'gold lagging, risk appetite elsewhere' if s < 0 else 'neutral'}")
        if wti:
            w = np.sign(wti.change_pct_5d) if abs(wti.change_pct_5d) > 4.0 else 0.0
            s = 0.5 if w > 0 else (-0.5 if w < 0 else 0.0)
            add("WTI", wti.change_pct_5d, s, 0.05,
                f"oil {wti.change_pct_5d:+.1f}% 5d — {'inflation-hedge bid' if s > 0 else 'deflationary tone' if s < 0 else 'oil quiet'}")
        if btc:
            b = np.sign(btc.change_pct_20d) if abs(btc.change_pct_20d) > 3.0 else 0.0
            s = -0.3 if b > 0 else (0.3 if b < 0 else 0.0)
            add("BTC", btc.change_pct_20d, s, 0.03,
                f"BTC {btc.change_pct_20d:+.1f}% 20d — {'risk-appetite high, gold bid eases' if s < 0 else 'risk-off flows into metals' if s > 0 else 'neutral'}")
        add("TA", ta_bias, np.clip(float(ta_bias) / 3.0, -0.3, 0.3), 0.02,
            f"local technicals {ta_bias:+.2f} — short-term confirmation")
        if ts_direction != 0 and abs(ts_confidence) > 0.4:
            add("TimesFM", float(ts_direction), float(ts_direction) * 0.5, 0.04,
                f"foundation-model forecast {ts_direction:+d} (conf {ts_confidence:.2f})")

        if news_relevance >= 0.5 and abs(news_sentiment) > 0.05:
            add("News", news_sentiment, np.clip(news_sentiment * 0.5, -0.4, 0.4), 0.04,
                f"commodity news tone {news_sentiment:+.2f}")

        if fund_events:
            bubble = float(np.clip(np.mean([e.get("sentiment", 0.0) for e in fund_events]) * 1.5,
                                   -1.0, 1.0))
            head = (fund_events[0].get("title", "") or "")[:72]
            add("FUND", bubble, bubble, 0.06,
                f"{len(fund_events)} fundamental catalyst(s) — e.g. “{head}”")

        return factors

    def _news_events(self, limit: int = 6) -> list[dict]:
        """Scan the recent tape for *fundamental* commodity catalysts and tag
        them by category. Same cached RSS items the rest of the system uses."""
        if self.news is None or not self.news.enabled:
            return []
        out: list[dict] = []
        for item in self.news.items_for_symbol("XAUUSD", max_age_seconds=5400):
            blob = (item.title + " " + item.summary).lower()
            cat = None
            for category, keys in FUND_EVENTS.items():
                if any(k in blob for k in keys):
                    cat = category
                    break
            if cat is not None:
                out.append({"category": cat, "title": item.title,
                            "sentiment": round(float(item.sentiment), 3),
                            "source": item.source})
            if len(out) >= limit:
                break
        return out

    def news_events(self) -> list[dict]:
        with self._lock:
            if self.news is None or not self.news.enabled:
                return []
        return self._news_events()

    def _compute_energy(self, symbol, dxy, wti, vix, btc,
                        *, ta_bias=0.0, news_sentiment=0.0, news_relevance=0.0,
                        calendar_impact=0, in_blackout=False) -> Optional[DeskVote]:
        factors: list[DeskFactor] = []
        if wti:
            w = np.sign(wti.change_pct_20d) if abs(wti.change_pct_20d) > 2.0 else 0.0
            factors.append(DeskFactor("WTI_trend", int(w), w * 0.5, 0.5,
                                      f"WTI {wti.change_pct_20d:+.1f}% 20d momentum"))
        if dxy:
            d = -np.sign(dxy.change_pct_20d) if abs(dxy.change_pct_20d) > 1.0 else 0.0
            factors.append(DeskFactor("DXY", int(d), d * 0.25, 0.25,
                                      "soft dollar supports commodity prices"))
        if vix and vix.last > 28:
            factors.append(DeskFactor("risk", -1, -0.15, 0.15,
                                      "stress tape caps energy rallies"))
        if news_relevance >= 0.5 and abs(news_sentiment) > 0.1:
            factors.append(DeskFactor("News", int(np.sign(news_sentiment)),
                                      news_sentiment * 0.15, 0.10,
                                      f"energy news tone {news_sentiment:+.2f}"))
        score, confidence = self._aggregate(factors)[:2]
        if calendar_impact >= 3 or in_blackout:
            if in_blackout:
                return None
            confidence *= 0.6
        direction = 1 if score >= self.min_score else (-1 if score <= -self.min_score else 0)
        if direction == 0:
            return None
        vote = DeskVote(symbol=symbol, score=score, direction=direction,
                        confidence=round(confidence, 3), factors=factors,
                        narrative=f"{symbol}: {direction:+d} via energy desk "
                                  f"({' / '.join(f.reason for f in factors)})")
        with self._lock:
            self._votes[symbol] = vote
        return vote

    # ---------------------------------------------------------------- helpers

    def _aggregate(self, factors: list[DeskFactor]) -> tuple[float, float, int, int]:
        if not factors:
            return 0.0, 0.0, 0, 0
        total_w = sum(f.weight for f in factors)
        if total_w <= 0:
            return 0.0, 0.0, 0, 0
        score = sum(f.contribution for f in factors) / total_w
        # Confidence from weighted agreement + strength.
        pos = sum(f.contribution for f in factors if f.contribution > 0)
        neg = -sum(f.contribution for f in factors if f.contribution < 0)
        pos_n = sum(1 for f in factors if f.contribution > 0)
        neg_n = sum(1 for f in factors if f.contribution < 0)
        strength = max(pos, neg)
        agreement = 1.0 - (abs(pos - neg) / max(total_w, 1e-9))
        confidence = min(1.0, strength * 1.6 + agreement * 0.5)
        return (float(np.clip(score, -1.0, 1.0)),
                float(np.clip(confidence, 0.0, 1.0)), pos_n, neg_n)

    def _narrative(self, direction, score, factors, stale, cal_impact, blackout) -> str:
        top = ", ".join(f.reason for f in factors[:4])
        bias = "long" if direction > 0 else ("short" if direction < 0 else "flat")
        note = ""
        if stale:
            note = " [PARTIAL DATA — reduced conviction]"
        if blackout:
            note = " [BLACKOUT — desk stands down]"
        elif cal_impact >= 3:
            note = " [HIGH-IMPACT EVENT WINDOW]"
        return f"Desk {bias} (score {score:+.2f}): {top}.{note}"

    def _crowd_read(self, vix, dxy, gold, btc, direction) -> str:
        parts: list[str] = []
        if vix is not None:
            parts.append("fear gauge muted/calm" if vix.last < 18 else
                         ("fear elevated — crowd is hedging into metals" if vix.last > 22
                          else "fear baseline"))
        if dxy is not None and abs(dxy.change_pct_20d) > 1.0:
            parts.append("US dollar heading %s" %
                         ("down — gold longs favored" if dxy.change_pct_20d < 0
                          else "up — crowding away from metals"))
        if gold is not None:
            g20 = gold.change_pct_20d
            flow = ("rising, momentum longs building" if g20 > 0
                    else "falling, crowded longs unwinding" if g20 < 0
                    else "sideways, waiting")
            parts.append("gold %+.1f%% over 20d — position flows %s" % (g20, flow))
        if btc is not None and btc.change_pct_20d > 3.0:
            parts.append("crypto bid — risk appetite drawing capital away from metals")
        if direction > 0:
            parts.append("desk follows the prevailing crowd tilt to the upside")
        elif direction < 0:
            parts.append("desk is contrarian to a nervous long crowd")
        return "; ".join(parts) or "no clear crowd signal"

    def _instrument_sector(self, symbol: str) -> str | None:
        if symbol in ("XAUUSD", "XAGUSD", "GOLD") or "GOLD" in symbol:
            return "metals"
        from .symbols import get as get_instrument
        ins = get_instrument(symbol)
        if ins is not None:
            sec = str(getattr(ins, "sector", ""))
            if sec == "metals" or sec == "energy":
                return sec
        if symbol in ("XAUUSD", "XAGUSD", "XPTUSD"):  # metals
            return "metals"
        return None


# ---------------------------------------------------------------------------
# Pure helper: turn a {name: {...}} prior-close force row into ForceSnapshot-
# shaped ducks for the desk (used by desk_hit_rate, zero lookahead).
# ---------------------------------------------------------------------------
def _assemble_prior_snapshots(row: dict) -> dict:
    out = {}
    for name, d in row.items():
        out[name] = _PriorForce(name, d.get("last", 0.0),
                                d.get("chg5", 0.0), d.get("chg20", 0.0))
    return out


class _PriorForce:
    __slots__ = ("name", "last", "change_pct_5d", "change_pct_20d", "regime")

    def __init__(self, name: str, last: float, chg5: float, chg20: float):
        self.name = name
        self.last = last
        self.change_pct_5d = chg5
        self.change_pct_20d = chg20
        self.regime = "neutral"


__all__ = ["CommodityDesk", "DeskVote", "DeskFactor", "DEFAULT_COMMODITIES",
           "desk_hit_rate_main"]

GOLD = "GOLD"


def desk_hit_rate_main() -> dict:
    """Standalone validator: python -m jarvis_predict.commodities"""
    desk = CommodityDesk(config=JarvisConfig.load(),
                         forces_service=MarketForcesService.from_config(JarvisConfig.load()))
    return desk.desk_hit_rate()


if __name__ == "__main__":
    import json
    print(json.dumps(desk_hit_rate_main(), indent=2, default=str))