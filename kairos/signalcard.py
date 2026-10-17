"""Shareable live signal cards — Instagram-square PNGs + WhatsApp captions.

One fresh engine call renders a branded 1080x1080 card (direction,
confidence, entry/target/stop) plus a plain-text caption. Meant for
sharing, never for auto-execution; every artifact carries the disclaimer.
"""

from __future__ import annotations

import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .config import CFG  # noqa: E402

BG, PANEL, ACCENT = "#0A1628", "#13233C", "#4C8DFF"
TXT, FAINT = "#F2F6FC", "#7E93B0"
UP, DOWN = "#12B76A", "#E03131"

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_S = 240.0


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def build_signal(symbol: str) -> dict[str, Any]:
    """Fresh engine verdict for one symbol, cached ~4 minutes."""
    key = symbol.strip().upper()
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    from .engine import ENGINE
    res = ENGINE.predict(key)
    s = res["signal"]
    bands = res.get("confidence_band") or {}
    h = s["horizon_days"]
    tgt = (res.get("targets") or {}).get(f"{h}d")
    long = s["score"] > 0
    # never publish a plan that contradicts its own direction: if the
    # horizon target sits on the loss side, use the band quantile instead
    wrong_side = tgt is None or (tgt <= res["price"] if long else tgt >= res["price"])
    if wrong_side:
        tgt = bands.get("p90") if long else bands.get("p10")
    sig = {"symbol": res["symbol"], "name": res["name"], "price": res["price"],
           "direction": s["direction"], "score": s["score"],
           "confidence": s["confidence"],
           "expected_return": s["expected_return"],
           "horizon_days": h,
           "regime": str(res["regime"]["regime"]),
           "target": tgt,
           "stop": bands.get("p10") if long else bands.get("p90"),
           "ts": time.strftime("%Y-%m-%d %H:%M")}
    _CACHE[key] = (time.time(), sig)
    return sig


def caption(sig: dict[str, Any]) -> str:
    long = sig["score"] > 0
    arrow = "LONG (buy)" if long else "SHORT (sell)"
    conf = f"{(sig['confidence'] or 0) * 100:.0f}%"
    er = (sig["expected_return"] or 0) * 100
    return (f"KAIROS LIVE SIGNAL — {sig['symbol']}\n"
            f"{arrow} · confidence {conf}\n"
            f"Entry {_fmt(sig['price'])} · Target {_fmt(sig['target'])}"
            f" · Stop {_fmt(sig['stop'])}\n"
            f"Horizon {sig['horizon_days']}d · expected {er:+.2f}%\n"
            f"Generated {sig['ts']} · model output, not investment advice")


def render_png(sig: dict[str, Any], path) -> str:
    long = sig["score"] > 0
    col = UP if long else DOWN
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1080)
    ax.set_ylim(0, 1080)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1080, 1080, color=BG))

    # header
    ax.text(70, 1000, "K A I R O S", fontsize=30, fontweight="bold",
            color=ACCENT, va="center")
    ax.text(330, 1000, "· LIVE SIGNAL", fontsize=16, color=FAINT, va="center")
    ax.text(1010, 1000, sig["ts"], fontsize=14, color=FAINT,
            va="center", ha="right")
    ax.plot([70, 1010], [960, 960], color=PANEL, lw=2)

    # symbol + name
    ax.text(70, 870, sig["symbol"], fontsize=64, fontweight="bold",
            color=TXT, va="center")
    ax.text(70, 815, (sig["name"] or "")[:44], fontsize=18, color=FAINT,
            va="center")

    # direction chip + expected return
    ax.add_patch(FancyBboxPatch((70, 700), 260, 74,
                                boxstyle="round,pad=0,rounding_size=14",
                                facecolor=col, edgecolor="none"))
    ax.text(200, 737, ("LONG ▲" if long else "SHORT ▼"), fontsize=26,
            fontweight="bold", color="white", ha="center", va="center")
    er = (sig["expected_return"] or 0) * 100
    ax.text(1010, 737, f"{er:+.2f}%", fontsize=40, fontweight="bold",
            color=col, ha="right", va="center")
    ax.text(1010, 690, f"in {sig['horizon_days']} days", fontsize=14,
            color=FAINT, ha="right", va="center")

    # confidence bar
    ax.text(70, 610, "CONFIDENCE", fontsize=14, color=FAINT, va="center")
    conf = max(0.0, min(1.0, sig["confidence"] or 0))
    ax.add_patch(FancyBboxPatch((70, 560), 860, 30,
                                boxstyle="round,pad=0,rounding_size=12",
                                facecolor=PANEL, edgecolor="none"))
    if conf > 0.04:
        ax.add_patch(FancyBboxPatch((70, 560), 860 * conf, 30,
                                    boxstyle="round,pad=0,rounding_size=12",
                                    facecolor=ACCENT, edgecolor="none"))
    ax.text(1010, 575, f"{conf * 100:.0f}%", fontsize=20,
            fontweight="bold", color=TXT, ha="right", va="center")

    # entry / target / stop panel
    ax.add_patch(FancyBboxPatch((60, 300), 960, 200,
                                boxstyle="round,pad=0,rounding_size=20",
                                facecolor=PANEL, edgecolor="none"))
    cells = (("ENTRY", _fmt(sig["price"]), TXT),
             ("TARGET", _fmt(sig["target"]), UP),
             ("STOP", _fmt(sig["stop"]), DOWN))
    for i, (lab, val, c) in enumerate(cells):
        x = 60 + 960 * (i + 0.5) / 3
        ax.text(x, 450, lab, fontsize=15, color=FAINT, ha="center",
                va="center")
        ax.text(x, 385, val, fontsize=27, fontweight="bold", color=c,
                ha="center", va="center")

    # regime strip + disclaimer
    ax.text(70, 250, f"REGIME: {sig['regime'].upper()}", fontsize=14,
            color=FAINT, va="center")
    ax.plot([70, 1010], [190, 190], color=PANEL, lw=2)
    ax.text(540, 120, "Generated by KAIROS research engine.\n"
                      "Model output — not investment advice.",
            fontsize=14, color=FAINT, ha="center", va="center")

    path = str(path)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def card_for(symbol: str, out_path=None) -> tuple[str, str]:
    """(png path, caption) for one symbol; out_path defaults to data/cards."""
    sig = build_signal(symbol)
    if out_path is None:
        d = CFG.data_dir / "cards"
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / f"{sig['symbol'].replace('.', '_')}.png"
    p = render_png(sig, out_path)
    return p, caption(sig)
