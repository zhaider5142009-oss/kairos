"""Prediction outcome feedback loop.

Closes the loop the way a real desk does: every prediction is snapshotted,
later resolved against realized prices, and the realized track record feeds
back into confidence calibration. Inspired by the outcome-aware feedback of
the v1 predictive trader, rebuilt on the v2 architecture.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG

JOURNAL = CFG.data_dir / "prediction_journal.jsonl"


@dataclass
class Snapshot:
    ts: str
    symbol: str
    horizon: int
    score: float
    confidence: float
    expected_return: float
    price: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "symbol": self.symbol, "horizon": self.horizon,
                "score": self.score, "confidence": self.confidence,
                "expected_return": self.expected_return, "price": self.price,
                "direction": self.direction, "resolved": False}


def record(res: dict[str, Any]) -> None:
    """Persist a prediction snapshot (called once per fresh, non-cached predict)."""
    if not res.get("ok"):
        return
    sig = res["signal"]
    snap = Snapshot(
        ts=datetime.now().isoformat(timespec="seconds"),
        symbol=res["symbol"], horizon=sig["horizon_days"], score=sig["score"],
        confidence=sig["confidence"], expected_return=sig["expected_return"],
        price=res["price"], direction=sig["direction"])
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap.to_dict()) + "\n")


def resolve_outcomes(md, lookback_days: int = 90) -> int:
    """Match unresolved snapshots against realized prices once the horizon has
    elapsed; rewrite the journal with outcomes. Returns count newly resolved."""
    if not JOURNAL.exists():
        return 0
    lines = [json.loads(l) for l in
             JOURNAL.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    newly = 0
    from .universe import resolve as resolve_inst
    for e in lines:
        if e.get("resolved"):
            continue
        age = (datetime.now() - datetime.fromisoformat(e["ts"])).days
        if age < int(np.ceil(e["horizon"] / 5)) + 1:
            continue                      # horizon not elapsed (calendar approx)
        inst = resolve_inst(e["symbol"])
        if inst is None:
            e.update(resolved=True, outcome="no-instrument")
            newly += 1
            continue
        df = md.history(inst)
        if df is None or len(df) < 10:
            continue
        # realized = close nearest to (snapshot date + horizon trading days)
        snap_dt = datetime.fromisoformat(e["ts"])
        idx = df.index[df["date"] >= pd.Timestamp(snap_dt.date())]
        if len(idx) == 0:
            continue
        i0 = int(idx[0])
        i1 = min(i0 + e["horizon"], len(df) - 1)
        realized = float(df["close"].iloc[i1] / df["close"].iloc[i0] - 1)
        pred_dir = 1 if e["score"] > 0.18 else -1 if e["score"] < -0.18 else 0
        correct = (pred_dir == 0) or (np.sign(e["score"]) == np.sign(realized))
        err = abs(realized - e["expected_return"])
        e.update(resolved=True, realized_return=round(realized, 5),
                 correct=bool(correct), abs_err=round(err, 5))
        newly += 1
    if newly:
        JOURNAL.write_text("\n".join(json.dumps(e) for e in lines) + "\n",
                           encoding="utf-8")
    return newly


def track_record() -> dict[str, Any]:
    """Aggregate realized performance — the system's own reliability report."""
    if not JOURNAL.exists():
        return {"ok": False, "reason": "no predictions recorded yet"}
    lines = [json.loads(l) for l in
             JOURNAL.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    done = [e for e in lines if e.get("resolved") and e.get("realized_return") is not None]
    if not done:
        return {"ok": True, "predictions": len(lines), "resolved": 0,
                "hit_rate": None, "reason": "no horizons elapsed yet"}
    hits = sum(1 for e in done if e["correct"])
    strong = [e for e in done if abs(e["score"]) > 0.5]
    strong_h = sum(1 for e in strong if e["correct"])
    mae = float(np.mean([e["abs_err"] for e in done])) if done else None
    # calibration: realized hit-rate per confidence bucket
    buckets: dict[str, list] = {}
    for e in done:
        b = f"{int(e['confidence'] * 5) / 5:.1f}"
        buckets.setdefault(b, []).append(1.0 if e["correct"] else 0.0)
    calib = {k: round(float(np.mean(v)), 3) for k, v in sorted(buckets.items())
             if len(v) >= 3}
    return {"ok": True, "predictions": len(lines), "resolved": len(done),
            "hit_rate": round(hits / len(done), 3),
            "strong_signal_hit_rate": round(strong_h / len(strong), 3) if strong else None,
            "mean_abs_error": round(mae, 5) if mae is not None else None,
            "calibration_by_confidence": calib,
            "recent": done[-8:]}


def confidence_adjustment() -> float:
    """Multiplier applied to live confidence based on realized track record.
    Institutional self-awareness: shrink when we're wrong, keep when right."""
    tr = track_record()
    if not tr.get("ok") or not tr.get("resolved") or tr["resolved"] < 10:
        return 1.0
    hr = tr["hit_rate"] or 0.5
    if hr >= 0.60:
        return 1.05
    if hr >= 0.52:
        return 1.0
    if hr >= 0.45:
        return 0.90
    return 0.75
