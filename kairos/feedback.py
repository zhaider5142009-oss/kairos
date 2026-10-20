"""Prediction outcome feedback loop.

Closes the loop the way a real desk does: every prediction is snapshotted,
later resolved against realized prices, and the realized track record feeds
back into confidence calibration. Inspired by the outcome-aware feedback of
the v1 predictive trader, rebuilt on the v2 architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .store import atomic_write

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


def _direction_threshold() -> float:
    """The same line the desk trades on — recorded here so scoring and
    sizing can never drift apart."""
    return float((CFG.section("autopilot") or {}).get("entry_threshold", 0.18))


def resolve_outcomes(md, lookback_days: int = 90) -> int:
    """Match unresolved snapshots against realized prices once the horizon has
    elapsed; rewrite the journal with outcomes. Returns count newly resolved."""
    if not JOURNAL.exists():
        return 0
    lines = [json.loads(l) for l in
             JOURNAL.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    newly = 0
    thr = _direction_threshold()
    from .universe import resolve as resolve_inst
    for e in lines:
        if e.get("resolved"):
            continue
        inst = resolve_inst(e["symbol"])
        if inst is None:
            e.update(resolved=True, outcome="no-instrument",
                     directional=bool(abs(e["score"]) > thr))
            newly += 1
            continue
        df = md.history(inst)
        if df is None or len(df) < 10:
            continue
        snap_dt = datetime.fromisoformat(e["ts"])
        idx = df.index[df["date"] >= pd.Timestamp(snap_dt.date())]
        if len(idx) == 0:
            continue
        i0 = int(idx[0])
        i1 = i0 + e["horizon"]
        if i1 > len(df) - 1:
            # The bars for this horizon have not printed yet. Clamping to the
            # last available bar scored a 5-day call on one flat bar, which
            # read as a perfect prediction.
            continue
        realized = float(df["close"].iloc[i1] / df["close"].iloc[i0] - 1)
        directional = abs(e["score"]) > thr
        pred_dir = 1 if e["score"] > thr else -1 if e["score"] < -thr else 0
        correct = bool(pred_dir != 0 and np.sign(e["score"]) == np.sign(realized))
        err = abs(realized - e["expected_return"])
        e.update(resolved=True, realized_return=round(realized, 5),
                 directional=directional, correct=correct, abs_err=round(err, 5))
        newly += 1
    if newly:
        atomic_write(JOURNAL, "\n".join(json.dumps(e) for e in lines) + "\n")
    return newly


def _is_directional(e: dict) -> bool:
    """Whether a resolved row was an actual call, not a HOLD.

    Rows written before this distinction existed carry no flag. Their *label*
    is what was actually claimed that day, so the label is the source of truth
    and the score is only a fallback for rows too old to have either. Reusing
    today's entry_threshold instead would let a config change reach backwards:
    lower it below the largest legacy HOLD score and 140 non-trades become 140
    directional calls — each still stamped correct=True by the pre-fix scorer —
    and the published hit rate reads 100% on a journal containing no calls.
    """
    if "directional" in e:
        return bool(e["directional"])
    label = str(e.get("direction", "")).upper()
    if label in ("BUY", "SELL", "LONG", "SHORT"):
        return True
    if label in ("HOLD", "NEUTRAL", "FLAT"):
        return False
    return abs(float(e.get("score", 0.0))) > _direction_threshold()


def _was_correct(e: dict) -> bool:
    """Whether a matured call was right, without trusting a legacy column.

    For flagged rows `correct` was written by the current scorer and means what
    it says. For unflagged rows it was written by the old accuracy rule — a
    HOLD inside the no-trade band scored as a win — so it is discarded and the
    verdict recomputed from the direction that was claimed against the sign of
    the return that actually happened. No threshold involved, so no config edit
    can rewrite last month's record.
    """
    if "directional" in e:
        return bool(e.get("correct"))
    realized = e.get("realized_return")
    if realized is None:
        return False
    label = str(e.get("direction", "")).upper()
    if label in ("BUY", "LONG"):
        return float(realized) > 0
    if label in ("SELL", "SHORT"):
        return float(realized) < 0
    return False


def track_record() -> dict[str, Any]:
    """Aggregate realized performance — the system's own reliability report."""
    if not JOURNAL.exists():
        return {"ok": False, "reason": "no predictions recorded yet"}
    lines = [json.loads(l) for l in
             JOURNAL.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    done = [e for e in lines if e.get("resolved")
            and e.get("realized_return") is not None]
    if not done:
        return {"ok": True, "predictions": len(lines), "resolved": 0,
                "hit_rate": None, "reason": "no horizons elapsed yet"}
    calls = [e for e in done if _is_directional(e)]
    n_hold = len(done) - len(calls)
    if not calls:
        return {"ok": True, "predictions": len(lines), "resolved": len(done),
                "directional_resolved": 0, "non_directional_resolved": n_hold,
                "hit_rate": None, "strong_signal_hit_rate": None,
                "mean_abs_error": round(float(np.mean([e["abs_err"] for e in done])), 5),
                "calibration_by_confidence": {},
                "reason": "every resolved snapshot was a HOLD — no directional "
                          "call has matured yet",
                # An empty list, not the HOLD rows. A non-trade shown under a
                # "correct ✓" column reads as evidence of skill; it is evidence
                # of nothing, and a diligence reader is entitled to not be
                # shown it as if it were a call.
                "recent": []}
    hits = sum(1 for e in calls if _was_correct(e))
    strong = [e for e in calls if abs(e["score"]) > 0.5]
    strong_h = sum(1 for e in strong if _was_correct(e))
    mae = float(np.mean([e["abs_err"] for e in calls]))
    # calibration: realized hit-rate per confidence bucket
    buckets: dict[str, list] = {}
    for e in calls:
        b = f"{int(e['confidence'] * 5) / 5:.1f}"
        buckets.setdefault(b, []).append(1.0 if _was_correct(e) else 0.0)
    calib = {k: round(float(np.mean(v)), 3) for k, v in sorted(buckets.items())
             if len(v) >= 3}
    return {"ok": True, "predictions": len(lines), "resolved": len(done),
            "directional_resolved": len(calls), "non_directional_resolved": n_hold,
            "hit_rate": round(hits / len(calls), 3),
            "strong_signal_hit_rate": round(strong_h / len(strong), 3) if strong else None,
            "mean_abs_error": round(mae, 5),
            "calibration_by_confidence": calib,
            "recent": [dict(e, correct=_was_correct(e)) for e in calls[-8:]]}


def confidence_adjustment() -> float:
    """Multiplier applied to live confidence based on realized track record.
    Institutional self-awareness: shrink when we're wrong, keep when right."""
    tr = track_record()
    if not tr.get("ok") or not tr.get("directional_resolved"):
        return 1.0
    if tr["directional_resolved"] < 10 or tr["hit_rate"] is None:
        return 1.0          # no matured calls yet is not evidence of either way
    hr = tr["hit_rate"]
    if hr >= 0.60:
        return 1.05
    if hr >= 0.52:
        return 1.0
    if hr >= 0.45:
        return 0.90
    return 0.75
