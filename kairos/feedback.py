"""Prediction outcome feedback loop.

Closes the loop the way a real desk does: every prediction is snapshotted,
later resolved against realized prices, and the realized track record feeds
back into confidence calibration. Inspired by the outcome-aware feedback of
the v1 predictive trader, rebuilt on the v2 architecture.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .store import atomic_write

JOURNAL = CFG.data_dir / "prediction_journal.jsonl"

# Serialises the two writers of the journal against each other: `record` adds a
# line, `resolve_outcomes` replaces the whole file. Without it a prediction
# appended between a resolve's read and its write is overwritten out of the
# record — see `_merge_for_write`.
_WRITE_LOCK = threading.Lock()


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
    with _WRITE_LOCK, JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(_suffix_if_unterminated() + json.dumps(snap.to_dict()) + "\n")


def _suffix_if_unterminated() -> str:
    """Close a half-written last line before appending onto it.

    Both writers end their lines with a newline, so this only fires on a
    journal that arrived from outside — copied to a fresh deploy, restored,
    hand-edited. Without it the append fuses two rows into one line no parser
    accepts, and that single bad line makes the whole record unreadable: the
    resolver raises before it scores anything, the desk raises behind it, and
    the autopilot's own handler swallows both, so the track record stops
    growing while the journal still shows yesterday's rows.
    """
    try:
        size = JOURNAL.stat().st_size
        if not size:
            return ""
        with JOURNAL.open("rb") as raw:
            raw.seek(-1, os.SEEK_END)
            return "" if raw.read(1) == b"\n" else "\n"
    except OSError:
        return ""


def _direction_threshold() -> float:
    """The same line the desk trades on — recorded here so scoring and
    sizing can never drift apart."""
    return float((CFG.section("autopilot") or {}).get("entry_threshold", 0.18))


def resolve_outcomes(md) -> int:
    """Match unresolved snapshots against realized prices once the horizon has
    elapsed; rewrite the journal with outcomes. Returns count newly resolved.

    There is deliberately no lookback window here. The parameter that used to
    accept one was never read, so it promised the caller a bound on how far the
    scan reached while the code actually walked the whole journal — and the
    journal only holds rows that are still unresolved, which is the queue the
    operator is waiting on regardless of its age.
    """
    if not JOURNAL.exists():
        return 0
    lines = _read_lines()
    newly = 0
    thr = _direction_threshold()
    from .universe import resolve as resolve_inst
    for e in lines:
        if e.get("resolved"):
            continue
        inst = resolve_inst(e["symbol"])
        if inst is None:
            # Leave it pending. Resolving here used to tombstone the row as
            # "no-instrument", so one cycle in which a symbol could not be
            # looked up — a watchlist name whose dynamic registry had not been
            # rebuilt, a journal copied to a fresh deploy without it — deleted
            # every call on that symbol from the forward record permanently,
            # and stamped `directional` from the band configured that day. An
            # unanswerable lookup is a reason to retry, not a verdict.
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
        # Judge the row on the claim it carries, not on the band configured
        # today. `directional` and `correct` are read back for the rest of the
        # record's life, so deciding them from a threshold the operator can
        # still move would let next week's tuning choose which of this week's
        # snapshots ever counted — and the gate the desk actually trades
        # against is a separate runtime value that this config key does not
        # even track. The band is a fallback for rows with no label on them.
        side = _claimed_side(e)
        if side is None:
            side = 1 if e["score"] > thr else -1 if e["score"] < -thr else 0
        err = abs(realized - e["expected_return"])
        e.update(resolved=True, realized_return=round(realized, 5),
                 directional=side != 0, correct=bool(side) and side * realized > 0,
                 abs_err=round(err, 5))
        newly += 1
    if newly:
        with _WRITE_LOCK:
            # Re-read rather than trust the copy the walk started from: a
            # prediction can be appended by another thread (or another process
            # running `kairos predict`) while this loop is fetching bars, and
            # the write below replaces the entire file. Merging onto what is on
            # disk keeps the newcomer and its turn; the lock keeps an append
            # out of the gap between this read and the replace.
            atomic_write(JOURNAL, "\n".join(
                json.dumps(e) for e in _merge_for_write(lines, _read_lines())) + "\n")
    return newly


def _read_lines() -> list[dict]:
    return [json.loads(ln) for ln in
            JOURNAL.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]


def _merge_for_write(updated: list[dict], on_disk: list[dict]) -> list[dict]:
    """Apply this pass's verdicts to whatever the journal holds right now.

    Positional, not keyed: the journal is append-only, so the file's first
    `len(updated)` lines are the rows this pass read and the rest arrived while
    it worked — which is exactly the ordering to write back. Keying the merge on
    the row's own fields would fail on two snapshots of one symbol in one
    second, and a different cache entry (`use_llm` on the dashboard, off in the
    autopilot) makes that pair ordinary rather than rare.
    """
    return list(updated) + on_disk[len(updated):]


def _claimed_side(e: dict) -> int | None:
    """+1 if the row claimed upside, -1 downside, None if it claimed nothing.

    Matching on a substring rather than an exact label because the ensemble's
    vocabulary is `STRONG BUY / BUY / HOLD / SELL / STRONG SELL`, and the
    strongest band is the one the published strong-signal rate is about: an
    exact match on "BUY" silently files a winning STRONG BUY as a loss.
    """
    label = str(e.get("direction", "")).upper()
    if "BUY" in label or "LONG" in label:
        return 1
    if "SELL" in label or "SHORT" in label:
        return -1
    return None


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
    if str(e.get("direction", "")).upper() in ("HOLD", "NEUTRAL", "FLAT"):
        return False
    side = _claimed_side(e)
    if side is not None:
        return True
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
    side = _claimed_side(e)
    if side is None:
        # A row too old to carry a label still records the signed score it was
        # judged on, and the sign is the claim — unlike the band, it does not
        # depend on a threshold anyone can move later.
        score = float(e.get("score", 0.0))
        side = 1 if score > 0 else -1 if score < 0 else None
    if side is None:
        return False
    return side * float(realized) > 0


def track_record() -> dict[str, Any]:
    """Aggregate realized performance — the system's own reliability report."""
    if not JOURNAL.exists():
        return {"ok": False, "reason": "no predictions recorded yet"}
    lines = [json.loads(ln) for ln in
             JOURNAL.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
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
