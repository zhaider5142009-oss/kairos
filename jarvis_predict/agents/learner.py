"""
Learning layer — rolling performance tracker for every prediction source.

The swarm is not just coordinated, it *learns*. The ModelLearner keeps a
sliding-window history of (prediction → realised outcome) for each model
and agent, computes hit-rates / information ratios, and produces adaptive
weight multipliers. These multipliers feed back into:

    - the ensemble's per-model weighting (so models that have been right
      recently count more),
    - the swarm consensus director (so agents that have been right count
      more).

The tracker is thread-safe, cheap (one dict + rolling lists), and fully
deterministic in offline mode (synthetic benchmark series in tests).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from typing import Any

LOGGER = logging.getLogger("jarvis.learn")


class ModelLearner:
    """Sliding-window accuracy tracker + adaptive weight multiplier producer."""

    # Relative weight of each realized step for agents vs models.
    def __init__(self, *, window: int = 120, min_samples: int = 8) -> None:
        self.window = max(16, int(window))
        self.min_samples = max(3, int(min_samples))
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=self.window))
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ record

    def observe(
        self,
        key: str,                 # model name or agent name
        direction: int,           # predicted direction (-1/0/+1)
        confidence: float,        # 0..1
        realized_return_pct: float,  # % the asset actually moved after the call
    ) -> None:
        """Record one prediction + its realised outcome."""
        if direction == 0:
            return
        hit = 1.0 if (direction > 0 and realized_return_pct > 0) or \
                     (direction < 0 and realized_return_pct < 0) else 0.0
        magnitude = abs(confidence) * abs(realized_return_pct)
        with self._lock:
            self._history[key].append({
                "hit": hit,
                "magnitude": magnitude,
                "t": time.time(),
            })

    # ------------------------------------------------------------------ read

    def metrics(self, key: str) -> dict[str, float]:
        """Recent accuracy stats for one model/agent. Empty dict if unknown."""
        with self._lock:
            h = list(self._history.get(key, ()))
        if not h:
            return {}
        n = len(h)
        hits = sum(1 for e in h if e["hit"])
        hit_rate = hits / n
        # Mean profit factor — magnitude-weighted direction score.
        signed = sum((e["hit"] * 2 - 1) * e["magnitude"] for e in h)
        return {
            "n": float(n),
            "hit_rate": round(hit_rate, 4),
            "profit_factor": round(signed / max(1e-9, n), 4),
        }

    def multiplier(self, key: str) -> float:
        """Adaptive weight multiplier in [0.25, 1.75]. Defaults to 1.0."""
        with self._lock:
            h = list(self._history.get(key, ()))
        if len(h) < self.min_samples:
            return 1.0
        n = len(h)
        hits = sum(1 for e in h if e["hit"])
        hit_rate = hits / n
        mean_conf = sum(e["magnitude"] for e in h) / n
        # Scale toward the edges based on hit-rate vs. 0.5 baseline.
        # Confident-and-correct → up to 1.75. Confident-and-wrong → 0.25.
        offset = (hit_rate - 0.5) * 2.0
        conf_boost = min(1.0, mean_conf) * 0.35
        raw = 1.0 + offset * 0.6 + (conf_boost if offset > 0 else -conf_boost)
        return max(0.25, min(1.75, raw))

    def all_multipliers(self, keys: list[str], prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        with self._lock:
            known = set(self._history.keys())
        for k in keys:
            if k in known:
                out[k] = self.multiplier(k)
        if prefix:
            out = {f"{prefix}{k}": v for k, v in out.items()}
        return out

    def summary(self) -> dict[str, Any]:
        """Per-key metric snapshot for the /agents endpoint."""
        with self._lock:
            keys = sorted(self._history.keys())
        return {k: self.metrics(k) for k in keys}

    # ------------------------------------------------------------------ misc

    def prune(self, older_than_s: float = 3600.0) -> int:
        """Drop entries older than `older_than_s` seconds. Returns removed count."""
        cutoff = time.time() - older_than_s
        removed = 0
        with self._lock:
            for k in list(self._history.keys()):
                dq = self._history[k]
                before = len(dq)
                while dq and dq[0]["t"] < cutoff:
                    dq.popleft()
                removed += before - len(dq)
                if not dq:
                    del self._history[k]
        return removed


# small stat helpers (avoid extra deps)
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


__all__ = ["ModelLearner", "_clamp"]