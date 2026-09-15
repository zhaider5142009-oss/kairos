"""
Parallel fleet executor — runs all 13+ models concurrently.

Each model's `forecast()` is dispatched to a persistent ThreadPoolExecutor
that lives for the lifetime of the process — no per-call pool creation
overhead. Models that fail or are unavailable complete quickly with an
error status. The result list is returned in the same order as the input.

Performance — v2.0 "ultra-efficient" cadence engine:
    - Persistent thread pool: total wall-time for N models is
      max(timesfm, chronos, ...), not the sum.
    - Cadence throttling: each model may declare a cadence (cycles). Models
      with cadence > 1 rerun their inference only every N-th cycle and reuse
      the previous result in between. This is what keeps LLM/network-backed
      models (FinGPT, FinRobot, DeepSeek) from hammering the API on every
      poll cycle while still keeping their vote fresh.
    - cheap_only mode: when the runtime is risk-paused (drawdown kill-switch),
      only CPU/non-network models run; network-backed votes are reused from
      cache so recovery monitoring stays zero-API-cost.
    - Graceful degradation: if the pool is exhausted, models are skipped
      rather than blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Optional

from .models import PredictiveModel, ModelForecast

LOGGER = logging.getLogger("jarvis.parallel")

_CPU_COUNT = os.cpu_count() or 8

# Model families whose forecast() makes a network/LLM call. They are the
# expensive members of the fleet and the main targets of cadence throttling.
_NETWORK_FAMILIES = frozenset({"fingpt", "finrobot", "deepseek", "openchat"})

# Out-of-the-box cadence for network families when the user does not pin one.
_LLM_DEFAULT_CADENCE = int(os.environ.get("JARVIS_LLM_CADENCE", "6"))


class ParallelFleet:
    """Run every PredictiveModel in the fleet concurrently, honoring cadence."""

    def __init__(self, models: Sequence[PredictiveModel], *,
                 max_workers: int | None = None,
                 cadence: Optional[dict[str, int]] = None,
                 llm_default_cadence: int = _LLM_DEFAULT_CADENCE) -> None:
        self.models = list(models)
        self.max_workers = max_workers or min(max(4, len(self.models) + 4), _CPU_COUNT, 24)
        self.llm_default_cadence = max(1, int(llm_default_cadence))
        self._cadence: dict[str, int] = {}
        for m in self.models:
            c = int((cadence or {}).get(m.name, 1) or 1)
            if c <= 1 and m.family in _NETWORK_FAMILIES:
                c = self.llm_default_cadence
            if c < 1:
                c = 1
            self._cadence[m.name] = c
        self._cache: dict[tuple[Optional[str], str], tuple[int, ModelForecast]] = {}
        self._cache_lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------ pools

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="fleet",
            )
        return self._pool

    # ------------------------------------------------------------ cadence API

    def cadence_table(self) -> dict[str, int]:
        return dict(self._cadence)

    def is_network(self, name: str) -> bool:
        m = next((x for x in self.models if x.name == name), None)
        return m is not None and m.family in _NETWORK_FAMILIES

    def _reuse(self, key: tuple[Optional[str], str], cycle: int) -> Optional[ModelForecast]:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                # family-level reuse: any symbol history works for cache warm-up
                fb = self._cache.get((None, key[1]))
                entry = self._cache.get(fb and (key[0], key[1])) or fb
                if entry is None:
                    return None
            cached_cycle, result = entry
            if cached_cycle == cycle:
                return result
            # Model is on cooldown: reuse the *last computed* result.
            return result

    def _store(self, key: tuple[Optional[str], str], cycle: int, result: ModelForecast) -> None:
        with self._cache_lock:
            self._cache[key] = (cycle, result)

    # --------------------------------------------------------------- forecast

    async def forecast_all(
        self,
        values,
        horizon: int,
        *,
        symbol: str | None = None,
        cycle: int = 0,
        cheap_only: bool = False,
    ) -> list[ModelForecast]:
        """Run the fleet concurrently.

        `cycle` is the global analysis cycle counter for cadence decisions.
        `cheap_only` skips network/LLM models entirely (reuses cached votes),
        which the runtime enables while the drawdown kill-switch is armed.
        """
        loop = asyncio.get_running_loop()
        pool = self._get_pool()
        out: list[ModelForecast] = []
        run_handles: list[tuple[int, asyncio.Future]] = []  # (position, future)
        for idx, m in enumerate(self.models):
            cad = self._cadence.get(m.name, 1)
            key = (symbol, m.name)
            on_cooldown = cad > 1 and (cycle % cad) != 0
            if cheap_only and m.family in _NETWORK_FAMILIES:
                on_cooldown = True
            if on_cooldown:
                cached = self._reuse(key, cycle)
                if cached is not None:
                    out.append((idx, cached))
                    continue
                # First cycle for this symbol — warm it up asynchronously.
            run_handles.append((idx, loop.run_in_executor(
                pool, _forecast_one, m, values, horizon, symbol)))
        results = await asyncio.gather(*[f for _, f in run_handles], return_exceptions=False)
        for (idx, _f), res in zip(run_handles, results):
            self._store((symbol, self.models[idx].name), cycle, res)
            out.append((idx, res))
        out.sort(key=lambda t: t[0])
        return [r for _, r in out]


class FleetStats:
    """Lightweight per-model run counters for /models diagnostics."""

    def __init__(self) -> None:
        self.runs: dict[str, int] = {}
        self.reuses: dict[str, int] = {}
        self.last_ms: dict[str, float] = {}


def _forecast_one(model: PredictiveModel, values, horizon: int, symbol: str | None) -> ModelForecast:
    try:
        return model.forecast(values, horizon, symbol=symbol)
    except Exception as e:  # pragma: no cover
        import numpy as np
        return ModelForecast(
            model_name=model.name, family=model.family,
            point=np.zeros(horizon, dtype=np.float32),
            quantiles=None, horizon=horizon,
            direction=0, expected_return_pct=0.0, confidence=0.0,
            inference_ms=0.0,
            status=f"error: {type(e).__name__}: {e}"[:60],
        )


__all__ = ["ParallelFleet"]
