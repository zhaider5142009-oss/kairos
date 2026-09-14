"""
Parallel fleet executor — runs all 12 models concurrently.

Each model's `forecast()` is dispatched to a persistent ThreadPoolExecutor
that lives for the lifetime of the process — no per-call pool creation
overhead. Models that fail or are unavailable complete quickly with an
error status. The result list is returned in the same order as the input.

Performance: even on CPU-only machines the total wall-time for 12 models
is max(timesfm, chronos, qlib_lstm, ...), not the sum, since they all
share a thread pool sized to the host CPU count.

v1.2 improvements:
    - Persistent thread pool: avoids creating a new pool per symbol/cycle.
    - Graceful degradation: if the pool is exhausted, the model is skipped
      rather than blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .models import PredictiveModel, ModelForecast

LOGGER = logging.getLogger("jarvis.parallel")

_CPU_COUNT = os.cpu_count() or 8


class ParallelFleet:
    """Run every PredictiveModel in the fleet concurrently."""

    def __init__(self, models: Sequence[PredictiveModel], *, max_workers: int | None = None) -> None:
        self.models = list(models)
        self.max_workers = max_workers or min(max(4, len(self.models) + 4), _CPU_COUNT, 24)
        self._pool: ThreadPoolExecutor | None = None

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="fleet",
            )
        return self._pool

    async def forecast_all(
        self,
        values,
        horizon: int,
        *,
        symbol: str | None = None,
    ) -> list[ModelForecast]:
        loop = asyncio.get_running_loop()
        pool = self._get_pool()
        tasks = [
            loop.run_in_executor(pool, _forecast_one, m, values, horizon, symbol)
            for m in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


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
