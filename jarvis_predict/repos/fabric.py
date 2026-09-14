"""RepoFabric — discovers local quant/trading repos and exposes them as adapters."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Dict, List, Optional

from .adapter import RepoAdapter
from .adapters import ADAPTER_TYPES, DEFAULT_PATHS


class RepoFabric:
    """Central integration point for all external algo-trading repos."""

    def __init__(self, paths: Optional[Dict[str, str]] = None,
                 enabled: Optional[List[str]] = None,
                 max_probe_workers: int = 5):
        self.paths: Dict[str, str] = {}
        for name, p in DEFAULT_PATHS.items():
            self.paths[name] = (paths or {}).get(name, p)
        self.enabled = enabled or list(ADAPTER_TYPES.keys())
        self.max_probe_workers = int(max_probe_workers)
        self.adapters: Dict[str, RepoAdapter] = {}
        self._build()

    def _build(self) -> None:
        for name in self.enabled:
            cls = ADAPTER_TYPES.get(name)
            if not cls:
                continue
            self.adapters[name] = cls(self, {"path": self.paths.get(name)})

    def get(self, name: str) -> Optional[RepoAdapter]:
        return self.adapters.get(name)

    def probe_all(self, max_workers: Optional[int] = None) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        workers = max_workers or self.max_probe_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(a.probe): name for name, a in self.adapters.items()}
            for fut, name in futs.items():
                try:
                    out[name] = fut.result(timeout=60)
                except Exception as e:  # noqa: BLE001
                    out[name] = {"error": str(e)}
        return out

    def summary(self) -> List[dict]:
        return [a.summary() for a in self.adapters.values()]

    def available(self) -> List[str]:
        return [n for n, a in self.adapters.items() if a.available()]

    def jobs(self) -> Dict[str, dict]:
        """Readiness claims each adapter makes (no probing on hot path)."""
        return {n: a.status() for n, a in self.adapters.items()}


def create_fabric(data_dir: Optional[str] = None) -> RepoFabric:
    return RepoFabric()