"""
Connector registry — builds the full connector layer from the catalog and
keeps health fresh with a background sweep.

    registry = ConnectorRegistry.from_config(config)
    registry.start()              # background health sweeper
    registry.snapshot()           # grouped table for /connectors
    registry.summary()            # counts by status
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from .base import (
    CATEGORIES, STATUS_OK, STATUS_UNCONFIGURED, STATUS_OFFLINE,
    Connector, ConnectorHealth,
)
from .catalog import catalog
from .impl import build_connector

LOGGER = logging.getLogger("jarvis.connectors.registry")

# Options that mains-config sections may inject (indexed by the section).
_CONFIG_SECTIONS = ("openrouter", "mt5")
_ENV_KEYS = ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")


class ConnectorRegistry:
    """All connectors, keyed by name, with a lazy background health sweep."""

    def __init__(self, config=None) -> None:
        self.config = config
        self._connectors: dict[str, Connector] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_sweep = 0.0
        self._sweep_interval = 45.0

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(cls, config=None) -> "ConnectorRegistry":
        """Build every catalog connector, resolving keys from config + env."""
        reg = cls(config)
        cn_section = {}
        if config is not None:
            try:
                cn_section = config.section("connectors") or {}
            except Exception:
                cn_section = {}
            if not isinstance(cn_section, dict):
                cn_section = {}

        # openrouter keys + MT5 path from the main config (shared sources).
        shared: dict[str, Any] = {}
        if config is not None:
            try:
                shared["openrouter_api_keys"] = list(
                    config.section("openrouter").get("api_keys", []) or [])
            except Exception:
                shared["openrouter_api_keys"] = []
            try:
                shared["mt5_terminal_path"] = str(
                    config.section("mt5").get("terminal_path",
                        "C:/Program Files/MetaTrader 5/terminal64.exe"))
            except Exception:
                shared["mt5_terminal_path"] = "C:/Program Files/MetaTrader 5/terminal64.exe"
        env_keys = [k for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
        env_single = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if env_keys:
            shared["openrouter_api_keys"] = env_keys + shared["openrouter_api_keys"]
        elif env_single:
            shared["openrouter_api_keys"] = [env_single] + shared["openrouter_api_keys"]

        for spec in catalog():
            spec = dict(spec)
            kind = spec.get("kind", "http")
            if kind == "openrouter":
                spec["api_keys"] = shared.get("openrouter_api_keys", [])
            if kind == "metatrader5":
                spec["terminal_path"] = shared.get("mt5_terminal_path",
                          "C:/Program Files/MetaTrader 5/terminal64.exe")
            if kind in ("key", "anthropic", "gemini", "tiingo", "telegram"):
                spec["api_key"] = _resolve_secret(spec, cn_section)
            if kind in ("discord", "slack", "webhook"):
                spec["webhook_url"] = _resolve_secret(
                    spec, cn_section, secret_key="url", env_key="")
            if kind == "smtp":
                spec["host"] = _resolve_secret(
                    spec, cn_section, secret_key="host", env_key="")
                spec["port"] = int((cn_section.get("email_smtp") or {}).get("port", 587))
            if kind == "openrouter":  # ensure placeholder junk never counts
                spec["api_keys"] = [k for k in spec["api_keys"] if k and "your_actual" not in k]
            connector = build_connector(spec["key"], spec)
            reg._connectors[connector.key] = connector

        LOGGER.info("Connector layer built: %d connectors / %d categories",
                    len(reg._connectors), len(CATEGORIES))
        return reg

    # -- public API -----------------------------------------------------------

    def get(self, key: str) -> Connector | None:
        return self._connectors.get(key)

    def keys(self) -> list[str]:
        return sorted(self._connectors)

    def all(self) -> list[Connector]:
        return sorted(self._connectors.values(), key=lambda c: c.key)

    def by_category(self) -> dict[str, list[Connector]]:
        out: dict[str, list[Connector]] = {}
        for c in self.all():
            out.setdefault(c.category, []).append(c)
        return out

    def start(self) -> None:
        """Begin the background health sweep (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sweep_loop,
                                        name="jarvis-connectors", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def sweep(self, force: bool = False) -> dict[str, ConnectorHealth]:
        """Check every stale connector concurrently (respecting per-connector
        TTLs). Never blocks a request path — runs in the background thread or
        on an explicit call. Returns the updated health map."""
        snapshot: dict[str, ConnectorHealth] = {}
        now = time.time()
        conns = [c for c in self.all() if not self._stop.is_set()]
        workers = min(8, len(conns) or 1)

        def _check(c):
            if self._stop.is_set():
                return c.key, None
            try:
                return c.key, c.check(force=force)
            except Exception:  # absolute safety
                return c.key, ConnectorHealth(status=STATUS_OFFLINE,
                                              detail="probe error")

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="jarvis-conn") as pool:
                for key, h in pool.map(_check, conns):
                    if h is not None:
                        snapshot[key] = h
        else:
            for key, h in map(_check, conns):
                if h is not None:
                    snapshot[key] = h
        self._last_sweep = now
        return snapshot

    def summary(self) -> dict[str, int]:
        """Counts by status (cached — no probing on request paths)."""
        counts = {"total": 0, "ok": 0, "degraded": 0, "unconfigured": 0,
                  "offline": 0, "mock": 0, "unknown": 0}
        for c in self.all():
            counts["total"] += 1
            counts[c.cached_health().status] = counts.get(
                c.cached_health().status, 0) + 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        """Grouped table for /connectors + the Connectors UI view.

        Instant: reads cached health (no inline network probes); the
        background sweep keeps the cache warm."""
        by_cat = self.by_category()
        cats = []
        for key, label in CATEGORIES:
            conns = by_cat.get(key, [])
            if not conns:
                continue
            ok = sum(1 for c in conns if c.cached_health().status == STATUS_OK)
            cats.append({
                "key": key,
                "label": label,
                "count": len(conns),
                "ok": ok,
                "connectors": [c.snapshot(probe=False) for c in conns],
            })
        return {
            "total": len(self._connectors),
            "categories": cats,
            "summary": self.summary(),
            "last_sweep": self._last_sweep,
        }

    # -- background loop ------------------------------------------------------

    def _sweep_loop(self) -> None:
        # First pass immediately (fast connectors only mark themselves),
        # then walk the network ones once so health is fresh after boot.
        self.sweep(force=True)
        while not self._stop.wait(self._sweep_interval):
            # Non-forcing sweep honors per-connector TTLs.
            self.sweep()


def _resolve_secret(spec: dict, cn_section: dict,
                    secret_key: str = "api_key", env_key: str | None = None) -> str:
    """Look up a secret from config.connectors.<key>.<secret_key> then env.

    env_key of "" means no env fallback (url/host connectors).
    """
    key = spec.get("key", "")
    env_key = env_key if env_key is not None else spec.get("key_env", "")
    configured = (cn_section.get(key) or {}).get(secret_key, "") if isinstance(cn_section, dict) else ""
    configured = str(configured or "").strip()
    if configured and configured not in ("x", "KEY"):
        return configured
    if env_key:
        val = os.environ.get(env_key, "").strip()
        if val and val not in ("your_actual_api_key_here", "xxx"):
            return val
    return ""


__all__ = ["ConnectorRegistry"]