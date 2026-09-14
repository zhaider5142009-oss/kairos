"""
Configuration loader.

Single source of truth: config.yaml next to the project root.
Override path with JARVIS_CONFIG env var or via JarvisConfig.load(path).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Locate the Jarvis project root.

    Order:
      1. JARVIS_ROOT environment variable
      2. Walk upward from this file (jarvis_predict/config.py -> ../.. = project root)
      3. CWD
    """
    env = os.environ.get("JARVIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "config.yaml").is_file():
            return ancestor
    return Path.cwd()


@dataclass(frozen=True)
class JarvisConfig:
    """Immutable configuration object. All paths resolved at load()."""

    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "JarvisConfig":
        if path is None:
            path = _project_root() / "config.yaml"
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls(raw=data)

    # --- convenient accessors -------------------------------------------------

    @property
    def root(self) -> Path:
        return Path(self.raw["paths"]["root"]).expanduser().resolve()

    @property
    def logs_dir(self) -> Path:
        return Path(self.raw["paths"]["logs_dir"]).expanduser().resolve()

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw["paths"]["cache_dir"]).expanduser().resolve()

    def section(self, *keys: str) -> dict[str, Any]:
        """Drill into nested sections with type-checked access."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"Missing config section: {'.'.join(keys)}")
            node = node[k]
        if not isinstance(node, dict):
            raise ValueError(f"Config section {'.'.join(keys)} is not a mapping")
        return node

    # --- helpers --------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check required sections exist with sane types. Returns a list of
        warning strings (empty = everything OK). Does not raise — callers
        can log warnings and continue, or raise for strict mode."""
        warnings: list[str] = []
        for section in ("mt5", "strategy", "ensemble", "timesfm",
                        "listener", "logging", "paths"):
            if not isinstance(self.raw.get(section), dict):
                warnings.append(f"Missing config section: {section}")
        mt5 = self.raw.get("mt5", {})
        order = mt5.get("order", {})
        if not isinstance(order.get("max_open_positions"), int) or order["max_open_positions"] < 1:
            warnings.append("mt5.order.max_open_positions must be int >= 1; defaulting to 3")
            self.raw.setdefault("mt5", {}).setdefault("order", {})["max_open_positions"] = 3
        if not isinstance(order.get("max_daily_loss"), (int, float)):
            warnings.append("mt5.order.max_daily_loss missing; defaulting to 50.0")
            self.raw.setdefault("mt5", {}).setdefault("order", {})["max_daily_loss"] = 50.0
        strat = self.raw.get("strategy", {})
        for key, default, lo in [("max_positions", 5, 1), ("max_drawdown_pct", 10.0, 1.0)]:
            val = strat.get(key)
            if not isinstance(val, (int, float)) or val < lo:
                warnings.append(f"strategy.{key} invalid; defaulting to {default}")
                strat[key] = default
        paths = self.raw.get("paths", {})
        for key in ("root", "logs_dir", "cache_dir"):
            if not paths.get(key):
                warnings.append(f"paths.{key} missing")
        return warnings

    def ensure_dirs(self) -> None:
        """Create on-disk directories that the config references. Idempotent."""
        for key in ("logs_dir", "cache_dir"):
            getattr(self, key).mkdir(parents=True, exist_ok=True)
        models = self.raw["paths"].get("model_cache")
        if models:
            Path(models).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def setup_logging(config: JarvisConfig) -> None:
    """Configure root logger with rotating file handler + console."""
    import logging
    from logging.handlers import RotatingFileHandler

    level = getattr(logging, config.raw["logging"]["level"].upper(), logging.INFO)
    rotate_mb = max(1, int(config.raw["logging"]["rotate_mb"]))
    keep = max(1, int(config.raw["logging"]["keep_files"]))
    config.ensure_dirs()

    root = logging.getLogger()
    root.setLevel(level)
    # Strip pre-existing handlers (re-entrancy safe)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = config.logs_dir / "jarvis.log"
    fh = RotatingFileHandler(
        log_file,
        maxBytes=rotate_mb * 1024 * 1024,
        backupCount=keep,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.info("Logging initialised at %s -> %s", level, log_file)


__all__ = ["JarvisConfig", "setup_logging"]
