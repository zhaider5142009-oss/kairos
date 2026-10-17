"""Configuration loader — single source of truth for keys, endpoints, tunables.

Resolution order for the config file: $KAIROS_CONFIG → ./config.yaml (cwd) →
repo root. The file is OPTIONAL: without it the engine runs on built-in
defaults, environment-variable keys, and public data sources. Put real keys in
config.yaml (git-ignored) or env vars — never in tracked source.
Runtime state (data/, reports/) lives next to the config file, or in the
current directory when no config file is present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _find_config() -> Path | None:
    env = os.environ.get("KAIROS_CONFIG")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    for p in (Path.cwd() / "config.yaml", ROOT / "config.yaml"):
        if p.is_file():
            return p
    return None


class Config:
    _instance: "Config | None" = None

    def __init__(self, path: Path | None = None) -> None:
        found = path or _find_config()
        self.path = found
        self._raw: dict[str, Any] = {}
        if found is not None:
            with open(found, "r", encoding="utf-8") as fh:
                self._raw = yaml.safe_load(fh) or {}
        self.root = found.parent if found is not None else Path.cwd()
        app = self._raw.get("app") or {}
        self.data_dir = self.root / app.get("data_dir", "data")
        self.cache_dir = self.root / app.get("cache_dir", "data/cache")
        self.reports_dir = self.root / app.get("reports_dir", "reports")
        for d in (self.data_dir, self.cache_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- accessors ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return self._raw.get(name, {}) or {}

    @property
    def apis(self) -> dict[str, Any]:
        return self.section("apis")

    @property
    def engine(self) -> dict[str, Any]:
        return self.section("engine")

    @property
    def risk(self) -> dict[str, Any]:
        return self.section("risk")

    @property
    def quantum(self) -> dict[str, Any]:
        return self.section("quantum")

    def openrouter_keys(self) -> list[str]:
        keys = self.apis.get("openrouter", {}).get("keys", [])
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key and env_key not in keys:
            keys = [env_key, *keys]
        return [k for k in keys if k]

    def openrouter_models(self) -> list[str]:
        return self.apis.get("openrouter", {}).get("council_models", [])

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


CFG = Config.get()
