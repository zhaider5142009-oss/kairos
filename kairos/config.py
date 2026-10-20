"""Configuration loader — single source of truth for keys, endpoints, tunables.

Resolution order for the config file: $KAIROS_CONFIG → ./config.yaml (cwd) →
repo root. The file is OPTIONAL: without it the engine runs on built-in
defaults, environment-variable keys, and public data sources. Put real keys in
config.yaml (git-ignored) or env vars — never in tracked source.
Runtime state (data/, reports/) lives next to the config file, or in the
current directory when no config file is present. The file is re-read when its
mtime changes, so a risk limit or an arming key edited while the tower runs
takes effect on the next use instead of the next restart.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

# how often a config read may cost a stat(); the tower runs for days
_RELOAD_CHECK_S = 1.0


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
        if found is not None and found.is_file():
            with open(found, "r", encoding="utf-8") as fh:
                self._raw = yaml.safe_load(fh) or {}
        self.root = found.parent if found is not None else Path.cwd()
        app = self._raw.get("app") or {}
        self.data_dir = self.root / app.get("data_dir", "data")
        self.cache_dir = self.root / app.get("cache_dir", "data/cache")
        self.reports_dir = self.root / app.get("reports_dir", "reports")
        for d in (self.data_dir, self.cache_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._file_mtime = self._stat()
        self._next_check = time.monotonic() + _RELOAD_CHECK_S

    def _stat(self) -> float | None:
        if self.path is None:
            return None
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def _maybe_reload(self) -> None:
        """Pick up an edited config.yaml without a restart.

        Sections read at decision time (risk limits, arming keys, thresholds)
        then bind on the next use. The app: paths are deliberately excluded —
        data_dir/cache_dir/reports_dir are bound into objects at import, so
        changing them still needs a restart.
        """
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + _RELOAD_CHECK_S
        mtime = self._stat()
        if mtime is None or mtime == self._file_mtime:
            return
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            # a half-written or broken file must not take the gate down: keep
            # the last good config and wait for the next edit.
            self._file_mtime = mtime
            return
        self._file_mtime = mtime
        self._raw = raw

    # -- accessors ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        self._maybe_reload()
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

    # -- writer -------------------------------------------------------------
    def store_secret(self, section: str, key: str, value: Any) -> bool:
        """Set one scalar inside a top-level block of the live config file.

        Line surgery rather than a yaml round-trip: config.yaml carries the
        operator's comments, and re-dumping it would silently erase them. The
        value is written to disk and never logged, journalled or returned —
        this is the only path a broker PIN should ever take.

        Returns False when there is no config file to write to.
        """
        if self.path is None or not self.path.is_file():
            return False
        # json scalars are valid YAML: a quoted PIN keeps its leading zeros
        text = (str(value).lower() if isinstance(value, bool)
                else str(value) if isinstance(value, (int, float))
                else json.dumps(str(value)))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        try:
            start = next(n for n, l in enumerate(lines)
                         if l.rstrip() == f"{section}:")
        except StopIteration:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"{section}:")
            lines.append(f"  {key}: {text}")
        else:
            end = start + 1
            while end < len(lines) and (not lines[end].strip()
                                        or lines[end].startswith((" ", "\t"))):
                end += 1
            hit = next((n for n in range(start + 1, end)
                        if lines[n].strip().startswith(f"{key}:")), None)
            if hit is not None:
                indent = lines[hit][:len(lines[hit]) - len(lines[hit].lstrip())]
                lines[hit] = f"{indent}{key}: {text}"
            else:
                at = end
                while at > start + 1 and not lines[at - 1].strip():
                    at -= 1
                indent = (lines[start + 1][:len(lines[start + 1])
                            - len(lines[start + 1].lstrip())]
                          if end > start + 1 and lines[start + 1].strip() else "  ")
                lines.insert(at, f"{indent}{key}: {text}")
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._file_mtime = self._stat()
        self._next_check = time.monotonic() + _RELOAD_CHECK_S
        return True

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


CFG = Config.get()
