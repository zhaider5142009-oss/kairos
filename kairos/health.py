"""Normalized system health, shared by the CLI and the control-tower API.

Design notes:
  * Market-data source state comes from MarketData.health(), which caches after
    the first probe, so repeated page loads don't re-hit the network.
  * LLM availability is reported at the config level (enabled + keys present),
    NOT via a live one-token call: a health endpoint must never block on a
    90-second provider timeout. Use the CLI `health` command for config truth.
"""

from __future__ import annotations

from typing import Any

from .config import CFG
from .universe import universe
from .wsl_bridge import WSL

_STATE_OK = ("ok", "up", "degraded")


def _public_url() -> str:
    """Written by tools/keepalive.ps1 — the empty string means the tunnel is
    down or still registering, which is exactly what the desk should show."""
    try:
        return (CFG.data_dir / "tunnel_url.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def system_health() -> dict[str, Any]:
    from .engine import ENGINE

    src = ENGINE.md.health()
    market_data = {
        "yahoo": "ok" if src.yahoo else "down",
        "tradingview": "ok" if src.tradingview else "down",
        "psx-local": "ok" if src.psx_local else "down",
    }

    or_api = CFG.apis.get("openrouter", {}) or {}
    enabled = bool(or_api.get("enabled", False)) and bool(CFG.openrouter_keys())
    models = len(CFG.openrouter_models())
    llm = {"ok": enabled,
           "provider": "openrouter" if enabled else "down",
           "models": models}

    ready = WSL.models_ready()
    wsl = {"available": WSL.wsl_available(),
           "models_ready": any(ready.values()),
           "detail": ready}

    data_ok = any(v in _STATE_OK for v in market_data.values())
    return {"ok": True,
            "market_data": market_data,
            "data_online": data_ok,
            "llm": llm,
            "wsl": wsl,
            "public_url": _public_url(),
            "universe_size": len(universe())}
