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

    from .llm import council_models, dead_seats, provider_status

    or_api = CFG.apis.get("openrouter", {}) or {}
    kimi_api = CFG.apis.get("moonshot", {}) or {}
    or_ok = bool(or_api.get("enabled", False)) and bool(CFG.openrouter_keys())
    kimi_ok = bool(kimi_api.get("enabled", False)) and bool(CFG.kimi_key())
    # Config says a key exists; only the calls know whether the account behind
    # it answers. Report both, because "kimi: true" next to a suspended account
    # is the silent rot this endpoint is supposed to make visible.
    pstat = provider_status()
    dead_key = {n: s["last_fatal"] for n, s in pstat.items() if s["last_fatal"]}
    or_live = or_ok and not dead_key.get("openrouter")
    kimi_live = kimi_ok and not dead_key.get("moonshot")
    providers = [n for n, ok in (("openrouter", or_live), ("kimi", kimi_live)) if ok]
    dead = dead_seats()
    llm = {"ok": bool(providers),
           "provider": " + ".join(providers) or "down",
           "models": len(council_models()),
           "kimi": kimi_live,
           "configured": {"openrouter": or_ok, "kimi": kimi_ok},
           "unanswered": dead_key,
           "dead_seats": dead}

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
