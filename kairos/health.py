"""Normalized system health, shared by the CLI and the control-tower API.

Design notes:
  * Market-data source state comes from MarketData.health(), which caches after
    the first probe, so repeated page loads don't re-hit the network.
  * LLM availability is reported at the config level (enabled + keys present),
    NOT via a live one-token call: a health endpoint must never block on a
    90-second provider timeout. Use the CLI `health` command for config truth.
  * A foundation model reaches a prediction by two independent routes — the GPU
    sidecar and the WSL bridge — so the union is what is reported. Consulting one
    route alone let a Chronos answering in 0.6 s be displayed as "WSL offline".
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


def gpu_state() -> dict[str, Any]:
    """Ask the DirectML sidecar's port, because the sidecar is a separate process
    and a health endpoint cannot infer it from anything else.

    Normalised here rather than at each reader: three surfaces used to decide on
    their own what a missing sidecar meant, and two of them concluded "no
    foundation model" from it while the third concluded "Chronos is on the WSL
    bridge" even when that bridge was down too. One shape, one meaning: `running`
    plus the reason it is not, when it is not.
    """
    try:
        from .gpu_dml import GPU
        h = GPU.health()
    except Exception as exc:
        return {"running": False, "backend": None, "model": None,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    if not h or not h.get("ok"):
        return {"running": False, "backend": None, "model": None,
                "error": (h or {}).get("error")
                or "nothing answering on the sidecar port"}
    return {"running": True, "backend": h.get("backend") or h.get("provider"),
            "model": h.get("model"), "provider": h.get("provider"),
            "batches": int(h.get("batches") or 0), "error": None}


def foundation_models(gpu: dict[str, Any],
                      wsl_ready: dict[str, bool]) -> dict[str, Any]:
    """Which foundation families can reach the next prediction, and by which route.

    The WSL bridge was the only route this question consulted, so a GPU-served
    Chronos answered in half a second while the desk printed "WSL offline" — a
    capability present but displayed as absent, which is how a model family stays
    switched off for weeks without anyone deciding to switch it off.
    """
    wsl_any = any(wsl_ready.get(k) for k in ("chronos", "timesfm", "qlib"))
    if gpu.get("running"):
        via = "gpu"
        label = f"{gpu.get('model') or 'Chronos'} on {gpu.get('backend') or 'GPU'}"
    elif wsl_any:
        via = "wsl"
        label = "WSL bridge: " + ", ".join(
            k for k in ("chronos", "timesfm", "qlib") if wsl_ready.get(k))
    else:
        via = "none"
        label = "no foundation model answering"
    return {"chronos": bool(gpu.get("running")) or bool(wsl_ready.get("chronos")),
            "timesfm": bool(wsl_ready.get("timesfm")),
            "qlib": bool(wsl_ready.get("qlib")),
            "any": via != "none", "via": via, "label": label,
            "gpu_error": gpu.get("error")}


def system_health() -> dict[str, Any]:
    from .engine import ENGINE

    src = ENGINE.md.health()
    market_data = {
        "yahoo": "ok" if src.yahoo else "down",
        "tradingview": "ok" if src.tradingview else "down",
        "psx-local": "ok" if src.psx_local else "down",
        "binance": "ok" if src.binance else "down",
    }
    # `binance` is listed but not counted below. It is the crypto *failover* and
    # the venue tape for orders, not a source of PSX or global history: letting it
    # alone turn the data banner green would advertise a live engine that has no
    # PSX quotes and no gold.
    primaries = {k: v for k, v in market_data.items() if k != "binance"}

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
    gpu = gpu_state()
    foundation = foundation_models(gpu, ready)

    data_ok = any(v in _STATE_OK for v in primaries.values())
    return {"ok": True,
            "market_data": market_data,
            "data_online": data_ok,
            "llm": llm,
            "wsl": wsl,
            "gpu": gpu,
            "foundation": foundation,
            "public_url": _public_url(),
            "universe_size": len(universe())}
