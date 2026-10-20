"""The health page must not call a suspended account live.

`/api/health` is the one endpoint the investor pack points a reviewer at for
provider rot ("the health endpoint reports dead_seats so rot is visible instead
of silent"). It was built entirely from config: a key present and enabled made
`kimi: true`, and the catalog check that finds retired slugs has nothing to say
about an account that is suspended. So the failure that actually happened — the
direct Moonshot key answering every request with `429 account … is suspended`
— was reported as healthy, and would have stayed reported as healthy forever.
"""

import pytest

from kairos.health import system_health
from kairos.llm import LLMResponse, ROUTER, provider_status


@pytest.fixture(autouse=True)
def clean_router_state():
    saved = (dict(ROUTER._cooldown_until), dict(ROUTER._last_fatal))
    ROUTER._cooldown_until.clear()
    ROUTER._last_fatal.clear()
    yield
    ROUTER._cooldown_until.clear()
    ROUTER._cooldown_until.update(saved[0])
    ROUTER._last_fatal.clear()
    ROUTER._last_fatal.update(saved[1])


def _fatal_leg(monkeypatch, error="HTTP 429 account is suspended"):
    """Replace the provider legs with one that fails the way a dead account does."""
    def fake(messages, model, max_tokens, temperature):
        return LLMResponse(False, "moonshot", model, error=error)
    monkeypatch.setattr(ROUTER, "_legs", lambda: [("moonshot", fake)])


# ------------------------------------------------------------------ the router
def test_a_fatal_failure_is_still_reported_after_the_cooldown_expires(monkeypatch):
    _fatal_leg(monkeypatch)
    out = ROUTER.chat("will anyone answer?", use_cache=False)
    assert out.ok is False
    st = provider_status()["moonshot"]
    assert st["cooling_s"] > 0
    assert "suspended" in st["last_fatal"]

    # Thirty minutes later the router is willing to pay for another request,
    # and the account is still suspended. Only remembering the reason keeps the
    # health page from declaring a dead provider healthy between attempts.
    ROUTER._cooldown_until["moonshot"] = 0.0
    st = provider_status()["moonshot"]
    assert st["cooling_s"] == 0
    assert "suspended" in st["last_fatal"]
    assert st["last_fatal_age_s"] >= 0


def test_a_success_does_not_leave_a_stale_grief(monkeypatch):
    """The reason is a fact about the last failure, not a permanent label."""
    _fatal_leg(monkeypatch)
    ROUTER.chat("first", use_cache=False)
    assert provider_status()["moonshot"]["last_fatal"]

    def good(messages, model, max_tokens, temperature):
        return LLMResponse(True, "moonshot", model, "42", latency_s=0.1)
    ROUTER._cooldown_until.pop("moonshot", None)   # the half hour passed
    monkeypatch.setattr(ROUTER, "_legs", lambda: [("moonshot", good)])
    ROUTER.chat("second", use_cache=False)
    assert provider_status()["moonshot"]["last_fatal"] == ""


# --------------------------------------------------------------------- health
def _patch_probes(monkeypatch, pstat):
    monkeypatch.setattr("kairos.llm.provider_status", lambda: pstat)
    monkeypatch.setattr("kairos.llm.dead_seats", lambda: [])
    monkeypatch.setattr("kairos.llm.council_models", lambda: ["a", "b", "c", "d",
                                                             "e", "f"])
    src = type("Src", (), {"yahoo": True, "tradingview": True, "psx_local": False})()
    monkeypatch.setattr("kairos.engine.ENGINE.md",
                        type("M", (), {"health": lambda self: src})(),
                        raising=False)
    # the rest of the page would shell out to WSL and resolve the tunnel
    monkeypatch.setattr("kairos.health.WSL.models_ready", lambda: {})
    monkeypatch.setattr("kairos.health.WSL.wsl_available", lambda: False)
    monkeypatch.setattr("kairos.health._public_url", lambda: None)
    monkeypatch.setattr("kairos.health.universe", lambda: [])


def _cfg(monkeypatch, moonshot_enabled=True):
    from kairos.config import CFG
    apis = {k: dict(v) for k, v in CFG.apis.items()}
    apis.setdefault("openrouter", {})["enabled"] = True
    apis["moonshot"] = dict(apis.get("moonshot", {}), enabled=moonshot_enabled,
                            api_key="kimi-placeholder")
    monkeypatch.setattr(CFG, "_raw", {**CFG._raw, "apis": apis})
    monkeypatch.setattr(CFG, "openrouter_keys", lambda: ["or-placeholder"])


def test_a_suspended_provider_is_not_reported_live(monkeypatch):
    _cfg(monkeypatch)
    _patch_probes(monkeypatch, {
        "openrouter": {"enabled": True, "cooling_s": 0, "last_fatal": "",
                       "last_fatal_age_s": None},
        "moonshot": {"enabled": True, "cooling_s": 1_240,
                     "last_fatal": "HTTP 429 account is suspended",
                     "last_fatal_age_s": 500}})
    llm = system_health()["llm"]
    assert llm["kimi"] is False                  # not live, though configured
    assert llm["configured"]["kimi"] is True     # and the difference is visible
    assert llm["unanswered"]["moonshot"] == "HTTP 429 account is suspended"
    assert llm["provider"] == "openrouter"       # the council still has a leg
    assert llm["models"] == 6                    # configured seats, unchanged


def test_openrouter_suffering_the_same_fate_is_named_too(monkeypatch):
    _cfg(monkeypatch)
    _patch_probes(monkeypatch, {
        "openrouter": {"enabled": True, "cooling_s": 1_800,
                       "last_fatal": "HTTP 402 insufficient balance",
                       "last_fatal_age_s": 3},
        "moonshot": {"enabled": True, "cooling_s": 0, "last_fatal": "",
                     "last_fatal_age_s": None}})
    llm = system_health()["llm"]
    assert llm["kimi"] is True
    assert llm["provider"] == "kimi"
    assert "insufficient balance" in llm["unanswered"]["openrouter"]


def test_nothing_configured_is_reported_as_down(monkeypatch):
    _cfg(monkeypatch, moonshot_enabled=False)
    _patch_probes(monkeypatch, {
        "openrouter": {"enabled": True, "cooling_s": 1_800,
                       "last_fatal": "HTTP 401 invalid authentication",
                       "last_fatal_age_s": 1}})
    llm = system_health()["llm"]
    assert llm["ok"] is False
    assert llm["provider"] == "down"
