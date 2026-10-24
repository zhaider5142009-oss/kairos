"""Router fail-over semantics — no network.

Two defects these guard:

1. A named council seat that failed on its own provider fell through to the
   next provider, which substituted its own model. The council then counted a
   second opinion from one model as a third seat, so the median was pulled by a
   duplicate instead of losing a vote.
2. Only the last attempted provider's error was returned, so a 429 from the
   fallback hid the 400 that actually killed the seat.
"""

from __future__ import annotations

import pytest

from kairos.config import CFG
from kairos.llm import LLMResponse, LLMRouter, dead_seats


@pytest.fixture
def router(monkeypatch):
    r = LLMRouter()
    r._cooldown_until.clear()
    # both providers "enabled" so fail-over order is exercised
    apis = dict(CFG.apis)
    for name in ("openrouter", "moonshot", "deepseek", "gemini"):
        apis[name] = dict(apis.get(name, {}), enabled=name in ("openrouter", "moonshot"))
    monkeypatch.setattr(CFG, "_raw", {**CFG._raw, "apis": apis})
    calls: list[str] = []

    def or_call(messages, model, max_tokens, temperature):
        calls.append("openrouter:" + model)
        return r._or_response

    def kimi_call(messages, model, max_tokens, temperature):
        calls.append("moonshot:" + model)
        return r._kimi_response

    monkeypatch.setattr(r, "_call_openrouter", or_call)
    monkeypatch.setattr(r, "_call_moonshot", kimi_call)
    r._or_response = LLMResponse(True, "openrouter", "x", "answer")
    r._kimi_response = LLMResponse(True, "moonshot", "kimi-k2.6", "kimi answer")
    r._calls = calls
    return r


def test_named_seat_is_not_substituted(router):
    """A dead llama seat loses its vote; it must not return as a Kimi vote."""
    router._or_response = LLMResponse(False, "openrouter", "meta-llm/x",
                                      error="HTTP Error 400: Bad Request")
    res = router.chat("q", model="meta-llm/llama-4-scout", use_cache=False)
    assert res.ok is False
    assert router._calls == ["openrouter:meta-llm/llama-4-scout"]


def test_kimi_prefix_routes_direct_only(router):
    res = router.chat("q", model="kimi:kimi-k3", use_cache=False)
    assert res.ok and res.provider == "moonshot"
    assert router._calls == ["moonshot:kimi-k3"]


def test_unnamed_request_still_fails_over(router):
    """The default route keeps its cross-provider fail-over."""
    router._or_response = LLMResponse(False, "openrouter", "d", error="429 slow down")
    res = router.chat("q", use_cache=False)
    assert res.ok and res.provider == "moonshot"
    assert len(router._calls) == 2


def test_every_leg_reason_is_returned(router):
    router._or_response = LLMResponse(False, "openrouter", "d", error="HTTP 400 retired")
    router._kimi_response = LLMResponse(False, "moonshot", "k", error="HTTP 429 balance")
    res = router.chat("q", use_cache=False)
    assert "openrouter: HTTP 400 retired" in res.error
    assert "moonshot: HTTP 429 balance" in res.error


def test_fatal_error_cools_the_provider_down(router):
    router._or_response = LLMResponse(False, "openrouter", "d",
                                      error="HTTP 429", status=429)
    assert router.chat("q", model="a/b", use_cache=False).ok is False
    before = len(router._calls)
    res = router.chat("q2", model="a/b", use_cache=False)
    assert res.ok is False and len(router._calls) == before      # leg not retried
    assert "cooling" in res.error


def test_non_fatal_error_does_not_cool(router):
    router._or_response = LLMResponse(False, "openrouter", "d", error="timed out")
    router.chat("q", model="a/b", use_cache=False)
    router.chat("q2", model="a/b", use_cache=False)
    assert len(router._calls) == 2


def test_dead_seats_flags_retired_slug(router, monkeypatch):
    monkeypatch.setattr(CFG, "_raw", {"apis": {
        "openrouter": {"enabled": True, "council_models": ["meta-llm/llama-3.3-70b-instruct"],
                       "fast_model": "deepseek/deepseek-chat-v3-0324", "deep_model": ""},
        "moonshot": {"enabled": False}}})
    monkeypatch.setattr("kairos.llm.catalog_ids",
                        lambda: ["deepseek/deepseek-chat-v3-0324", "meta-llama/llama-4-scout"])
    monkeypatch.setattr("kairos.llm._catalog", None)
    assert dead_seats() == ["meta-llm/llama-3.3-70b-instruct"]


def test_dead_seats_stays_quiet_when_the_provider_is_unreachable(monkeypatch):
    def boom():
        raise OSError("no route to host")
    monkeypatch.setattr(CFG, "_raw", {"apis": {
        "openrouter": {"enabled": True, "council_models": ["meta-llm/llama-3.3-70b-instruct"]}}})
    monkeypatch.setattr("kairos.llm.catalog_ids", boom)
    monkeypatch.setattr("kairos.llm._catalog", None)
    assert dead_seats() == []
