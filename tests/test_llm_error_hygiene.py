"""Provider text and error hygiene — no network.

What a provider hands us is shown on the desk, returned by /api/health and
parsed by the council, so three things must hold about it:

1. A provider names its own account inside HTTP error bodies — a 429 from
   Moonshot carries the organisation id and the access-key id. Those strings
   must not leave the process, and redacting them must not cost the caller the
   status code or the reason, because `FATAL_CODES` and the word hints together
   are what put a dead provider into cooldown.
2. A model hosted behind OpenRouter can end its answer with its own stop-token
   markup; parsed as part of the answer it rides into the verdict.
3. A reasoning model that spends max_tokens on hidden thinking returns an empty
   content field. Reported as a success it raises models_responded and joins no
   median — a phantom seat, the same class of defect the router fail-over fix
   removed.

The fake keys here are built at runtime: a literal provider-key string in this
file would trip the credential tripwire in test_no_secret_leak.py.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from kairos.config import CFG
from kairos.llm import LLMRouter, _answered, _clean, _fatal, _safe_error

ORG_ID = "org-" + "7f3a9c21be4448e0b6d1c5a0e9f2b731"
AK_ID = "ak-" + "2b7c9d3e1f4a86520c9d7e3b"
OR_KEY = "sk-or-v1-" + "9c1f8a7b6d5e4c3b2a190f8e7d6c5b4a"
GOOGLE_KEY = "AIza" + "SyD9c8b7a6f5e4d3c2b1a908877665544"

RAW_429 = ('HTTP 429 {"error":{"message":"insufficient balance","organization":"'
           + ORG_ID + '","internal_line":"L234","id":"' + AK_ID + '"}}')


def _apis(**changed):
    """Config with the given provider blocks replaced — nothing else touched."""
    apis = {k: dict(v) for k, v in CFG.apis.items()}
    for name, patch in changed.items():
        apis[name] = dict(apis.get(name, {}), **patch)
    return {**CFG._raw, "apis": apis}


# ------------------------------------------------------------------ redaction
def test_account_identifiers_are_redacted():
    out = _safe_error(RAW_429)
    assert ORG_ID not in out and AK_ID not in out
    assert "org-" not in out and "ak-" not in out


def test_bearer_and_long_key_shapes_are_redacted():
    assert "9c1f8a7b6d5e4c3b" not in _safe_error(f"failed: Bearer {OR_KEY}")
    assert "SyD9c8b7a6f5e4d3c2b1a" not in _safe_error(f"rejected key {GOOGLE_KEY}")


def test_status_code_and_reason_survive_redaction():
    """Redaction must not cost the caller the diagnosis or the cooldown."""
    out = _safe_error(RAW_429)
    assert "429" in out and "insufficient balance" in out
    assert _fatal(out, 429) is True
    assert _fatal(_safe_error("HTTP 401 invalid authentication"), 401) is True
    assert _fatal(_safe_error("timed out")) is False


def test_redaction_happens_before_truncation():
    """Truncating first would leave a readable head of a long token: the length
    rule cannot match a fragment, so a partial key would escape the cap."""
    token = "9" + "b" * 39
    body = "y" * 179 + " " + token          # the token straddles the 200-char cut
    out = _safe_error(body)
    assert len(out) <= 200
    assert "bbb" not in out
    assert out.count("y") == 179            # the diagnosis itself is kept


def test_the_sentence_explaining_the_failure_survives():
    """An over-broad rule would erase the reason the operator needs to read."""
    for msg in ("HTTP 503 upstream model capacity exceeded, retry later",
                 "insufficient balance, please top up your account",
                 "invalid authentication: this API key has no access",
                 "unsupportedmodelandquantizationcombination"):
        assert _safe_error(msg) == msg


# ---------------------------------------------------------------- stop tokens
def test_stop_token_markup_is_stripped():
    assert _clean("OK<|close|>") == "OK"
    assert _clean('{"score": 0.6}<|end|>') == '{"score": 0.6}'
    assert _clean("a<|b|><|c|>d") == "ad"
    assert _clean("  <|think|>  ") == ""
    assert _clean(None) == ""
    assert _clean("SELL if the range breaks below 100") == \
        "SELL if the range breaks below 100"


def test_blank_completion_is_a_failure_not_a_vote():
    res = _answered("openrouter", "a/b", "", 1.0)
    assert res.ok is False and res.error == "empty completion"
    assert _fatal(res.error) is False          # must not cool the provider down
    assert _answered("openrouter", "a/b", "<|close|>", 1.0).ok is False
    res = _answered("moonshot", "k", "OK<|close|>", 2.0)
    assert res.ok and res.text == "OK" and res.latency_s == 2.0


# ----------------------------------------------------------------- real legs
class _Body:
    def __init__(self, payload):
        self._b = json.dumps({"choices": [{"message": {"content": payload}}]}).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int, body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.moonshot.ai/v1/chat/completions",
                                  code, "Forbidden", hdrs=None,
                                  fp=io.BytesIO(json.dumps(body).encode()))


def test_moonshot_error_response_reaches_the_caller_clean(monkeypatch):
    """End to end through the real leg: the account id must not escape."""
    monkeypatch.setattr(CFG, "_raw", _apis(moonshot={"enabled": True,
                                                    "api_key": "test-kimi-key"}))
    router = LLMRouter()
    router._cooldown_until.clear()

    def boom(req, timeout=None):
        raise _http_error(429, {"error": {"message": "insufficient balance",
                                          "organization": ORG_ID}})

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    res = router._call_moonshot([{"role": "user", "content": "hi"}],
                                "kimi-k2.6", 8, 0.3)
    assert res.ok is False
    assert ORG_ID not in res.error
    assert res.status == 429
    assert "429" in res.error and _fatal(res.error, res.status) is True


def test_openrouter_blank_completion_is_not_success(monkeypatch):
    monkeypatch.setattr(CFG, "_raw", _apis(openrouter={"enabled": True}))
    router = LLMRouter()
    router._or_key_health.clear()
    monkeypatch.setattr(router, "_or_key", lambda: "test-or-key")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Body(""))
    res = router._call_openrouter([{"role": "user", "content": "hi"}],
                                  "moonshotai/kimi-k3", 8, 0.3)
    assert res.ok is False and res.error == "empty completion"


def test_openrouter_stop_token_never_reaches_the_council(monkeypatch):
    monkeypatch.setattr(CFG, "_raw", _apis(openrouter={"enabled": True}))
    router = LLMRouter()
    router._or_key_health.clear()
    monkeypatch.setattr(router, "_or_key", lambda: "test-or-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Body('{"score": 0.6}<|close|>'))
    res = router._call_openrouter([{"role": "user", "content": "hi"}],
                                  "moonshotai/kimi-k3", 8, 0.3)
    assert res.ok and res.text == '{"score": 0.6}'


def test_council_does_not_count_a_blank_seat(monkeypatch):
    """The phantom-vote fix has to show up in models_responded, not just the leg."""
    router = LLMRouter()
    answers = {"a/b": _answered("openrouter", "a/b", "", 0.1),
               "kimi:k/c": _answered("moonshot", "k/c", '{"score": 0.4}<|close|>', 0.1)}

    def fake_chat(prompt, model=None, **kw):
        return answers[model]

    monkeypatch.setattr(router, "chat", fake_chat)
    out = router.council("q", models=["a/b", "kimi:k/c"], max_tokens=8)
    assert out["models_responded"] == 1
    assert out["models_total"] == 2
    assert out["consensus"].get("score") == 0.4
    assert [r["model"] for r in out["responses"]] == ["k/c"]
