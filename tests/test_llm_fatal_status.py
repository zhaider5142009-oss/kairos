"""A provider's status code, not the digits in its prose, decides whether it is dead.

Three defects these guard, and the one contract that fixes all three:

1. `str(urllib.error.HTTPError)` is only "HTTP Error 400: Bad Request". Three of
   the four legs kept exactly that and threw the body away, so the desk and
   `/api/health` reported a status line with no reason in it — a retired model, a
   suspended account and a malformed request were the same print — and the
   provider's own sentence ("Insufficient Balance"), which is the thing that
   would never recover, never reached the rule that decides whether to cool the
   leg down.
2. What stood in for a code was substring matching on digits, which cannot tell a
   status from a token count. On the one leg that did read the body — Kimi — a 400
   saying "you requested 4296 tokens" contains "429", so a healthy provider was
   cooled for half an hour for being asked for too many tokens.
3. Making the legs read the body without a code would move that false positive
   from one leg to all four, which is why the code comes first.

So `LLMResponse.status` carries the number the provider actually answered with,
`FATAL_CODES` decides from it, and the words are what a leg with no code to offer
falls back on — read as words, never as digits.
"""

from __future__ import annotations

import ast
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kairos import llm, netio
from kairos.config import CFG

ORG_ID = "org-" + "7f3a9c21be4448e0b6d1c5a0e9f2b731"
TOKEN_MSG = ("This model's maximum context length is 8192 tokens, however you "
             "requested 4296 tokens. Please reduce the length of the messages.")
LLM_SOURCE = Path(__file__).resolve().parents[1] / "kairos" / "llm.py"


def _apis(**changed):
    apis = {k: dict(v) for k, v in CFG.apis.items()}
    for name, patch in changed.items():
        apis[name] = dict(apis.get(name, {}), **patch)
    return {**CFG._raw, "apis": apis}


def _http_error(url: str, code: int, body: dict | str,
                reason: str = "Bad Request") -> urllib.error.HTTPError:
    raw = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(url, code, reason, hdrs=None,
                                  fp=io.BytesIO(raw.encode()))


def _raises(exc: BaseException):
    """urlopen stand-in that fails with exactly this exception."""
    def go(req, timeout=None):
        raise exc
    return go


class _Body:
    """A 2xx answer whose JSON payload is exactly this object."""

    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _router_for(monkeypatch, provider: str, **api_patch):
    """A router whose named leg will really run, with the network still walled."""
    monkeypatch.setattr(CFG, "_raw", _apis(**{provider: api_patch}))
    r = llm.LLMRouter()
    r._cooldown_until.clear()
    r._or_key_health.clear()
    if provider == "openrouter":
        monkeypatch.setattr(r, "_or_key", lambda: "test-or-key")
    return r


# ------------------------------------------------------------------ status_of
def test_status_of_reads_the_code_a_provider_answered_with():
    assert netio.status_of(
        _http_error("https://x.test/v1", 429, {}, "Too Many Requests")) == 429
    assert netio.status_of(netio.HttpStatus(503, "https://x.test/v1", "no capacity")) == 503


def test_status_of_is_zero_when_nothing_answered_with_a_code():
    assert netio.status_of(TimeoutError("timed out")) == 0
    assert netio.status_of(urllib.error.URLError("no route to host")) == 0
    assert netio.status_of(ValueError("HTTP 429 said so in prose")) == 0


def test_status_of_does_not_trust_anything_that_merely_has_a_code_attribute():
    """`code` is not an HTTP convention — a database error carries one too."""

    class _Sqliteish(Exception):
        code = 101

    assert netio.status_of(_Sqliteish()) == 0


# --------------------------------------------------------------------- _fatal
def test_fatality_is_decided_by_the_code():
    for code in sorted(llm.FATAL_CODES):
        assert llm._fatal("", code) is True
    for code in (400, 404, 408, 500, 503):
        assert llm._fatal("", code) is False


def test_the_words_still_decide_when_no_code_arrived():
    assert llm._fatal("no keys") is True
    assert llm._fatal("no api_key") is True
    assert llm._fatal("HTTP Error 400: Bad Request") is False
    assert llm._fatal(llm._answered("openrouter", "a/b", "", 1.0).error) is False


def test_no_hint_is_a_bare_number():
    """The list must never grow a digit string back: prose contains digits."""
    assert not [h for h in llm._FATAL_HINTS if h.isdigit()]
    assert llm._fatal("you requested 4296 tokens", 400) is False
    assert llm._fatal("HTTP Error 401: Unauthorized", 401) is True


# ------------------------------------------------------------------- real legs
def test_a_leg_keeps_the_status_as_a_number(monkeypatch):
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 429,
        {"error": {"message": "insufficient balance", "organization": ORG_ID}})))
    res = r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert res.ok is False and res.status == 429
    assert ORG_ID not in res.error                    # the body is still redacted
    assert llm._fatal(res.error, res.status) is True


def test_a_leg_keeps_the_sentence_the_provider_wrote(monkeypatch):
    """HEAD returned "HTTP Error 400: Bad Request" and lost the reason forever."""
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 400,
        {"error": {"message": TOKEN_MSG}})))
    res = r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert res.status == 400 and "4296 tokens" in res.error
    assert "8192 tokens" in res.error
    assert llm._fatal(res.error, res.status) is False


def test_asking_for_too_many_tokens_does_not_cool_the_provider(monkeypatch):
    """End to end: this is the false positive that froze a leg for half an hour."""
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 400,
        {"error": {"message": TOKEN_MSG}})))
    res = r.chat("q", model="a/b", use_cache=False)
    assert res.ok is False
    assert r._cooling("openrouter") is False
    assert "openrouter" not in r._last_fatal


def test_the_kimi_leg_was_already_misreading_its_own_body(monkeypatch):
    """Moonshot was the one leg that did read the body, so this false positive was
    live: a 400 quoting a token count contains "429" and cooled a healthy key for
    half an hour, on a rule that had no idea which number it had matched."""
    r = _router_for(monkeypatch, "moonshot", enabled=True, api_key="test-kimi-key")
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://api.moonshot.ai/v1/chat/completions", 400,
        {"error": {"message": TOKEN_MSG}})))
    leg = r._call_moonshot([{"role": "user", "content": "hi"}], "kimi-k2.6", 8, 0.3)
    assert leg.status == 400 and "4296 tokens" in leg.error
    assert llm._fatal(leg.error, leg.status) is False
    res = r.chat("q", model="kimi:kimi-k2.6", use_cache=False)
    assert res.ok is False
    assert r._cooling("moonshot") is False
    assert "moonshot" not in r._last_fatal


def test_the_reason_is_the_sentence_not_the_json_wrapper(monkeypatch):
    """Each leg's report is shown in 90 characters: a JSON envelope spends them on
    punctuation and cuts the sentence off before the part that says what to do."""
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 429,
        {"error": {"message": "Rate limit exceeded: 10 requests per minute",
                   "type": "rate_limit_error"}})))
    res = r.chat("q", model="a/b", use_cache=False)
    assert "Rate limit exceeded" in res.error
    assert '{"error"' not in res.error
    assert r._cooling("openrouter") is True          # and it is still a 429


def test_a_body_that_is_not_json_is_still_the_reason(monkeypatch):
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 503,
        "Upstream model capacity exceeded, retry later", "Service Unavailable")))
    res = r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert res.status == 503
    assert "capacity exceeded" in res.error
    assert llm._fatal(res.error, res.status) is False


def test_a_refused_key_is_still_taken_out_of_the_rotation(monkeypatch):
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 401,
        {"error": {"message": "No endpoints found matching your IP address"}},
        "Unauthorized")))
    res = r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert res.status == 401
    assert r._or_key_health["test-or-key"] == 0.0


def test_a_number_in_a_400_body_does_not_retire_a_working_key(monkeypatch):
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://openrouter.ai/api/v1/chat/completions", 400,
        {"error": {"message": TOKEN_MSG}})))
    r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert r._or_key_health.get("test-or-key", 1.0) > 0


def test_a_gemini_refusal_names_the_reason(monkeypatch):
    """The leg that never read a body: a disabled key was reported as a bare 403."""
    r = _router_for(monkeypatch, "gemini", enabled=True, api_key="test-gemini-key")
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://generativelanguage.googleapis.com/v1beta/x", 403,
        {"error": {"code": 403, "message": "API key not valid. "
                                           "Please pass a valid API key.",
                   "status": "PERMISSION_DENIED"}}, "Forbidden")))
    res = r._call_gemini("q", "gemini-2.0-flash", 8, 0.3)
    assert res.status == 403 and "API key not valid" in res.error
    assert res.latency_s > 0                           # it timed like the others
    assert llm._fatal(res.error, res.status) is True


def test_a_zero_balance_answer_reaches_the_rule_that_acts_on_it(monkeypatch):
    """The provider's sentence used to be discarded, so an account with no credit
    was asked again every cycle forever; now the words arrive and are enough."""
    r = _router_for(monkeypatch, "deepseek", enabled=True, api_key="test-ds-key")
    monkeypatch.setattr(urllib.request, "urlopen", _raises(_http_error(
        "https://api.deepseek.com/chat/completions", 400,
        {"error": {"message": "Insufficient Balance"}})))
    res = r._call_deepseek([{"role": "user", "content": "hi"}], "d", 8, 0.3)
    assert "Insufficient Balance" in res.error
    assert llm._fatal(res.error, res.status) is True


def test_a_wire_failure_still_has_no_status_to_act_on(monkeypatch):
    r = _router_for(monkeypatch, "moonshot", enabled=True, api_key="test-kimi-key")
    monkeypatch.setattr(urllib.request, "urlopen", _raises(TimeoutError("timed out")))
    res = r._call_moonshot([{"role": "user", "content": "hi"}], "kimi-k2.6", 8, 0.3)
    assert res.status == 0 and llm._fatal(res.error, res.status) is False
    assert "timed out" in res.error


def test_a_payload_that_changed_shape_says_so(monkeypatch):
    """A 200 with no `choices` reached the desk as `'choices'` — a quoted key with
    nothing to tell the operator about the response that produced it."""
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Body({"result": "ok"}))
    res = r._call_openrouter([{"role": "user", "content": "hi"}], "a/b", 8, 0.3)
    assert res.ok is False and res.status == 0
    assert "KeyError" in res.error and "'choices'" in res.error
    assert llm._fatal(res.error, res.status) is False


def test_an_unreadable_body_does_not_lose_the_code(monkeypatch):
    """A 401 whose body cannot be read is still a 401, and is cooled as one."""
    r = _router_for(monkeypatch, "moonshot", enabled=True, api_key="test-kimi-key")
    err = _http_error("https://api.moonshot.ai/v1/chat/completions", 401, "", "Unauthorized")
    err.close()                                        # body already drained
    monkeypatch.setattr(urllib.request, "urlopen", _raises(err))
    res = r._call_moonshot([{"role": "user", "content": "hi"}], "kimi-k2.6", 8, 0.3)
    assert res.status == 401 and llm._fatal(res.error, res.status) is True


# ------------------------------------------------------------------- structure
def _leg_exception_handlers() -> list[list[ast.AST]]:
    tree = ast.parse(LLM_SOURCE.read_text(encoding="utf-8"))
    out: list[list[ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_call_"):
            out.extend([ast.walk(sub) for sub in ast.walk(node)
                        if isinstance(sub, ast.ExceptHandler)])
    return out


def test_every_leg_failure_path_goes_through_the_one_builder():
    """A leg that builds its own error string is a leg that loses the code again."""
    assert _leg_exception_handlers(), "no leg catches a failure any more"
    for body in _leg_exception_handlers():
        names = {n.func.id for n in body if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)}
        assert "_failed" in names, f"a leg handler returns without _failed: {names}"


def test_the_cooldown_is_decided_from_the_response_not_its_text_alone():
    """`_fatal(resp.error)` would silently drop the status back on the floor."""
    tree = ast.parse(LLM_SOURCE.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_fatal"]
    assert calls, "nothing decides fatality any more"
    assert all(len(c.args) == 2 for c in calls if not c.keywords), (
        "a fatality decision that reads only the message")


def test_the_guard_also_bites_on_a_leg_that_only_forgets_the_status():
    """A guard that passes for the wrong reason is the bug this file exists over."""
    src = ('def _call_x(a, b):\n'
           '    try:\n        pass\n'
           '    except Exception as exc:\n'
           '        return LLMResponse(False, "p", m, error=str(exc))\n')
    tree = ast.parse(src)
    handlers = [ast.walk(sub) for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("_call_")
                for sub in ast.walk(node) if isinstance(sub, ast.ExceptHandler)]
    names = {n.func.id for body in handlers for n in body
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_failed" not in names


@pytest.mark.parametrize("code", [401, 402, 403, 429])
def test_the_documented_fatal_codes_are_the_codes_the_router_cools_on(monkeypatch, code):
    assert code in llm.FATAL_CODES
    r = _router_for(monkeypatch, "openrouter", enabled=True)
    monkeypatch.setattr(r, "_call_openrouter",
                        lambda m, mo, mt, t: llm.LLMResponse(
                            False, "openrouter", mo, error=f"HTTP {code}",
                            status=code))
    r.chat("q", model="a/b", use_cache=False)
    assert r._cooling("openrouter") is True
