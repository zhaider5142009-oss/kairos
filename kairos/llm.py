"""LLM intelligence layer: multi-provider router + model council.

Providers are ranked by live health (probed 2026-10-19):
  1. OpenRouter — 3 live keys, round-robin, access to 300+ models
  2. Kimi / Moonshot — direct 262k-context reasoning models (kimi-k2.6,
     kimi-k2.7-code); global endpoint api.moonshot.ai, the .cn endpoint
     rejects this key
  3. DeepSeek   — valid key, zero balance (disabled, auto-probe on enable)
  4. Gemini     — project API disabled (kept for future enablement)
  5. Z.AI / Grok — dead keys (kept for future rotation)

Council protocol: the same structured prompt goes to N diverse models;
responses must parse as JSON; the median numeric fields become the
council verdict, with dissent recorded. Prefix a council entry with
"kimi:" to send it straight to Moonshot instead of OpenRouter.
Caching: disk-backed, keyed by route+model+prompt hash, tagged with the
provider that actually answered.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CFG

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KAIROS/2.0"
KIMI_PREFIXES = ("kimi:", "moonshot:")
PROVIDER_COOLDOWN_S = 1800.0
# errors that will not fix themselves between two cycles
_FATAL_HINTS = ("401", "402", "403", "429", "invalid authentication",
                "insufficient balance", "suspended", "no api_key", "no keys")

# A provider echoes its own account identifiers into the HTTP error body — a 429
# from Moonshot names the organisation and the access-key id. That string is
# returned by /api/health and shown on the desk, so it must not leave the process
# verbatim. Status codes survive; identifier shapes do not.
#
# The unnamed-blob rule requires a digit. Without it, any long run of letters is
# mistaken for a secret and the operator loses the sentence that explains why the
# provider failed — and a dead provider with no readable reason is worse than a
# redacted one. Real account ids are hex or base62 and carry digits.
_ID_SHAPES = re.compile(
    r"(?:org|ak|user|key|token|proj|team|sub)[-_][A-Za-z0-9]{6,}"
    r"|sk-[A-Za-z0-9]{8,}"
    r"|Bearer\s+[A-Za-z0-9._-]{8,}"
    r"|\b(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{24,}\b")


def _safe_error(text: str) -> str:
    return _ID_SHAPES.sub("[id]", text)[:200]


def _clean(text: str) -> str:
    """Drop a model's own stop-token markup before anyone parses the answer.

    kimi-k3 through OpenRouter answers "OK<|close|>"; left in, the trailing token
    rides into the verdict parser and the council vote.
    """
    return re.sub(r"<\|[^|>]{0,32}\|>", "", text or "").strip()


def _answered(provider: str, model: str, text: str,
              latency_s: float) -> LLMResponse:
    """A blank completion is a failure, not a vote.

    A reasoning model can spend max_tokens on hidden thinking and return an empty
    content field. Reported as success it raises models_responded while
    contributing nothing to the median — the same phantom-seat defect the router
    fail-over fix removed, reachable on every leg instead of one.
    """
    body = _clean(text)
    if not body:
        return LLMResponse(False, provider, model, error="empty completion",
                           latency_s=latency_s)
    return LLMResponse(True, provider, model, body, latency_s=latency_s)


@dataclass
class LLMResponse:
    ok: bool
    provider: str
    model: str
    text: str = ""
    error: str = ""
    latency_s: float = 0.0
    cached: bool = False


class LLMRouter:
    def __init__(self) -> None:
        self.cache_dir = CFG.data_dir / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._or_key_index = 0
        self._or_key_health: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}
        # name -> (reason, when): cooldown expires, a suspended account does not
        self._last_fatal: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------ utils
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> tuple[str, str] | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                age = time.time() - p.stat().st_mtime
                if age < 3600 * 6:
                    j = json.loads(p.read_text(encoding="utf-8"))
                    return j.get("text"), j.get("provider", "openrouter")
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, text: str, provider: str) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps({"text": text, "provider": provider,
                            "ts": time.time()}), encoding="utf-8")
        except Exception:
            pass

    def _or_key(self) -> str | None:
        keys = CFG.openrouter_keys()
        live = [k for k in keys
                if self._or_key_health.get(k, 1.0) > 0] or keys
        if not live:
            return None
        key = live[self._or_key_index % len(live)]
        self._or_key_index += 1
        return key

    # ------------------------------------------------------------- providers
    def _call_openrouter(self, messages: list[dict], model: str,
                         max_tokens: int, temperature: float) -> LLMResponse:
        api = CFG.apis.get("openrouter", {})
        key = self._or_key()
        if not key:
            return LLMResponse(False, "openrouter", model, error="no keys")
        url = api.get("base_url", "https://openrouter.ai/api/v1") + "/chat/completions"
        body = json.dumps({
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kairos.local",
            "X-Title": "KAIROS Quant System",
            "User-Agent": _UA,
        }, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=int(api.get("timeout_s", 90))) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            text = payload["choices"][0]["message"]["content"] or ""
            return _answered("openrouter", model, text, time.time() - t0)
        except Exception as exc:
            if "401" in str(exc) or "403" in str(exc):
                self._or_key_health[key] = 0.0
            return LLMResponse(False, "openrouter", model,
                               error=_safe_error(str(exc)), latency_s=time.time() - t0)

    def _call_deepseek(self, messages: list[dict], model: str,
                       max_tokens: int, temperature: float) -> LLMResponse:
        api = CFG.apis.get("deepseek", {})
        url = api.get("base_url", "https://api.deepseek.com") + "/chat/completions"
        body = json.dumps({"model": model or api.get("model", "deepseek-chat"),
                           "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api.get('api_key', '')}",
            "Content-Type": "application/json", "User-Agent": _UA}, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            return _answered("deepseek", model,
                             payload["choices"][0]["message"]["content"],
                             time.time() - t0)
        except Exception as exc:
            return LLMResponse(False, "deepseek", model, error=_safe_error(str(exc)))

    def _call_moonshot(self, messages: list[dict], model: str,
                       max_tokens: int, temperature: float) -> LLMResponse:
        """Kimi (Moonshot) direct. OpenAI-shaped request, its own rules:

        * temperature must be inside (0, 1] — a 1.4 "creative" setting is an
          HTTP 400 there, so it is clamped rather than failed;
        * OpenRouter slugs mean nothing here, so an unknown name falls back
          to the configured Kimi model instead of burning a request.
        """
        api = CFG.apis.get("moonshot", {})
        key = (api.get("api_key") or os.environ.get("MOONSHOT_API_KEY")
               or os.environ.get("KIMI_API_KEY") or "")
        if not key:
            return LLMResponse(False, "moonshot", model, error="no api_key")
        m = _kimi_model(model, api)
        url = api.get("base_url", "https://api.moonshot.ai/v1") + "/chat/completions"
        body = json.dumps({
            "model": m, "messages": messages,
            "max_tokens": max_tokens,
            "temperature": min(1.0, max(0.01, float(temperature))),
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "User-Agent": _UA}, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(
                    req, timeout=int(api.get("timeout_s", 90))) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            msg = payload["choices"][0]["message"]
            # reasoning models can answer with an empty content and the text
            # in reasoning_content; never report a blank success
            return _answered("moonshot", m, msg.get("content"), time.time() - t0)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            return LLMResponse(False, "moonshot", m,
                               error=_safe_error(f"HTTP {exc.code} {detail}"),
                               latency_s=time.time() - t0)
        except Exception as exc:
            return LLMResponse(False, "moonshot", m, error=_safe_error(str(exc)),
                               latency_s=time.time() - t0)

    def _call_gemini(self, prompt: str, model: str,
                     max_tokens: int, temperature: float) -> LLMResponse:
        api = CFG.apis.get("gemini", {})
        url = (f"{api.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')}"
               f"/models/{model or api.get('model', 'gemini-2.0-flash')}:generateContent"
               f"?key={api.get('api_key', '')}")
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"temperature": temperature,
                                                "maxOutputTokens": max_tokens}}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "User-Agent": _UA}, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            parts = payload["candidates"][0]["content"].get("parts", [])
            return _answered("gemini", model, parts[0].get("text") if parts else None,
                             time.time() - t0)
        except Exception as exc:
            return LLMResponse(False, "gemini", model, error=_safe_error(str(exc)))

    # ------------------------------------------------------------------ chat
    def _legs(self) -> list[tuple[str, Any]]:
        """Enabled providers in fail-over order."""
        want = (("openrouter", self._call_openrouter),
                ("moonshot", self._call_moonshot),
                ("deepseek", self._call_deepseek),
                ("gemini", self._call_gemini))
        return [(n, fn) for n, fn in want
                if CFG.apis.get(n, {}).get("enabled", n == "openrouter")]

    def _cooling(self, name: str) -> bool:
        return time.time() < self._cooldown_until.get(name, 0.0)

    def chat(self, prompt: str, model: str | None = None,
             system: str | None = None, max_tokens: int = 1400,
             temperature: float = 0.4, use_cache: bool = True) -> LLMResponse:
        m = model or CFG.apis.get("openrouter", {}).get("fast_model",
                                                        "deepseek/deepseek-chat-v3-0324")
        direct = None
        if m.startswith(KIMI_PREFIXES):
            direct, m = "moonshot", m.split(":", 1)[1]
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        route = direct or ("named" if model else "chain")
        cache_key = hashlib.sha256(
            f"{route}|{m}|{prompt}|{system}|{max_tokens}".encode()).hexdigest()[:24]
        if use_cache:
            hit = self._cache_get(cache_key)
            if hit:
                text, provider = hit
                return LLMResponse(True, provider, m, text, cached=True)

        legs = self._legs()
        if direct:
            legs = [(n, fn) for n, fn in legs if n == direct]
        elif model:
            # A named council seat must be answered by that seat. Falling over
            # to the next provider substitutes a different model, so a dead
            # llama seat would return as a second Kimi vote and the median
            # would quietly double-count one opinion.
            legs = [(n, fn) for n, fn in legs if n == "openrouter"]
        errs: list[str] = []
        for name, fn in legs:
            if self._cooling(name):
                errs.append(f"{name}: cooling")
                continue
            if name == "gemini":
                resp = fn(prompt, m, max_tokens, temperature)
            else:
                resp = fn(messages, m, max_tokens, temperature)
            if resp.ok:
                self._cache_put(cache_key, resp.text, resp.provider)
                # a recovered provider must not stay flagged: a 429 limit is fatal too
                self._last_fatal.pop(name, None)
                return resp
            # every leg's reason, in order — the last provider's balance error
            # must not hide the primary provider's actual failure
            errs.append(f"{name}: {resp.error[:90]}")
            if _fatal(resp.error):
                # a dead key or an empty balance will not recover between
                # cycles — stop paying 0.5s per call for a guaranteed 401/429
                self._cooldown_until[name] = time.time() + PROVIDER_COOLDOWN_S
                self._last_fatal[name] = (resp.error[:120], time.time())
        return LLMResponse(False, "chain", m,
                           error=" | ".join(errs)[:220] or "no provider answered")

    # --------------------------------------------------------------- council
    def council(self, prompt: str, system: str | None = None,
                models: list[str] | None = None, max_tokens: int = 1200,
                temperature: float = 0.3) -> dict[str, Any]:
        models = models if models is not None else council_models()
        results: list[LLMResponse] = []
        for m in models:
            resp = self.chat(prompt, model=m, system=system,
                             max_tokens=max_tokens, temperature=temperature)
            if resp.ok:
                results.append(resp)
        verdict = {"responses": [], "consensus": {}, "dissent": [],
                   "models_responded": len(results), "models_total": len(models)}
        parsed: list[tuple[str, dict[str, Any]]] = []
        for r in results:
            data = _extract_json(r.text)
            verdict["responses"].append({
                "provider": r.provider, "model": r.model,
                "latency_s": round(r.latency_s, 2), "cached": r.cached,
                "raw": (data if data else r.text[:600])})
            if isinstance(data, dict):
                parsed.append((f"{r.provider}/{r.model}", data))
        if parsed:
            first = parsed[0][1]
            numeric_fields = [k for k in first
                              if isinstance(first[k], (int, float))]
            consensus: dict[str, Any] = {}
            for k in numeric_fields:
                vals = [p[k] for _, p in parsed if isinstance(p.get(k), (int, float))]
                if vals:
                    consensus[k] = _median(vals)
            text_fields = [k for k in first if isinstance(first[k], str)]
            for k in text_fields:
                vals = [str(p.get(k, "")) for _, p in parsed if p.get(k)]
                consensus[k] = _mode(vals) if vals else ""
            verdict["consensus"] = consensus
            # dissent: numeric views far from median, attributed to the model
            # that held it — an unattributed dissent cannot be learned from
            for k, v in consensus.items():
                if not isinstance(v, (int, float)):
                    continue
                for name, p in parsed:
                    pv = p.get(k)
                    if isinstance(pv, (int, float)) and abs(pv - v) > max(0.15, abs(v) * 0.6):
                        verdict["dissent"].append(
                            {"model": name, "field": k, "value": pv, "consensus": v})
                        break
        return verdict

    def health_probe(self) -> dict[str, Any]:
        """One-token probe of every enabled provider."""
        out: dict[str, Any] = {}
        probe_prompt = "Reply with exactly: OK"
        if CFG.apis.get("openrouter", {}).get("enabled", True):
            m = CFG.apis.get("openrouter", {}).get("fast_model",
                                                   "deepseek/deepseek-chat-v3-0324")
            r = self.chat(probe_prompt, model=m, max_tokens=8, use_cache=False)
            out["openrouter"] = {"ok": r.ok, "model": m, "latency_s": round(r.latency_s, 2),
                                 "error": r.error[:120]}
        if CFG.apis.get("moonshot", {}).get("enabled", False):
            m = _kimi_model(None, CFG.apis.get("moonshot", {}))
            r = self.chat(probe_prompt, model=f"kimi:{m}", max_tokens=8, use_cache=False)
            out["moonshot"] = {"ok": r.ok, "model": m, "latency_s": round(r.latency_s, 2),
                               "error": r.error[:120]}
        return out


def _fatal(error: str) -> bool:
    e = (error or "").lower()
    return any(h in e for h in _FATAL_HINTS)


def provider_status() -> dict[str, Any]:
    """Enabled providers, whether each is cooling, and why it last died.

    "A key is configured" and "the account answers" are different facts: the
    direct Kimi key sits enabled in config while the account behind it is
    suspended, so a health page built from config alone reports a provider that
    has not answered a request in days as live.
    """
    out: dict[str, Any] = {}
    now = time.time()
    for name, _fn in ROUTER._legs():
        left = ROUTER._cooldown_until.get(name, 0.0) - now
        why, when = ROUTER._last_fatal.get(name, ("", 0.0))
        out[name] = {"enabled": True,
                     "cooling_s": round(left) if left > 0 else 0,
                     "last_fatal": why,
                     "last_fatal_age_s": round(now - when) if when else None}
    return out


# Retiring a model slug is routine for a vendor and silent for the caller: the
# seat just stops answering, and the council quietly loses a vote nobody
# notices. One cached catalog read turns that into a visible misconfiguration.
CATALOG_TTL_S = 6 * 3600.0
_catalog: tuple[float, list[str]] | None = None


def catalog_ids() -> list[str]:
    api = CFG.apis.get("openrouter", {})
    keys = CFG.openrouter_keys()
    if not keys:
        raise RuntimeError("no OpenRouter key configured")
    req = urllib.request.Request(
        api.get("base_url", "https://openrouter.ai/api/v1") + "/models",
        headers={"Authorization": f"Bearer {keys[0]}", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return sorted({d["id"] for d in json.loads(raw)["data"]})



def dead_seats(force: bool = False) -> list[str]:
    """Configured OpenRouter slugs the provider no longer recognises."""
    global _catalog
    now = time.time()
    if _catalog and (not force and now - _catalog[0] < CATALOG_TTL_S):
        ids = set(_catalog[1])
    else:
        try:
            ids = set(catalog_ids())
        except Exception:
            return []          # provider unreachable: never claim a slug is dead
        _catalog = (now, sorted(ids))
    api = CFG.apis.get("openrouter", {})
    want = set(api.get("council_models", []) or [])
    for field in ("fast_model", "deep_model"):
        if api.get(field):
            want.add(str(api[field]))
    return sorted(s for s in want if s and not s.startswith(KIMI_PREFIXES)
                  and s not in ids)


def _kimi_model(model: str | None, api: dict[str, Any]) -> str:
    """Resolve a Kimi model name; anything that is not a Moonshot model
    (OpenRouter slugs, blank, a stale name) falls back to the configured one."""
    m = (model or "").strip()
    for cut in ("/", ":"):
        if cut in m:
            m = m.split(cut)[-1]
    if m.startswith(("kimi", "moonshot")):
        return m
    return api.get("model", "kimi-k2.6")


def council_models() -> list[str]:
    """Diverse model list for the council: OpenRouter pool + direct Kimi seats.

    Diversity is the point of a council, so a configured Moonshot model
    joins it with the "kimi:" route prefix instead of replacing a seat.
    """
    models = list(CFG.openrouter_models())
    api = CFG.apis.get("moonshot", {})
    if api.get("enabled", False):
        seats = api.get("council_models") or [api.get("model", "kimi-k2.6")]
        models += [f"kimi:{m}" for m in seats if m]
    return models


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def _mode(vals: list[str]) -> str:
    if not vals:
        return ""
    best, best_n = "", 0
    for v in set(vals):
        n = vals.count(v)
        if n > best_n:
            best, best_n = v, n
    return best


def _extract_json(text: str) -> Any:
    """Robustly pull the first JSON object out of an LLM response."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
    return None


ROUTER = LLMRouter()


def health_probe() -> dict[str, Any]:
    return ROUTER.health_probe()
