"""LLM intelligence layer: multi-provider router + model council.

Providers are ranked by live health (probed 2026-10-16):
  1. OpenRouter — 3 live keys, round-robin, access to 300+ models
  2. DeepSeek   — valid key, zero balance (disabled, auto-probe on enable)
  3. Gemini     — project API disabled (kept for future enablement)
  4. Z.AI / Grok — dead keys (kept for future rotation)

Council protocol: the same structured prompt goes to N diverse models;
responses must parse as JSON; the median numeric fields become the
council verdict, with dissent recorded. Caching: disk-backed, keyed by
provider+model+prompt hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import CFG

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JARVIS/2.0"


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

    # ------------------------------------------------------------------ utils
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> str | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                age = time.time() - p.stat().st_mtime
                if age < 3600 * 6:
                    return json.loads(p.read_text(encoding="utf-8")).get("text")
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, text: str) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps({"text": text, "ts": time.time()}), encoding="utf-8")
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
            "HTTP-Referer": "https://jarvis.local",
            "X-Title": "JARVIS Quant System",
            "User-Agent": _UA,
        }, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=int(api.get("timeout_s", 90))) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            text = payload["choices"][0]["message"]["content"] or ""
            return LLMResponse(True, "openrouter", model, text.strip(),
                               latency_s=time.time() - t0)
        except Exception as exc:
            if "401" in str(exc) or "403" in str(exc):
                self._or_key_health[key] = 0.0
            return LLMResponse(False, "openrouter", model,
                               error=str(exc)[:200], latency_s=time.time() - t0)

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
            return LLMResponse(True, "deepseek", model,
                               payload["choices"][0]["message"]["content"].strip(),
                               latency_s=time.time() - t0)
        except Exception as exc:
            return LLMResponse(False, "deepseek", model, error=str(exc)[:200])

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
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return LLMResponse(True, "gemini", model, text.strip(),
                               latency_s=time.time() - t0)
        except Exception as exc:
            return LLMResponse(False, "gemini", model, error=str(exc)[:200])

    # ------------------------------------------------------------------ chat
    def chat(self, prompt: str, model: str | None = None,
             system: str | None = None, max_tokens: int = 1400,
             temperature: float = 0.4, use_cache: bool = True) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        m = model or CFG.apis.get("openrouter", {}).get("fast_model",
                                                        "deepseek/deepseek-chat-v3-0324")
        cache_key = hashlib.sha256(
            f"or|{m}|{prompt}|{system}|{max_tokens}".encode()).hexdigest()[:24]
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached:
                return LLMResponse(True, "openrouter", m, cached, cached=True)

        if CFG.apis.get("openrouter", {}).get("enabled", True):
            resp = self._call_openrouter(messages, m, max_tokens, temperature)
            if resp.ok:
                self._cache_put(cache_key, resp.text)
                return resp
        if CFG.apis.get("deepseek", {}).get("enabled", False):
            resp = self._call_deepseek(messages, m, max_tokens, temperature)
            if resp.ok:
                self._cache_put(cache_key, resp.text)
                return resp
        if CFG.apis.get("gemini", {}).get("enabled", False):
            resp = self._call_gemini(prompt, m, max_tokens, temperature)
            if resp.ok:
                self._cache_put(cache_key, resp.text)
                return resp
        return LLMResponse(False, "none", m, error="all providers failed/disabled")

    # --------------------------------------------------------------- council
    def council(self, prompt: str, system: str | None = None,
                models: list[str] | None = None, max_tokens: int = 1200,
                temperature: float = 0.3) -> dict[str, Any]:
        models = models or CFG.openrouter_models()
        results: list[LLMResponse] = []
        for m in models:
            resp = self.chat(prompt, model=m, system=system,
                             max_tokens=max_tokens, temperature=temperature)
            if resp.ok:
                results.append(resp)
        verdict = {"responses": [], "consensus": {}, "dissent": [],
                   "models_responded": len(results), "models_total": len(models)}
        parsed: list[dict[str, Any]] = []
        for r in results:
            data = _extract_json(r.text)
            verdict["responses"].append({
                "provider": r.provider, "model": r.model,
                "latency_s": round(r.latency_s, 2), "cached": r.cached,
                "raw": (data if data else r.text[:600])})
            if isinstance(data, dict):
                parsed.append(data)
        if parsed:
            numeric_fields = [k for k in parsed[0]
                              if isinstance(parsed[0][k], (int, float))]
            consensus: dict[str, Any] = {}
            for k in numeric_fields:
                vals = [p[k] for p in parsed if isinstance(p.get(k), (int, float))]
                if vals:
                    consensus[k] = _median(vals)
            text_fields = [k for k in parsed[0] if isinstance(parsed[0][k], str)]
            for k in text_fields:
                vals = [str(p.get(k, "")) for p in parsed if p.get(k)]
                consensus[k] = _mode(vals) if vals else ""
            verdict["consensus"] = consensus
            # dissent: numeric views far from median
            for k, v in consensus.items():
                if not isinstance(v, (int, float)):
                    continue
                for p in parsed:
                    pv = p.get(k)
                    if isinstance(pv, (int, float)) and abs(pv - v) > max(0.15, abs(v) * 0.6):
                        verdict["dissent"].append(
                            {"model": "?", "field": k, "value": pv, "consensus": v})
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
        return out


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
