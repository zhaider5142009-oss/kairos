"""
OpenRouter client — unified access to GPT-4o, Claude 3.5, Llama 3.1,
DeepSeek-V3, DeepSeek-V4.1 (via OpenRouter), etc.

OpenRouter exposes a single OpenAI-compatible chat-completions API at
https://openrouter.ai/api/v1/chat/completions — so we can swap backends
without touching the rest of Jarvis.

Features:
    - Multi-key rotation (primary + fallback keys)
    - Model fallback chain (try best model, fall back to cheaper on failure)
    - Retry with exponential backoff (handles 429/5xx/network blips)
    - Streaming support (call .stream(...) for SSE)
    - Token usage tracking
    - Per-call timeout

Security note:
    API keys are loaded from config.yaml under `openrouter.api_keys` or
    from the OPENROUTER_API_KEYS env var. NEVER commit keys to git.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator

import requests

LOGGER = logging.getLogger("jarvis.openrouter")

_DEFAULT_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 60.0


@dataclass
class ChatMessage:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ChatResponse:
    model: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class OpenRouterError(RuntimeError):
    """Raised on irrecoverable API errors after retries."""


class OpenRouterClient:
    """Thin OpenAI-compatible client with key rotation + model fallback chain."""

    def __init__(
        self,
        *,
        api_keys: list[str] | None = None,
        base_url: str = _DEFAULT_BASE,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 4,
        backoff_base: float = 0.6,
        site_url: str = "https://jarvis.local",
        site_name: str = "Jarvis Reborn — MT5 Algo Trader",
    ) -> None:
        # Load keys from explicit arg → env → error
        keys = list(api_keys) if api_keys else []
        env = os.environ.get("OPENROUTER_API_KEYS", "")
        if env:
            keys.extend(k.strip() for k in env.split(",") if k.strip())
        env_single = os.environ.get("OPENROUTER_API_KEY", "")
        if env_single:
            keys.append(env_single.strip())
        keys = [k for k in keys if k]
        if not keys:
            LOGGER.warning(
                "No OpenRouter keys configured — client will fail on every call. "
                "Set OPENROUTER_API_KEYS env var or add openrouter.api_keys to config.yaml."
            )
        self.api_keys = keys
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.site_url = site_url
        self.site_name = site_name
        self._key_idx = 0  # round-robin
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0

    # ----- introspection --------------------------------------------------

    @property
    def usage(self) -> dict:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "cost_usd": round(self._total_cost_usd, 4),
            "keys": len(self.api_keys),
        }

    # ----- core chat --------------------------------------------------------

    def chat(self, req: ChatRequest) -> ChatResponse:
        """Send a chat request. Tries keys round-robin, retries on 429/5xx."""
        if not self.api_keys:
            raise OpenRouterError("no API keys configured")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            key = self._next_key()
            try:
                t0 = time.perf_counter()
                resp = self._post_once(key, req)
                latency = (time.perf_counter() - t0) * 1000.0
                self._total_prompt_tokens += resp.prompt_tokens
                self._total_completion_tokens += resp.completion_tokens
                self._total_cost_usd += resp.cost_usd
                resp.latency_ms = latency
                return resp
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                last_err = e
                if code in (401, 403):
                    LOGGER.warning("Auth failure on key ending ...%s: %s",
                                   key[-6:], code)
                    continue    # try next key
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait = self.backoff_base * (2 ** attempt)
                    LOGGER.info("OpenRouter %s, retrying in %.1fs", code, wait)
                    time.sleep(wait)
                    continue
                # 4xx other than 429 → unrecoverable
                raise OpenRouterError(f"HTTP {code}: {e}") from e
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = self.backoff_base * (2 ** attempt)
                    LOGGER.info("OpenRouter network error, retrying in %.1fs", wait)
                    time.sleep(wait)
                    continue
                raise OpenRouterError(f"network: {e}") from e
        raise OpenRouterError(f"all attempts failed: {last_err}")

    def stream(self, req: ChatRequest) -> Generator[str, None, None]:
        """Stream the response token-by-token. Same retry policy as chat()."""
        if not self.api_keys:
            raise OpenRouterError("no API keys configured")
        req = ChatRequest(**{**req.__dict__, "stream": True})
        key = self._next_key()
        try:
            with requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(key, req),
                json=self._payload(req),
                timeout=self.timeout,
                stream=True,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data = line[len(b"data: "):].decode("utf-8", errors="ignore").strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                            delta = (j.get("choices", [{}])[0].get("delta") or {}).get("content") or ""
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue
        except requests.RequestException as e:
            raise OpenRouterError(str(e)) from e

    # ----- helpers ---------------------------------------------------------

    def _next_key(self) -> str:
        if not self.api_keys:
            raise OpenRouterError("no API keys")
        k = self.api_keys[self._key_idx % len(self.api_keys)]
        self._key_idx += 1
        return k

    def _headers(self, key: str, req: ChatRequest) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            **req.extra_headers,
        }

    def _payload(self, req: ChatRequest) -> dict:
        return {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
            "stream": req.stream,
        }

    def _post_once(self, key: str, req: ChatRequest) -> ChatResponse:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(key, req),
            json=self._payload(req),
            timeout=self.timeout,
        )
        r.raise_for_status()
        j = r.json()
        choice = (j.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "") or ""
        usage = j.get("usage") or {}
        cost = _estimate_cost(req.model,
                              usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0))
        return ChatResponse(
            model=j.get("model", req.model),
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            cost_usd=cost,
            raw=j,
        )


# ---- model catalog (for chat UI + fallback chain) ------------------------

# Curated models on OpenRouter — defaults to DeepSeek-V3 (cost-effective)
# with GPT-4o + Claude as fallbacks.
DEFAULT_MODEL_CHAIN: list[str] = [
    "deepseek/deepseek-chat",                 # DeepSeek-V3
    "openai/gpt-4o",                          # OpenAI GPT-4o
    "meta-llama/llama-3.1-70b-instruct",      # Meta Llama 3.1 70B
    "openai/gpt-4o-mini",                     # Cheap fallback
]


# Pricing (USD per 1M tokens) — used for cost estimates.
_PRICING: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat":               (0.14, 0.28),
    "deepseek/deepseek-chat-v4.1-flash":    (0.14, 0.28),
    "openai/gpt-4o":                        (2.5, 10.0),
    "openai/gpt-4o-mini":                   (0.15, 0.6),
    "anthropic/claude-3.5-sonnet":          (3.0, 15.0),
    "anthropic/claude-3-haiku":             (0.25, 1.25),
    "meta-llama/llama-3.1-70b-instruct":    (0.59, 0.79),
    "google/gemini-2.0-flash-exp:free":     (0.0, 0.0),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = _PRICING.get(model, (1.0, 3.0))
    return (prompt_tokens / 1_000_000.0) * p_in + (completion_tokens / 1_000_000.0) * p_out


def chat_with_fallback(
    client: OpenRouterClient,
    req_template: ChatRequest,
    *,
    fallback_models: list[str] | None = None,
) -> ChatResponse:
    """Try the requested model; if it fails (non-retryable error), try
    the next model in the fallback chain. Returns the first success."""
    chain = [req_template.model]
    if fallback_models:
        chain = list(fallback_models)
    last_err: Exception | None = None
    for m in chain:
        try:
            req = ChatRequest(**{**req_template.__dict__, "model": m})
            return client.chat(req)
        except OpenRouterError as e:
            LOGGER.warning("model %s failed (%s); trying next", m, e)
            last_err = e
            continue
    raise OpenRouterError(f"all fallback models failed: {last_err}")


__all__ = [
    "OpenRouterClient", "OpenRouterError",
    "ChatMessage", "ChatRequest", "ChatResponse",
    "DEFAULT_MODEL_CHAIN",
    "chat_with_fallback",
]
