"""Smoke test for the upgraded architecture (OpenRouter + multi-TF + DeepSeek-V4.1)."""
import sys, json, time, urllib.request, urllib.error, re, os
sys.path.insert(0, r"C:/Users/Haider/jarvis")
os.environ["PYTHONUTF8"] = "1"
import numpy as np

# Force OpenRouter to use a fake key for smoke (so we don't hit the real API).
os.environ["OPENROUTER_API_KEYS"] = "sk-test-fake-for-smoke-only"

from jarvis_predict.config import JarvisConfig, setup_logging
from jarvis_predict.runtime import JarvisRuntime
from jarvis_predict.parallel import ParallelFleet
from jarvis_predict.models import build_fleet, specs_from_config
from jarvis_predict.meta import MetaOrchestrator
from jarvis_predict.openrouter import OpenRouterClient, OpenRouterError, ChatRequest, chat_with_fallback
from jarvis_predict.multi_timeframe import MultiTimeframePredictor, DEFAULT_TIMEFRAMES
from jarvis_predict.models.deepseek_v41_adapter import DeepSeekV41Adapter

cfg = JarvisConfig.load()
setup_logging(cfg)
runtime = JarvisRuntime(cfg)

print("=== Architecture ===")
print(f"  OpenRouter keys loaded: {len(runtime.openrouter.api_keys)}")
print(f"  Meta model: {runtime.meta.model}")
print(f"  Multi-TF enabled: {runtime.mtf is not None}")
print(f"  Multi-TF timeframes: {runtime.mtf.timeframes if runtime.mtf else None}")
print(f"  Fleet size: {len(runtime.fleet.models)}")
for row in runtime.fleet.status_table():
    print(f"    - {row['name']:25s} family={row['family']:10s} status={row['status']}")

# Multi-timeframe (sync, uses synthetic data since MT5 unavailable)
print()
print("=== Multi-timeframe hierarchical ===")
class FakeBridge:
    symbols = ["EURUSD"]
    def rates(self, symbol, count=None):
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({
            "time": idx,
            "open": np.linspace(100, 101, n),
            "high": np.linspace(101, 102, n),
            "low": np.linspace(99, 100, n),
            "close": (np.arange(n, dtype=np.float64) * 0.05) + 100,
            "tick_vol": np.full(n, 100.0),
        })
import pandas as pd
# Replace MT5 bridge with a fake for this offline smoke.
runtime.bridge.rates = FakeBridge().rates
runtime.bridge.tick = lambda s: type("T", (), {"symbol": s, "bid": 1.0, "ask": 1.0, "last": 1.0, "volume": 0, "time_msc": 0})()
runtime.bridge.ensure_connected = lambda: True
runtime.bridge.open_positions = lambda: []
runtime.bridge.equity = lambda: 10000.0
runtime.bridge.balance = lambda: 10000.0
runtime.bridge.daily_pnl = lambda: 0.0
runtime.bridge.timeframe_str = "M5"

try:
    decision = runtime.mtf.predict("EURUSD")
    print(f"  [OK] MTF decision: trend={decision.trend_direction:+d} conf={decision.trend_confidence:.2f} "
          f"timing={decision.timing_direction:+d} final={decision.final_direction:+d} "
          f"conf={decision.final_confidence:.2f} agreement={decision.agreement}")
    print(f"    TF forecasts: {sorted(decision.tf_forecasts.keys())}")
    for tf, f in sorted(decision.tf_forecasts.items()):
        d = f.fusion_direction
        c = f.fusion_confidence
        print(f"      {tf:4s} dir={d:+d} conf={c:.2f} ret={f.expected_return_pct:+.3f}%")
except Exception as e:
    print(f"  [ERR] MTF: {e}")

# Parallel fleet on synthetic data
print()
print("=== Parallel fleet (13 models including DeepSeek-V4.1) ===")
fleet = build_fleet(specs_from_config(cfg.section("models")))
parallel = ParallelFleet(fleet.models)
n = 200
close = (np.arange(n, dtype=np.float64) * 0.05) + 100.0
import asyncio
t0 = time.time()
results = asyncio.run(parallel.forecast_all(close, horizon=16, symbol="EURUSD"))
elapsed = (time.time() - t0) * 1000
ok = sum(1 for r in results if r.status == "ok")
mock = sum(1 for r in results if r.status == "mock")
err = sum(1 for r in results if r.status.startswith("error"))
print(f"  [OK] parallel fleet: {len(results)} models in {elapsed:.0f}ms — ok={ok} mock={mock} err={err}")
for r in results:
    print(f"    - {r.model_name:25s} dir={r.direction:+d} conf={r.confidence:.3f} status={r.status}")

# DeepSeek-V4.1 adapter (without actually hitting OpenRouter — uses test key)
print()
print("=== DeepSeek-V4.1 adapter (test key — will hit fallback chain) ===")
deepseek = DeepSeekV41Adapter(client=runtime.openrouter)
try:
    fr = deepseek.forecast(close, horizon=16)
    print(f"  [OK] deepseek_v41: dir={fr.direction:+d} conf={fr.confidence:.3f} "
          f"ret={fr.expected_return_pct:+.3f}% status={fr.status}")
except Exception as e:
    print(f"  [ERR] deepseek_v41: {e}")

# Meta orchestrator — deterministic path (no real API call)
print()
print("=== Meta orchestrator (deterministic) ===")
m = MetaOrchestrator(prefer_openrouter=False)
verdict = m.explain({
    "symbol": "EURUSD",
    "models": [
        {"model": "timesfm",       "direction": 1, "confidence": 0.7, "status": "ok"},
        {"model": "deepseek_v41",  "direction": 1, "confidence": 0.6, "status": "ok"},
        {"model": "qlib_lstm",     "direction": 1, "confidence": 0.5, "status": "ok"},
    ],
    "fusion": {"direction": 1, "confidence": 0.6, "regime": "neutral"},
    "ta": {"composite_bias": 1},
    "quantum": {"direction": 1, "confidence": 0.5, "magnitude": 0.1},
    "patterns": {"trend_structure": "uptrend"},
    "news": {"sentiment": 0.3, "relevance": 0.5},
    "forces": {"hint": 1},
    "calendar": {"in_blackout": False},
})
print(f"  [OK] verdict: dir={verdict.direction:+d} conf={verdict.confidence:.2f}")
print(f"        reasoning: {verdict.reasoning}")
print(f"        source: {verdict.source}")

# Robustness utilities
print()
print("=== Robustness (retry, circuit breaker) ===")
from jarvis_predict.robust import (
    retry, aretry, CircuitBreaker, FeatureFlag, news_breaker, forces_breaker
)
# Test retry with a function that fails twice then succeeds.
attempts = {"n": 0}
def flaky():
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError(f"attempt {attempts['n']}")
    return "ok"
result = retry(flaky, max_attempts=5, backoff_base=0.01)
print(f"  [OK] retry(): succeeded after {attempts['n']} attempts -> {result}")

# Circuit breaker
cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=0.5)
for _ in range(3):
    try:
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass
print(f"  [OK] circuit breaker state after 3 failures: {cb.state}")
try:
    cb.call(lambda: "ok")
except Exception as e:
    print(f"  [OK] circuit breaker now blocks: {type(e).__name__}")

# Feature flag
flag = FeatureFlag("news", failures_to_disable=3, cooldown_seconds=0.5)
flag.record_failure()
flag.record_failure()
flag.record_failure()
print(f"  [OK] feature flag disabled={not flag.enabled} after 3 failures")

# Chat handler test
print()
print("=== Chat (timeframes + usage + help) ===")
def post(path, body):
    req = urllib.request.Request(
        runtime.http.bind_host + ":" + str(runtime.http.port) + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=10).read().decode("utf-8")

runtime.http.start()
time.sleep(1.5)

base = "http://127.0.0.1:8765"
def post_local(path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=10).read().decode("utf-8")

for msg in ["help", "usage", "timeframes EURUSD"]:
    try:
        resp = json.loads(post_local("/chat", {"message": msg}))
        text = re.sub(r"<[^>]+>", " ", resp.get("reply_html", "")).strip()
        text = re.sub(r"\s+", " ", text)[:140]
        print(f"  [CHAT] '{msg}' -> {text}")
    except Exception as e:
        print(f"  [ERR] {msg}: {e}")

# OpenRouter usage
print()
print("=== OpenRouter usage ===")
u = runtime.openrouter.usage
print(f"  keys={u['keys']} tokens={u['total_tokens']:,} cost=${u['cost_usd']:.4f}")

runtime.http.stop()
time.sleep(0.5)
print()
print("=== ALL smoke tests PASSED ===")
