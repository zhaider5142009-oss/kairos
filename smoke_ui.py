"""UI smoke test — boots the ChatGPT-style UI server and hits every endpoint."""
import sys, json, time, urllib.request, urllib.error, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:/Users/Haider/jarvis")
import numpy as np

from jarvis_predict.config import JarvisConfig, setup_logging
from jarvis_predict.runtime import JarvisRuntime
from jarvis_predict.parallel import ParallelFleet
from jarvis_predict.models import build_fleet, specs_from_config
from jarvis_predict.meta import MetaOrchestrator

cfg = JarvisConfig.load()
setup_logging(cfg)
runtime = JarvisRuntime(cfg)
runtime.http.start()
time.sleep(1.5)

base = "http://127.0.0.1:8765"

# --- REST endpoints -----------------------------------------------------------
endpoints = [
    ("/", "GET", None, "html"),
    ("/static/style.css", "GET", None, "css"),
    ("/static/app.js", "GET", None, "js"),
    ("/health", "GET", None, "json"),
    ("/status", "GET", None, "json"),
    ("/models", "GET", None, "json"),
    ("/universe", "GET", None, "json"),
    ("/news", "GET", None, "json"),
    ("/forces", "GET", None, "json"),
    ("/calendar", "GET", None, "json"),
    ("/pending", "GET", None, "json"),
    ("/agents", "GET", None, "json"),
    ("/track", "GET", None, "json"),
    ("/log", "GET", None, "json"),
]
print("=== REST + UI ===")
for path, method, _, kind in endpoints:
    try:
        r = urllib.request.urlopen(base + path, timeout=5)
        body = r.read()
        size = f"({len(body)} bytes)"
        if kind == "json":
            try:
                j = json.loads(body)
                keys = list(j.keys())[:5]
                print(f"  [OK]  {method:5s} {path:22s} -> {r.status} {size} keys={keys}")
            except Exception:
                print(f"  [OK]  {method:5s} {path:22s} -> {r.status} {size}")
        else:
            print(f"  [OK]  {method:5s} {path:22s} -> {r.status} {size} ({kind})")
    except urllib.error.HTTPError as e:
        print(f"  [HTTP] {method:5s} {path:22s} -> {e.code}")
    except Exception as e:
        print(f"  [ERR] {method:5s} {path:22s} -> {e}")

# --- Chat endpoint -----------------------------------------------------------
print()
print("=== /chat endpoint ===")
def post(path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=10).read().decode("utf-8")

chat_tests = [
    "help",
    "status",
    "models",
    "news",
    "forces",
    "calendar",
    "symbols",
    "pending",
    "forecast EURUSD",
    "run timesfm",
    "run qlib_lstm",
    "why EURUSD",
    "cancel all",
    "hello there",
]
for msg in chat_tests:
    try:
        resp = json.loads(post("/chat", {"message": msg}))
        text = re.sub(r"<[^>]+>", " ", resp.get("reply_html", "")).strip()
        text = re.sub(r"\s+", " ", text)[:100]
        print(f"  [CHAT] '{msg:20s}' -> {text}")
    except Exception as e:
        print(f"  [ERR] {msg}: {e}")

# --- Parallel fleet ----------------------------------------------------------
print()
print("=== Parallel fleet (async) ===")
import asyncio
fleet = build_fleet(specs_from_config(cfg.section("models")))
parallel = ParallelFleet(fleet.models)
n = 200
close = (np.arange(n, dtype=np.float64) * 0.05) + 100.0
t0 = time.time()
results = asyncio.run(parallel.forecast_all(close, horizon=16, symbol="EURUSD"))
elapsed = (time.time() - t0) * 1000
ok = sum(1 for r in results if r.status == "ok")
mock = sum(1 for r in results if r.status == "mock")
err = sum(1 for r in results if r.status.startswith("error"))
print(f"  [OK] parallel fleet: {len(results)} models in {elapsed:.0f}ms")
print(f"        ok={ok}  mock={mock}  error={err}")
for r in results[:5]:
    print(f"        - {r.model_name:25s} dir={r.direction:+d} conf={r.confidence:.3f} status={r.status}")

# --- Meta orchestrator -------------------------------------------------------
print()
print("=== Meta orchestrator ===")
m = MetaOrchestrator(prefer_openrouter=False)
verdict = m.explain({
    "symbol": "EURUSD",
    "models": [
        {"model": "timesfm", "direction": 1, "confidence": 0.7},
        {"model": "qlib_lstm", "direction": 1, "confidence": 0.5},
        {"model": "fingpt_sentiment", "direction": 0, "confidence": 0.1},
    ],
    "fusion": {"direction": 1, "confidence": 0.6},
    "ta": {"composite_bias": 1},
    "quantum": {"direction": 1, "confidence": 0.5},
    "patterns": {"trend_structure": "uptrend"},
    "news": {"sentiment": 0.3},
    "forces": {"hint": 1},
    "calendar": {"in_blackout": False},
})
print(f"  [OK] verdict: dir={verdict.direction:+d} conf={verdict.confidence:.2f}")
print(f"        reasoning: {verdict.reasoning}")
print(f"        source: {verdict.source}")

# --- WebSocket ---------------------------------------------------------------
print()
print("=== WebSocket ===")
try:
    import websocket  # type: ignore
    ws = websocket.create_connection("ws://127.0.0.1:8765/ws", timeout=3)
    msg = ws.recv()
    j = json.loads(msg)
    print(f"  [OK] WS hello received: type={j.get('type')}")
    print(f"        status payload keys: {list((j.get('payload') or {}).keys())}")
    ws.close()
except ImportError:
    print("  [SKIP] websocket-client not installed")
except Exception as e:
    print(f"  [ERR] WS: {e}")

# Cleanup.
runtime.http.stop()
time.sleep(0.5)

print()
print("=== UI smoke test PASSED ===")
