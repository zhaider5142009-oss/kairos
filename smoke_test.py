"""Smoke test for the 12-model predictive fleet."""
import sys, json
sys.path.insert(0, r"C:/Users/Haider/jarvis")
import numpy as np

from jarvis_predict.config import JarvisConfig
from jarvis_predict.models import (
    PredictiveModel, ModelForecast, MockModel,
    ModelSpec, ModelFleet, build_fleet, specs_from_config,
    list_qlib_algorithms,
)
from jarvis_predict.models.timesfm_adapter import TimesFMAdapter
from jarvis_predict.models.chronos_adapter import ChronosAdapter
from jarvis_predict.models.qlib_adapter import QlibAdapter
from jarvis_predict.models.fingpt_adapter import FinGPTForecasterAdapter, FinGPTSentimentAdapter
from jarvis_predict.models.finrobot_adapter import FinRobotAnalystAdapter
from jarvis_predict.ensemble import EnsembleFusion, MarketContext
from jarvis_predict.signals import SignalGenerator
from jarvis_predict.indicators import TechnicalIndicators
from jarvis_predict.patterns import detect_patterns
from jarvis_predict.quantum import QuantumDecisionEngine
from jarvis_predict.core import ForecastResult
from jarvis_predict.runtime import JarvisRuntime

cfg = JarvisConfig.load()
print("[OK] config loaded")

model_cfg = cfg.section("models")
print("[OK] declared models in config.yaml:")
for name, entry in model_cfg.items():
    if isinstance(entry, dict):
        print(f"     - {name:25s} family={entry.get('family'):10s} enabled={entry.get('enabled')} weight={entry.get('weight')}")

specs = specs_from_config(model_cfg)
print(f"[OK] parsed {len(specs)} ModelSpec entries")

fleet = build_fleet(specs)
print(f"[OK] fleet built: {len(fleet.models)} models")
for row in fleet.status_table():
    avail = "[OK]" if row["available"] else "[--]"
    print(f"     {avail} {row['name']:25s} family={row['family']:10s} status={row['status']}")

n = 200
close = (np.arange(n, dtype=np.float64) * 0.05) + 100.0
print("[OK] running forecast on all models...")
results = fleet.forecast_all(close, horizon=16, symbol="EURUSD")
print(f"[OK] got {len(results)} model forecasts")
for r in results:
    print(f"     - {r.model_name:25s} dir={r.direction:+d} ret={r.expected_return_pct:+.3f}% conf={r.confidence:.3f} status={r.status:10s} ms={r.inference_ms:.1f}")

ti = TechnicalIndicators.from_config(cfg)
indicators_snap = ti.compute(open_=close, high=close+0.05, low=close-0.05, close=close, volume=np.full(n, 2000.0))
ps = detect_patterns(open_=close, high=close+0.05, low=close-0.05, close=close)
qd = QuantumDecisionEngine(seed=42)
q = qd.decide(per_asset_signal=[0.7, 0.5, 0.6], per_asset_phase=[0, 0, 0],
              correlation_matrix=np.array([[1, 0.3, 0.5], [0.3, 1, 0.4], [0.5, 0.4, 1]]))

canonical = next((r for r in results if r.family == "timesfm"), results[0])
forecast = ForecastResult(
    point=canonical.point,
    quantiles=canonical.quantiles,
    horizon=canonical.horizon,
    context_len=n,
    direction=canonical.direction,
    expected_return_pct=canonical.expected_return_pct,
    confidence=canonical.confidence,
    inference_ms=canonical.inference_ms,
)

ctx = MarketContext(
    symbol="EURUSD",
    forecast=forecast,
    indicators=indicators_snap,
    quantum=q,
    pattern=ps,
    news_sentiment=0.3, news_relevance=0.5,
    intermarket_hint=1, calendar_impact=1, in_blackout=False,
    model_forecasts=results,
)
ens = EnsembleFusion.from_config(cfg)
fusion = ens.fuse(ctx)
print(f"[OK] fusion: direction={fusion.direction:+d} confidence={fusion.confidence:.3f} regime={fusion.regime}")
print("     model_consensus:", fusion.model_consensus)

sg = SignalGenerator.from_config(cfg)
sig = sg.generate(symbol="EURUSD", fusion=fusion, ctx=ctx, current_price=float(close[-1]))
d = sig.to_dict()
print(f"[OK] signal: direction={sig.direction:+d} entry={sig.entry_price:.5f} SL={sig.stop_loss:.5f} TP={sig.take_profit:.5f} R={sig.r_multiple:.2f}")
print(f"[OK] signal JSON: {len(json.dumps(d))} bytes")
print(f"     per-model forecasts in signal: {len(sig.models)}")
for m in sig.models[:14]:
    print(f"       - {m['model']:25s} dir={m['direction']:+d} conf={m['confidence']:.3f} status={m['status']}")
if len(sig.models) > 14:
    print(f"       ... and {len(sig.models) - 14} more")

runtime = JarvisRuntime(cfg)
print("[OK] JarvisRuntime constructed")
print(f"     fleet size: {len(runtime.fleet.models)}")
try:
    app_obj = getattr(runtime.http, "app", None)
    inner = getattr(app_obj, "app", app_obj)
    routes = sorted({r.path for r in getattr(inner, "routes", [])})
    print(f"     http routes ({len(routes)}): {routes[:6]}{'...' if len(routes) > 6 else ''}")
except Exception as e:
    print(f"     http routes: (unreadable: {e})")

print()
print("=== ALL 12-model fleet tests passed ===")
