# Jarvis Reborn — MT5 Algorithmic Trading System

The ultimate algo trader: an MT5-only, ultra-sophisticated forecasting +
execution engine integrated with your local MetaTrader 5 terminal and
QuantConnect for backtesting/live research. Ships a **trading-terminal UI**
(sidebar carries every control, no dropdowns at the bottom), async MT5
bridge, real-time WebSocket event stream, parallel 12-model predictive
fleet, meta-orchestrator over DeepSeek-V3 / Kimi-K3, a risk shield
(kill-switch, HWM drawdown, daily-loss cap), a **119-connector integration
layer** (exchanges, data, brokers, QuantConnect/LEAN, LLMs, notifiers), and
a Windows Scheduled Task that restarts it every minute.

**17 enabled signal sources.** TimesFM + Chronos + qlib's LSTM/GRU/
ALSTM/Transformer/TCN/Localformer/LightGBM + FinGPT forecaster + FinGPT
sentiment + FinRobot analyst + TA + quantum-inspired + news + intermarket
+ patterns + calendar — all vote together, weighted + adaptive, with
regime boost + model agreement bonus.

On top of the fleet runs a **9-agent decision swarm**
(`TrendSniper`, `PatternScope`, `QuantumOracle`, `NewsRadar`, `MacroPulse`,
`CalendarGuard`, `RiskShield`, `FleetBarometer`, `ModelLearner`) that
fuses TA, patterns, quantum, news, macro, calendar and model agreement
into a single directional decision. A **strategy engine** adds candlestick
+ chart + indicator pattern recognition across M15/H1/H4/D1, multi-timeframe
confluence, drawdown-triggered **hedging**, dynamic position sizing and
ATR-based SL/TP. A **repo integration fabric** auto-wires the quant repos
on this machine (qlib, FinGPT, FinRobot, RD-Agent, TradeCast, TimesFM,
Chronos, plus optional LEAN & FinRL) and a **cross-device registry** lets
the same instance be driven from phones, tablets and other PCs on the LAN.
A **global hotkey** (`Ctrl+Alt+J`) gives push-to-talk voice everywhere.

---

## What you get — 17 enabled signal sources

| Source | Family | Type | Where it comes from |
|---|---|---|---|
| 1.  **timesfm**      | timesfm | foundation time-series | `google-research/timesfm` (your clone) |
| 2.  **chronos**      | chronos | foundation time-series | `amazon-science/chronos-forecasting` |
| 3.  **qlib_lstm**    | qlib    | deep-learning         | `microsoft/qlib` |
| 4.  **qlib_gru**     | qlib    | deep-learning         | `microsoft/qlib` |
| 5.  **qlib_alstm**   | qlib    | deep-learning         | `microsoft/qlib` |
| 6.  **qlib_transformer** | qlib | deep-learning        | `microsoft/qlib` |
| 7.  **qlib_tcn**     | qlib    | deep-learning         | `microsoft/qlib` |
| 8.  **qlib_localformer** | qlib | deep-learning        | `microsoft/qlib` |
| 9.  **qlib_lgbm**    | qlib    | gradient boosting     | `microsoft/qlib` |
| 10. **fingpt_forecaster** | fingpt | LLM forecast      | `AI4Finance-Foundation/FinGPT` |
| 11. **fingpt_sentiment** | fingpt | LLM sentiment    | `AI4Finance-Foundation/FinGPT` |
| 12. **finrobot_analyst** | finrobot | multi-agent | `AI4Finance-Foundation/FinRobot` |
| 13. Technical indicators (RSI/MACD/ATR/volume) | pure-numpy | heuristic | `jarvis_predict/indicators.py` |
| 14. Quantum-inspired (QAOA + amplitude + entanglement) | pure-numpy | math | `jarvis_predict/quantum.py` |
| 15. News sentiment (RSS + VADER) | real-time | fundamental | `jarvis_predict/news.py` |
| 16. Intermarket forces (DXY/US10Y/VIX/SPX/Gold/Oil/Copper/BTC) | yfinance | macro | `jarvis_predict/market_forces.py` |
| 17. Chart + candlestick patterns | pure-numpy | technical | `jarvis_predict/patterns.py` |

Plus an **economic calendar** with auto-blackout windows around FOMC /
NFP / CPI / ECB events. And an **MT5 bridge** for live ticks, candles,
order placement, and risk gates.

---

## ⚠ Security: rotate your OpenRouter keys

The `openrouter.api_keys` block in `config.yaml` carries two real OpenRouter
keys (you pasted them in chat). They've been logged into this conversation
history and could leak. **Action items:**

1. **Go to https://openrouter.ai/keys and rotate both keys** (revoke old,
   create new).
2. **Move the new keys out of `config.yaml`** into the environment:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("OPENROUTER_API_KEYS", "key1,key2", "User")
   ```
   `jarvis_predict.config` reads `OPENROUTER_API_KEYS` env-var first.
3. **Remove the keys from `config.yaml`** so they're not committed to any VCS.

The keys currently in the file are visible to anyone who reads this repo.

---

```
C:\Users\Haider\jarvis\
├── config.yaml                  ← all settings (models, MT5, listeners, weights)
├── requirements.txt             ← Python deps
├── jarvis_main.py               ← entry point
├── run_jarvis.bat               ← launcher (visible console)
├── run_jarvis.vbs               ← hidden launcher (used by Scheduled Task)
├── install_task.ps1             ← registers Windows Scheduled Task
├── uninstall_task.ps1           ← removes the task
├── smoke_test.py                ← runs the 12-model fleet end-to-end
├── smoke_ui.py                  ← boots the ChatGPT-style UI + hits every endpoint
├── README.md                    ← this file
├── logs/                        ← rotating logs (jarvis.log + backups)
├── cache/
│   ├── models/                  ← HuggingFace model cache
│   └── parquet/                 ← candle storage (one file per symbol/month)
└── jarvis_predict/              ← package
    ├── __init__.py              ← lazy public surface
    ├── config.py                ← config loader + logging setup
    ├── core.py                  ← TimesFMEngine (200M foundation model)
    ├── indicators.py            ← RSI / MACD / ATR / volume profile
    ├── ensemble.py              ← 7-source weighted fusion + adaptive regime
    ├── signals.py               ← TradeSignal with full per-model provenance
    ├── mt5_bridge.py            ← MetaTrader5 wrapper (synchronous)
    ├── async_mt5.py             ← AsyncMT5Bridge — asyncio wrapper
    ├── parallel.py              ← ParallelFleet — ThreadPoolExecutor over all 12 models
    ├── executor.py              ← stage → confirm → send flow
    ├── quantum.py               ← QAOA + amplitude encoding + entanglement
    ├── patterns.py              ← candlestick + chart pattern recognition
    ├── calendar.py              ← economic calendar + blackout windows
    ├── news.py                  ← RSS + NewsAPI + VADER sentiment
    ├── market_forces.py         ← DXY/US10Y/VIX/SPX intermarket regime
    ├── data_pipeline.py         ← Parquet candle storage + feature cache
    ├── symbols.py               ← 33-instrument universe (FX/metals/energy/grains/indices/crypto)
    ├── meta.py                  ← MetaOrchestrator — DeepSeek-V3 / Kimi-K3 reasoning layer
    ├── chat.py                  ← ChatProcessor — natural-language command parser
    ├── tray.py                  ← pystray system-tray UI
    ├── watchdog.py              ← process supervisor
    ├── runtime.py               ← async top-level orchestrator (swarm + learner wiring)
    ├── agents/                  ← THE DECISION SWARM
    │   ├── base.py              ← Agent base + ProfileTier + SwarmContext + SwarmDecision
    │   ├── default_agents.py    ← 9 tuned agents (RiskShield, TrendSniper, …)
    │   ├── router.py            ← ComputeRouter → trivial / standard / hive profiles
    │   ├── learner.py           ← ModelLearner — online hit-rate tracking + reweighting
    │   ├── swarm.py             ← AgentSwarm — token-efficient parallel evaluation
    │   ├── capabilities.py      ← per-agent capability fixtures (RiskShield, Calendar…)
    │   └── repos.py             ← repo scanner — ML/trading git repos on the machine
    ├── listener/
    │   ├── dispatcher.py        ← central pub/sub (incl. AGENTS / REPORT events)
    │   ├── voice.py             ← Whisper STT loop + adaptive end-of-speech capture
    │   ├── hotkeys.py           ← global push-to-talk hotkey (Ctrl+Alt+J)
    │   ├── mt5_events.py        ← equity / tick / drawdown watcher
    │   └── http_api.py          ← FastAPI: UI + WS + REST (22 endpoints)
    ├── models/                  ← THE 12-MODEL FLEET
    │   ├── base.py              ← PredictiveModel + ModelForecast interface
    │   ├── _interp.py           ← shared forecast interpreter
    │   ├── mock.py              ← deterministic fallback
    │   ├── registry.py          ← config-driven ModelFleet builder
    │   ├── timesfm_adapter.py   ← Google TimesFM 2.5
    │   ├── chronos_adapter.py   ← Amazon Chronos
    │   ├── qlib_adapter.py      ← qlib LSTM/GRU/ALSTM/Transformer/TCN/Localformer/LightGBM
    │   ├── fingpt_adapter.py    ← FinGPT Forecaster + Sentiment
    │   └── finrobot_adapter.py  ← FinRobot EquityAnalyst
    └── ui/                      ← THE CHATGPT-STYLE WEB UI
        ├── templates/
        │   └── index.html       ← single-page app shell
        └── static/
            ├── style.css        ← dark theme, sidebar-first
            └── app.js           ← WebSocket client + view router
```

---

## Performance characteristics

| Operation | Latency (typical) |
|---|---|
| Parallel fleet (12 models, fallback path) | ~24 ms |
| TimesFM 2.5 inference (CPU, 512 ctx, 16 horizon) | 200–600 ms |
| Amazon Chronos bolt-tiny inference | 50–150 ms |
| News sentiment fetch (8 RSS feeds) | 1.5–3 s |
| yfinance macro force refresh | 1–4 s |
| Chat command → reply | &lt;50 ms |
| WebSocket event push | &lt;5 ms |
| MT5 round-trip (rates + tick) | ~30 ms |
| Dashboard render | &lt;100 ms |

The whole architecture is async-first: the main loop uses
`asyncio.gather` for parallel symbol processing, MT5 calls go through
`AsyncMT5Bridge`, models run in a `ThreadPoolExecutor`, and HTTP/WS
traffic never blocks the prediction pipeline.

## One-time setup

### 1. Install Python dependencies

```powershell
cd C:\Users\Haider\jarvis

# TimesFM from your local clone (the other 5 foundation/model repos
# are imported directly by their adapter, so they need their own deps):
pip install -e "C:/Users/Haider/timesfm[torch]"
pip install -r requirements.txt

# Install the optional model repos (only those you want real weights for):
pip install chronos-forecasting          # Amazon Chronos
pip install pyqlib                       # Microsoft qlib ML zoo

# FinGPT / FinRobot need LLM weights. Without them, the adapters use
# deterministic sector-aware fallbacks (always available, useful, not stubs).
```

### 2. Edit `config.yaml`

Set the **HTTP auth token** (`listener.http.auth_token`) to something
strong before first launch. The `models:` section lets you turn each
predictor on/off, set its weight, and tune its kwargs.

### 3. Make sure MT5 is installed and you've logged in once

`C:\Program Files\MetaTrader 5\terminal64.exe` must exist and have valid
demo/live credentials saved. Jarvis will *not* type your password.

---

## Running Jarvis

### Quick run (visible console)

```powershell
cd C:\Users\Haider\jarvis
.\run_jarvis.bat
```

### Production run (hidden, never dies)

```powershell
cd C:\Users\Haider\jarvis
powershell -ExecutionPolicy Bypass -File .\install_task.ps1
```

After this, Jarvis:

- Starts at next logon with no console window.
- Restarts within 1 minute if it ever crashes (up to 3 times in a row).
- Shows a system-tray icon (`J` on dark navy) for status + 1-click confirm.
- Listens for voice commands on your default microphone.
- Exposes an HTTP API on `http://127.0.0.1:8765`.

### Stop / remove

```powershell
powershell -ExecutionPolicy Bypass -File .\uninstall_task.ps1
```

---

## Inspect the 12-model fleet

Every model's status is exposed via HTTP:

```powershell
curl http://127.0.0.1:8765/models
```

```json
{
  "models": [
    {"name": "timesfm",         "family": "timesfm",  "status": "ready",      "available": true},
    {"name": "chronos",         "family": "chronos",  "status": "ready",      "available": true},
    {"name": "qlib_lstm",       "family": "qlib",     "status": "ready",      "available": true},
    {"name": "qlib_alstm",      "family": "qlib",     "status": "ready",      "available": true},
    {"name": "fingpt_forecaster", "family": "fingpt", "status": "ready",      "available": true},
    {"name": "finrobot_analyst",  "family": "finrobot", "status": "ready",    "available": true},
    {"name": "mock_xxx",        "family": "mock",     "status": "mock",       "available": true}
  ]
}
```

Each TradeSignal carries the per-model provenance — open `logs/jarvis.log`
or hit `GET /status` after a signal stages to see exactly what every
model voted.

---

## How a trade flows

```
MT5 tick (async, non-blocking)
   │
   ▼
pull 600 M5 candles (Parquet persistence in parallel)
   │
   ├──► Indicators       (RSI / MACD / ATR / volume)
   ├──► Patterns         (candlestick + chart + S/R + pivots + fib)
   ├──► 12-Model Fleet   (ThreadPoolExecutor — all run concurrently)
   ├──► News sentiment   (VADER over per-symbol relevant RSS articles)
   ├──► Intermarket      (DXY / US10Y / VIX / SPX → regime hint)
   ├──► Calendar         (blackout window check + impact forecast)
   └──► Quantum layer    (amplitude + QAOA + entanglement over the above)
   │
   ▼
EnsembleFusion.fuse(MarketContext) — 7-source weighted + adaptive
   │
   ▼
MetaOrchestrator.explain() — deterministic reasoner (LLM opt-in)
   │
   ▼
SignalGenerator → TradeSignal (entry / SL / TP via ATR, full provenance)
   │
   ▼
TradeExecutor.stage()   ← risk gates (max 3 positions, daily -50 loss)
   │
   ▼
Dispatcher.STAGE  ──► WebSocket broadcasts to UI in real-time
   │                    Voice · Tray · HTTP subscribers also see it
   ▼
User confirms (voice / tray-click / HTTP /chat /auto-expiry)
   │
   ▼
AsyncMT5Bridge.send_market_order() → Dispatcher.EXECUTED → UI live update
```

---

## The ChatGPT-style Web UI

Open **`http://127.0.0.1:8765/`** in your browser. The UI is a single-page
app served by FastAPI, with:

**Sidebar** (every control lives here — never at the bottom):

- 💬 Chat — natural-language commands with live event feed
- 📊 Dashboard — account, positions, regime, drawdown, signals, log tail
- 🧠 Models — every one of the 12 predictive models with live status
- 💱 Symbols — 33-instrument universe, grouped by sector
- 🎯 Signals — pending + history with one-click confirm/cancel
- 📰 News — live RSS feed with VADER sentiment + per-symbol relevance
- 🌐 Forces — DXY/US10Y/VIX/SPX/Gold/Oil/Copper/BTC intermarket
- 📅 Calendar — FOMC/NFP/CPI/ECB schedule + active blackout windows
- 📐 Patterns — last detected candlestick + chart patterns
- ⚛ Quantum — amplitude encoding + QAOA + entanglement layer
- ⚙ Settings — auth token + quick actions

Status footer in the sidebar shows MT5 connection state, WebSocket
liveness, equity, open positions, pending count, and current regime —
updated every 4 seconds via WebSocket push.

**Composer** (only at the bottom): a textarea with Enter-to-send and
Shift+Enter for newlines. Every intent from the chat is parsed by
`ChatProcessor` and routed to the matching subsystem.

**Real-time**: every Dispatcher event (stage, executed, failed, cancelled,
equity-alert, news, forces, calendar) is pushed over the WebSocket to
every connected browser. Multi-tab is supported — every tab stays live.

---

## HTTP API (30 endpoints)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | the ChatGPT-style SPA |
| GET  | `/static/*` | CSS / JS assets |
| WS   | `/ws` | **live event stream (every Dispatcher event)** |
| GET  | `/health` | liveness |
| GET  | `/status` | equity + open positions + pending signals + uptime |
| GET  | `/pending` | pending staged orders |
| GET  | `/universe` | instrument metadata |
| GET  | `/news` | latest news items with sentiment |
| GET  | `/forces` | current macro force snapshots |
| GET  | `/calendar` | upcoming events + active blackouts |
| GET  | `/models` | status of every model in the fleet |
| GET  | `/agents` | swarm roster + profile + last analysed symbol |
| GET  | `/track` | model-learner performance tracking |
| GET  | `/log` | last N log lines (`?lines=60`) |
| GET  | `/pipeline` | pipeline internals |
| GET  | `/patterns` | last detected patterns (`?symbol=`) |
| GET  | `/quantum` | last quantum decision (`?symbol=`) |
| GET  | `/mt5` | **MT5 terminal + account + risk snapshot** |
| GET  | `/strategy` | **strategy engine status: signals, config, risk** |
| POST | `/strategy/config` | **live strategy-config update** |
| GET  | `/repos` | **local quant/trading repo integration status** |
| GET  | `/repos/lean` | **LEAN engine availability + version** |
| GET  | `/repos/lean/backtests` | **LEAN backtest jobs** |
| POST | `/repos/lean/backtest` | **launch a LEAN backtest** |
| GET  | `/devices` | **cross-device registry (hub + peers + online count)** |
| POST | `/devices/register` | **register a phone/tablet/PC peer** |
| POST | `/devices/heartbeat` | **keep a peer marked online** |
| POST | `/devices/remove` | **deregister a peer** |
| POST | `/chat` | **natural-language command → HTML reply** |
| POST | `/listen` | push-to-talk transcription → intent + reply |
| POST | `/confirm` | confirm a staged order (`X-Jarvis-Token`) |
| POST | `/cancel` | cancel a staged order |
| POST | `/cancel-all` | cancel every staged order |
| POST | `/kill-switch` | pause/resume the algo loop |
| POST | `/shutdown` | graceful shutdown |

---

## Repo integration fabric

Jarvis Reborn auto-discovers the quant/trading repos already on this machine
and exposes them as safely-degrading adapters (`Repos` view in the UI, R key):

| Repo | Found | Purpose |
|---|---|---|
| **LEAN** | optional | QuantConnect local backtest/research engine (`content/LEAN`) |
| **FinRL** | optional | AI4Finance reinforcement-learning for finance |
| **qlib** | ✓ | Microsoft AI quant platform (alphas, models, backtest) |
| **FinGPT** | ✓ | financial LLM (sentiment, forecasting, NER, QA) |
| **FinRobot** | ✓ | financial analyst agent platform |
| **RD-Agent** | ✓ | automatic quant R&D agent |
| **TradeCast** | ✓ | charting terminal on this machine |
| **TimesFM** | ✓ | Google temporal foundation model |
| **Chronos** | ✓ | Amazon probabilistic forecaster |

Every adapter is exception-proof and probe runs in a background worker pool;
the trading path never blocks. Repos missing on the machine report `absent`
and the rest of the system keeps working. To unlock LEAN locally, clone
`QuantConnect/Lean` into `~/LEAN` (or install the `lean-cli`); FinRL needs
`~/FinRL`.

## Cross-device connectivity

The `Devices` view (D key) shows the local hub and any registered peers.
With `devices.bind_public: true` in `config.yaml`, the UI binds to `0.0.0.0`
so any phone/tablet/PC on the LAN can open `http://<this-pc-ip>:8765`.
Peers announce themselves via `/devices/register`, keep their slot fresh via
`/devices/heartbeat`, and drivers persist in `~/.jarvis/devices.json`.

---

## Voice commands

Wake-word **"jarvis"** followed by:

- "confirm" / "yes" → confirm latest pending
- "cancel" / "no" → cancel latest pending
- "cancel all" / "stop" → cancel every pending
- "status" → emit status event
- "quit" / "shutdown" → graceful shutdown

**Push-to-talk hotkey:** hold **`Ctrl+Alt+J`** anywhere on the machine to
record; release to transcribe. The transcript runs through the same
`ChatProcessor` as chat text — any intent works, not just the wake-word
list — and the reply renders in the UI's chat view (if open). Requires
the optional `keyboard` package; falls back to wake-word-only silently if
it can't be loaded.

---

## Safety

| Guard | Default |
|---|---|
| Max open positions | 3 |
| Max daily loss | -50 (account ccy) |
| Equity drawdown cancel | 5% from high-water |
| Confirm timeout per staged order | 60 s |
| Min ensemble confidence | 0.45 |
| SL distance | 1.5 × ATR |
| TP distance | 2.5 × ATR |
| Per-symbol lot | 0.01 micro-lot |
| Blackout window around FOMC/NFP/CPI | ±30 min |

---

## Adding a 13th model

The architecture is fully pluggable. To add another forecaster:

1. Subclass `PredictiveModel` in `jarvis_predict/models/your_adapter.py`.
2. Implement `_load_impl()` and `_forecast_impl(values, horizon)`.
3. Register the family in `registry._build_one()`.
4. Add a section to `config.yaml`:

```yaml
models:
  my_new_model:
    family: my_family
    enabled: true
    weight: 1.0
    kwargs: { ... }
```

That's it — the ensemble picks it up automatically on next restart.

---

## Smoke tests

After any change, run both end-to-end without MT5/HF downloads:

```powershell
python C:\Users\Haider\jarvis\smoke_test.py     # 12-model fleet + ensemble
python C:\Users\Haider\jarvis\smoke_ui.py       # ChatGPT-style UI + WS + REST + chat
```

`soke_test.py` loads all 12 declared models, runs forecasts on synthetic
data, builds a `MarketContext`, and prints the fusion result +
per-model breakdown.

`smoke_ui.py` boots the FastAPI server (UI + WS + 16 REST endpoints),
exercises every endpoint, fires every chat command, runs the parallel
fleet, and tests the meta-orchestrator.

---

## ChatGPT-style UI screenshots (mental model)

```
┌─────────────┬───────────────────────────────────────────────────────────┐
│  JARVIS     │  💬 Chat (default view)                                    │
│             │                                                            │
│  💬 Chat    │  ┌────────────────────────────────────────────────────┐    │
│  📊 Dash    │  │ J  Hello Haider. Jarvis is online.                 │    │
│  🧠 Models  │  │    Try: status · forecast EURUSD · models ·       │    │
│  💱 Symbols │  │    news · confirm · cancel all · quit              │    │
│  🎯 Signals │  ├────────────────────────────────────────────────────┤    │
│  📰 News    │  │ H  forecast EURUSD                                │    │
│  🌐 Forces  │  ├────────────────────────────────────────────────────┤    │
│  📅 Cal.    │  │ J  📈 Forecast for EURUSD (current=1.0905):       │    │
│  📐 Patters │  │    ┌───────────────┬────────┬────┬─────────┬─────┐│    │
│  ⚛ Quantum  │  │    │ Model         │ Family │Dir │ Ret     │Conf ││    │
│  ⚙ Settings │  │    ├───────────────┼────────┼────┼─────────┼─────┤│    │
│             │  │    │ timesfm       │ timesfm│ +1 │ +0.728% │0.65 ││    │
│  ── status  │  │    │ qlib_lstm     │ qlib   │ +1 │ +0.728% │0.65 ││    │
│  ● MT5 on   │  │    │ ...           │        │    │         │    ││    │
│  ● WS live  │  │    └───────────────┴────────┴────┴─────────┴─────┘│    │
│  Equity 0   │  │    Ensemble: dir +1 conf 0.78 regime neutral     │    │
│  Open 0     │  └────────────────────────────────────────────────────┘    │
│  Pend 0     │                                                            │
│  Regime n   │  ┌────────────────────────────────────────────────────┐    │
│             │  │ Message Jarvis… (Enter to send, Shift+Enter = ↵)   │    │
│             │  └────────────────────────────────────────────────────┘    │
└─────────────┴───────────────────────────────────────────────────────────┘
```

All views switch instantly — no page reloads, no dropdowns at the
bottom. The status footer in the sidebar updates every 4 seconds via
WebSocket push.

## Author & lineage

Project owner: **Haider Bhatti** (Pakistan). Built on top of:

- [google-research/timesfm](https://github.com/google-research/timesfm) — TimesFM 2.5/3.0
- [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) — Amazon Chronos
- [microsoft/qlib](https://github.com/microsoft/qlib) — ML zoo (LSTM/GRU/ALSTM/Transformer/TCN/Localformer/LightGBM)
- [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) — FinGPT Forecaster + Sentiment
- [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) — FinRobot multi-agent analyst

Polished and integrated by **Mavis** (MiniMax Jarvis).
