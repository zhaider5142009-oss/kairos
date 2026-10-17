# KAIROS

**K**nowledge-**A**ugmented **I**ntelligence for **R**eturns & **O**ptimization **S**ystems — formerly *JARVIS v1/v1.5*, rebuilt from first principles as an institutional-grade market prediction and execution engine.

KAIROS predicts commodities, FX, global indices, crypto and the Pakistan Stock Exchange (all 484 listed names + blue-chip dossiers), sizes positions with layered risk gates, and executes through paper, MT5, and JS Global bridges.

---

## What it does

| Capability | Detail |
|---|---|
| **Prediction** | 10+ model battery: ARIMA, GARCH(1,1) MLE, Kalman filter, Ornstein-Uhlenbeck, Holt-Winters, Fourier-momentum, GBM/RF/ExtraTrees/Ridge classifiers, LSTM + GRU, Chronos-Bolt + TimesFM-2.5 + qlib LightGBM (WSL) |
| **Strategy library** | 100 strategies in 10 families, each voting with direction + strength + rationale |
| **Fusion** | Regime-affinity model weighting, disagreement shrinkage, strategy consensus blend |
| **Regime engine** | Hurst/ADX/vol-percentile regimes + KSE-100 bull/bear multiplier + FOMC/NFP/SBP-CPI event calendar |
| **Risk** | Historical/parametric/Cornish-Fisher VaR, CVaR, Kelly (capped), vol-targeted sizing, fail-closed risk gate |
| **Portfolio** | Cardinality-constrained mean-variance via quantum-inspired parallel-tempering annealer + coherence check |
| **LLM council** | Multi-model deliberation (OpenRouter) with median consensus — optional per prediction |
| **Execution** | Paper broker (slippage + commission + immutable JSONL journal), MT5 bridge (commodities/FX), JS Global ticket desk (PSX, human-confirm) |
| **Validation** | Walk-forward backtests with hysteresis, cost hurdle, vol targeting; paper sessions with P&L verdicts |

## Quick start

```bash
pip install -r requirements.txt

# one-shot predictions
python -m kairos predict GOLD WTI EURUSD
python -m kairos predict OGDC --council          # with LLM quant council

# PSX cross-sectional leaderboard (all 484 stocks)
python -m kairos scan

# quantum-assisted portfolio
python -m kairos portfolio GOLD,WTI,EURUSD,SPX,OGDC,HBL --capital 1000000

# paper trading validation (does the core signal turn profit?)
python -m kairos backtest GOLD,SILVER,WTI,EURUSD --days 120

# place one gated paper order
python -m kairos paper GOLD --notional 10000

# MT5 bridge check + order latency benchmark
python -m kairos mt5 --benchmark

# the web control tower (or just double-click the KAIROS desktop shortcut)
python -m kairos serve
```

## The web control tower

`python -m kairos serve` (or the **KAIROS** desktop shortcut) opens http://127.0.0.1:8600 — a single-page command center:

- **Home** — system health, live watchlist, plain-language quick asks
- **Predict** — full dossier per instrument: verdict, targets, models, strategies, risk, council
- **PSX Scan** — factor leaderboard over the whole exchange
- **Portfolio** — annealed allocation with per-instrument metrics
- **Paper Trading** — gated orders, account state, walk-forward validation, immutable journal
- **News** — seven-feed macro/Pakistan radar
- **System** — health, MT5 bridge, tradable universe browser

## Foundation models via WSL (optional, ~4 GB)

```bash
python -m kairos setup-wsl     # installs Chronos-Bolt, TimesFM-2.5, qlib into WSL
```

Everything degrades gracefully — KAIROS runs fully without WSL, without the LLM council, and with any single data source down.

## Configuration

Copy `config.example.yaml` to `config.yaml` and fill in your keys (OpenRouter etc.). The file is git-ignored — **never commit real keys**.

## Architecture

```
python -m kairos             CLI entry (predict/scan/portfolio/backtest/paper/mt5/feedback/serve/…)
launch_kairos.bat            one-click launcher (desktop shortcut target)
config.yaml                  all keys + engine tunables (git-ignored)
kairos/
  config.py universe.py      config singleton · 80-instrument registry
  data.py                    Yahoo v8 + TradingView PSX scanner + local PSX API + news + CSV store
  features.py                78 features · patterns · regimes · cross-sectional factors
  models_stat.py models_ml.py   statistical + ML + neural batteries
  strategies.py              100-strategy library (@strategy registry)
  ensemble.py                regime-adaptive fusion
  risk.py                    VaR/CVaR/Kelly/vol targeting
  quantum.py                 annealer portfolio + coherence check
  kse_regime.py calendar.py  KSE-100 regime · macro event calendar
  engine.py                  PredictionEngine orchestration
  feedback.py                prediction journal → realized outcomes → confidence calibration
  backtest.py                walk-forward validation + paper sessions
  execution.py               OrderIntent → RiskGate → PaperBroker / Mt5Bridge / JsGlobalTicket + audit journal
  wsl_bridge.py              Chronos/TimesFM/qlib subprocess bridge
  llm.py report.py cli.py    LLM council router · Rich terminal dossiers · command line
  api/server.py api/ui.html  FastAPI control tower + web UI
  wsl_scripts/kairos_models.py WSL-side model server
data/                        store cache, journal, paper account, LLM cache
reports/                     saved JSON dossiers
```

## Doctrine

1. **Never fabricate data** — missing source is reported, not guessed.
2. **Fail closed** — execution blocked unless every gate passes.
3. **Prove it on paper first** — no live order until walk-forward + paper sessions are green.
4. **Everything journaled** — append-only JSONL audit trail with idempotent client order IDs.

## Lineage

JARVIS v1/v1.5 (MiniMax build) → **KAIROS v2** (this repo). Full v1.5 source preserved in git history on GitHub: `zhaider5142009-oss/jarvis-reborn`.
