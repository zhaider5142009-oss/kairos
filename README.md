# KAIROS

[![CI](https://github.com/zhaider5142009-oss/kairos/actions/workflows/ci.yml/badge.svg)](https://github.com/zhaider5142009-oss/kairos/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**K**nowledge-**A**ugmented **I**ntelligence for **R**eturns & **O**ptimization **S**ystems — formerly *JARVIS v1/v1.5*, rebuilt from first principles as an institutional-grade market prediction and execution engine.

KAIROS forecasts **crypto** (22 majors: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC and more), commodities, FX, global indices and the Pakistan Stock Exchange (all 484 listed names + blue-chip dossiers), sizes positions with layered risk gates, and executes through paper, MT5, and JS Global bridges. A **100-instrument** tradable universe, all free and open source under MIT.

> **Status, stated honestly.** There is **no forward track record yet**: of every prediction
> that has matured, all were non-trades, so no hit rate exists. The walk-forward sweep is
> profitable on 8 of 14 instruments and **loses money on PSX equities net of local costs**.
> Numbers are regenerated from the journals by one command — see
> [docs/investors/TRACK_RECORD.md](docs/investors/TRACK_RECORD.md). Not a fund, not advice,
> no return promised.

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
| **LLM council** | Multi-model deliberation across OpenRouter + direct Kimi/Moonshot, median consensus with dissent recorded — optional per prediction |
| **Fugu conductor** | One fused verdict per instrument from all experts — ensemble score, model agreement, regime, realized calibration, doctrine R:R geometry, risk posture — emitting probability-weighted edge in R-multiples; the entry bar adapts automatically to KAIROS' own realized hit-rate |
| **Execution** | Paper broker (true cash ledger, slippage + market-aware commission + immutable JSONL journal), MT5 bridge (commodities/FX, attach-only — never launches or blocks the terminal), JS Global bridge (PSX: read-only sync + **double-keyed order desk** — stage→ack with the broker PIN, human tickets by default), PMX futures scaffold (tickets-only, account-ready) |
| **Command plane** | Fugu-style orchestration tailored to a quant desk: plain language → a 1–4 step plan over a frozen registry of 12 read-only tools → executed under a wall-clock budget → answered with the trace. Cross-step references resolve from the previous step's own output. **No tool in the registry can place, cancel, arm or move anything**, so no plan — human, deterministic or model-invented — can reach money; verbs that ask for movement get directions instead. Every figure is rendered by the tool that produced it; the LLM may change the voice, never the numbers, and the plane answers in full with no provider enabled |
| **Validation** | Walk-forward backtests with hysteresis, cost hurdle, vol targeting; paper sessions with a proper cash ledger + PSX commission model; realized outcomes feed back every 12 cycles so the system's own hit-rate tunes its entry bar |

## Quick start

```bash
# clone, then install the package (editable) with core + test extras
git clone https://github.com/zhaider5142009-oss/kairos.git && cd kairos
pip install -e ".[test]"          # or: pip install -r requirements.txt

# (optional) full battery: neural models, quantum, MT5, signal cards
pip install -e ".[all]"
# (optional) just the shareable PNG cards: pip install -e ".[card]"

# or invoke the console script once installed
kairos --help

# FIRST RUN — three commands, no documentation required
kairos setup      # writes git-ignored config.yaml from the example, pulls keys
                  # from the environment, prompts (masked) for the rest
kairos doctor     # what it actually checked: python floor, deps, the config file
                  # AND whether .gitignore really covers it, key slots still
                  # placeholder, writable state dirs, market feeds, live posture
kairos up         # start the tower and open the desk

# ask the engine a question in plain language — it plans which experts to run,
# executes them read-only, and prints the trace
kairos ask "screen the PSX and tell me which name is worth trading today"
kairos ask "backtest HBL over 120 days" --plain

# one-shot predictions
python -m kairos predict BTC ETH SOL          # crypto
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

# JS Global orders (PSX) — stage first, then ack with the broker's own PIN.
# Staging itself refuses until js_global.allow_orders is true.
python -m kairos order OGDC --side SELL --qty 50 --price 124.50
python -m kairos order --ack <staged-id>      # prompts for the PIN, never echoes it
python -m kairos order --list                 # gate status + what is pending

# MT5 bridge check + order latency benchmark (attaches to a running terminal only)
python -m kairos mt5 --benchmark

# Fugu conductor: fused multi-expert verdict + probability-weighted edge
python -m kairos fugu            # whole board (autopilot watchlist)
python -m kairos fugu GOLD       # one instrument in depth

# the web control tower (or just double-click the KAIROS desktop shortcut)
python -m kairos serve
```

## The web control tower

`python -m kairos serve` (or the **KAIROS** desktop shortcut) opens http://127.0.0.1:8600 — a single-page command center:

- **Home** — ChatGPT-style chat console: ask anything ("where is gold this week?", "best Pakistani stocks?", "allocate 1M across GOLD WTI SPX") and the full engine answers in bubbles with inline verdict cards. Harder questions are compiled into expert steps by the command plane and the reply carries an expandable trace — which tool ran, what it argued, how long it took, and what it said
- **Predict** — full dossier per instrument: verdict, targets, models, strategies, risk, council
- **PSX Scan** — factor leaderboard over the whole exchange
- **Portfolio** — annealed allocation with per-instrument metrics
- **Paper Trading** — gated orders, account state, walk-forward validation, immutable journal
- **Trade Like a Pro** — the autonomous desk: start/stop the autopilot, arm LIVE execution, broker-connection strip (MT5 + JS Global), kill switch, live signal board, positions and audit feed. Every held position is actively managed — hard stops, targets, a trailing ratchet in R-multiples and venue reconciliation — and four circuit breakers (−3% day, 60% gross exposure, 4 straight losses, 8% cross-day drawdown from the equity peak) pause new entries while exits keep running. The watchlist itself rotates daily onto the PSX cross-sectional factor leaders — auto-admitted to the registry and screened for price floor and traded volume before they're ever tradable. Fresh entries additionally pass the **Fugu conductor** gate: no trade unless the fused verdict shows ≥0.15R probability-weighted edge and clears the adaptive conviction bar
- **Fugu conductor panel** — live board of fused expert verdicts per instrument: direction, conviction, P(win), edge in R-multiples, R:R geometry and the exact reasons a call was taken or passed; refreshes every two minutes and is served at `/api/fugu`
- **Shareable live signals** — one click per signal: an Instagram-square 1080×1080 PNG card (`/api/signal-card?symbol=…`) plus a WhatsApp-ready caption copied to your clipboard; every card carries the not-investment-advice disclaimer
- **JS Global live book** — one click ("Sync from JS Global" / `kairos livebook --sync`) logs into your broker's Web Trade Cast **read-only** and imports your real holdings (quantity, average, pending sells, sold-out positions auto-released); KAIROS then reviews every position through the engine (live price, P&L, call, recommended stop, status) and produces broker-ready SELL tickets awaiting your confirmation. In LIVE mode that same synced book is treated as a real venue position set, so stops, targets and trailing ratchets apply to your actual holdings

- **JS Global order desk** — the double-keyed stage→ack panel under the live book: fill symbol/side/qty/price, stage it, then release it to the broker by typing the trading PIN (or check *execute now* to use a stored one). Shows gate status, the broker's own price band per scrip, staged orders with age, and can cancel a live order by number. Disabled entirely while `allow_orders` is false — staging just refuses
- **News** — seven-feed macro/Pakistan radar
- **System** — health, MT5 bridge, tradable universe browser

## Foundation models via WSL (optional, ~4 GB)

```bash
python -m kairos setup-wsl     # installs Chronos-Bolt, TimesFM-2.5, qlib into WSL
```

Everything degrades gracefully — KAIROS runs fully without WSL, without the LLM council, and with any single data source down.

## Always-on & deployment

**On your PC (the trading brain):** `tools/kairos-daemon.bat` is a 30-second
watchdog — it keeps the control tower alive, mints a one-time access token to
`data/kairos_token.txt`, and (if `tools/cloudflared.exe` is present) opens a
free Cloudflare quick tunnel so the same tower is reachable from anywhere.
`tools/mkstartup.ps1` registers it hidden in the Windows Startup folder, so
every boot ends with a live KAIROS. The tunnel is owned by
`tools/keepalive.ps1`, which the watchdog calls every ~2 minutes: it asks the
**published URL** whether the tower is really reachable (a quick tunnel process
can sit alive for hours printing `Unauthorized: Tunnel not found`) and relaunches
`cloudflared` when it isn't, then writes the fresh hostname to
`data/tunnel_url.txt`. `/api/health` reports it as `public_url` and the desk
shows it as a clickable card — the URL changes on every re-registration, which is
the price of a free, account-less tunnel. The server writes its own log through
`serve --logfile` (4 MB, one backup), so `data/serve.log` can no longer grow to
gigabytes. On server boot the **paper autopilot auto-resumes**
(`autopilot.auto_start_paper`) — the self-improvement loop never has to be
re-armed by hand — and a cycle that outlives its own interval doubles the next
wait instead of stacking runs on a small CPU. Live mode still requires an
explicit `confirm_live`, and PSX still issues human-confirmed tickets unless you
turn every order key (see below).

**Remote access security:** when `KAIROS_TOKEN` is set, every `/api/` call
(reads included — the live book carries real holdings) demands
`X-KAIROS-Token`; the browser desk prompts once and remembers it.
`KAIROS_AUTH=writes` relaxes reads to public if you prefer a shared dashboard.

**Free hosted mirror (Render):** `render.yaml` + `Dockerfile` are ready —
connect the repo on render.com (free plan), it builds with core deps only,
carries **no** `config.yaml` (so no broker/LLM secrets ever leave your PC) and
runs the full paper loop + desk with a generated `KAIROS_TOKEN`. Free tier
sleeps after inactivity; the PC daemon is the permanent engine.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zhaider5142009-oss/kairos)

### Where it runs

Every route is free and needs no account on our side; the engine is the same
code in all of them, and none of them receives a `config.yaml`.

| surface | command / URL | what you get |
|---|---|---|
| Your machine | `pip install -e .` then `kairos up` | everything, including WSL foundation models and the broker bridge |
| Container registry | `docker run -p 8600:8600 -e KAIROS_TOKEN=… ghcr.io/zhaider5142009-oss/kairos:latest` | the tower + paper loop; the CI builds and *starts* this image on every merge |
| Render (free web service) | the button above, or "New +" → Blueprint → this repo | same image, HTTPS URL, sleeps when idle |
| Any "deploy from container" host | point it at the GHCR tag | Railway/Fly/Koyeb/Kubernetes — one env var, `PORT` |
| Read without installing | <https://zhaider5142009-oss.github.io/kairos/> | the landing page, the generated track record and the whitepaper as plain text |

There is deliberately **no PyPI route**: the name `kairos` on PyPI is someone
else's time-series storage library (Aaron Westendorf, 0.10.1), so `pip install
kairos` would install an unrelated package into a machine that then trusts it
with broker credentials. Install from the checkout, or pull the container.

One switch is a GitHub setting, not code, and it is one click:

- **Pages** → Settings → Pages → Build and deployment → Source: **GitHub
  Actions**. Until it is set, the Pages workflow's deploy job fails with a clear
  error instead of reporting a site that does not exist.

The container needs no such click: `ghcr.io/zhaider5142009-oss/kairos:latest`
answers an unauthenticated manifest request with `200`, so `docker pull` works
for anyone, and that is the route the free hosts use.

Both are gated rather than assumed: the Pages workflow refuses to publish a page
that advertises a guaranteed return, and the Container workflow refuses to leave
an image published if it cannot answer `/api/health` or if `/api/autopilot`
answers without a token.

### Placing JS Global orders from a server

`kairos/jsgtrade.py` speaks the broker's own Web Trade Cast order API
(`PlaceOrder` / `CancelOrder`, plus the per-scrip upper-cap / lower-lock band
it publishes). It is **double-keyed by design** — three independent acts are
needed before real money moves, and none of them is a code change:

| key | where | what it means |
|---|---|---|
| `js_global.allow_orders` | config.yaml or `JSG_ALLOW_ORDERS` | your standing yes to touch the broker at all |
| the trading `PIN` | desk button **Store PIN for unattended live** (writes git-ignored `config.yaml`), `JSG_PIN`, or typed per order | the broker's own second factor — **Forget PIN** is the instant undo |
| `autopilot.auto_place` | config.yaml | only for autonomous PSX fills; without it the loop still writes tickets |

Flow: **stage → ack**. `kairos order OGDC --side SELL --qty 50 --price 124.50`
(or the desk's *JS Global order desk* card) records a pending order in
`data/jsg_pending.json` with a 15-minute TTL and no network call. Nothing
reaches the broker until it is acked — by you typing the PIN, or by an armed
autopilot holding a stored PIN. Every order passes the kill switch, the
per-order notional cap and the broker's price band first; a POST that dies in
transit discards the staged copy instead of risking a duplicate.

A headless deploy carries the same values as env secrets (`JSG_PORTAL`,
`JSG_LOGIN`, `JSG_PASSWORD`, `JSG_PIN`, `JSG_ALLOW_ORDERS`,
`JSG_ORDER_ACCOUNT`) so no `config.yaml` has to exist on the server. **Know
the tradeoff:** storing a PIN anywhere — Render secrets included — means that
server can move money without you. Keep `allow_orders: false` and
`auto_place: false` on any machine you don't physically control, keep the
per-order notional cap small, and remember that the kill switch blocks
staging, acking and cancelling alike.

## Configuration

`kairos setup` writes `config.yaml` from `config.example.yaml`, filling each provider's key
slot from the environment (`OPENROUTER_API_KEY`, `MOONSHOT_API_KEY`, `DEEPSEEK_API_KEY`) and
prompting with masked input for the rest. Doing it by hand is still fine — copy the example
and fill in your keys.

**`config.yaml` is git-ignored and is the only file that may hold a credential.** It is
never committed, printed, or logged: `doctor` reports which slots are filled and which are
still placeholders, and counts keys without reading one aloud. `kairos doctor` fails loudly
if the ignore rule ever stops covering the file — that rule is the only thing between your
keys and a public repository. MT5 and JS Global credentials belong in the same file, under
their own blocks.

## Architecture

```
python -m kairos             CLI entry (setup/doctor/up/ask · predict/scan/portfolio/backtest/paper/mt5/fugu/autopilot/rules/share/livebook/order/feedback/serve/…)
launch_kairos.bat            one-click launcher (desktop shortcut target; icon in tools/kairos.ico)
config.yaml                  all keys + engine tunables (git-ignored; created by `kairos setup`)
kairos/
  config.py universe.py      config singleton · 100-instrument registry (22 crypto + commodities/FX/indices/PSX)
  onboard.py                 setup · doctor · up — the first run, answered by code instead of by a document
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
  rules.py                   JARVIS v1 doctrine merged: risk-per-trade, R:R floor, ATR stops, wisdom corpus
  signalcard.py              shareable live signal cards: 1080×1080 PNG + WhatsApp caption
  livebook.py                your REAL JS Global holdings — engine verdicts + exit tickets
  jsgsync.py                 read-only Web Trade Cast sync: login → import real portfolio
  jsgtrade.py                double-keyed order desk: stage → ack → PlaceOrder/CancelOrder, broker price band + notional cap
  fugu.py                    Fugu conductor: fused expert verdicts, calibrated P(win), edge in R, adaptive entry bar
  command.py                 read-only command plane: plan → 12-tool registry → trace; refuses movement before planning
  pmx.py                     PMX (Pakistan Mercantile Exchange) futures scaffold — venue + human-confirmed tickets, account-ready
  backtest.py                walk-forward validation + paper sessions
  execution.py               OrderIntent → RiskGate → PaperBroker / Mt5Bridge (attach-only) / JsGlobalTicket + audit journal
  autopilot.py               autonomous loop: scan → verdicts → Fugu gate → doctrine-sized orders → gated execution
  wsl_bridge.py              Chronos/TimesFM/qlib subprocess bridge
  llm.py report.py cli.py    LLM council router · Rich terminal dossiers · command line
  api/server.py api/ui.html  FastAPI control tower + web UI
  wsl_scripts/kairos_models.py WSL-side model server
data/                        store cache, journal, paper account, LLM cache
reports/                     saved JSON dossiers
```

### Performance on small hardware

Measured on a 4-core / 8 GB desktop (2026-10-19), because the 24/7 box *is* the
trading box:

| cost | before | after |
|---|---|---|
| one full battery refit (stat+ML+neural, 730 bars) | ~45 s | ~45 s, **once per bar set** |
| repeat cycle on unchanged daily bars | ~65 s/instrument | ~20 s/instrument (battery memoised) |
| noise written to the daemon log per ML stack | ~2 900 warnings | 0 |
| `data/serve.log` growth | unbounded (33 MB/day) | 4 MB + one backup |

- `engine._battery()` memoises the refits on `(symbol, horizon, bar count,
  last bar date, last close)` — a moved last close re-anchors, so cached levels
  can never price a trade off a stale anchor. Bounded to 64 entries.
- BLAS/OpenMP/torch threads are capped at 4 from `kairos/cli.py` **before**
  numpy loads; oversubscription on 4 cores makes a refit minutes slower.
- loky workers get `PYTHONWARNINGS=ignore::UserWarning` (they carry their own
  filter state and flooded stdout).
- `torch` stays lazy: boot is 131 MB, the first neural fit takes it to ~460 MB.
- Scan-admitted PSX tickers persist to `data/psx_registered.json` and re-admit
  at import — a restarted daemon can still analyse and exit yesterday's watchlist.

## Doctrine

1. **Never fabricate data** — missing source is reported, not guessed.
2. **Fail closed** — execution blocked unless every gate passes.
3. **Prove it on paper first** — no live order until walk-forward + paper sessions are green.
4. **Everything journaled** — append-only JSONL audit trail with idempotent client order IDs.
5. **Real money turns more than one key** — a PSX order needs `allow_orders` **and** the broker PIN **and**, for autonomy, `auto_place`; tickets are the default, a PIN is never logged or echoed, and an order whose POST died in transit is discarded rather than re-sent.

## Verification & commercial interest

Every published figure is regenerated from the on-disk journals, so the claims and the code
cannot drift apart:

```bash
python tools/make_track_record.py   # rebuilds docs/investors/TRACK_RECORD.md
python tools/kit_stats.py --write   # re-points the investor pack at those figures
python -m kairos.cli feedback       # realized record, live
```

[docs/investors/](docs/investors/) holds the ledger, a one-pager, the whitepaper and the
deck. Read the ledger first: there is **no forward track record yet** — 1,249 predictions
journaled, 140 horizons matured, all 140 non-trades — and 6 of 14 instruments in the
walk-forward sweep lost money. That is the honest state of the project, and it is the reason
the risk layer exists.

KAIROS is research infrastructure, not a fund: no pooled capital, no managed accounts, no
licence, no return promise. Open to licensing, acquisition, or a bounded pilot inside a
licensed firm — contact in the one-pager.

## License

Released under the [MIT License](LICENSE). Free to use, modify and redistribute
with attribution. See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved and
[SECURITY.md](SECURITY.md) to report a vulnerability privately.

> KAIROS is decision-support and paper-execution software, provided as is with
> no warranty. It makes no guarantee of profitable or accurate forecasts.
> Trading crypto, commodities, FX and equities carries substantial risk of
> loss. Nothing here is financial advice.

## Lineage

JARVIS v1/v1.5 (MiniMax build) → **KAIROS v2** (this repository). The v1 sources were
committed here as `jarvis/` before the rename, so the lineage is auditable in this repo's
own git history — the useful parts of it (doctrine, regime work, execution discipline) are
merged into `kairos/`, and the rest was rebuilt from first principles.

**Single project (2026-10-17 merge disposition).** Every jarvis component found on this machine was audited against KAIROS:
- *Merged in*: `trading_rules.py` doctrine → `kairos/rules.py` (risk-per-trade, R:R floor, ATR stops, Kelly, wisdom corpus — enforced by the autopilot's pre-trade plan check); v1 state (paper ledgers, backtest reports) mirrored to `data/v1_legacy/`.
- *Superseded, not merged*: `profit_engine` (regime/pattern/sizing → features+risk+engine), `stock_analysis` (indicators → features.py), `trading_signals` & `predictive_trader` & `paper_trader` (→ autopilot + execution.py), `psx.py` (→ data.py + universe), `chronos/timesfm/quantum bridges` (→ wsl_bridge + quantum), `model_router` (→ llm.py tiers), `jarvis_commands`/`connectors`/`job_finder`/`code_lab`/VoiceAI app (general assistant, out of trading scope).
- *Archived*: everything else lives at `D:\JARVIS_ARCHIVE_2026-10-17\` with a README; this repo is the one KAIROS.
