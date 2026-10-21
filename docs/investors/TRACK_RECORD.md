# KAIROS — verification ledger

_Rebuilt from the on-disk journals by `tools/make_track_record.py`. Unaudited. Nothing here is a forecast or a promise of return._

## 1. Realized forward predictions

| measure | value |
|---|---|
| rebuilt from the journals on | 2026-10-22 |
| predictions journaled | 1687 |
| horizons matured | 140 |
| of which directional calls | 0 |
| of which non-trades (HOLD) | 140 |
| directional hit rate | not yet measurable |

**There is no directional hit rate yet.** Every matured snapshot was a non-trade, and a non-trade cannot be a correct prediction — the scorer used to count them as wins, which is fixed as of v2.5.0. A forward record of live calls is what this table will show, and it accumulates on its own as horizons elapse.

375 directional calls are journaled and still waiting out their horizon, and the tenth of them cannot resolve before **2026-10-25**, five bars after it was written. This table publishes a hit rate the day it can.

## 2. Walk-forward strategy sweep

One composite signal per instrument, blended from 4 components. Its weights, entry threshold, regime gate and short mode are chosen by a 2,592-point grid search on a rolling 252-bar training window, re-fitted every 30 bars, and traded only out-of-sample with PSX commission and slippage modelled. This sweep does not evaluate the 100-strategy library: that votes inside the live engine, and none of the returns below come from it.

_Sweep run on **2026-10-21** (`reports/spectrum_v261.log`), the same day this ledger was rebuilt._

| instrument | verdict | net return | buy & hold | Sharpe | hit rate | trades | max drawdown |
|---|---|---|---|---|---|---|---|
| WHEAT | profitable | +26.8% | +19.0% | +1.60 | 0.517 | 24 | -6.0% |
| GOLD | profitable | +29.3% | +11.6% | +1.57 | 0.630 | 18 | -9.2% |
| SILVER | profitable | +43.5% | +36.1% | +1.44 | 0.615 | 39 | -22.3% |
| HBL | profitable | +12.4% | +14.8% | +1.39 | 0.505 | 12 | -8.9% |
| WTI | profitable | +16.0% | +50.2% | +0.77 | 0.400 | 10 | -12.8% |
| KSE100 | unprofitable | -1.8% | +6.6% | +0.52 | 0.537 | 13 | -15.8% |
| MCB | unprofitable | -1.1% | +8.1% | +0.42 | 0.419 | 11 | -16.9% |
| EURUSD | unprofitable | -1.3% | -5.5% | +0.19 | 0.510 | 17 | -3.1% |
| LUCK | unprofitable | -6.1% | -16.7% | +0.03 | 0.509 | 10 | -7.7% |
| BTC | unprofitable | -3.9% | -26.8% | -0.04 | 0.280 | 21 | -9.3% |
| NATGAS | unprofitable | -1.5% | -9.1% | -0.18 | 0.182 | 5 | -5.9% |
| COPPER | unprofitable | -9.1% | +41.9% | -0.33 | 0.494 | 24 | -18.0% |
| OGDC | unprofitable | -14.3% | +12.1% | -0.53 | 0.414 | 12 | -21.4% |
| BRENT | unprofitable | -9.4% | +58.4% | -0.69 | 0.405 | 14 | -15.1% |

Run summary: **5/14 instruments profitable out-of-sample**, the other 9 unprofitable. Skipped for lack of history, and not in the table: ENGRO.

- **Global**: mean net return +10.0% across 9 instruments; beat buy-and-hold on 67% of them.
- **PSX**: mean net return -2.2% across 5 instruments; beat buy-and-hold on 20% of them.

_Backtests are not returns. They include costs but not every market impact, and the traded parameter set is chosen from 2,592 candidates in every calibration window, so selection bias is stated here as a number rather than as a disclaimer. That is why the sweep is reported per instrument instead of as an average, and why section 1 — the forward record — is the figure that matters._

## 3. Live paper book

- Capital Rs 1,000,000 · equity Rs 999,992 · realized P&L Rs -13.05 since 2026-10-21T18:28:41
- Autopilot cycles run: **224** · mode `paper` · live-armed: False
- Open paper positions: 0 · fills: 2

## 4. Funded account (read-only, from the broker)

- Holdings tracked: 0 · equity not fetched in this snapshot
- KAIROS observes this book; real orders require three independent keys (`autopilot.auto_place`, `js_global.allow_orders`, the broker PIN) and the kill switch to be clear.

## 5. What is and is not established

| claim | status |
|---|---|
| The system produces structured, risk-gated signals across PSX, FX, commodities and crypto | **established** — running continuously, journalled |
| Out-of-sample strategy selection beats buy-and-hold on some instruments | **partly established** — see the sweep; commodity books strongest |
| PSX equities are profitable net of local costs at these horizons | **not established** — fee and slippage drag exceeded signal on the equity book |
| A realized forward track record | **does not exist yet** — no directional call has matured |
| Any return promise to an investor | **explicitly not made** |

## 6. How this file is verified

- `data/prediction_journal.jsonl` — every prediction, timestamped, resolved only when the horizon's bars exist.
- `data/paper_account.json` and `data/autopilot.json` — the paper book and every gate decision the loop made.
- `data/audit_journal.jsonl` — order staging, arming, kill-switch and venue-loss events, append-only.
- Rebuilding this ledger is one command; it reads only those files.

### What the audit journal actually contains

| measure | value |
|---|---|
| lines recorded | 172 |
| distinct event types | 22 |
| live-mode loop starts | 20 |
| lines inside 30s of a live-mode start (the automated-test signature) | 100 |
| venue acknowledgements carrying a real order number | **0** |

**No order has ever reached a broker.** Every `AUTOPILOT_PLACED` line in the journal answers with the test broker's constant order number `1` at a fixed fixture price, and the single `JSG_STAGED`/`JSG_PLACED` pair came from an offline unit test whose HTTP session was mocked — that pair is annotated in the journal itself.

Of the 172 lines, **100 were written by the automated test suite**, not by the trading process. The suite used to import the operator's live configuration, so the module-bound journal object inside a test process appended to the production file, armed `live` mode against stubbed brokers, and left its daemon threads running after the stubs were torn down. Nothing was placed and the paper book was never touched, but the trail was polluted, which is a defect in an append-only audit log and is not something to leave unmentioned.

The fix is structural, not a cleanup of the file: `tests/conftest.py` now points every test at a throwaway config with an unroutable portal URL and empty credentials, blocks the transport layer so no test can reach a network, and stops every autopilot instance built during a test at teardown. The history is left in place — the containment event itself is the last line of the journal, because rewriting an audit trail to hide its own contamination would destroy the thing it is supposed to prove.

