# KAIROS — verification ledger

_Rebuilt from the on-disk journals by `tools/make_track_record.py`. Unaudited. Nothing here is a forecast or a promise of return._

## 1. Realized forward predictions

| measure | value |
|---|---|
| rebuilt from the journals on | 2026-10-21 |
| predictions journaled | 1437 |
| horizons matured | 140 |
| of which directional calls | 0 |
| of which non-trades (HOLD) | 140 |
| directional hit rate | not yet measurable |

**There is no directional hit rate yet.** Every matured snapshot was a non-trade, and a non-trade cannot be a correct prediction — the scorer used to count them as wins, which is fixed as of v2.5.0. A forward record of live calls is what this table will show, and it accumulates on its own as horizons elapse.

## 2. Walk-forward strategy sweep

100 candidate strategies per instrument, parameters chosen on a rolling training window and evaluated out-of-sample with PSX commission and slippage modelled.

| instrument | verdict | net return | buy & hold | Sharpe | hit rate | trades | max drawdown |
|---|---|---|---|---|---|---|---|
| WTI | profitable | +25.1% | +61.8% | +1.68 | 0.577 | 11 | -7.4% |
| GOLD | profitable | +28.7% | +13.4% | +1.61 | 0.594 | 50 | -11.6% |
| SILVER | profitable | +44.5% | +34.5% | +1.46 | 0.622 | 39 | -21.8% |
| HBL | profitable | +9.3% | +14.1% | +1.19 | 0.446 | 12 | -8.9% |
| WHEAT | profitable | +19.5% | +31.3% | +1.16 | 0.519 | 18 | -7.1% |
| NATGAS | profitable | +4.2% | +5.9% | +1.09 | 0.222 | 5 | -2.1% |
| BRENT | profitable | +6.8% | +64.0% | +0.65 | 0.522 | 18 | -12.1% |
| KSE100 | profitable | +0.0% | +7.9% | +0.58 | 0.534 | 11 | -14.9% |
| MCB | unprofitable | -0.2% | +6.6% | +0.56 | 0.515 | 14 | -18.5% |
| EURUSD | unprofitable | -0.6% | -4.1% | +0.29 | 0.523 | 15 | -4.0% |
| BTC | unprofitable | -0.8% | -29.5% | +0.12 | 0.349 | 16 | -9.1% |
| COPPER | unprofitable | -4.7% | +42.3% | -0.04 | 0.495 | 20 | -15.9% |
| LUCK | unprofitable | -9.2% | -16.0% | -0.07 | 0.431 | 14 | -10.7% |
| OGDC | unprofitable | -14.1% | +11.4% | -0.42 | 0.429 | 14 | -21.3% |

Run summary: **8/14 instruments profitable out-of-sample**, 6 not.

- **Global**: mean net return +13.6% across 9 instruments; beat buy-and-hold on 44% of them.
- **PSX**: mean net return -2.8% across 5 instruments; beat buy-and-hold on 20% of them.

_Backtests are not returns. They include costs but not every market impact, and the parameter search is a known overfitting risk; that is why the sweep is reported per instrument instead of as an average._

## 3. Live paper book

- Capital Rs 1,000,000 · equity Rs 999,992 · realized P&L Rs -13.05 since 2026-10-21T18:28:41
- Autopilot cycles run: **202** · mode `paper` · live-armed: False
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
| lines recorded | 168 |
| distinct event types | 22 |
| live-mode loop starts | 20 |
| lines inside 30s of a live-mode start (the automated-test signature) | 100 |
| venue acknowledgements carrying a real order number | **0** |

**No order has ever reached a broker.** Every `AUTOPILOT_PLACED` line in the journal answers with the test broker's constant order number `1` at a fixed fixture price, and the single `JSG_STAGED`/`JSG_PLACED` pair came from an offline unit test whose HTTP session was mocked — that pair is annotated in the journal itself.

Of the 168 lines, **100 were written by the automated test suite**, not by the trading process. The suite used to import the operator's live configuration, so the module-bound journal object inside a test process appended to the production file, armed `live` mode against stubbed brokers, and left its daemon threads running after the stubs were torn down. Nothing was placed and the paper book was never touched, but the trail was polluted, which is a defect in an append-only audit log and is not something to leave unmentioned.

The fix is structural, not a cleanup of the file: `tests/conftest.py` now points every test at a throwaway config with an unroutable portal URL and empty credentials, blocks the transport layer so no test can reach a network, and stops every autopilot instance built during a test at teardown. The history is left in place — the containment event itself is the last line of the journal, because rewriting an audit trail to hide its own contamination would destroy the thing it is supposed to prove.

