# KAIROS
### Autonomous prediction and execution for Pakistan and global markets

*One page. Every number is reproducible from the repository and dated 2026-10-24 — regenerate the ledger before sending. Nothing here is a promise of return.*

---

## What it is

A single-operator, always-on trading system that forecasts PSX equities, FX, commodities
and crypto, then places its own orders — paper today, live behind three independent arming
keys — through a regulated Pakistani broker and an MT5 bridge.

It is **research infrastructure**, not a fund. KAIROS does not hold client money, is not
SECP-licensed as an investment adviser, and has never managed third-party capital.

| | |
|---|---|
| Codebase | 17,226 lines of Python across 42 modules, public repo |
| Strategy library | 100 strategies in 10 families, voting into the live consensus |
| Attribution | each vote weighted by its own realized excess over the base rate of its bucket — per instrument, horizon and regime, measured out-of-sample; the engine runs a named flat poll rather than trust stale weights |
| Signal council | 6 configured seats across 2 providers, median consensus with named-seat integrity; a blank or dead seat casts no vote |
| Execution | paper book + JS Global PSX order bridge + MT5 (JS Markets) |
| Uptime | 24/7 daemon with watchdog and public token-gated API |
| Tests | 675 passing, money-path and integrity cases included |

## What has actually been demonstrated

Walk-forward, out-of-sample, costs modelled — one blended signal per instrument, its weights chosen from 2,592 candidates on a rolling window, not the 100-strategy library — **5 of 14 instruments profitable**:

| instrument | net | buy & hold | Sharpe |
|---|---|---|---|
| SILVER | +43.5% | +36.1% | 1.44 |
| GOLD | +29.3% | +11.6% | **1.57** |
| WHEAT | +26.8% | +19.0% | 1.60 |
| WTI | +16.0% | +50.2% | 0.77 |
| HBL | +12.4% | +14.8% | 1.39 |

The other 9 — BTC, BRENT, COPPER, EURUSD, KSE100, LUCK, MCB, NATGAS and OGDC — were **not**
profitable and are reported as losses, per instrument, in the dated ledger.
Backtests are not returns; the parameter search is a known overfitting risk, which is why
this is published per instrument and never as an average.

## What has *not* been demonstrated — stated plainly

- **No forward track record yet.** 3,558 predictions journaled, 263 horizons matured, and all
  263 were non-trades. A directional hit rate does not exist until live calls mature.
- **PSX equities are net-negative out-of-sample.** Local commission and slippage exceeded
  the signal: mean −2.2% across the 5-stock book, versus +10.0% across the global book.
- **No audited performance, no AUM, no third-party capital, no regulatory approval.**

The system publishes these limits itself. It previously reported a 100% hit rate built out
of non-trades; that scorer was found, fixed and regression-tested in v2.5.0, and the
confidence bonus it fed into live entries was removed. That correction is the most useful
thing on this page: it shows the audit found the flaw before a live account paid for it.

## Architecture worth paying for

1. **Regime-aware ensemble** — features, patterns and regime classification feed a model
   battery; disagreement between models is a signal, not noise.
2. **LLM council with seat integrity** — a named seat may only be answered by that seat, so a
   dead model shrinks the council instead of silently substituting another opinion.
3. **Self-calibrating from realized outcomes** — per-symbol and per-model weights updated
   from the journal; confidence is capped at 1.0 until 10 directional calls have matured.
4. **Fail-closed execution** — kill switch, per-order notional cap, broker price band,
   cash-ledger verification, no shorting on a cash market, and stage→ack with a 15-minute
   TTL. Live placement needs three independent keys; removing any one returns to
   human-confirmed tickets.
5. **Venues are swappable** — PSX today, MT5 now, PMX scaffolded; the risk layer does not
   know or care which broker is underneath.

## What we are open to

| Route | Shape |
|---|---|
| **Acquisition / licensing** | The engine, the risk layer and the broker bridges as a technology asset for a broker, exchange or asset manager's digital desk. |
| **Strategic partnership** | A licensed firm provides the regulated wrapper; KAIROS provides the signal and execution stack. Requires a compliance review before anything is signed. |
| **Advisory / pilot** | A named institution funds a bounded pilot with a real forward record as the only deliverable. |
| **Employment / collaboration** | The operator is available to build this inside a firm that can hold the licence. |

We are **not** offering securities, not accepting pooled capital, and not taking money to
trade on someone's behalf. Any structure that needs those things needs a licensed
vehicle first — and we will not propose one without counsel.

## Verify it yourself

- Repo: `github.com/zhaider5142009-oss/kairos` (mirror: `jarvis-reborn`)
- Rebuild the ledger: `python tools/make_track_record.py` → `docs/investors/TRACK_RECORD.md`
- Every prediction: `data/prediction_journal.jsonl`, append-only, resolvable to the bars
- Every order decision, arming event and kill-switch state: `data/audit_journal.jsonl`
- Reproduce a sweep: `python -m kairos.cli backtest <symbol>`

**Contact:** z.haider5142009@gmail.com · GitHub `zhaider5142009-oss`
