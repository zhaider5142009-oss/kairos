# KAIROS — technical whitepaper
**A self-auditing prediction and execution stack for Pakistan and global markets**

Version 2.5.0 · 2026-10-19 · single-operator, publicly verifiable

> **Reading note.** This document describes a system, not a track record. Every quantitative
> claim is either reproduced from an on-disk journal or explicitly labelled as a backtest.
> There is no forward performance record in existence yet, and section 8 explains why that
> fact was itself a bug we had to find.

---

## 1. Problem

Two markets, one gap.

**PSX** is a cash-and-carry equity market with high retail participation, thin intraday
liquidity outside the index heavyweights, and a cost structure (commission, CPDC/CDNS fees,
slippage on a 10-tick book) that quietly destroys most signals that look good gross. Public
tooling is either charting UIs or manual order entry.

**Global** FX/commodity/crypto signal infrastructure is abundant, but almost none of it is
wired to a Pakistani broker, and none of it publishes its losses.

KAIROS was built to occupy the intersection: institutional-grade signal plumbing that is
honest about what it has not proven, and that can place a real order on a real Pakistani
account without a human in the loop *only when three independent keys are turned*.

## 2. System overview

```
        data               features              decision             execution
  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐
  │ Yahoo        │   │ indicators       │   │ 100-strategy     │   │ paper book     │
  │ TradingView  │→  │ pattern sets     │→  │ walk-forward     │→  │ MT5 bridge     │
  │ PSX local    │   │ regime class     │   │ model battery    │   │ JS Global PSX  │
  │ news         │   │ factor panel     │   │ LLM council      │   │ risk layer     │
  └──────────────┘   └──────────────────┘   │ quantum weights  │   └────────────────┘
        ↑                                   └──────────────────┘            │
        │              prediction journal ←── gates, kill switch ←──────────┘
        └────────── outcome resolution ←───── calibration ←─────────────────┘
```

| layer | module | what it guarantees |
|---|---|---|
| Data | `kairos/data.py` | multi-source history, circuit breaker on dead local services |
| Features | `kairos/features.py` | deterministic, versioned feature frames |
| Models | `kairos/models_stat.py`, `models_ml.py` | statistical + ML + neural battery |
| Strategy | `kairos/strategies.py` | 100 strategies, per-instrument selection |
| Council | `kairos/llm.py` | named-seat LLM consensus with provider integrity |
| Ensemble | `kairos/ensemble.py`, `quantum.py` | fused expert scores, portfolio weights |
| Risk | `kairos/risk.py` | position sizing, drawdown guard, price band |
| Execution | `kairos/execution.py`, `jsgtrade.py` | paper + live, fail-closed |
| Feedback | `kairos/feedback.py` | outcome resolution, calibration, honesty gates |
| Orchestration | `kairos/fugu.py`, `autopilot.py` | 24/7 loop, self-tuning entry bar |
| Surface | `kairos/cli.py`, `api/server.py`, `api/ui.html` | 33 token-gated endpoints, operator desk |

## 3. Signal construction

**Ensemble, not oracle.** Each instrument is scored by the strategy library on a rolling
training window; parameters are chosen out-of-sample, never in-sample. The model battery
returns a score in [−1, +1] and a confidence, and disagreement across models is carried
forward as a feature rather than averaged away.

**The LLM council.** Up to six models vote through a router. Three properties matter:

1. *Named-seat integrity.* If seat `meta-llama/llama-4-scout` is dead, the council shrinks.
   It does not fall over to another provider, because that would substitute a different
   model into the same seat and the median would quietly double-count one opinion.
   Fail-over to another provider is only allowed on an unnamed free-text route.
2. *Catalog anti-rot.* Vendor slugs retire (`meta-llm/llama-3.3-70b-instruct`,
   `deepseek/deepseek-r1`). The router compares configured seats against the live catalog
   every six hours and the health endpoint reports `dead_seats` so rot is visible instead of
   silent.
3. *Provider cooldown.* A fatal hint (401/402/403/429, insufficient balance) cools that
   provider for 30 minutes instead of burning a request per call.

**Dissent is attributed** as `provider/model`, so a split council can be traced to who
disagreed, which turned out to matter for debugging more than for alpha.

## 4. Execution: fail-closed by construction

Live order placement requires **all** of:

| key | where | effect if absent |
|---|---|---|
| `autopilot.auto_place` | config / env | loop emits a ticket, places nothing |
| `js_global.allow_orders` | config / env | order bridge refuses to stage |
| broker trading PIN | config only, never logged | staging possible, submission impossible |
| kill switch clear | runtime | hard stop, audited |
| broker price band + free cash | live book | order rejected before submission |

Then, and only then, does a stage→ack handshake with a 15-minute TTL submit. The per-order
notional cap is enforced inside `jsgtrade.stage()`, not in the UI, so a compromised or
buggy caller cannot bypass it.

**The PIN is stored through a dedicated one-line config writer** (`Config.store_secret`) that
mutates the git-ignored `config.yaml` in place. The value is never logged, journalled,
echoed by the API, or returned by any endpoint — the desk reports only `has_pin: true/false`.
"Forget PIN" is one click and returns the system to human-confirmed tickets instantly.
Comment lines in the operator's config are preserved because the writer does line surgery
rather than a YAML round-trip.

**Venue-aware risk posture.** Free cash and mark-to-market holdings are read from the real
book keyed on the account number. Guards that fail closed rather than guess: portal down →
ticket only; insufficient cash → ticket only; shorting a cash market → refused; selling more
than is held → refused.

## 5. Self-improvement, with a leash

Outcome resolution walks the journal, and a snapshot is only graded when the bars for its
horizon actually exist. Directional accuracy is measured against the entry threshold: inside
the band is a non-trade and is excluded from the hit rate entirely.

Calibration then feeds back three ways:
- per-model skill weights from realized outcomes,
- per-symbol weights, since the system is genuinely better at SILVER than at OGDC,
- a confidence multiplier that is **pinned to 1.0 until at least 10 directional calls have
  matured**, then 1.05× above an 80% hit rate and down to 0.75× below 50%.

The leash is the point: self-improvement that raises confidence on evidence that does not
exist is a way to lose money faster than the market teaches you.

## 6. Operations

- Runs 24/7 as a supervised daemon with a watchdog; the API server is restarted by the loop
  if it dies, and a tunnel keeper re-establishes the public HTTPS endpoint.
- Efficiency work measured and cut the scan cost of a dead localhost dependency from ~21 s
  per four symbols to a single 2.2 s probe followed by an instant refusal (900 s penalty).
- Logs rotate; state persists across restarts; RAM, startup and per-cycle cost were profiled
  on a 2015-era i5 / 8 GB machine, which is a deliberate constraint — if it runs there, the
  infrastructure argument is cheap.
- 113 regression tests, including the money path and the integrity cases below.

## 7. What the numbers say, and what they do not

Walk-forward, costs modelled, 8/14 instruments profitable — commodities strongest
(SILVER +44.5%, GOLD +28.7% beating buy-and-hold with Sharpe 1.61 and 1.68-class risk
adjustment), and six instruments reported as losses.

**PSX equities lost money net of local costs** (−2.8% mean across the equity book, and a
~8% fee drag observed on the active stock trials). This is the single most commercially
important finding in the project: the edge, if any, is in the global and commodity books,
and any PSX strategy must be sized against costs before it is sized against conviction.

The live paper book is essentially flat (realized P&L −Rs 13.05 on Rs 1,000,000 over 154
autopilot cycles) with two fills. That is a system working as designed on a low-conviction
market, not a strategy proving itself.

## 8. The integrity bugs, documented because they are the credibility

An investor kit is only worth its willingness to publish its own audit. Nine defects were
found and fixed; four of them were in the machinery that produces the numbers an investor
would read, four were on the money path or the trail that records it, and one was a
credential sitting in a committed file.

| defect | consequence had it shipped | fix |
|---|---|---|
| Non-trades scored as correct predictions; horizons clamped to the last available bar | the headline read **"100% hit rate"**; the calibration loop fed a 1.05× confidence boost into live entries on evidence that was entirely fabricated | a call must exceed the entry threshold and its horizon bars must exist; confidence pinned to 1.0 until 10 real calls mature; 8 new regression tests |
| WSL bridge reported setup success on a non-zero exit | a broken foundation-model bridge would look healthy forever | exit code is checked, `check=False` made explicit |
| Provider fail-over substituted models into named council seats | one provider's opinion counted twice inside a "median consensus" | named seats route to that seat only |
| Configuration was read once at import and never re-read | a risk limit tightened mid-run — the daily loss cap, the arming keys — bound only after a restart, so an operator pulling the brake did not stop the next order | the config is re-read on mtime change, throttled, and a half-written file keeps the last good values; the risk gate now reads its limits per check — 13 tests |
| The duplicate-order idempotency set grew without bound | after enough cycles the process that guards live orders exhausts its own memory | the seen-order window is a bounded LRU |
| Parallel arrays (weights, symbols, forecasts) were zipped silently | a short selection vector would drop the tail instruments from the book while still reporting their count | every such zip is now length-checked and fail-closed — 9 tests |
| The test suite imported the operator's live config | a `pytest` run armed live mode against stubbed brokers, appended to the production audit journal, and left daemon threads alive after the stubs were torn down — the operator could no longer trust their own order trail | `tests/conftest.py`: a throwaway config with an unroutable URL and empty credentials, the HTTP transport blocked for the session, every autopilot stopped at teardown; plus a credential-leak tripwire that reads `git ls-files` — 3 tests |
| A restarted autopilot reported live from a stale state file | the desk told the operator "LIVE" while the process was running paper — the opposite of the truth at exactly the moment it mattered | restart clamps `mode`, `armed_live` and `running` before it reports anything |
| The operator's real broker account number was hard-coded in two test fixtures | a live account identifier in a public repository, next to code that names its venue | replaced with a placeholder, and a tripwire test now fails if any live credential value or provider-key-shaped string appears in tracked source |

Nothing moved money. The journal now says so itself: rebuilt from `data/audit_journal.jsonl`,
the track record publishes the count of lines carrying the test signature and the count of
venue acknowledgements with a real order number, which is zero. That history is left in
place rather than cleaned — a rewritten audit trail cannot prove anything, so the containment
event was appended instead.

The first is the one to judge us by. It was not an investor's bug or a reviewer's bug — our
own scorer produced it, our own audit caught it, and it was fixed before a live rupee could
be mis-sized because of it.

## 9. Reproducing every claim

```bash
git clone https://github.com/zhaider5142009-oss/kairos && cd kairos
cp config.example.yaml config.yaml      # add your own keys; config.yaml is git-ignored
pip install -r requirements.txt
python -m pytest tests -q                    # 113 tests
python -m kairos.cli backtest GOLD WTI HBL   # walk-forward sweep, costs modelled
python -m kairos.cli health                  # subsystem health, dead council seats
python -m kairos.cli feedback                # realized track record, live from the journal
python tools/make_track_record.py            # rebuilds docs/investors/TRACK_RECORD.md
```

Journals are append-only JSONL and map to the bars they were graded on:
`data/prediction_journal.jsonl`, `data/audit_journal.jsonl`, `data/paper_account.json`.

## 10. Status and honest limits

- Single operator, no team, no outside capital, no audit, no regulatory licence.
- Not an offer of securities. Not investment advice. No return is promised or implied.
- Live execution has never been exercised at scale; the arming path exists, is tested
  against staging, and is deliberately gated.
- The forward record does not exist yet, and this whitepaper will be boring until it does.

**Contact:** z.haider5142009@gmail.com · `github.com/zhaider5142009-oss/kairos`
