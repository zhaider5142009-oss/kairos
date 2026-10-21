# KAIROS — technical whitepaper
**A self-auditing prediction and execution stack for Pakistan and global markets**

Version 2.6.0 · 2026-10-20 · single-operator, publicly verifiable

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
4. *A blank answer is not a vote.* A reasoning model can spend its whole token budget on
   hidden thinking and return an empty completion. Reported as success, that inflates the
   count of responding seats while contributing nothing to the median — a phantom opinion
   inside a consensus. A blank completion is therefore a failure: the seat loses its vote,
   and because the failure is not fatal the seat is retried next cycle instead of cooling
   its provider.

**Six configured seats are not six live seats.** The direct Moonshot key on this machine has
been suspended by the provider, so its two configured seats lose their vote on every call
until that is resolved, and one Kimi seat hosted behind OpenRouter answers intermittently at
small token budgets. Nothing here is presented as better than it is: `provider_status()`
reports the suspension, `dead_seats()` reports retired slugs, and every council record
publishes `models_responded` beside `models_total`, so the size of the council that actually
voted is part of the evidence rather than an assumption.

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
- 218 regression tests, including the money path and the integrity cases below.

## 7. What the numbers say, and what they do not

Walk-forward, costs modelled, 8/14 instruments profitable — commodities strongest
(SILVER +44.5%, GOLD +28.7% beating buy-and-hold with Sharpe 1.61 and 1.68-class risk
adjustment), and six instruments reported as losses.

**PSX equities lost money net of local costs** (−2.8% mean across the equity book, and a
~8% fee drag observed on the active stock trials). This is the single most commercially
important finding in the project: the edge, if any, is in the global and commodity books,
and any PSX strategy must be sized against costs before it is sized against conviction.

The live paper book is essentially flat (realized P&L −Rs 13.05 on Rs 1,000,000 over 189
autopilot cycles) with two fills. That is a system working as designed on a low-conviction
market, not a strategy proving itself.

## 8. The integrity bugs, documented because they are the credibility

An investor kit is only worth its willingness to publish its own audit. Nineteen defects were
found and fixed; four of them were in the machinery that produces the numbers an investor
would read, five were on the money path or the trail that records it, two were
credentials — or the guard that was supposed to catch them — sitting in tracked files, and
the last nine came from probing the newest provider, re-reading our own published pack,
testing the fix that was supposed to keep the ledger honest, asking what the health page
actually knows, and then asking the same question of the tooling we wrote to keep our own
figures honest. That second group is worth naming, because nothing about it was
flattering: three of them were in code we had already declared finished, two were in the
numbers we had already printed, and three were inside an earlier repair or the guardrail we
built for one — which is why the last ones were found at all.

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
| The operator's real broker account number was hard-coded in test fixtures, in the shipped `config.example.yaml`, and in the order module's own docstring | a live account identifier in a public repository, next to code that names its venue | replaced with a synthetic sentinel in all eight places, and a tripwire test fails if any live credential value appears in tracked source |
| The credential tripwire could only see credentials that were present in `config.yaml` | a PIN literal in five test assertions and the account number in the shipped example were invisible to it — an earlier pass even reported the account number as replaced while it still sat in eight tracked files | the guard now matches the **shape** of a short numeric secret bound to a secret-shaped name, with an explicit sentinel allow-list, so an unknown value still fails the run; this was caught by re-scanning immediately before tagging v2.6.0, not by the suite |
| The prediction journal and the paper book were rewritten in place | an interrupted write left a truncated file, and every loader below it read "unreadable" as "no state yet" — the next cycle opened at full capital with no positions and no realized P&L, a reset that looks exactly like a healthy account, on the same equity number live sizing divides by | both files are written through a temp file and renamed into place; an unreadable book is now moved aside, journalled, and engages the kill switch instead of silently restarting — 7 tests |
| A provider's HTTP error body was passed through verbatim | the Moonshot account ids that come back inside a `429` were returned by `/api/health` and rendered on the desk, so a status page meant for operators leaked identifiers meant for the provider | every provider error is now run through a shape-based redactor **before** the length cap, keeping the status code and the sentence so the cooldown logic and the operator both still work |
| A model's own stop-token markup was kept in its answer | a hosted Kimi seat answers `OK<|close|>`; the token went on to be read as part of the verdict by the parser and published as part of the vote | special-token markup is stripped from every provider's text before anyone parses it |
| An empty completion was reported as a successful answer | a reasoning model that spends its token budget on hidden thinking returns a blank `content`, and the council counted it as a responding seat while it contributed nothing to the median — a phantom opinion in a "consensus". Reachable on every provider leg but one, which already guarded itself | one shared success constructor, so a blank or markup-only completion is a **failure**: the seat loses its vote instead of skewing it, and the failure is non-fatal so a flaky model is retried next pass rather than cooling its provider for half an hour — 11 tests |
| The paper book rolled its daily figure in memory only | state was written on a fill, so through a quiet stretch — most of this account — the ledger on disk kept the previous day's stamp and the previous day's loss, and the published track record quoted them. A daily loss limit is a brake; a stale stamp means nobody can tell from the file which day a number belongs to | the roll is persisted when the date changes, and a test asserts the desk polling the book does not rewrite it |
| Nothing in the suite compared the pack's figures with the repository | our own published deck quoted a line count, a module count and a test total that were three passes old under a footer saying they were verified that day, and a `pytest` run could not have noticed: the numbers were prose in a markdown table | every quoted figure is now checked twice — codebase figures against the working tree, evidence figures against the regenerated ledger — and one command rewrites them all, so "verified today" is a claim the suite holds up rather than one we type |
| The fix for the first defect in this table trusted today's configuration | to decide whether an old journal row was a call or a non-trade it re-read the row's score against whatever `entry_threshold` is set **now**, and the hit arithmetic still read the pre-fix `correct` column. The threshold is a knob the operator turns, so turning it down reaches backwards and edits history: those rows become wins that were never played, and the published 100% hit rate returns from a journal containing no calls. Turning it up deletes calls that were made and won | the label written that day decides whether a row claimed a direction, and a legacy row's verdict is recomputed from the sign of the realized return against that claim. Neither reads today's config, so no future tuning can rewrite the record — 4 more tests. Found because the new figure test failed on the commit that added it, which sent us back through the ledger code it protects. Worth adding that the first cut of this repair was itself wrong: it matched the label `BUY` exactly, and the ensemble says `STRONG BUY` / `BUY` / `HOLD` / `SELL` / `STRONG SELL`, so the strongest band — the one the strong-signal rate is measured on — would have been filed as a loss it had won. Caught by re-reading the fix against the vocabulary it reads, two commits later, and now covered by two more tests. The write path had the same flaw — `directional` and `correct` were stamped at resolution time from the band configured that day, and those columns are read back for the rest of the record's life, so tuning the gate next week would still have chosen which of this week's snapshots ever counted. Resolution now stamps the claim the row carries, with the band kept only as a fallback for rows too old to be labelled |
| `/api/health` built its provider list from the config file | a key that is present and enabled made `kimi: true`, and the catalog check only spots retired slugs — so the one provider failure we actually have, a Moonshot account answering every request with `429 account … is suspended`, was published as healthy. This is the endpoint the pack names as the place where rot becomes visible, and it was blind to the rot we had | the router keeps the last account-level failure per provider — redacted, timestamped, and cleared the moment the provider answers again, because a 429 rate limit is also a fatal hint and a recovered provider must not stay flagged — and health reports `configured`, `unanswered` and the remaining cooldown as three separate facts, so "a key exists" and "the account answers" can no longer be confused for one badge — 5 tests |
| The guardrail that was supposed to stop stale figures had its own stale figures | the anti-staleness test and the command that rewrites the pack each carried a **hand-written list of the phrasings** a number could appear in — "218 regression tests", "# 218 tests" — and neither list contained the word "passing", so the deck and the one-pager printed one test total and the landing page another while the suite had moved twice since; and the checker explicitly *exempted* the phrase the rewriter rewrote, so the headline count was refreshed but never verified. A checker that enumerates the same vocabulary as the writer cannot find a phrase neither of them knows | the vocabulary is now one table (`tools/kit_stats.RULES`) that the rewriter and the test read together, so adding a layout makes them agree by construction; a second test fails if a number sits beside a figure word with no rule behind it — write "999 passed" into the deck and the suite goes red instead of publishing it; the strategy count joined the checked figures, and the exemption is gone |

Nothing moved money. The journal now says so itself: rebuilt from `data/audit_journal.jsonl`,
the track record publishes the count of lines carrying the test signature and the count of
venue acknowledgements with a real order number, which is zero. That history is left in
place rather than cleaned — a rewritten audit trail cannot prove anything, so the containment
event was appended instead.

One limit we will not paper over: the account-number and PIN literals are gone from the
current tree, but they remain in the pushed git history, and a release that deleted history
would also delete the evidence the audit rests on. Those values must be rotated before any
live arming — and they never were reachable by a scanner that only knew what was in a config
file, which is why the guard now matches shapes rather than known values, to stop the next
one.

The same pre-release scan caught our own deck quoting line counts and a test total that were
three passes old, under a footer that said the figures were verified that day. That was a
document defect rather than a code one — and "be more careful next time" is not a fix for a
class of error that lives in prose, so it is now structural: `python tools/kit_stats.py
--write` recomputes every figure in the pack from the repository and the regenerated ledger,
and a test fails if any of them disagrees. A stale number in this pack is a broken build
rather than an unnoticed sentence.

The first is the one to judge us by. It was not an investor's bug or a reviewer's bug — our
own scorer produced it, our own audit caught it, and it was fixed before a live rupee could
be mis-sized because of it.

## 9. Reproducing every claim

```bash
git clone https://github.com/zhaider5142009-oss/kairos && cd kairos
cp config.example.yaml config.yaml      # add your own keys; config.yaml is git-ignored
pip install -r requirements.txt
python -m pytest tests -q                    # 218 tests
python -m kairos.cli backtest GOLD WTI HBL   # walk-forward sweep, costs modelled
python -m kairos.cli health                  # subsystem health, dead council seats
python -m kairos.cli feedback                # realized track record, live from the journal
python tools/make_track_record.py            # rebuilds docs/investors/TRACK_RECORD.md
python tools/kit_stats.py                    # the only figures this pack may quote
python tools/kit_stats.py --write            # point every quoted figure at them
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
