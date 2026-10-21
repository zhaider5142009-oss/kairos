# KAIROS — technical whitepaper
**A self-auditing prediction and execution stack for Pakistan and global markets**

Version 2.6.2 · 2026-10-22 · single-operator, publicly verifiable

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
| Strategy | `kairos/strategies.py` | 100 strategies voting direction, strength and rationale into the live consensus |
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
- 319 regression tests, including the money path and the integrity cases below.

## 7. What the numbers say, and what they do not

Walk-forward, costs modelled, 5/14 instruments profitable — one blended signal per instrument, its weights selected from 2,592 candidates rather than the 100-strategy library — commodities strongest
(SILVER +43.5% against +36.1% buy-and-hold, and GOLD +29.3% against +11.6% with Sharpe 1.57),
and 9 instruments reported as losses.

**PSX equities lost money net of local costs** (−2.2% mean across the equity book, and a
~8% fee drag observed on the active stock trials). This is the single most commercially
important finding in the project: the edge, if any, is in the global and commodity books,
and any PSX strategy must be sized against costs before it is sized against conviction.

The live paper book is essentially flat (realized P&L −Rs 13.05 on Rs 1,000,000 over 226
autopilot cycles) with two fills. That is a system working as designed on a low-conviction
market, not a strategy proving itself.

## 8. The integrity bugs, documented because they are the credibility

An investor kit is only worth its willingness to publish its own audit. Thirty-nine defects
are tabled below, each with what it would have cost had it shipped. Eight of them were in the
machinery that produces the numbers an investor would read — the scorer that turns a journal
into a hit rate, the command that regenerates the evidence, the ledger repair meant to keep
that history honest, and the guardrail built to keep the pack's figures current. That group is
worth naming first, because nothing about it was flattering: three were in code we had already
declared finished, two were in numbers we had already printed, and three were inside an earlier
repair or the guardrail we built for one — which is why the last ones were found at all.
Nineteen more were on the money path or the trail that records it, nine were a provider or a
bridge answering in a way the desk then reported as health, and three were credentials — or the
guard that was supposed to catch them — sitting in tracked files.

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
| The guardrail that was supposed to stop stale figures had its own stale figures | the anti-staleness test and the command that rewrites the pack each carried a **hand-written list of the phrasings** a number could appear in — "N regression tests", "# N tests" — and neither list contained the word "passing", so the deck and the one-pager printed one test total and the landing page another while the suite had moved twice since; and the checker explicitly *exempted* the phrase the rewriter rewrote, so the headline count was refreshed but never verified. A checker that enumerates the same vocabulary as the writer cannot find a phrase neither of them knows | the vocabulary is now one table (`tools/kit_stats.RULES`) that the rewriter and the test read together, so adding a layout makes them agree by construction; a second test fails if a number sits beside a figure word with no rule behind it — write "999 passed" into the deck and the suite goes red instead of publishing it; the strategy count joined the checked figures, and the exemption is gone. This row's own examples are written as patterns rather than numbers because of a defect found while regenerating the v2.6.1 pack: the rewriter matches on adjacency, so it cannot tell a live figure from a historical value quoted inside the defect narrative, and it had been quietly overwriting the very numbers this table cites as evidence of the mistake — every earlier version of this row claimed the old lists contained whatever the suite size happened to be that day |
| The figure guard did not read the file investors actually receive | every check covered the markdown and the HTML, because those are what the rewriting command touches, and the PowerPoint deck — the copy attached to the release and forwarded to a stranger — was outside the guard entirely. It was further stale than the prose beside it: an old line count, an old module count, the same old test total in two places, an old journaled count, an old cycle count, and a version stamp two releases behind, under a footer claiming the figures were verified against the repository on a day they had not been | the deck is read as text now: a `.pptx` is a zip of XML and its runs flatten into the same vocabulary the rules already match, so the checked figures and the stamped date and version are compared against it before it can ship. Read-only deliberately — rewriting run boundaries inside a binary is how a deck gets corrupted, so the test names the value to rebuild with rather than editing the file quietly. Two phrasings had to join the stamp table first: a version written as "KAIROS vN.N.N" and a footer saying "verified against the repository on" a date. A checker that only knows the word "Version" reports a stale version as *absent*, not as wrong, which is the same failure as a checker that only knows "tests" and not "passing". Guarding the deck then cost the deck two of its numbers: the journal grows on its own every ten minutes, and a file nobody re-runs on that clock must not print a count it cannot keep current, so the slides now carry the figures that are stable inside a release and point at the dated ledger for the ones that tick. |
| The command that regenerates the evidence was reading the wrong sweep log | ties in the walk-forward log choice were broken on row count alone over a sorted glob, so the run re-executed after this release's strategy repair lost to a four-day-old run of the same width. The pack's headline count, its per-venue means and several instruments' verdicts were therefore produced by code the same release admits was defective, under a footer saying the figures were verified that day | ties go to the newest file by modification time, the ledger names the log it read and the afternoon it ran, the skipped instrument is printed outside the denominator instead of inflating it, and the verdicts that moved are listed in the release note rather than corrected quietly |
| Every rule in the guard checked a digit, and the pack's worst false claim was a noun | the deck's cover and the ledger both wrote the sweep as `Walk-forward sweep, N strategies per instrument`. The sweep fits one blended signal, not the library, and the count beside the false noun was itself a verified figure — so a complete refresh cycle passed it. The same shape hid the headline ratio and the book means: each was read only when one particular word sat next to the number, and a sentence that put the descriptor a clause away was invisible to the rewriter and the checker alike | a ratio is read by its denominator and the nearest win/loss word decides which side it counts, a mean is read by the book named beside it, and `method_claims()` denies any sweep sentence that credits the strategy library with the evidence. Each rule ships with a test written in the opposite direction, so the guard fails if it stops seeing one of the shapes |
| A call whose symbol the registry could not name was deleted from the record | `feedback.resolve_outcomes` marked any snapshot it could not look up **resolved**, with `outcome=no-instrument` and `directional` decided from the entry threshold configured *that day*. It recorded no realized return, so the row appeared in neither the numerator nor the denominator — the forward record just became smaller, permanently, and raised nothing. One cycle in which the dynamic PSX registry had not been rebuilt, or a journal copied to a fresh deploy without it, would erase every pending call on those names at once. At the time it was found, most of the pending calls sat on four watchlist symbols, and the first batch of them was days away from becoming the kit's first measurable hit rate | an unanswerable lookup now leaves the row pending and retries it on the next pass, which is already what an unmatured horizon does, and the row's own label is the only thing that decides whether it ever claimed a direction. The ledger prints the queue as well as the answered part — how many calls are still waiting and the first date the tenth of them can resolve — so a record that has stalled is visible instead of merely empty. One test pins the behaviour both ways: unresolved while the lookup fails, then scored normally once the symbol answers again |
| A snapshot journaled while the record was being rewritten was deleted from it | `resolve_outcomes` reads the journal, walks its own copy for as long as the bars take, then overwrites the **whole file** from that copy — while `record` appends to the same file from any thread that finishes a prediction, which includes a visitor on the public dashboard, the desk's own auto-refresh, and a second terminal running `kairos predict`. A row landing inside that window was not marked unresolved or mis-scored, it was never in the file again. The atomic rename shipped in v2.5.0 made this *harder to notice*: the file is always well-formed, merely short one call. And resolution had had no reason to write at all, because nothing had matured yet — so the defect armed itself on the first day the record becomes measurable instead of announcing itself in the weeks before | the write merges onto a fresh read of the file rather than onto the stale copy, and both writers take one lock so an append cannot land between that read and the rename. The merge is positional, because the journal is append-only: keying it on a row's own fields collides on two snapshots of one symbol in the same second, and different cache entries — `use_llm` on the desk, off in the autopilot — make that pair ordinary rather than rare, which is how the first version of this fix lost a verdict it had just computed |
| The pass that grades the forward record said nothing when it failed | the resolve pulse sat inside `except Exception: pass`, in a thread nobody watches, so a failure there neither stopped the desk nor appeared anywhere — while the ledger's own queue sentence ("*N* calls waiting until *date*") stayed true-looking and simply aged. A stalled track record and an unripe one render as the same words, and only one of them is honest | the failure is recorded in the autopilot state as `feedback_error`, appended to the audit journal as its own event, cleared when a pass succeeds, and served by `/api/autopilot`. The ledger reads that field before it prints the queue's date, so a record that stopped maturing says so on the page rather than leaving a reader to work it out |
| One copied journal line could make the whole record unreadable | `record` appended without checking that the line below it was finished. A track record restored from backup, copied onto a fresh deploy, or hand-edited — all three are paths this pack describes — welded the new snapshot onto the last row into a single line no parser accepts. The damage was not one row: the resolver raises before it scores anything and the desk raises behind it, and the swallow above hid both, so the record stopped growing while still displaying yesterday's calls | the appender closes an unterminated last line before writing, so a copied journal repairs itself on the next snapshot with nothing lost on either side of it |
| A portal that answered with nothing was read as an account that owned nothing | the function that reads the funded book documents that an outage-shaped zero must never reach the risk guards — and then honoured that only for an outage which **raised**. `ledger_balance` answers a mis-keyed balance query with `0.0` rather than an error, and `open_positions` turns a transport failure into an empty list, so a portal in pieces hands back free cash of zero, no holdings and `ok: true` without raising anything. That is the exact shape the guard existed for, arriving through the one door it did not watch | a book that cannot describe a funded account is now a venue loss like any other: journalled once, stood the auto-place path down to a human-confirmed ticket, and never reaches the arithmetic that sizes an order |
| One degraded read at the open switched the daily loss breaker off until the close | the PKR day's profit is measured against the first equity snapshot of the calendar day, because the portal reports no day figure. A zero stamped there is never lifted: every later reading compares against it, the comparison declines to run on a zero base, and the day reports flat forever after. So a portal that answered nothing once at the open, followed by a healthy book and a real five percent loss, showed a day P&L of zero — and both loss brakes, the autopilot's and the pre-trade gate's, are comparisons against that one number. Nothing raised and nothing was logged; the desk displayed a calm day on an account that was losing money | only a positive equity may set the baseline, and a baseline an earlier build latched at zero re-stamps from the first usable read rather than poisoning the day. One test walks the whole shape: outage at the open, healthy book, real loss, breaker fires |
| The pre-trade gate accepted a negative equity as a usable figure | the gate's own rule is that it needs **positive knowledge**, which it tested as "an equity that is neither absent nor zero" — then every check below compared against equity only `if equity > 0`. A margin call is a real account state, and it was the one state that satisfied the first test and opted out of every check that followed. An equity that was not a number at all raised out of the middle of the gate instead of answering no | the figure is read once, must parse, and must be positive; anything else is a refusal that names what it saw. A test walks four unusable readings and requires all four to stop the order |
| A live cycle with no risk figures at all reported no blockers | each circuit breaker divides by equity, so a cycle holding none — every venue unreadable — came back with an empty blocker list and entries unblocked. The order itself still failed closed at the gate, so nothing moved; but the posture is the thing `/api/autopilot` and the desk publish, and "no brake is running" was displayed as "nothing is wrong" | an unusable equity is now itself a blocker, so a cycle with no brakes says so on the page instead of leaving the reader to work out that the clean report was an empty one |
| An order whose numbers were not numbers cleared every guard on it | the four rounds above all refuse an unusable *account*, but the order itself was never asked to be a number. `price <= 0` is False for a NaN, `qty * price > cap` is False, `lo <= price <= hi` is False, `quantity <= 0 or quantity > max_qty` is False — so one NaN arriving from a prediction, a config arithmetic slip or a hand-typed ticket satisfied the notional cap, the per-order cap, the broker's price band and the quantity bounds at once, and went to the portal as a well-formed request | finiteness is checked **before** bounds, because a comparison cannot detect the one value that answers every comparison the same way. Both doors are guarded — the shared pre-trade gate and the broker bridge, which refuses before it stages anything — and a test walks NaN and infinite quantities and prices through each |
| A price band the bridge could not read was filed as a price that passed | the broker's per-instrument band table is a portal call, and a failed one is cached for five minutes so a dying portal cannot lock the account out through repeated logins. While it is cached-as-failed the band check simply does not run — and the order trail, which is the evidence an auditor reads, recorded the stage exactly as it records a stage the band approved | an order the band could not be checked against is journalled as its own event, so the trail distinguishes "inside the broker's limits" from "nobody knew", and the notional cap that does not need the portal still applies |
| A hole in the vendor's price series became a bar | Yahoo's chart payload spells a missing close as a literal `NaN`, which `json` parses without complaint and which the loader's `if close is None` never catches. The row went into the history as a real session, and because the loader backfills a missing open, high and low from that row's close, one hole became four unusable numbers in one bar — and the bar sits at the end of the series, where every price, return and range is read from | every OHLC cell is converted through one helper that returns a number or nothing, so a `NaN`, an `Infinity` or a string is a missing bar rather than a poisoned one; a chart of nothing but holes is no history at all |
| The desk quoted a price it could not see | `predict` read the last close into `price` and returned `ok: true` with it, whatever it held. A NaN price is the number that sizes a live order and stamps the entry of the paper book the published record is graded from, and the round above shows how it travels — an infinite close was worse, it reached the feature engine and raised `ValueError` out of a histogram, taking the whole prediction down with an unhandled exception instead of an answer | the price is checked where it is read, before anything is fitted, and an unusable one is a refusal that names what it saw; a test proves the model battery never runs on a series the desk cannot price |
| The screen built to reject unusable scrips called an unusable scrip liquid | the PSX microstructure screen exists to keep the autopilot off penny stocks and dead names, and it asked `px < min_px` of the last close. A NaN answers no, so an unreadable price cleared the penny check on its way to a real order — while the volume check three lines below it *did* test for NaN, which is how the asymmetry survived review. A history read that raised was swallowed into the same "tradable" answer | a screen that cannot see answers *reject*: an unusable price and an unreadable history are both rejection reasons now, journalled as skipped entries, and a test pins all three paths — unusable price, raising read, and a genuinely liquid name that must still pass |
| A price of exactly zero divided the position sizer, and one symbol's bad print ended a whole cycle | zero is finite, so it survived the round above — and every figure in the sizing block is a quotient by the price: `notional / price` for a paper or CFD ticket, `notional // price` for PSX shares, and the bank-size formula's `price × stop distance`. A vendor serving a halted or delisted name as `0.0` therefore raised out of the middle of the instrument loop, past the one `try` that only wraps the prediction call, and the lap that died was the lap in which *every other* position's stop, trail and target went unmanaged | a quote at or below zero is refused as a hole rather than read as a cheap instrument, before anything is fitted or sized. The feature engine had already been quietly substituting `1.0` for one — the same mistake wearing a different face, and the reason a zero could look like a number right up to the division |
| A broker view that failed to answer was filed as an account holding nothing | the zero-equity guard catches an outage only when it arrives shaped like a zero, and this one did not: both holdings views wrapped their portal call in `except Exception: data = None`, so a timeout, a gateway error or a body that was not a list contributed no rows and raised nothing. With the cash leg still readable the funded book came back plausible and **flat** — no zero for the guard to see, no exception for the desk to stand down on — and every exposure, sizing and reconciliation decision that lap went on to make was made about an account nobody had actually read | the two views are read through one helper that keeps three answers apart: a list is the holdings, a null on a successful call is the broker's own "nothing here", and anything else raises as a venue loss. The import that prunes the watched book against what the broker reports now stops before it prunes anything |
| MetaTrader's "the request failed" was translated into "you hold nothing" | the terminal's own API distinguishes the two — it answers None when a retrieve could not be made and an empty sequence when the account really is flat — and the bridge collapsed the first into the second. Both are read downstream as facts: gross exposure summed to nothing, the simultaneous-position cap cleared, and the account snapshot published a position count of zero for a terminal that had not answered at all | None travels all the way to the pre-trade gate, which already refuses an order on a missing position snapshot, and the account snapshot carries the unknown instead of inventing a count |
| The desk showed a flat book for a venue that had not answered, then forgot the trades it was managing | the posture's exposure and position count were a sum and a length over that same collapsed list, so the live page printed zero percent gross exposure on a fused read while the cap that consumes the figure had just cleared it. Reconciliation — whose documented rule is that an unreachable venue must not erase memory — applied that rule to the *terminal* but not to the *read*, and dropped the managed entries whose stop, trail and target nobody then owned | an unreadable position list is its own answer: exposure and count publish as unknown, the posture carries a blocker naming the fused read so it can never report a clean risk picture it did not have, reconciliation declines to forget anything, and the desk prints "unreadable" where it used to print a zero |

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
python -m pytest tests -q                    # 319 tests
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
