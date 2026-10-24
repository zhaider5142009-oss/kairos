# KAIROS — investor deck
*12 slides. Say the true thing on every one of them. Figures dated 2026-10-24; refresh `TRACK_RECORD.md` before presenting.*

---

## 1. Title

**KAIROS**
Autonomous prediction and execution for Pakistan and global markets.

Built by one operator, running 24/7, publishing its own losses.

> *Note: open with "I am not here to raise money to trade for you." It buys the room for the next eleven slides.*

---

## 2. The gap

- PSX: ~380 listed companies, retail-dominated, thin books outside the heavyweights, no
  institutional-grade systematic tooling available to a small account.
- Global signal infrastructure exists in abundance and connects to no Pakistani broker.
- Nobody in either market publishes an honest failure record.

**The wedge is not the alpha. The wedge is the plumbing plus the honesty.**

---

## 3. What it is

| | |
|---|---|
| Code | 18,618 lines Python · 45 modules · public repo |
| Strategies | 100 voting into the live consensus; the sweep fits one blended signal, not the library |
| Council | 6 configured LLM seats across 2 providers; a dead seat shrinks the council |
| Execution | paper + JS Global PSX bridge + MT5 |
| Uptime | 24/7 daemon, watchdog, token-gated public API |
| Tests | 958 passing, money path included |

Not a fund. No client money. No licence. No AUM.

---

## 4. Proof: out-of-sample, costs modelled

One composite signal per instrument: four components blended, their weights chosen by a 2,592-point grid search on a rolling 252-bar window, re-fitted every 30 bars, traded only out-of-sample with PSX commission and slippage modelled. It is not the 100-strategy library, and none of these returns come from it.

4 / 14 instruments profitable:

SILVER **+46.2%** · GOLD **+24.5%** (vs +8.9% buy-and-hold, Sharpe 1.32) · WHEAT **+27.2%** (Sharpe 1.59) · WTI **+7.3%** · HBL **+12.4%**

Reported as losses: OGDC −15.8% · BRENT 2.9% · COPPER −6.5% · LUCK −7.0% · BTC −5.8% · KSE100 −5.0% · NATGAS −15.7% · EURUSD −1.3% · MCB −3.0%

> *Note: read the losses before the winners. It is the only slide that makes the winners credible.*

---

## 5. The uncomfortable finding

**PSX equities lost money net of local costs.** −3.7% mean across the equity book against
+8.8% across the global book. Observed fee-and-slippage drag on active stock trials: ~8%.

That is not a bug, it is a market structure fact, and it is the reason most PSX "signals"
never survive to a real account. Any PSX strategy here is sized against cost before
conviction.

---

## 6. What is *not* proven

- **No forward track record.** 4,671 predictions journaled, 779 horizons matured, all 779 were
  non-trades. A directional hit rate does not yet exist.
- Paper book flat: −Rs 13.05 realized on Rs 1,000,000 across 492 cycles, 2 fills.
- Live execution path is built and gated, not exercised at scale.
- No audit, no team, no third-party capital.

> *Note: this slide has ended more conversations than any other. Keep it. A prospect who walks away here was never going to fund slide 4.*

---

## 7. The integrity story — why to trust slide 4

Our own scorer reported **100% hit rate**. It was counting non-trades as correct predictions,
and grading un-printed horizons on the last flat bar. It fed a 1.05× confidence bonus into
live entry thresholds.

We found it, fixed it, pinned confidence to 1.0 until 10 real calls mature, and wrote 8
regression tests so it cannot come back. Ten more defects went out in the next four passes,
five of them on the money path or on the trail that records it: a config that froze its risk
limits at process start, an unbounded duplicate-order guard, ledgers rewritten in place so an
interrupted write reset the paper book to full capital, a test suite that could arm live mode
and write into the production audit journal, and a restart that reported **LIVE** from a stale
state file while running paper. The tenth was about us rather than the code: the tripwire
that is supposed to catch a leaked credential only compared against values present in
`config.yaml`, so a PIN an operator keeps in their head was invisible to it — and a fix we had
already reported as landed had not landed. It was caught by re-scanning by hand before
tagging v2.6.0. The rule now matches the **shape** of a short numeric secret, not just a
value we happen to know. A fifth pass, run after that tag, took out five more: a provider's
HTTP error body that leaked our own account identifiers onto the health page, a model's
stop-token markup parsed as part of its vote, a blank completion counted as a council seat
that had said nothing, a paper ledger whose daily figure rolled in memory and never reached
the file, and the reason any of them could be missed at all — nothing in the build compared
the figures this deck prints with the repository they are supposed to come from. Now a test
does, and one command regenerates them. That test failed on the very next commit — adding
four tests moved the quoted total — and going back through the ledger code it protects turned
out a sixth: the earlier fix that stopped non-trades being scored as wins decided whether a
row was a call from its score against whatever entry threshold is configured *today*, so
turning that knob down would have reached backwards and republished a perfect hit rate built
from trades that were never made. The record now reads the label written that day and
recomputes the verdict from the sign of the realized return; no configuration change can
rewrite history. A seventh pass asked what our own health page actually knows, and then what
our own checker knows. The health endpoint built its provider list from the config file, so a
key that is present made the provider read live — while the account behind one of them answers
every request with a suspension error and two council seats have not spoken in days. It now
reports "a key exists" and "the account answers" as two facts. The checker was worse in a way
that only matters if you trust it: it and the rewriting command each carried their own list of
the phrasings a figure could appear in, and neither list contained the word "passing" — so the
deck, the one-pager and the landing page published three test totals that were two and four passes
old, under that same "verified today" footer, while the machinery built to prevent exactly that
looked straight through them. The vocabulary now lives in one table that both the rewriter and
the test read, a fourth figure (the strategy count) joined it, and a new test fails if a number
sits next to a figure word with no rule behind it. The journal was left as it is — the containment is
recorded in it, because a rewritten audit trail proves nothing.

The quietest one was found while checking whether those pending calls would actually be scored.
A snapshot whose symbol the registry could not name was marked **resolved** with no realized
return, so it left the record entirely — silently, permanently, and most of the pending calls sat
on four watchlist symbols whose registry entries are rebuilt at runtime. An unanswerable lookup
now leaves the row pending and retries it, which is what an unmatured horizon already does, and the
ledger prints the queue beside the answered part: how many calls are waiting, and the first date
the tenth of them can resolve.

Two more came out of the pass that produced this version, and both were found by reading, not by
losing money. One strategy scored its three stack conditions with `+` over numpy scalars, where `+`
on booleans is xor, so its score could only ever be True or False: the long branch was unreachable
and the library carried a guaranteed bearish vote into the consensus on every symbol with 52 bars
or more. Of the 40 symbols here long enough to fire it, 15 now vote long where none ever could.
A test walks the package's syntax tree for that sum shape, because it never raises — it only votes.
The other was this pack's own footer: the figures were regenerated on every pass while the date
printed beside them, and the version the whitepaper named, were typed by hand and had gone stale.
Both are generated from the ledger and the package now, and the new guard failed the first time it
ran — which is the only proof of a staleness check that counts. The same pass found the pack's
PowerPoint, the copy that actually gets attached to an email, sitting outside the guard altogether
and staler than the markdown it shipped beside; its text is now read out of the zip and put through
the same rules. A version written as "v2.6.0" was invisible to a rule that only recognised the word
"Version", which is the same blind spot as a test count written as "passing".

Three more came the day after, from checking whether the queue of pending calls would survive
being graded. The command that scores them reads the journal, walks it, and writes the **whole
file** back; the thing that adds new predictions appends to that same file from any thread
serving a dashboard. Any call recorded during a rewrite was simply not in the file afterwards.
It had never happened, and it could not have: nothing had matured yet, so the resolver had no
reason to write — the defect was scheduled to switch on the day the first hit rate became
measurable. Fixing it needed the write to merge onto a fresh read, and the test that proved the
old behaviour then found the second one, in the same line of the same file: the appender never
checked that the line below it was finished, so a journal copied onto a fresh deploy would weld
two rows into one the parser rejects, and the resolver and the desk both raise on that. The
third was that neither of those failures could have been seen — the pulse that grades the record
swallowed its exceptions in a thread nobody watches, next to a ledger sentence that reads the
same whether the record is unripe or stalled. It is recorded, served on the desk and printed on
the page now.

Four more came from the live risk path, and they share one shape. Every guard that protects an
account compares against equity, so each was written as `if equity > 0` — and a broker portal
that answers with nothing (zero cash, no holdings, no error raised) produces an equity of zero,
which reads to those guards as *nothing to check*. Such a read at the open also set that day's
baseline, and a baseline of zero is never recovered: the account reports a flat day from then on,
and both loss brakes are comparisons against that single figure. One degraded morning read
therefore switched the daily-loss breaker off for the rest of a session in which the account was
losing money, with nothing raised and a calm number on the desk. The pre-trade gate below it had
the same blind spot pointed the other way: it accepted a **negative** equity as usable knowledge,
which is the one account state that should stop everything. Each now refuses rather than skips,
and a cycle running with no brakes reports that as a blocker instead of reporting health.

Six more came from one level further up the same path, and they share a different shape: a
guard that asks a question **of a number**, asked of a value that is not one. `price <= 0`,
`qty * price > cap`, `lo <= price <= hi` and the penny-stock screen's `px < min_px` are all
*false* of a `NaN`, so one unusable figure cleared the notional cap, the broker's band and the
liquidity screen in the same order — and the reason a `NaN` was reachable at all is that the
vendor's chart payload spells a hole as a literal `NaN`, which parses as a number and which the
loader's `if close is None` never catches, so the hole landed at the end of the series where
every price is read. A price of exactly zero is finite, and it survived that check: the position
sizer divides by the price, so one bad print raised out of the instrument loop and ended the
cycle — the cycle in which every *other* position's stop and target were meant to be managed.
Finiteness is now tested **before** bounds, on both the shared gate and the order bridge; a
price at or below zero is refused where it is read, before anything is fitted or sized; and a
screen that cannot see now answers *reject* instead of *pass*.

Three more came from the same path read one level further out, and they fit neither shape above:
an **outage answering in a way that reads as a fact about the account**. MetaTrader spells a
failed retrieve as None and a genuinely flat account as an empty list, and the bridge translated
the first into the second; the broker's two holdings views wrapped their portal call in
`except Exception: data = None`, so a timeout contributed no rows and — with the cash leg still
readable — handed back a funded book that was plausible and empty. What sits downstream of that
translation is the sum and the length the desk publishes: a fused read printed zero percent gross
exposure and cleared the position cap, while reconciliation, whose own documented rule is that an
unreachable venue must not erase memory, applied that rule to the *terminal* but not to the *read*
and dropped live trades whose stop nobody then owned. None is now its own answer all the way to
the gate: an unknown exposure blocks entries and says so on the page, a view that could not be
read raises instead of arriving empty, and forgetting a managed trade requires a read to base it
on.

The last round walked one level further out again, to the trail itself. Every append to the
audit journal ran inside `except Exception: pass`, so a full disk or a file held open ends the
record and nothing notices — and the desk, which reads the same file, prints "no entries yet",
the same words a quiet, healthy session prints. A half-written last line welded the next record
onto it, costing the trail both rows, and one unparseable line made the reader give up on the
whole file. That matters more here than anywhere else in the system, because the journal is the
artifact an investor is being asked to believe: the claim "every decision, order and fill is
appended before it happens" is only worth anything if the appender cannot fail quietly. A write
that does not land now returns a no and names the event that could not be recorded, and **both
live order doors refuse the trade when the answer is no** — the PSX bridge before a ticket
exists, the MT5 bridge before an order request is assembled. An order with no record behind it
cannot be audited, explained or reversed, so it does not go out.

The round before last asked whether a check could be *fooled*. The last one asked whether the
checks are bolted to anything, and five of them were not. **One limit had two names and only
one of them was armed**: `rules.max_daily_loss_pct` is a percentage and feeds the breaker that
pauses entries, `risk.daily_loss_limit_pct` is a fraction and feeds the gate that refuses the
order, and nothing connected them — an operator who tightened the doctrine to half a percent
kept trading through a ten-percent loss, because the knob they turned was not the knob on the
door. The two readings now resolve through one function that returns the tighter value and
**names the key that produced it**, in the refusal on the page, in the terminal and in the
audit trail, so a brake that fired is a brake you can find. The same function makes an
unreadable limit a limit of zero: pasted nonsense stops the next order instead of removing it.

Three shared that shape one level down. The gate validated the numbers in an *order* and read
its own *limits* by trust, so a NaN quantity cap cleared every order and a string in the risk
block raised out of the gate instead of answering no. The posture that decides whether an order
may go out at all converted account figures with bare float calls, so a feed returning `n/a`
ended the cycle that manages every held stop, and a NaN day P&L answered *no* to the single
comparison that asks whether we are past the brake — which is the same fail-open, on the other
side of the ledger. A figure that is not one is now either a zero where zero already means "no
usable figure", or its own blocker where zero would be a lie: an unreadable day reports the
loss brake as blind, and a held position with no readable price reports exposure as
understated rather than as free. And in the feature engine, `float(last_close) or 1.0` turned a
halted print into a one-rupee price and scaled every normalized feature by it — a *value*, which
is what a model consumes without asking — while the same zero inside `np.log` became `-inf` and
made a histogram raise, so one delisted scrip could end the scan for every instrument behind
it. A hole is now a hole: the denominator is absent, the features built on it are absent, and
the log helpers drop non-positive prints before any statistic touches them.

This round walked to the file in between — the queue where a staged PSX order waits for a human
to type the PIN. Every write there rebuilds the whole file from the last read, and that read
answered an empty list to any file it could not parse, which is precisely what a plain
whole-file write leaves behind after a crash. So the panel printed "no staged orders" — the same
words an empty queue prints — while the next ticket anybody staged rewrote the file from that
empty read and destroyed the ones still inside it: an order somebody had staged and was waiting
on, gone, with nothing on the page saying it had ever been there. An ack had the same shape at a
shorter range. It removed its ticket by rewriting the list it had read *before* the broker POST,
so anything staged while the broker was busy came back into the file, and the write came before
the record, so a full disk also cost the line saying this order may or may not have reached the
broker — the one event the trail most needs. Fifteen minutes of expiry ate tickets with no event
behind them at all. A queue that cannot be read is now a refusal with a path in it; the file is
only ever replaced by an atomic rename; a removal re-reads under a lock; the unknown-outcome
record is written first; and an order that went out while its staged copy could not be cleared
says so on the answer instead of leaving a re-ackable ticket quiet. The same rule reached one
level over: an autopilot timestamp it could not parse was invented as *today*, which froze the
daily watchlist rotation and left a held entry with no maturity exit, each of them looking
exactly like a system behaving normally. **A component that cannot answer a question is not
allowed to answer it** — that is the whole rule, and it now holds all the way down to the file a
real order waits in.

This round moved off the money path and onto the system's own bookkeeping, and started by
building the thing the integrity story had been missing: a way to say **which** of the votes
deserved weight. The library was a flat poll, and the journal stored only the fused composite,
so nothing in the system could ever be credited or demoted from evidence — every calibration
above graded the answer and none graded the voter. `tools/strategy_attribution.py` re-derives
the features at every fifth bar from only the bars that existed *then*, takes each strategy's
vote, and grades it once that horizon's bars print. `kairos/attribution.py` turns the resulting
counts into weights, bounded to 0.5–1.5 and shrunk toward 1.0 by sample size, measured against
the up-rate of its own bucket — same instrument, same horizon, same regime — because a strategy
that was right 70% of the time on a tape that rose 70% of the time has proved nothing. The
artifact in this build carries 1,029 graded slices and 31,188 directional votes over 14
instruments; the weights actually applied span 0.79 to 1.18, 24 of the 100 strategies still have
too little evidence to be weighed at all and stay at 1.0, and 15 never cast a directional vote
in the walk. **The engine refuses the artifact** — and says so, on the desk, in the prediction
and in the CLI dossier, running a flat poll instead — if it is stale, incomplete, or measured
against a library of a different size. A weight you cannot switch off is a weight you cannot
audit.

Two defects came out of *using* it rather than writing it. The first walk printed a raw hit rate
over 100% and raised nothing: a counter was adding a long vote into the **hits** column and a
short vote into the **longs** column, so the arithmetic was impossible and every number built on
it still looked plausible. The second run died on its very first slice, inside a helper that had
quietly pre-created an empty bucket, and the exit code reported success. Both shapes are gated
now: the tool refuses to publish an artifact in which hits outnumber votes, names the bucket, and
writes nothing, so whatever weights were in force stay in force; and a test asserts the column
layout itself rather than trusting the totals.

The desk was the quieter find. Clicking any tab did nothing, because a `)` was missing from the
JS Global order panel's ternary and the entire inline `<script>` failed to parse — so nav,
health, the watchlist and every handler were gone, while the API kept answering and the page kept
rendering. **A classic script that dies on a syntax error leaves no trace but a window with no
functions on it**, and nothing in the suite had ever read the HTML: the shipped product's only
interface had been inert since the order desk landed. It has a bracket walk as a test now, one
that replays the defect instead of describing it and fails with the offending HTML line. The same
round caught another in that category: `high_volatility` was a numpy bool, the CLI dossier writes
with `default=str`, and a quiet market was saved as the *string* "False" — so a saved report read
back as True where the live object said False, and the field whose entire job is to say whether
the market is calm meant the opposite on disk. A flag that changes meaning on its way through a
disk is a component nobody ever asked whether it worked.

**A system that audits its own performance numbers is the product.** Everything else is a
chart.

---

## 8. Architecture that transfers

1. Fail-closed execution: three independent arming keys, kill switch, notional cap enforced
   in the order bridge, price band, cash-ledger check, no shorting a cash market,
   stage→ack with TTL.
2. Council integrity: a dead seat shrinks the council instead of substituting another
   provider's opinion into a "median".
3. Self-calibration from realized outcomes, with a leash that refuses to trust thin evidence.
4. Attribution that can be switched off, inside a condition with two named axes: each
   strategy's vote is weighted by its own realized excess over the base rate of its bucket,
   measured out-of-sample — and the bucket is `Range :: Risk-Off`, the instrument's own
   regime joined with a cross-asset reading of seven weighted prints scored point-in-time,
   because skill earned while credit is tightening does not transfer into a melt-up. An
   absent print is not a neutral one: below 55% of the panel's weight the macro axis abstains
   and skill pools across the whole condition rather than crediting a regime nobody
   measured. The same reading sizes the position (risk-off ×0.75, risk-on ×1.10), is carried
   in every journal row, and is checked against the record: of the 21 joint conditions in the
   1,029 graded slices, 14 carry enough votes to move a weight. The engine still refuses the
   weights and runs a flat poll if the evidence is stale, incomplete, measured against a
   library of a different size, or classified from a panel too thin to read. The divisors are
   declared judgment constants, deliberately not fitted to the sample.
5. Venue-agnostic risk layer — PSX cash, MT5 commodities and FX, PMEX futures. The
   bridge attaches only to the account it was configured for, and its link cost is a
   measured number (`tools/mt5_probe.py`), not an adjective.
6. Secrets handling: broker PIN storable in git-ignored config only, never logged or echoed,
   one-click disarm.

---

## 9. Who this is for

| buyer | why it is worth something to them |
|---|---|
| **PSX broker / digital desk** | an execution + risk stack that already speaks their portal, as a differentiator for their own clients |
| **Asset manager / mutual fund** | signal research and an instrumented journal they can diligence — costs and all |
| **Exchange / market infrastructure** | a working proof that retail-accessible systematic tooling can be built locally |
| **Strategic technology acquirer** | the only Pakistani-market stack of this shape that is open enough to verify |

---

## 10. Business shape

- **Now:** licensing / acquisition of the engine, or a bounded pilot inside a licensed firm.
- **Pilot deliverable:** a real forward record — 100+ matured directional calls on a named
  book, published either way.
- **Deliberately not for sale:** pooled capital, managed accounts, or a return promise.
  Those require a licensed vehicle, and proposing one without counsel would be the fastest
  way to lose the only asset that matters here: credibility.

---

## 11. Roadmap, evidence-first

| step | gate |
|---|---|
| 1. Grow the forward record | 100 matured directional calls; hit rate published however it lands |
| 2. Fix the cost problem | a PSX equity book that survives fees out-of-sample, or PSX equities are dropped |
| 3. Pilot on a real but bounded mandate | one institution, one instrument class, kill switch in their hands |
| 4. Licensed structure | only with counsel and a regulator-appropriate wrapper |

---

## 12. Close

**Verify, don't trust.**

```
github.com/zhaider5142009-oss/kairos
python tools/make_track_record.py     # rebuilds the ledger from the journals
python tools/kit_stats.py --write     # re-points every figure in this kit at it
```

z.haider5142009@gmail.com

> *Note: ask for one specific thing — a 20-minute technical review with someone who can diligence a codebase. Not money.*

---

## Presenter rules

1. Never say "returns", "guaranteed", "passive income", "100%" or "AI beats the market".
2. Lead every number with its caveat once, out loud.
3. If asked "how much can I make", answer: "there is no forward record yet; that is what a
   pilot is for."
4. If asked for money, answer: "I'm not raising a fund; I'm looking for a licensed partner
   and a pilot."
5. Send only what the deck already says. No bespoke projections, ever.
