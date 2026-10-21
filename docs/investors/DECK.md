# KAIROS — investor deck
*12 slides. Say the true thing on every one of them. Figures dated 2026-10-21; refresh `TRACK_RECORD.md` before presenting.*

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
| Code | 12,354 lines Python · 37 modules · public repo |
| Strategies | 100 voting into the live consensus; the sweep fits one blended signal, not the library |
| Council | 6 configured LLM seats across 2 providers; a dead seat shrinks the council |
| Execution | paper + JS Global PSX bridge + MT5 |
| Uptime | 24/7 daemon, watchdog, token-gated public API |
| Tests | 283 passing, money path included |

Not a fund. No client money. No licence. No AUM.

---

## 4. Proof: out-of-sample, costs modelled

One composite signal per instrument: four components blended, their weights chosen by a 2,592-point grid search on a rolling 252-bar window, re-fitted every 30 bars, traded only out-of-sample with PSX commission and slippage modelled. It is not the 100-strategy library, and none of these returns come from it.

5 / 14 instruments profitable:

SILVER **+43.5%** · GOLD **+29.3%** (vs +11.6% buy-and-hold, Sharpe 1.57) · WHEAT **+26.8%** (Sharpe 1.60) · WTI **+16.0%** · HBL **+12.4%**

Reported as losses: OGDC −14.3% · BRENT −9.4% · COPPER −9.1% · LUCK −6.1% · BTC −3.9% · KSE100 −1.8% · NATGAS −1.5% · EURUSD −1.3% · MCB −1.1%

> *Note: read the losses before the winners. It is the only slide that makes the winners credible.*

---

## 5. The uncomfortable finding

**PSX equities lost money net of local costs.** −2.2% mean across the equity book against
+10.0% across the global book. Observed fee-and-slippage drag on active stock trials: ~8%.

That is not a bug, it is a market structure fact, and it is the reason most PSX "signals"
never survive to a real account. Any PSX strategy here is sized against cost before
conviction.

---

## 6. What is *not* proven

- **No forward track record.** 1,525 predictions journaled, 140 horizons matured, all 140 were
  non-trades. A directional hit rate does not yet exist.
- Paper book flat: −Rs 13.05 realized on Rs 1,000,000 across 210 cycles, 2 fills.
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
4. Venue-agnostic risk layer — PSX today, MT5 now, PMX scaffolded.
5. Secrets handling: broker PIN storable in git-ignored config only, never logged or echoed,
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
