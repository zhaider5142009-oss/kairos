# KAIROS — outreach drafts
*Copy ready to send. Nothing in this file leaves the machine without the operator's explicit
approval for that specific channel, one at a time.*

> **Before sending anything.**
> - Pull the recipient's address from the firm's own website, the day you send. Do not reuse
>   an address from a list, a scraped sheet, or a previous campaign.
> - Attach only `ONE_PAGER.md` (as PDF) and link the public repo. The whitepaper goes on
>   request, not as a first attachment.
> - If a number is not in `TRACK_RECORD.md`, it does not exist. Do not add one.
> - One channel approved = one channel sent. Approvals do not cascade.

---

## A. Cold email — brokerage / asset manager (Tier 1)

**Subject:** Open-source systematic stack for PSX — code review, not a pitch

> Dear [name / institutional equities team],
>
> I built and run KAIROS, an open-source prediction and execution engine that covers PSX
> equities alongside FX, commodities and crypto, and places its own orders through a JS
> Global bridge under a fail-closed risk layer. It has run continuously since [date] and
> publishes its own results, including the instruments where it lost money.
>
> Two things in it may be worth twenty minutes of your time:
>
> 1. A walk-forward sweep in which one blended signal per instrument is fitted from 2,592
>    candidate parameter sets on a rolling window and traded only out-of-sample, with
>    Pakistani commission and slippage modelled. 4 of 14 instruments came out net-positive; the PSX
>    equity book came out **negative**, mean −3.7%, because fee and slippage drag exceeded
>    the signal. That cost finding is published rather than buried.
> 2. An integrity bug I found in my own scorer: it was grading non-trades as correct
>    predictions and reporting a 100% hit rate, then feeding a confidence bonus into live
>    entry thresholds. Fixed, regression-tested, and written up in the docs.
>
> There is deliberately no performance promise in this message and no request for capital.
> The system has no forward track record yet, and I would rather be judged on the code and
> the journals than on a projection.
>
> Repo: [github URL]. Verification ledger: `docs/investors/TRACK_RECORD.md`, rebuilt with
> one command from the on-disk journals.
>
> If a technical review is useful, I will take questions about anything in the codebase.
>
>Regards,
>[Name] · [phone] · z.haider5142009@gmail.com

*Why it is written this way: a BD inbox deletes "seeking investment". A quant or risk
person reads a published failure and replies.*

## B. LinkedIn DM — analyst / risk manager at a target firm (warm, individual)

> Hi [name] — I noticed you work on [equities research / risk] at [firm]. I've spent the
> last [n] months building an open-source systematic stack for PSX that also trades global
> commodities, and it runs 24/7 on a 2015-era desktop with a paper book and a gated live
> order bridge.
>
> I'm not fundraising. I'd value a code-level critique from someone who does this for a
> living — specifically on my cost modelling, because the equity book loses money net of
> PSX fees and I want to know whether that's my model or the market.
>
> Repo: [URL]. Happy to share the walk-forward journals if that's easier than GitHub.

*Rules: individual names and individual roles only. No connection-request broadcast, no
"Dear Hiring Manager", no mass DM. Max ~15 per week, different copy per firm.*

## C. Show HN

**Title:** I found a 100% hit rate in my own backtester — it was counting non-trades as wins

> Body:
> I built a prediction/execution engine for the Pakistan stock exchange plus global
> commodities. It reported a 100% hit rate for weeks. It was wrong in an interesting way:
> the scorer treated every "no trade" inside the entry band as a correct prediction, and if
> the future bars for a 5-day horizon hadn't printed yet, it clamped the evaluation to the
> last flat bar and graded it as resolved. 140 "correct predictions", zero actual calls.
> Worse, that fabricated accuracy fed a 1.05× confidence bonus into live entry thresholds.
>
> Fix: a snapshot only counts if the score exceeds the entry threshold, only resolves if the
> horizon's bars exist, and the confidence multiplier is pinned to 1.0 until 10 real
> directional calls have matured. Eight regression tests. The honest record today: 4,031
> predictions journaled, 404 horizons matured, 0 directional calls, hit rate not yet measurable.
>
> Also in there: a council of LLM seats where a dead seat must shrink the council rather
> than silently substitute a different provider's opinion into the median.
>
> Repo + the ledger generator that rebuilds every published number from JSONL journals.
> Ask me anything, including about the 10 instruments where the strategy sweep lost money.

## D. r/algotrading / r/quant — methodology post, no pitch

**Title:** Walk-forward sweep of PSX equities against a 2,592-point parameter grid: the fee wall, and a scoring bug that faked 100% accuracy

> [Method: rolling training window, out-of-sample evaluation, commission + CPDC/CDNS +
> slippage on 10-tick books. Results per instrument including the losers. The non-trade
> scoring bug and the horizon-clamping bug, with the patch and the tests.]
>
> I'm not sharing a strategy for sale and there's no link to a product. Code is on GitHub in
> my profile if the method is useful. Main question for this sub: for markets where round
> trip cost is 1-2% of notional, what horizon and turnover do you actually find survives?

## E. X / LinkedIn public post — the recurring cadence

> **Week 1 — the confession.** Thread on the 100% hit-rate bug. 5 posts, one idea each:
> what it reported → why it was wrong → what it fed into live entries → the fix → the honest
> number today (0 directional calls matured).
>
> **Week 2 — the market-structure finding.** PSX equity book loses ~3.7% mean net of costs
> out-of-sample while the commodity book makes +13.6%. One chart, one table, no advice.
>
> **Week 3 — the engineering.** The three-key live-arming design and why a broker PIN never
> touches a log, an API response or a chat window.
>
> **Then weekly.** One timestamped signal card from `kairos.cli share`, plus any journal
> update. Forward record reported as it grows, wins and losses with the same sentence length.

## F. University / demo-day talk (15 min)

> Title: "What my own audit caught before my broker did."
> Structure: live desk demo → the 100% hit-rate bug → the fee wall → fail-closed execution →
> what a real track record takes. Ends with the repo and one question: who wants to try to
> break the risk layer.

---

## Prohibited, in every channel

| never | why |
|---|---|
| "guaranteed / consistent / passive income" | untrue and, in most jurisdictions, actionable |
| any forward projection, even illustrative | reads as a promise once it is screenshotted |
| the 100% or any hit rate | there is no directional record yet; the ledger says so |
| Sharpe or returns from a single best instrument without the 10 losers | cherry-picking is the one thing that ends a diligence conversation |
| screenshots of the funded account, the portal, or any credential | the operator's money and identity, not marketing material |
| pooled capital, managed accounts, "invest with me" | a regulated activity KAIROS is not licensed to perform |
| mass DM, purchased lists, engagement bots | destroys the credibility that is the entire pitch |

## Send log

Record every send here as it happens — channel, recipient, date, version of the kit used.
This is the audit trail a real diligence process will ask for, and it is also how we know
which wording converted.

| date | channel | recipient | version | reply |
|---|---|---|---|---|
| 2026-10-20 | GitHub release (public repo, tag v2.6.0) | open audience — no named recipient | v2.6.0 kit | — |
| 2026-10-21 | GitHub release (public repo, tag v2.6.1) | open audience — no named recipient | v2.6.1 kit, deck attached | — |
| 2026-10-21 | GitHub release (public repo, tag v2.6.2) | open audience — no named recipient | v2.6.2 kit, corrected deck attached | — |

The link every later channel points at:
<https://github.com/zhaider5142009-oss/kairos/releases/tag/v2.6.2> — release notes, the
whitepaper, the regenerated track record and the deck as an attachment. The deck attached there is
the one this repository's current code generates, and its slides were checked against the ledger
before upload by reading the published asset back out of the release.

Earlier releases are left exactly as they were published, because a corrected history is not a
history. v2.6.0's attached deck carried figures stale by several passes. v2.6.1's body and deck
quoted a ledger rebuilt from the wrong sweep log — a run four days older than the release, from
before the strategy repair it describes — and v2.6.2's note names the instruments whose verdict
moved instead of quietly reprinting them. Anyone who downloaded the v2.6.1 deck should discard it
and take the one attached here.

Nothing else has been published: the landing page is unhosted, and there has been no Show HN,
Reddit, LinkedIn, X, cold email, brokerage or fund contact. Each of those still needs its own
approval.

**Same-day asset refresh.** The suite grew by one test after v2.6.2's deck was uploaded, so the
attachment was replaced with the deck the repository now verifies against itself. The release
*body* still prints the totals that were true when it was written, and that is the documented
convention for a dated note — the attachment is the copy a stranger forwards, so it is the one
that has to match the code.

**It was not refreshed a second time, and that is deliberate.** After the release shipped, three
more defects were found on the record path itself — a whole-file rewrite that could drop a
prediction journaled while it ran, an append that could make the journal unreadable, and the
swallowed exception that would have hidden either one. The deck in the repository carries the
larger totals its own guard now checks against; the published asset stays as uploaded, because
its note prints the figures that were true when it was written and refreshing it again would
make the attachment disagree with the body beside it. The next release carries the current deck.
