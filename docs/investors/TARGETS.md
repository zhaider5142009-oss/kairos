# KAIROS — target list and channel plan
*Who to talk to, in what order, and what "talk to" means here.*

> **Rules before anything leaves this machine.**
> 1. Contact details are **not** in this file on purpose. Emails and forms change, and a
>    guessed address that bounces or lands in the wrong department burns the one shot.
>    Pull the address from the firm's own website at send time, from the page that is
>    explicitly about business development, institutional sales or careers.
> 2. Nothing is sent or posted without the operator's explicit per-channel approval.
> 3. Nothing is claimed that `docs/investors/TRACK_RECORD.md` does not already show.
> 4. No pooled capital, no managed-account offers, no projections — see slide 10.

---

## Tier 1 — people who can say yes in a week

| target | why | route | what to send |
|---|---|---|---|
| **PSX-listed brokerage groups with a proprietary/digital desk** — Arif Habib Ltd, AKD Securities, Topline Securities, Next Capital, AFG (Alfa) Securities, Mir Corporate, Bahria Capitals | they already own the regulated wrapper, the client base and the portal integration; an execution + risk stack is a buying decision, not a licensing debate | BD / institutional-equities address from the firm's own contact page; or a second-degree intro via their quant/risk staff on LinkedIn | one-pager + repo link + the integrity slide |
| **SECP-regulated asset managers** — Meezan Asset Management, EFU Asset Management, UBS Fund Management (Pakistan), Alfalah Asset Management, Habib Asset Management | systematic overlay research and an instrumented journal are diligence-able products for them; several already run in-house quant teams | research/due-diligence contact on their funds' site; their published "professional investors" desk | one-pager + whitepaper section 4 and 8 |
| **The exchange itself (PSX) and its market-technology arm** | a locally built retail-accessible systematic stack is a market-development story they actively tell; also the fastest way to learn what is permitted | their published innovation/market-development channel, or via a member broker | whitepaper + roadmap slide 11 |

**Reality check:** the big brokerages will not answer a cold email from a student-age
account. Warm intros (LinkedIn to their analysts and risk managers, not to "info@") convert
roughly 5–10× better and cost nothing.

## Tier 2 — capital that understands a pre-record system

| target | fit | caveat |
|---|---|---|
| **i2i Ventures, VentureDive, Plan9, Indus Ventures, Careef Capital** | Pakistan/South-Asia funds that have backed fintech and market-structure startups; some have explicit fintech vehicles | most want a company with users and revenue, not a trading engine; expect "show us adoption first" |
| **500 Global, Flat6Labs, Shorooq Partners, Global Ventures, 212 Founders** (region-wide) | fintech mandates, faster first meetings than local funds | verify each vehicle is still active in Pakistan before writing to it |
| **Family offices and HNI investor communities in Karachi/Lahore/Islamabad** | the only pool that will fund a bounded pilot without a company structure | **highest compliance risk.** No pooled capital, no managed accounts, no performance promise — talk about a technology pilot, not a return |

**Do not** approach: SECP-registered investment advisers offering "portfolio management" to
retail (that is the regulated activity we cannot perform), crowdfunding platforms, or any
channel whose terms forbid solicitation of trading products.

## Tier 3 — distribution that creates inbound instead of chasing outbound

This is where "investors" actually find a project like this, and it is free.

| channel | play | why it works here |
|---|---|---|
| **GitHub** | polish the README with the ledger at the top, tag releases, add topics (`algorithmic-trading`, `psx`, `quant`, `risk-management`), keep commits legible | every institutional technologist who is interested will end up here; the repo *is* the pitch |
| **Hacker News — Show HN** | lead with the integrity bug: "I found a 100% hit rate in my own backtester. It was counting non-trades as wins." | this is exactly the story HN rewards; a "I built a trading bot that makes money" post is flagged |
| **r/algotrading, r/quant (Reddit)** | post the *methodology and the failure analysis*, never the pitch; link the repo in profile | the sub bans return claims on sight and upvotes honest walk-forward write-ups |
| **LinkedIn** | 3 posts: (1) the integrity bug, (2) PSX fee drag measurement, (3) fail-closed live-order design | this is where Pakistani brokerage and asset-management people actually are |
| **X / Twitter fintech-PK and algo-trading circles** | the shareable signal card KAIROS already generates (`kairos.cli share`) as proof-of-life, one per week | visual, verifiable, timestamped, low effort |
| **TradingView public scripts / MQL5 & cTrader marketplaces** | publish an *indicator* version (signal only, no order routing) with the walk-forward method documented | the largest genuine audience of self-directed investors, and a licensing conversation starter |
| **Value Investing Pakistan and PSX investor communities** | participate as a researcher publishing methods and losses; no solicitation | credibility in the local retail market compounds faster than any ad |
| **University / hackathon / startup events in PK** (LUMS, NUST, IBA fintech circles, Plan9 programmes) | a 15-minute demo of the desk and the audit journal | produces the warm intros Tier 1 needs |

## Sequence

1. **Week 1 — make it findable.** README + ledger published, GitHub topics and tags,
   whitepaper linked, one LinkedIn post (the integrity bug).
2. **Week 2 — make some noise.** Show HN, one r/algotrading methodology post, first signal
   card on X.
3. **Week 3 — go warm.** 10–15 individual LinkedIn messages to analysts/risk managers at the
   Tier 1 firms; ask for a technical review, not money.
4. **Week 4 — go cold, carefully.** 5–8 BD emails with one-pager + whitepaper attached.
5. **Ongoing — publish the record as it grows.** Every month, the ledger updates itself and
   the campaign re-attaches the newest version. Silence is fine; spin is not.

## What "success" looks like, in order

| milestone | signal |
|---|---|
| inbound repo stars / issues from practitioners | Tier 3 is working |
| a reply from someone who read the code and asked a specific question | worth more than a meeting |
| one technical review meeting | Tier 1 conversion |
| one bounded pilot with a named institution | the whole point |
| an offer to buy or license | only meaningful after the forward record exists |
