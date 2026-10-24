# Investor & diligence pack

Everything here is generated from, or checkable against, the on-disk journals. If a number
appears in one document and not in another, the journals win — and the mistake is a bug.

| file | for whom | what it is |
|---|---|---|
| `TRACK_RECORD.md` | everyone | the machine-generated ledger. Source of truth for every figure quoted elsewhere. Regenerate: `python tools/make_track_record.py` |
| `ONE_PAGER.md` | first contact | one page, all caveats intact |
| `DECK.md` | a meeting | 12 slides with speaker notes and presenter rules |
| `WHITEPAPER.md` | technical reviewer | architecture, integrity bugs found and fixed, reproduction steps |
| `TARGETS.md` | internally | who to approach, in what order, through which channel |
| `OUTREACH.md` | internally | drafts per channel, prohibited claims, send log |
| `landing.html` | public web | self-contained page; deploy only with explicit approval |
| `KAIROS_Investor_Deck.pptx` | a meeting, by email | the same slides as a file. `--write` cannot rewrite a binary, so it carries only the figures that are stable inside a release and points at `TRACK_RECORD.md` for the ones that tick |

## Standing rules for this pack

1. **No forward track record exists.** 3,965 predictions journaled, 404 horizons matured,
   and all 404 were non-trades. Never quote a hit rate; the ledger says "not yet measurable"
   and so must every other document.
2. **Backtests are labelled backtests**, per instrument, including the 10 unprofitable ones.
3. **No return promise, projection, or illustration of future performance** in any channel.
4. **Nothing is sent or published without the operator's approval for that specific
   channel**, one at a time. Approvals do not cascade.
5. **No pooled capital, no managed accounts, no "invest with me."** KAIROS is not licensed
   to hold or manage third-party money; the honest commercial shapes are licensing,
   acquisition, a bounded pilot inside a licensed firm, or the operator's own employment.
6. **The integrity bugs stay in the documents.** Publishing the 100%-hit-rate fabrication is
   the most persuasive thing in the pack, because it is the one claim a reviewer can verify
   from the git history and the tests.
7. **A slide may not hold a number that moves.** The journal grows every ten minutes, and a
   binary deck cannot be re-run on that clock, so the deck quotes the dated ledger for
   predictions journaled and autopilot cycles instead of printing them. Every figure the deck
   does print is checked against the repository by `tests/test_kit_figures.py`.

## Update discipline

Nothing in this directory is allowed to be typed. Regenerate the ledger, then let the pack
pull its figures from it:

```bash
python tools/make_track_record.py
python tools/kit_stats.py            # the only figures this pack may quote
python tools/kit_stats.py --write    # point every quoted figure at them
python -m pytest tests/test_kit_figures.py -q
```

`tests/test_kit_figures.py` is not a formality. It compares each codebase figure against the
repository and each evidence figure against the regenerated ledger, so a number that has
moved and was not re-checked fails the build instead of going out in an email. That test
exists because the published v2.6.0 deck quoted a line count and a test total three passes
old under a footer saying they were verified that day.

`--write` rewrites the sweep as well as the codebase counts: each instrument's net return,
buy-and-hold, drawdown and Sharpe, the `profitable`/`unprofitable` verdict words, the HTML
row colour classes, the headline ratio and the loser count — in whichever shape the sentence
writes them, because "4 of 14 profitable", "profitable on 4 of 14 instruments" and "10 of 14
instruments in the sweep lost money" are one claim and the first version of the rule only knew
the glued form — and the per-venue means, which a venue phrase beside the number identifies as
readily as the word "mean" does. It attributes a figure to the nearest instrument name before
it, which is also a writing rule — a number that sits further from its own instrument than from
another one gets charged to the wrong one and the lint says so. Two claims the rewriter cannot
write, and so refuses: a sentence that denies a figure this directory publishes, and a sentence
that credits the strategy library with the sweep's own evidence — the one false statement here
that no digit could reveal, because the number beside it was correct. Both are reported, with
any contradicted figure, by running `python tools/kit_stats.py` with no arguments. Two things
stay outside the rewriter's reach on purpose: the dated release notes under `releases/` (frozen
at publication, so they keep the numbers that were true that day) and
`KAIROS_Investor_Deck.pptx` (a binary no tool here edits), which is re-patched by hand.
