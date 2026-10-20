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

## Standing rules for this pack

1. **No forward track record exists.** 1,018 predictions journaled, 140 horizons matured,
   and all 140 were non-trades. Never quote a hit rate; the ledger says "not yet measurable"
   and so must every other document.
2. **Backtests are labelled backtests**, per instrument, including the 6 unprofitable ones.
3. **No return promise, projection, or illustration of future performance** in any channel.
4. **Nothing is sent or published without the operator's approval for that specific
   channel**, one at a time. Approvals do not cascade.
5. **No pooled capital, no managed accounts, no "invest with me."** KAIROS is not licensed
   to hold or manage third-party money; the honest commercial shapes are licensing,
   acquisition, a bounded pilot inside a licensed firm, or the operator's own employment.
6. **The integrity bugs stay in the documents.** Publishing the 100%-hit-rate fabrication is
   the most persuasive thing in the pack, because it is the one claim a reviewer can verify
   from the git history and the tests.

## Update discipline

When the ledger changes, regenerate it and re-check the three documents that quote it:

```bash
python tools/make_track_record.py
grep -rn "890|140\|140\|+44.5\|−2.8" docs/investors --include=*.md
```

Any figure that has moved but was not re-checked is a misstatement waiting to be screenshotted.
