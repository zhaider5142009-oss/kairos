"""The investor pack's numbers must be the repository's numbers.

This exists because the published v2.6.0 deck quoted a line count, a module count
and a test total that were three passes old, under a footer that said the figures
were verified that day. Nothing in the suite noticed: the numbers were prose in a
markdown table, and the only generated document in the pack was the track record.

That first fix had its own flaw, found by re-reading the pack against it. The
rewriting command and this test each carried their own hand-written list of the
phrasings a figure could appear in — "161 regression tests", "# 161 tests" — and
neither list contained "121 passing" or "<b>87</b><small>passing tests", which is
how the deck, the one-pager and the landing page went on publishing three test
totals that were two and four passes old while the machinery that was supposed to
prevent exactly that looked straight through them. A checker that enumerates the
same vocabulary as the writer cannot find a phrase neither one knows.

So the vocabulary now lives in one place, `tools/kit_stats.RULES`, which this test
imports: adding a layout makes the rewriter and the checker agree by construction.
And `test_no_figure_phrase_lacks_a_rule` is the part neither of the old files had —
a scan for digits standing next to a word that can only mean a figure, failing if
no rule covers that spot. Write "999 passed" into the deck and the suite goes red
instead of publishing it.

Two kinds of figure, both pinned to the same command:

* Codebase figures (Python lines, modules, tests, strategies) change only when
  code is committed, so they are compared against the working tree right now.
* Evidence figures (predictions journaled, horizons matured, cycles, fills) move
  while the daemon runs, so they are pinned to the generated ledger instead —
  TRACK_RECORD.md, rebuilt by one command. If the ledger is regenerated and the
  deck is not, this fails.

Either way the pack cannot drift silently, which is the point: an outside party
can check every number in it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from kairos.config import ROOT

_TOOLS = str(ROOT / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import kit_stats  # noqa: E402  (the figure table is the thing under test)

PACK_DIR = ROOT / "docs" / "investors"
PACK = [PACK_DIR / name for name in
        ("DECK.md", "ONE_PAGER.md", "WHITEPAPER.md", "landing.html", "README.md",
         "OUTREACH.md")] + [ROOT / "README.md"]

# A figure can sit inside markup, so the scan that looks for one has to cross the
# tags that the layout rules have to spell out.
SEP = r"(?:[\s>.,:;/-]|</?b>|<small>|<em>|</em>)*"


def _text(p: Path) -> str:
    body = p.read_text(encoding="utf-8", errors="replace")
    # Figures get wrapped, and a blockquote continuation marker lands between the
    # number and the word it belongs to. Match on the flattened text so a line
    # break cannot hide a stale figure.
    return re.sub(r"\s+", " ", body.replace("\n>", " "))


def _digits(match: re.Match) -> int:
    """The number a rule matched, dropping whatever the rule re-emits around it."""
    tail = re.search(r"[\d,]+$", match.group(0))
    return int(tail.group().replace(",", ""))


def _rule_spans(body: str) -> list[tuple[int, int, str]]:
    """Where each rule sits in this text, as digit spans plus its figure key."""
    out: list[tuple[int, int, str]] = []
    for pattern, key, _keep in kit_stats.RULES:
        for m in re.finditer(pattern, body):
            tail = re.search(r"[\d,]+$", m.group(0))
            out.append((m.start() + tail.start(), m.start() + tail.end(), key))
    return out


@pytest.fixture(scope="module")
def figures():
    """Every quotable figure, straight from the command that rewrites the pack."""
    f = kit_stats.figures()
    # a collector that quietly returned nothing would turn every comparison
    # below into "the pack is wrong", which is the loudest possible lie
    assert f["lines_comma"] > 5_000 and f["tests"] > 100, f
    assert f["modules"] > 10 and f["strategies"] > 0, f
    return f


@pytest.mark.parametrize("path", PACK, ids=lambda p: p.name)
def test_pack_figures_match_the_repository(path, figures):
    body = _text(path)
    bad = _stale_figures(body, figures)
    assert not bad, f"{path.relative_to(ROOT)} contradicts the repository: {bad}"


def _stale_figures(body: str, figures) -> list[str]:
    """Every place the rules find a figure that the repository no longer agrees with."""
    bad: list[str] = []
    for pattern, key, _keep in kit_stats.RULES:
        want = figures[key]
        for m in re.finditer(pattern, body):
            if _digits(m) != want:
                bad.append(f"{m.group(0)!r} — the repository says {want:,} {key}")
    return bad


def test_the_powerpoint_deck_quotes_the_same_figures(figures):
    """The .pptx is the copy that gets emailed, and `--write` cannot rewrite it.

    A slide puts "121" and "passing tests" in separate text runs, so the pack's
    rewriter has no safe way to touch the binary and this is the only check it
    gets — read-only, and it names the value the slide has to be rebuilt with.
    It found the deck six passes stale: 10,691 lines, 35 modules, 121 tests and a
    v2.5.0 stamp, under a footer claiming the figures were verified that day.
    """
    path = PACK_DIR / "KAIROS_Investor_Deck.pptx"
    if not path.is_file():
        pytest.skip("no pptx in this checkout")
    body = kit_stats.pptx_text(path)
    assert len(body) > 2_000, "the extractor read an empty deck"
    bad = _stale_figures(body, figures)
    want = kit_stats.stamps(kit_stats.ledger())
    for pattern, key, _keep in kit_stats.STAMP_RULES:
        for m in re.finditer(pattern, body):
            tail = re.search(r"(\d{4}-\d{2}-\d{2}|\d+\.\d+\.\d+)$", m.group(0))
            if tail.group() != want[key]:
                bad.append(f"{m.group(0)!r} — {key} is {want[key]}")
    assert not bad, (f"{path.relative_to(ROOT)} contradicts the repository; rebuild "
                     f"it from DECK.md: {bad}")


@pytest.mark.parametrize("path", PACK, ids=lambda p: p.name)
def test_no_figure_phrase_lacks_a_rule(path, figures):
    """A digit next to a figure word must be a figure the rules rewrite.

    `figures` is only here to make the message actionable: the value is the one
    the missing rule would have to write.
    """
    body = _text(path)
    spans = _rule_spans(body)
    bad: list[str] = []
    for key, width, word in kit_stats.FIGURE_WORDS:
        for m in re.finditer(
                rf"(?<!every )(?<![\d,]){width}(?={SEP}(?:{word}))", body):
            if any(a < m.end() and m.start() < b for a, b, _k in spans):
                continue
            bad.append(f"{m.group(0)!r} before a figure word has no rule in "
                       f"tools/kit_stats.RULES (it must read {figures[key]:,})")
    assert not bad, f"{path.relative_to(ROOT)}: {bad}"


@pytest.mark.parametrize("path", PACK, ids=lambda p: p.name)
def test_the_stamped_date_and_version_are_not_typed(path):
    """A date or version in the pack is a figure like any other.

    The v2.6.0 deck printed "Figures dated 2026-10-20" over a ledger rebuilt on
    the 21st, and the whitepaper still described a release that was no longer the
    one the kit shipped with. Neither was a number, so neither was checked — the
    stamp now comes from the ledger's own rebuild date and from `kairos.__version__`,
    which is what makes a stale one a test failure instead of a sentence.
    """
    body = _text(path)
    want = kit_stats.stamps(kit_stats.ledger())
    bad: list[str] = []
    for pattern, key, _keep in kit_stats.STAMP_RULES:
        for m in re.finditer(pattern, body):
            tail = re.search(r"(\d{4}-\d{2}-\d{2}|\d+\.\d+\.\d+)$", m.group(0))
            if tail.group() != want[key]:
                bad.append(f"{m.group(0)!r} — {key} is {want[key]}")
    assert not bad, f"{path.relative_to(ROOT)} carries a stale stamp: {bad}"


def test_the_pack_cites_the_ledger_as_the_source():
    """A figure an investor can verify has to say where it comes from."""
    assert "make_track_record" in _text(PACK_DIR / "README.md")


def test_the_pack_quotes_the_ledger_for_every_instrument_figure():
    """The copied sweep table cannot drift from the generated one.

    `RULES` rewrites a count attached to a figure word, so "259 passing" is
    maintained and "GOLD +28.7%" was not — while that pair sits, hand-copied, in
    the deck, the one-pager, the whitepaper, the landing page and the slides of
    the binary. A fresh sweep run moves every one of them.
    """
    assert kit_stats.stale_sweep_quotes() == []


def test_an_instrument_figure_the_ledger_no_longer_prints_is_a_failure():
    """The check can fire, on the value the ledger actually has."""
    real = kit_stats.sweep()
    sym = sorted(real)[0]
    bad = kit_stats.stale_sweep_quotes({"x.md": f"{sym} went up +99.9% last.\n"})
    assert any(sym in m for m in bad), bad
    ok = kit_stats.stale_sweep_quotes(
        {f"{sym}.md": f"{sym} {real[sym]['return']}, buy & hold "
                      f"{real[sym]['benchmark']}, drawdown {real[sym]['drawdown']}.\n"})
    assert ok == []


def test_a_percentage_after_a_different_instrument_is_not_claimed_by_the_first():
    """Pairing stops at the next name, so a row cannot borrow another's return."""
    real = kit_stats.sweep()
    a, b = sorted(real)[:2]
    straight = {f"{a}{b}.md": f"{a} {real[a]['return']}, {b} {real[b]['return']}.\n"}
    assert kit_stats.stale_sweep_quotes(straight) == []
    swapped = {f"{a}{b}.md": f"{a} {real[b]['return']}, {b} {real[a]['return']}.\n"}
    assert kit_stats.stale_sweep_quotes(swapped), \
        "each instrument's figure was read as the other's, and both passed"


def test_a_venue_mean_is_checked_against_the_book_it_names():
    """A mean is identified by the book beside it, not by the word "mean".

    The pack wrote "+13.6% across the global book" and the guard let it through,
    because it only knew two shapes: a number followed by the word "mean", and the
    ledger's own "mean net return +13.6%". A sentence that names the venue instead
    is still quoting that venue's mean, and it still has to move when the ledger
    does. Binding the number to the book also catches a figure put in the wrong
    sentence, which a set of allowed magnitudes cannot see.
    """
    means = kit_stats._venue_means(kit_stats.LEDGER.read_text(encoding="utf-8"))
    good = {"x.md": f"{means['Global']}% across the global book, and "
                    f"{means['PSX']}% across the equity book.\n"}
    assert kit_stats.stale_sweep_quotes(good) == []
    wrong_book = {"x.md": f"{means['Global']}% across the equity book.\n"}
    assert kit_stats.stale_sweep_quotes(wrong_book), wrong_book
    stale = {"x.md": f"{means['Global']}% mean across the global book, against "
                     f"13.6% across the equity book.\n"}
    assert kit_stats.stale_sweep_quotes(stale), stale


def test_a_stale_sweep_headline_or_loser_count_is_a_failure():
    """The ratio and the loser count are two claims about one table.

    A run that flips a verdict moves both, and a pack that refreshes only the
    flattering one says "9 of 14 profitable" over "the other 6 unprofitable"
    while fourteen rows sit under it. Both are counted off the parsed ledger,
    not read back out of the prose that is being checked.
    """
    wins, total, losers = kit_stats._ledger_headline()
    good = {"x.md": f"{wins} of {total} instruments profitable, and the "
                    f"other {losers} unprofitable ones lost money.\n"}
    assert kit_stats.stale_sweep_quotes(good) == []
    stale_wins = {"x.md": f"{int(wins) - 1} of {total} instruments profitable.\n"}
    assert kit_stats.stale_sweep_quotes(stale_wins), stale_wins
    stale_losers = {"x.md": f"the other {int(losers) + 1} unprofitable ones.\n"}
    assert kit_stats.stale_sweep_quotes(stale_losers), stale_losers


def test_no_denial_in_the_pack_is_contradicted_by_the_pack():
    """A claim that the pack omits something is a figure like any other.

    The published v2.6.1 release note said the kit contained no hit rate and no
    return figure, in the same document that promised to publish a hit rate and
    two sentences above a table whose columns are net return and hit rate. Every
    check in this file asks whether a quoted number is right; none asked whether
    a sentence about what is *absent* is.
    """
    assert kit_stats.denials() == []


def test_a_quantified_denial_of_a_published_figure_is_a_failure():
    """The check can fire, which is the only thing that makes the one above real.

    The false sentence has to be denied over the whole corpus ("anywhere in this
    pack"); "no hit rate exists" is a claim about the market, and no text file can
    refute it. And the narrowing word has to sit next to the noun it narrows: the
    release note's own cause clause, that zero directional calls have matured,
    made the sentence read as scoped to the person who wrote it while it stayed
    a universal claim about the pack.
    """
    figures = ("| instrument | net return | hit rate |\n"
               "| GOLD | +28.7% | 0.594 |\n")
    bad = kit_stats.denials({"a.md": "no hit rate appears anywhere in this pack.\n",
                             "b.md": figures})
    assert any("hit rate" in m for m in bad), bad
    scoped = kit_stats.denials({"a.md": "no forward hit rate appears anywhere"
                                        " in this pack.\n", "b.md": figures})
    assert scoped == []
    unquantified = kit_stats.denials({"a.md": "of every prediction matured, all were"
                                              " non-trades, so no hit rate exists.\n",
                                      "b.md": figures})
    assert unquantified == []


def test_a_denial_split_across_a_wrapped_line_still_reads_as_quantified():
    """Release bodies wrap mid-sentence, and a line-by-line checker loses them.

    The first version of `denials()` bounded sentences on a newline and reported
    the real false claim as consistent, because "…appears" ended the line and
    "anywhere in this pack" began the next one.
    """
    bodies = {"a.md": "so no hit rate, accuracy or return figure appears\n"
                      "anywhere in this pack, and none is implied here.\n",
              "b.md": "| instrument | net return | hit rate |\n"
                      "| GOLD | +28.7% | 0.594 |\n"}
    bad = kit_stats.denials(bodies)
    assert any("hit rate" in m for m in bad), bad


def test_a_release_note_is_frozen_and_not_rewritten():
    """What GitHub published stays as published.

    `--write` must not chase a note that has already been read: its journal
    counts tick every ten minutes, and a corrected note is edited in place and
    marked, not silently updated. So the only rule release notes are held to is
    the one above.
    """
    assert not any(n in kit_stats.PACK_FILES for n in kit_stats.release_notes())


def test_a_figure_command_value_is_reachable_from_the_pack(figures):
    """Every key the rules rewrite must be a figure this test can resolve.

    A typo in a rule's key would otherwise make the rewriter crash on `--write`
    and the checker pass on an empty rule list — the failure mode being a pack
    that silently stops being maintained.
    """
    keys = {key for _p, key, _k in kit_stats.RULES}
    assert keys <= set(figures), keys ^ set(figures)
    words = {key for key, _w, _word in kit_stats.FIGURE_WORDS}
    assert words <= set(figures), words ^ set(figures)
    stamp_keys = {key for _p, key, _k in kit_stats.STAMP_RULES}
    assert stamp_keys <= set(kit_stats.stamps(kit_stats.ledger())), stamp_keys
