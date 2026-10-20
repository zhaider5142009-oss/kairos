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
    bad: list[str] = []
    for pattern, key, _keep in kit_stats.RULES:
        want = figures[key]
        for m in re.finditer(pattern, body):
            if _digits(m) != want:
                bad.append(f"{m.group(0)!r} — the repository says {want:,} {key}")
    assert not bad, f"{path.relative_to(ROOT)} contradicts the repository: {bad}"


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


def test_the_pack_cites_the_ledger_as_the_source():
    """A figure an investor can verify has to say where it comes from."""
    assert "make_track_record" in _text(PACK_DIR / "README.md")


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
