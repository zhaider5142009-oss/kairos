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
instead of publishing it. The binary is scanned by that lint too, not just held to
the figure comparison: see `LINTED` below for what that found.

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
import zipfile
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
DECK_PPTX = PACK_DIR / "KAIROS_Investor_Deck.pptx"
# The binary gets the same vocabulary lint as the text files. It used to get only
# the figure comparison, and that turned out to be the weaker half: slide 5 wrote
# "17,212 lines, 42 modules, 665 tests" and no rule ends in a bare "lines," or a
# bare "tests", so of that triple only the module count was ever compared. The
# other two sat one refresh behind while the same sentence restamped "41" — a
# guard that reads the forms it knows reports an unknown form as absent, and the
# deck is the copy that gets emailed.
LINTED = PACK + [pytest.param(DECK_PPTX, marks=pytest.mark.skipif(
    not DECK_PPTX.is_file(), reason="no pptx in this checkout"))]

# A figure can sit inside markup, so the scan that looks for one has to cross the
# tags that the layout rules have to spell out.
SEP = r"(?:[\s>.,:;/-]|</?b>|<small>|<em>|</em>)*"


def _text(p: Path) -> str:
    if p.suffix == ".pptx":
        # the runs come out of the zip already flattened, so the same rules that
        # read the markdown read the binary without a second vocabulary
        return kit_stats.pptx_text(p)
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


def _stale_stamps(body: str, want: dict) -> list[str]:
    """Every stamped figure written somewhere other than the way the kit says it.

    The whole match is compared, not a digit tail. This check used to pull the
    last date-or-version-shaped run out of the match and call it a day, so a
    stamped value could be wrong in whatever part of itself the tail regex did not
    recognise: a rupee figure with a comma and decimals is nothing
    `(\\d{4}-\\d{2}-\\d{2}|\\d+\\.\\d+\\.\\d+)$` can see, and neither was a stale sign
    in front of it. Comparing `group(0)` against `prefix + value` needs no
    vocabulary of shapes — it asks whether the text is exactly what `--write` would
    have produced, which is the only question worth asking.
    """
    bad: list[str] = []
    for pattern, key, keep in kit_stats.STAMP_RULES:
        for m in re.finditer(pattern, body):
            expected = ("".join(m.group(i) or "" for i in range(1, keep + 1))
                        + want[key])
            if m.group(0) != expected:
                bad.append(f"{m.group(0)!r} — {key} is {want[key]!r}")
    return bad


def test_the_powerpoint_deck_quotes_the_same_figures(figures):
    """The .pptx is the copy that gets emailed, and `--write` cannot rewrite it.

    A slide puts "121" and "passing tests" in separate text runs, so the pack's
    rewriter has no safe way to touch the binary and this is the only check it
    gets — read-only, and it names the value the slide has to be rebuilt with.
    It found the deck six passes stale: 10,691 lines, 35 modules, 121 tests and a
    v2.5.0 stamp, under a footer claiming the figures were verified that day.
    """
    path = DECK_PPTX
    if not path.is_file():
        pytest.skip("no pptx in this checkout")
    body = kit_stats.pptx_text(path)
    assert len(body) > 2_000, "the extractor read an empty deck"
    bad = _stale_figures(body, figures)
    bad += _stale_stamps(body, kit_stats.stamps(kit_stats.ledger()))
    assert not bad, (f"{path.relative_to(ROOT)} contradicts the repository; rebuild "
                     f"it from DECK.md: {bad}")


def test_a_deck_figure_in_a_prose_sentence_is_reachable(figures):
    """A figure the guard cannot read is a figure the deck may print stale forever.

    Slide 5 held the repository triple as prose — "17,212 lines, 42 modules, 665
    tests" — and a rule only owns a digit that ends in the word it knows, so of
    those three claims the module count alone was compared. The deck went on
    shipping a line count and a test total that no pass had ever looked at, in the
    same sentence that got its "41" reported. The wording now ends in "lines of
    Python" and "passing tests", which the rules do end in, and this checks both
    homes for each figure: the stat card and the sentence. Reword either into a
    shape no rule finishes with, or delete one, and this fails.
    """
    if not DECK_PPTX.is_file():
        pytest.skip("no pptx in this checkout")
    body = kit_stats.pptx_text(DECK_PPTX)
    sites = ((f"{figures['lines_comma']:,} lines of Python", "9,999 lines of Python"),
             (f"{figures['modules']} modules", "99 modules"),
             (f"{figures['tests']} passing tests", "999 passing tests"))
    for good, wrong in sites:
        assert body.count(good) >= 2, f"{good!r} is quoted in one place only"
        assert _stale_figures(body.replace(good, wrong, 1), figures), \
            f"corrupting {good!r} left the deck guard green"
    # The money sentence is quoted once, and corrupting each of its three figures
    # in place has to fire the stamp guard — the sentence is reachable, not merely
    # matched by a synthetic one. "The money" is now a statement in three parts,
    # because the trading result alone does not add up to the move in equity.
    want = kit_stats.stamps(kit_stats.ledger())
    money = want["realized_money"]
    assert f"Paper book: {money} trading P&L" in body, "the money sentence moved"
    assert f"FX translation {want['fx_money']}" in body, "the translation left the deck"
    # the deck closes the sentence "…in total", the pack leads with "total …", and
    # both shapes are stamps: the guard owns the money either way.
    assert f"{want['total_money']} in total" in body, "the total left the deck"
    for figure in (money, want["fx_money"], want["total_money"]):
        assert _stale_stamps(body.replace(figure, "−Rs 13.05", 1), want), \
            f"corrupting {figure!r} left the guard green"


def test_the_deck_is_flattened_in_play_order_not_part_order(tmp_path):
    """A moved slide keeps its part name, and the sweep rules read what came before.

    `pptx_text` used to sort `ppt/slides/slideN.xml` by N, which is the order the
    parts were written in, not the order the deck plays them: PowerPoint reuses a
    part name when a slide is dragged elsewhere, so a reordered deck flattens into
    a sequence no audience ever sees. That matters here because a figure is
    attributed to the nearest instrument name *before* it — a rearrangement can put
    "+43.5%" ahead of its own SILVER and leave the return compared by no rule, and
    the guard calls an unattributed figure absent rather than wrong. So the order
    is resolved through presentation.xml, and this is the case where the two orders
    disagree.
    """
    def slide(*runs: str) -> str:
        body = "".join(f"<a:t>{r}</a:t>" for r in runs)
        return f'<?xml version="1.0"?><p:sld><p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>'

    # part 1 is the return and part 2 the instrument, but the deck plays part 2 first
    deck = tmp_path / "reordered.pptx"
    with zipfile.ZipFile(deck, "w") as z:
        z.writestr("ppt/presentation.xml",
                   '<p:presentation><p:sldIdLst>'
                   '<p:sldId id="256" r:id="rId9"/>'
                   '<p:sldId id="257" r:id="rId7"/></p:sldIdLst></p:presentation>')
        z.writestr("ppt/_rels/presentation.xml.rels",
                   '<Relationships>'
                   '<Relationship Id="rId7" Target="slides/slide1.xml"/>'
                   '<Relationship Id="rId9" Target="slides/slide2.xml"/>'
                   '</Relationships>')
        z.writestr("ppt/slides/slide1.xml", slide("+43.5%"))
        z.writestr("ppt/slides/slide2.xml", slide("SILVER"))
    body = kit_stats.pptx_text(deck)
    assert body.index("SILVER") < body.index("+43.5%"), \
        f"the deck was read in part-name order, so its return precedes its instrument: {body!r}"
    sym = kit_stats._attributed(body, sorted(kit_stats.sweep(), key=len, reverse=True),
                                body.index("+43.5%"))
    assert sym == "SILVER", f"the sweep guard reads {body!r} as {sym!r}, not SILVER"


@pytest.mark.parametrize("path", LINTED, ids=lambda p: p.name)
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
def test_the_stamped_date_version_and_rupee_are_not_typed(path):
    """A stamped figure in the pack is a figure like any other.

    The v2.6.0 deck printed "Figures dated 2026-10-20" over a ledger rebuilt on
    the 21st, and the whitepaper still described a release that was no longer the
    one the kit shipped with. Neither was a count, so neither was checked — the
    stamp now comes from the ledger's own rebuild date and from `kairos.__version__`,
    which is what makes a stale one a test failure instead of a sentence.

    The realized rupee joined them for the same reason. The pack described a book
    as "essentially flat (−Rs 13.05 realized on Rs 1,000,000)" in four files while
    that figure was the dollar arithmetic of one leg, and the ledger it sits beside
    said −Rs 3,501.69 after the restatement. It is not a count, so no digit rule
    could hold it, and "realized" is a scope word, so the phrase lint never looked
    in that sentence either — the one shape of number in the pack that no guard had
    ever been pointed at.
    """
    body = _text(path)
    bad = _stale_stamps(body, kit_stats.stamps(kit_stats.ledger()))
    assert not bad, f"{path.relative_to(ROOT)} carries a stale stamp: {bad}"


def test_the_realized_money_is_a_figure_the_pack_can_get_wrong():
    """The money figure, and the parse it comes from, both shown to be reachable.

    Three ways this goes wrong, all of them live before this test existed: the pack
    prints a magnitude the ledger no longer agrees with; the pack prints the right
    digits under the wrong sign; and the command that reads the figure out of the
    ledger stops at the first comma, so a −Rs 3,501.69 book is read as "−3" and
    every comparison below passes on the sign and the leading digit of a number
    three thousand four hundred and ninety-eight rupees too small.
    """
    want = kit_stats.stamps(kit_stats.ledger())
    live = want["realized_money"]
    assert re.fullmatch(r"−?[A-Za-z$€£₹¥]{1,3} [\d,]+\.\d{2}", live), live
    label = live.split(" ")[0].lstrip("−")

    # the figure the pack prints and the figure the ledger holds are the same money
    def _amount(s: str) -> float:
        return float(re.sub(r"[A-Za-z$€£₹¥]{1,3}", "", s)
                     .replace("−", "-").replace(",", "").replace(" ", ""))

    assert _amount(live) == _amount(kit_stats.ledger()["realized_pnl"]), live

    for sentence in (f"Paper book: {live} realized on {label} 1,000,000 across 545 cycles",
                     f"The book is flat (realized P&L {live} on {label} 1,000,000 over 545)"):
        assert _stale_stamps(sentence, want) == [], sentence
        wrong = sentence.replace(live, "−Rs 13.05", 1)
        assert wrong != sentence, "the stale example is already the live figure"
        assert _stale_stamps(wrong, want), \
            f"the pack could print {live!r} as −Rs 13.05 and nothing would fire"
        # right digits, wrong direction
        if live.startswith("−"):
            unsigned = sentence.replace(live, live[1:], 1)
            assert _stale_stamps(unsigned, want), \
                "a minus in front of a gain reads as a figure, not as a claim"


def test_the_translation_and_the_total_are_figures_the_pack_can_get_wrong():
    """The pack's paper-book sentence is three figures, and all three are checked.

    `realized P&L` alone cannot be reconciled with the book's cash: a foreign leg
    held across a cross change also revalues its own notional, and on the
    published round trip that was −Rs 2,760.38 beside a −Rs 3,501.69 trading
    result. Three files described the book with the trading figure alone, which
    made a 0.63% loss read as 0.35%, and no rule could have caught it: the
    translation had not been a figure in the ledger long enough to have a shape.
    All three are stamps now, and the parity is the check — a pack that quotes a
    translation and a total that do not add up to the trading result is quotable
    arithmetic that no reader can verify.
    """
    want = kit_stats.stamps(kit_stats.ledger())
    assert want["fx_money"] and want["total_money"], want
    sentence = (f"Paper book: {want['realized_money']} trading P&L and FX "
                f"translation {want['fx_money']} on Rs 1,000,000, "
                f"{want['total_money']} in total, 2 fills.")
    assert _stale_stamps(sentence, want) == [], sentence

    def _amount(s: str) -> float:
        return float(re.sub(r"[A-Za-z$€£₹¥]{1,3}", "", s)
                     .replace("−", "-").replace(",", "").replace(" ", ""))

    assert abs(_amount(want["realized_money"]) + _amount(want["fx_money"])
               - _amount(want["total_money"])) < 0.01, \
        "the ledger's own three figures do not add up"

    for key in ("fx_money", "total_money"):
        stale = sentence.replace(want[key], "−Rs 13.05", 1)
        assert _stale_stamps(stale, want), f"{key} could be printed as −Rs 13.05"
        unsigned = sentence.replace(want[key], want[key].lstrip("−"), 1)
        if want[key].startswith("−"):
            assert _stale_stamps(unsigned, want), \
                f"a sign-flipped {key} reads as a figure, not as a claim"

    # A rupee-only book owes no translation, and the stamp is empty rather than
    # zero: the pack may not quote one the ledger does not print.
    plain = {"realized_pnl": "-3501.69", "ledger_generated": "2026-10-25"}
    bare = kit_stats.stamps(plain)
    assert bare["fx_money"] == "" and bare["total_money"] == ""
    assert _stale_stamps(sentence, bare), \
        "a translation was quotable over a ledger that reports none"


def test_a_book_in_dollars_is_not_published_in_rupees(monkeypatch, tmp_path):
    """The label is the book's, and the guard reads it from the book.

    `make_track_record` printed "Rs" as a literal over whatever the paper account
    held, and this guard had "Rs" baked into both its pattern and its value — so
    a book funded in dollars (`paper.currency: USD`) would have gone out with a
    rupee label that the one checker pointed at the money figure agreed with. The
    currency comes from the account state, and it travels inside the stamp.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "paper_account.json").write_text(
        '{"currency": "USD", "capital": 100000.0, "cash": 99349.4, '
        '"realized_pnl": -650.25}', encoding="utf-8")
    monkeypatch.setattr(kit_stats, "ROOT", tmp_path)
    want = kit_stats.stamps({"realized_pnl": "-650.25",
                             "ledger_generated": "2026-10-25"})
    assert want["realized_money"] == "−$ 650.25", want
    good = "Paper book: −$ 650.25 realized on $ 100,000 across 12 cycles"
    assert _stale_stamps(good, want) == [], good
    assert _stale_stamps(good.replace("$", "Rs"), want), \
        "a rupee label over a dollar book passed the money guard"


def test_the_ledger_prints_the_currency_the_book_holds(monkeypatch, tmp_path):
    """A label the generator got wrong is a defect, not the standard.

    The guard's expected label comes from `data/paper_account.json`, the file the
    ledger is built from, so a ledger restamped in another currency is reported
    instead of being adopted as the truth.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "paper_account.json").write_text(
        '{"currency": "PKR"}', encoding="utf-8")
    monkeypatch.setattr(kit_stats, "ROOT", tmp_path)
    monkeypatch.setattr(kit_stats, "LEDGER", tmp_path / "TRACK_RECORD.md")
    (tmp_path / "TRACK_RECORD.md").write_text(
        "- Capital $ 100,000 · cash $ 99,349 · realized P&L −$ 650.25 since now\n",
        encoding="utf-8")
    assert kit_stats.ledger()["currency_label"] == "$", "the label was not read"
    assert kit_stats._book_currency() == "PKR"
    assert kit_stats.stamps()["realized_money"] == "−Rs 650.25"


def test_a_non_trade_count_is_a_figure_that_can_go_stale(figures):
    """The pack's other ledger count: how much of the matured record was a non-call.

    It was not a figure at all until a published copy of it had been stale in six
    files for four refresh passes. The guard rewrites a digit beside a word it
    knows, and "non-trades" was not in the vocabulary, so the number named a real
    ledger column while no rule owned it — and every other figure in the same
    sentence was restamped correctly on the pass that walked over it.
    """
    n = figures["non_trades_comma"]
    m = figures["matured_comma"]
    current = _stale_figures(f"{m:,} horizons matured, all {n:,} were non-trades, "
                             f"and every one of the {n:,} was a non-trade.", figures)
    assert current == []
    # The wrong numbers are built from the live ones rather than copied out of an
    # old ledger. A hand-typed historical value is a figure like any other: 263 sat
    # in this sentence as "the stale example" for three refreshes and then became
    # the live count, at which point the sentence had three stale digits in it and
    # the test that claimed to prove two started failing for a reason nobody could
    # read off the assertion.
    wrong_m, wrong_n = m - 1, n - 2
    stale = _stale_figures(f"{wrong_m:,} horizons matured, all {wrong_n:,} were non-trades, "
                           f"and every one of the {wrong_n:,} was a non-trade.", figures)
    assert len(stale) == 3, stale
    reported = " ".join(stale)
    assert f"{wrong_m:,}" in reported and f"{wrong_n:,}" in reported, stale


_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
DEFECT_TOTAL = re.compile(r"\b([A-Za-z]+(?:-[A-Za-z]+)?) defects\b")


def _spelled(n: int) -> str:
    if n in _ONES:
        return next(k for k, v in _ONES.items() if v == n)
    tens, units = divmod(n, 10)
    word = next(k for k, v in _TENS.items() if v == tens * 10)
    return word if not units else f"{word}-{next(k for k, v in _ONES.items() if v == units)}"


def _defect_tally(body: str) -> tuple[str, int, int]:
    """The total section 8 states in words, and the rows it actually tables."""
    sec = body.split("## 8. ", 1)[1]
    lines = sec.splitlines()
    head = next(i for i, row in enumerate(lines) if row.startswith("| defect "))
    rows = 0
    for line in lines[head + 2:]:
        if not line.strip().startswith("|"):
            break
        rows += 1
    m = DEFECT_TOTAL.search(sec)
    assert m, "section 8 no longer states a defect total"
    return m.group(1), _spelled_number(m.group(1)), rows


def _spelled_number(word: str) -> int:
    total = 0
    for part in word.lower().split("-"):
        value = _TENS.get(part, _ONES.get(part))
        assert value is not None, f"not a number word: {part!r}"
        total += value
    return total


def test_the_stated_defect_total_is_the_number_of_defect_rows():
    """How many bugs the kit admits to is a figure, and it was typed.

    The table grew by a row and the sentence at its head did not, which is the
    same failure the rest of this file exists to stop — except spelled out in
    words, so no digit rule could see it. An investor who counts the rows finds
    the difference, and the count is the credibility claim.
    """
    body = (PACK_DIR / "WHITEPAPER.md").read_text(encoding="utf-8")
    word, claimed, rows = _defect_tally(body)
    assert claimed == rows, (f"section 8 says {word!r} defects and tables {rows} "
                             f"rows ({claimed} == {rows} required)")


def test_the_defect_total_check_can_fire():
    """A total read off a sentence has to disagree when the sentence is wrong."""
    body = (PACK_DIR / "WHITEPAPER.md").read_text(encoding="utf-8")
    word, claimed, rows = _defect_tally(body)
    off = body.replace(f"{word} defects", f"{_spelled(claimed + 1)} defects", 1)
    assert _defect_tally(off)[1] == claimed + 1
    assert _defect_tally(off)[1] != _defect_tally(off)[2]


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


def test_every_sweep_figure_the_deck_prints_can_be_named():
    """A deck percentage the sweep guard cannot attribute is one that goes stale green.

    The instrument returns and the two book means are hand-copied into the slides,
    and the guard finds their owner either from the nearest instrument name in a
    window or from the book word beside the number. Both are properties of the
    layout, not of the file, and the sweep test above only asks the guard to find
    nothing — so a slide that printed "+43.5%" in one text box and SILVER in
    another, in the other order or past the window, would leave that return
    compared by no rule while the run stayed green. That is the same failure slide
    5's repository triple had, and here nothing but the layout keeps it from
    happening. So corrupt each signed percentage where it sits and require the
    guard to name it.
    """
    if not DECK_PPTX.is_file():
        pytest.skip("no pptx in this checkout")
    # normalized the way `stale_sweep_quotes` normalizes, so these offsets are the
    # text the guard reads and a corrupted copy needs no second treatment
    body = re.sub(r"\s+", " ",
                  kit_stats.pptx_text(DECK_PPTX).replace("\n>", " ").replace("**", " "))
    name = DECK_PPTX.relative_to(ROOT).as_posix()
    sites = [(kit_stats._attributed(body, sorted(kit_stats.sweep(), key=len, reverse=True),
                                    m.start()), m) for m in kit_stats.PCT.finditer(body)]
    assert sites, "the deck prints no signed percentages, so this checks nothing"
    for sym, m in sites:
        bad = body[:m.start()] + "+77.7%" + body[m.end():]
        hits = kit_stats.stale_sweep_quotes({name: bad})
        assert any("+77.7%" in h for h in hits), \
            (f"{sym or 'unattributed'} figure {m.group(0)!r} corrupted at {m.start()} "
             f"and the guard reported nothing: the deck prints a sweep figure in a "
             f"shape no rule can reach")
    assert kit_stats.stale_sweep_quotes({name: body}) == []


def test_a_sweep_ratio_is_checked_wherever_its_word_sits():
    """A headline count is the same claim whether "profitable" precedes or follows.

    The rule that checked it needed the descriptor glued to the numbers, so the
    pack's "profitable on 8 of 14 instruments" and "6 of 14 instruments in the
    walk-forward sweep lost money" were invisible to every pass. The number that
    counts the losers has to be read as the loser count, and a winner count
    written where the sentence means losses has to fail the build.
    """
    wins, total, losers = kit_stats._ledger_headline()
    good = {"x.md": f"profitable on {wins} of {total} instruments, and "
                    f"{losers} of {total} in the sweep lost money.\n"}
    assert kit_stats.stale_sweep_quotes(good) == []
    wrong_side = {"x.md": f"{wins} of {total} instruments lost money.\n"}
    assert kit_stats.stale_sweep_quotes(wrong_side), wrong_side
    stale = {"x.md": f"profitable on {losers} of {total} instruments.\n"}
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


def test_a_sweep_sentence_may_not_credit_the_strategy_library():
    """How the evidence was produced is a claim, and the guard only read digits.

    The deck's own cover held "Walk-forward sweep, 100 strategies per instrument"
    through a full refresh cycle. The sweep fits one blended signal, not the
    library — but every rule that reads the pack compares a number, so the false
    noun passed while the 100 beside it was verified as the strategy count.
    """
    false = {"x.md": "Walk-forward sweep, 100 strategies per instrument, "
                     "costs modelled.\n"}
    assert kit_stats.method_claims(false), false
    disclosed = {"x.md": "One composite signal per instrument. It is not the "
                         "100-strategy library, and none of these returns come "
                         "from it.\n"}
    assert kit_stats.method_claims(disclosed) == []
    card = {"x.md": "100 strategies voting into the live consensus; the sweep "
                    "fits one signal.\n"}
    assert kit_stats.method_claims(card) == []
    assert kit_stats.method_claims() == []


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
