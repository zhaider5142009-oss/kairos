"""Print the figures the investor pack is allowed to quote.

Every number in docs/investors/ is either derived from the repository or read out
of the generated ledger, and so is the date stamped beside them and the version
they describe. None of them may be typed from memory — the published
v2.6.0 deck carried a line count and a test total three passes old under a footer
that said the figures were verified that day, and nothing caught it because the
numbers were prose. tests/test_kit_figures.py now fails if the pack and these
values disagree, so the answer to "what should the deck say" is this command.

    python tools/kit_stats.py
    python tools/kit_stats.py --markdown     # same table, paste-ready

Evidence figures come from docs/investors/TRACK_RECORD.md. Refresh that first,
from the live journals, with:

    python tools/make_track_record.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "docs" / "investors" / "TRACK_RECORD.md"


def codebase() -> dict[str, int]:
    r = subprocess.run(["git", "ls-files", "kairos/*.py"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    files = [f for f in r.stdout.split() if f.endswith(".py")]
    lines = sum(len((ROOT / f).read_text(encoding="utf-8",
                                         errors="replace").splitlines())
                for f in files)
    return {"lines": lines, "modules": len(files),
            "tests": _collected_tests(), "strategies": _strategy_count(),
            "grid": _grid_points()}


def _grid_points() -> int:
    """How many candidates one calibration window tries.

    This is the sweep's actual method, and the pack used to describe it as
    "100 candidate strategies per instrument" — the wrong count and the wrong
    object, since `walk_forward()` blends four components into one signal and
    never imports the strategy library. The number comes from `_grid()` so that
    widening the search changes the published sentence by itself.
    """
    from kairos.backtest import SMOOTH_SPANS, _grid
    return len(_grid()) * len(SMOOTH_SPANS)


def _strategy_count() -> int:
    from kairos.strategies import STRATEGIES
    return len(STRATEGIES)


def _collected_tests() -> int:
    """Ask pytest how many cases it would run, parametrization included."""
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    total = 0
    for line in r.stdout.splitlines():
        m = re.match(r"^\S+\.py: (\d+)$", line.strip())
        if m:
            total += int(m.group(1))
    return total


def ledger() -> dict[str, str]:
    body = LEDGER.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for label, key in (("predictions journaled", "journaled"),
                       ("horizons matured", "matured"),
                       ("of which directional calls", "directional"),
                       ("of which non-trades (HOLD)", "non_trades")):
        m = re.search(rf"\| {re.escape(label)} \| ([\d,]+) \|", body)
        out[key] = m.group(1) if m else "MISSING"
    for pattern, key in ((r"cycles run: \*\*([\d,]+)\*\*", "cycles"),
                         (r"fills: (\d+)", "fills"),
                         (r"realized P&L Rs (-?[\d.]+)", "realized_pnl")):
        m = re.search(pattern, body)
        out[key] = m.group(1) if m else "MISSING"
    m = re.search(r"^\| rebuilt from the journals on \| (.+?) \|", body, re.M)
    out["ledger_generated"] = m.group(1) if m else "not stamped"
    return out


# The pack quotes the same facts in four different layouts. Each rule replaces
# only the digits immediately before a phrase that can be nothing else, so prose
# like "every 12 cycles" or "8 new regression tests" is not a target. `keep` is
# the number of leading groups to carry over — the shell-comment rule has to
# re-emit its "# " or the code block loses its comment marker.
#
# The layouts are enumerated because a rule this loose could rewrite the wrong
# digits, and rewriting is the dangerous half. That looseness is also how three
# stale test counts survived the v2.6.0 pack: the deck and the one-pager wrote
# "121 passing" and the landing page "<b>87</b><small>passing tests", and the
# only phrasings covered were "N regression tests" and "# N tests". Adding a
# layout here is not enough by itself — tests/test_kit_figures.py reads this
# same table, so the checker and the rewriter cannot drift into two different
# vocabularies, and it fails if a figure word in the pack has no rule behind it.
PACK_DIR = "docs/investors"
PACK_FILES = [f"{PACK_DIR}/{n}" for n in
              ("DECK.md", "ONE_PAGER.md", "WHITEPAPER.md", "landing.html",
               "README.md", "OUTREACH.md", "TARGETS.md")] + ["README.md"]
RULES = [
    (r"[\d,]{4,9}(?=</b><small>lines of Python)", "lines_comma", 0),
    (r"[\d,]{4,9}(?=[\s>]*lines[\s>]*(?:of[\s>]*)?Python)", "lines_comma", 0),
    (r"\b\d{2,3}(?=[\s>]*modules)", "modules", 0),
    (r"\b\d{2,4}(?=[\s>]*regression[\s>]*tests)", "tests", 0),
    (r"(#\s*)\d{2,4}(?=[\s>]*tests)", "tests", 1),
    (r"(?<=<b>)\d{2,4}(?=</b><small>passing)", "tests", 0),
    (r"\b\d{2,4}(?=[\s>]*passing[\s>]*tests)", "tests", 0),
    (r"\b\d{2,4}(?=[\s>]*passing,)", "tests", 0),
    (r"(?<=<b>)\d{1,4}(?=</b><small>strateg)", "strategies", 0),
    (r"(?<=\| Strategies \| )\d{1,4}(?=,[\s>]*selected)", "strategies", 0),
    (r"(?<![\d,])\d{1,4}(?=[\s>-]*strateg)", "strategies", 0),
    (r"(?<![\d,])[\d,]{3,7}(?=[\s>]*predictions[\s>]*journaled)", "journaled_comma", 0),
    (r"(?<![\d,])[\d,]{2,7}(?=[\s>]*horizons[\s>]*matured)", "matured_comma", 0),
    (r"(?<!every )\b\d{2,6}(?=[\s>]*autopilot[\s>]*cycles)", "cycles", 0),
    (r"(?<!every )\b\d{2,6}(?=[\s>]*cycles)", "cycles", 0),
    (r"(?<![\d,])\d{1,7}(?=[\s>]*fills)", "fills", 0),
    (r"(?<![\d,])[\d,]{3,5}(?=[\s>]*-?[\s>]*point[\s>]*grid)", "grid", 0),
    # The non-trade count is a ledger figure like any other, and the pack quoted it
    # in two shapes — "all 140 were non-trades" and "every one of the 140 was a
    # non-trade". Neither had a rule, so the number sat behind the words that name
    # it and no pass moved it: the same sentence shipped four refreshes after the
    # ledger's matured count had gone from 140 to 263. The number has to be glued to
    # its word for a rule to own it, which is what these two shapes require.
    (r"(?<![\d,])[\d,]{2,7}(?=[\s>]*(?:were[\s>]*)?non-trades)", "non_trades_comma", 0),
    (r"(?<![\d,])[\d,]{2,7}(?=[\s>]*was a non-trade)", "non_trades_comma", 0),
]

# Words that mean exactly one figure wherever the pack pairs them with a number.
# This is the lint's vocabulary, and it is deliberately wider than RULES: a new
# phrasing must fail the suite until a rule exists for it, rather than print a
# number nothing checks. `width` mirrors the digit width of the rules that write
# that figure — a one-digit number in front of "regression tests" is a defect
# row's own count ("wrote 8 regression tests"), not the suite size, and must not
# be pulled into either the rewrite or the lint.
FIGURE_WORDS = [
    ("lines_comma", r"[\d,]{4,9}", r"lines(?:[\s>]*(?:of[\s>]*)?)?Python"),
    ("modules", r"\d{2,3}", r"modules"),
    ("tests", r"\d{2,4}", r"regression[\s>]*tests|tests[\s>]*passing"
                       r"|passing[\s>]*tests|passing"),
    ("strategies", r"\d{1,4}", r"strateg(?:y|ies)"),
    ("journaled_comma", r"[\d,]{3,7}", r"predictions[\s>]*journaled"),
    ("matured_comma", r"[\d,]{2,7}", r"horizons[\s>]*matured"),
    ("cycles", r"\d{2,6}", r"(?:autopilot[\s>]*)?cycles"),
    ("fills", r"\d{1,7}", r"fills"),
    ("grid", r"[\d,]{3,5}", r"-?[\s>]*point[\s>]*grid"),
    ("non_trades_comma", r"[\d,]{2,7}", r"(?:were )?non-trades|was a non-trade"),
]


# Two figures in the pack are strings, not counts: the day the evidence was
# rebuilt, and the version the kit describes. Both were typed. The v2.6.0 deck
# published "Figures dated 2026-10-20" beside a ledger that said 2026-10-21, and
# the whitepaper still described a release two tags old — a date nobody regenerates
# drifts in silence, which is the same failure the numeric rules above exist to
# stop. The date is taken from the ledger's own rebuild stamp rather than from the
# clock, so the pack can never claim a day the evidence was not rebuilt.
STAMP_RULES = [
    (r"(\bdated )\d{4}-\d{2}-\d{2}", "pack_date", 1),
    (r"(Version )\d+\.\d+\.\d+", "version", 1),
    (r"(Version \d+\.\d+\.\d+ · )\d{4}-\d{2}-\d{2}", "pack_date", 1),
    # The PowerPoint stamps its version as "KAIROS v2.5.0", not "Version 2.5.0",
    # and its footer says "verified against the repository on <date>" rather than
    # "dated <date>". Both phrasings sat inside the guardrail's own vocabulary
    # blind spot: a checker that lists the forms it knows reports a stale version
    # as absent, not as wrong.
    (r"(KAIROS v)\d+\.\d+\.\d+", "version", 1),
    (r"(KAIROS v\d+\.\d+\.\d+ · )\d{4}-\d{2}-\d{2}", "pack_date", 1),
    (r"(repository on )\d{4}-\d{2}-\d{2}", "pack_date", 1),
]


# The pack's twelfth artifact is a PowerPoint, and it is the copy that gets
# attached to an email, so it is the one that actually reaches a reader. `--write`
# cannot reach it back: a slide splits "121" and "passing tests" into separate
# text runs, and rewriting run boundaries inside a zip is how a deck gets
# corrupted. So this is a read-only check, and it earned its place — when the
# markdown was found to be four passes stale, the .pptx turned out to be six,
# quoting 10,691 lines, 35 modules, 121 tests and a v2.5.0 stamp under a footer
# that said the figures were verified the day the deck was built.
#
# The consequence for the deck's own text: a binary cannot chase a number that
# moves every ten minutes, so it quotes the figures that are stable inside a
# release (lines, modules, tests, strategies, matured horizons, fills) and points
# at the dated ledger for the ones that tick (predictions journaled, autopilot
# cycles). That is why `journaled` and `cycles` never appear in a slide.
DECK_PPTX = f"{PACK_DIR}/KAIROS_Investor_Deck.pptx"


def pptx_text(path: Path) -> str:
    """Every text run in a .pptx, in slide order, flattened so the rules match.

    Stdlib only: the figures live in `<a:t>` elements and a zip is a zip.
    """
    import zipfile
    from html import unescape
    with zipfile.ZipFile(path) as z:
        slides = [n for n in z.namelist()
                  if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        slides.sort(key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[-1]).group()))
        runs = []
        for name in slides:
            xml = z.read(name).decode("utf-8", errors="replace")
            runs.append(" ".join(re.findall(r"<a:t>(.*?)</a:t>", xml)))
    return re.sub(r"\s+", " ", unescape(" ".join(runs)))


def stamps(facts: dict | None = None) -> dict[str, str]:
    """The string figures, resolved from the ledger and the package."""
    facts = ledger() if facts is None else facts
    return {"pack_date": str(facts["ledger_generated"]).strip(),
            "version": _package_version()}


# A denial is a figure. "No hit rate appears anywhere in this pack" describes the
# pack, so the pack can check it — and the published v2.6.1 release note failed
# that check. It denied that any hit rate or return figure appeared anywhere in
# the kit, in the same paragraph it used to describe a sweep table whose columns
# are net return and hit rate. The sentence came out of a real rule (no *forward*
# hit rate until ten directional calls have matured) and dropped the word that
# made it true. Nothing caught it because it was not a number: every rule above
# asks whether a quoted value is right, and none asks whether a claim of absence
# is.
#
# Only a denial with a corpus-wide quantifier is checked. "No forward hit rate
# exists" is a claim about the market, which no text file can refute; "no hit rate
# appears anywhere in this pack" is a claim about these files, so these files
# decide it. The narrowing word has to sit between "no" and the noun it narrows,
# because if a scope word anywhere in the sentence were enough, the clause that
# *caused* the denial ("zero directional calls have matured") would excuse the
# denial itself — which is exactly what happened.
#
# Release notes are frozen. `--write` never touches them, so a note keeps the
# figures it was published with rather than chasing a journal that ticks every
# ten minutes. A denial is the one thing inside a note that does not go stale: it
# is as false the week after publication as the day of it, so it is the one thing
# worth checking on a document nobody edits twice.
RELEASE_DIR = f"{PACK_DIR}/releases"

DENIAL_NOUNS = [
    ("hit rate", r"\|[^\n]{0,80}hit[\s>]*rate[^\n]{0,80}\|"
                 r"|hit[\s>]*rate[^\S\n]{0,12}[\d.]+"),
    ("accuracy", r"\|[^\n]{0,80}accura[^\n]{0,80}\|"
                 r"|accura[^\S\n]{0,20}[+-]?\d"),
    ("return figure", r"\|[^\n]{0,80}net[\s>]*return[^\n]{0,80}\|"
                      r"|net[\s>]*return[^\S\n]{0,12}[+-]?\d"),
]
DENIAL = re.compile(r"\bno (?P<qual>[\w*`-]{0,30}?[ ])?(?P<noun>hit rate|"
                    r"accuracy|return figures?)\b", re.I)
QUANTIFIER = re.compile(r"anywhere in (?:this|the) \w+|\bin (?:this|the) "
                        r"(?:pack|kit|deck|release)\b", re.I)
SCOPE_WORDS = frozenset({"forward", "paper", "live", "realized", "matured",
                         "out-of-sample", "backtest", "backtests", "directional",
                         "trade", "trading", "yet", "so far", "currently"})


def release_notes() -> list[str]:
    """Every GitHub release body kept in the repository, newest tag first."""
    d = ROOT / RELEASE_DIR
    return sorted((f"{RELEASE_DIR}/{p.name}" for p in d.glob("v*.md")),
                  key=lambda n: n.rsplit("/", 1)[-1], reverse=True)


def corpus() -> dict[str, str]:
    """Everything the pack publishes, keyed by the file it came from.

    The generated ledger is in it because a claim about what the pack contains is
    a claim about the ledger, and the deck is in it because the deck is the copy
    that gets attached to an email.
    """
    names = (*PACK_FILES, LEDGER.relative_to(ROOT).as_posix(), DECK_PPTX,
             *release_notes())
    out: dict[str, str] = {}
    for name in dict.fromkeys(names):
        path = ROOT / name
        if not path.is_file():
            continue
        out[name] = (pptx_text(path) if path.suffix == ".pptx" else
                     path.read_text(encoding="utf-8", errors="replace"))
    return out


def _sentence(text: str, start: int, end: int) -> str:
    left = max((text.rfind(c, 0, start) for c in ".!?"), default=-1)
    rights = [i for i in (text.find(c, end) for c in ".!?") if i != -1]
    return text[left + 1:min(rights, default=len(text))]


def denials(body_by_file: dict[str, str] | None = None) -> list[str]:
    """Claims that the pack omits a figure, contradicted by the pack itself."""
    bodies = corpus() if body_by_file is None else body_by_file
    everywhere = "\n".join(bodies.values())
    printed = {noun: re.search(figure, everywhere)
               for noun, figure in DENIAL_NOUNS}
    out: list[str] = []
    for name, body in bodies.items():
        # Flattened, not merely space-collapsed: the release note wrapped the
        # sentence between "appears" and "anywhere in this pack", and a checker
        # that reads line by line reads a denial with no quantifier in it.
        flat = re.sub(r"\s+", " ", body.replace("\n>", " "))
        for m in DENIAL.finditer(flat):
            sentence = _sentence(flat, m.start(), m.end())
            if not QUANTIFIER.search(sentence):
                continue
            words = [w.strip("*`.,").lower() for w in (m["qual"] or "").split()]
            if words and all(w in SCOPE_WORDS for w in words):
                continue
            for noun, _figure in DENIAL_NOUNS:
                if not printed[noun] or noun.split()[0] not in sentence.lower():
                    continue
                out.append(f'{name}: "…{m.group(0)}…" denies any {noun}, but '
                           f'{printed[noun].group(0).strip()!r} publishes one. '
                           f'Scope the claim ("no {words[0] if words else "forward"} '
                           f'{noun}") or drop it.')
    return out


# The sweep fits one blended signal per instrument. It does not fit the
# 100-strategy library, and that is the most load-bearing sentence in the pack's
# evidence section — the deck's own cover printed "Walk-forward sweep, 100
# strategies per instrument" through a full refresh cycle, because every rule in
# this file checks a *number* and that defect was a noun. How the evidence was
# produced is a claim, so it is checked as one. Two shapes carry it: a per-strategy
# count attached to a single instrument, and a sweep verb attached to the library;
# a sentence that says "not" is the pack's own disclosure, not a contradiction.
LIBRARY_CREDIT = re.compile(
    r"\b\d{1,3}[\s\-]?strateg\w*\s+per\s+instrument\b"
    r"|\bsweep\b[^.!?]{0,40}\bfit(?:s|ting)?\b[^.!?]{0,30}strateg"
    r"|\bsweep\b[^.!?]{0,40}\b(?:uses|used|ran|trades|traded)\b[^.!?]{0,30}strateg",
    re.I)
DENIED = re.compile(r"\b(?:not|never|nor|neither)\b", re.I)


def method_claims(body_by_file: dict[str, str] | None = None) -> list[str]:
    """Pack sentences that credit the strategy library with the sweep's evidence."""
    bodies = corpus() if body_by_file is None else body_by_file
    out: list[str] = []
    for name, raw in bodies.items():
        if name.endswith(".md") and "/releases/" in name:
            continue
        flat = re.sub(r"\s+", " ", raw.replace("\n>", " ").replace("**", " "))
        for m in LIBRARY_CREDIT.finditer(flat):
            sentence = _sentence(flat, m.start(), m.end())
            if DENIED.search(sentence):
                continue
            out.append(f'{name}: "{sentence.strip()}" credits the strategy library '
                       f'with the sweep. The sweep fits one blended signal per '
                       f'instrument, not the library.')
    return out


# The sweep table under the headline is the part an investor actually reads, and
# it was copied by hand into five layouts — deck prose, a one-pager table, a
# whitepaper sentence, a landing-page table and the slides. Nothing checked those
# copies: the rules above rewrite a count attached to a figure *word* ("259
# passing"), and a return attached to an instrument name is not that. So the
# sweep is parsed back out of the generated ledger and every signed percentage
# and Sharpe is bound to the nearest instrument name before it, which has to be
# one of the figures the ledger prints for that instrument.
#
# Pairing on "nearest preceding name, no other name in between" is what makes it
# survive all five layouts without a layout rule each, and it is why a percentage
# after the last instrument of a row cannot be claimed by the next row.
SWEEP_ROW = re.compile(r"^\|\s*([A-Z][\w\d]*)\s*\|\s*(\w+)\s*\|\s*([+-][\d.]+%)\s*\|"
                       r"\s*([+-][\d.]+%)\s*\|\s*([+-][\d.]+)\s*\|\s*([\d.]+)\s*\|"
                       r"\s*(\d+)\s*\|\s*([+-][\d.]+%)\s*\|$")
# The sweep ratio, in every shape the pack writes it: "5 / 14 instruments
# profitable", "profitable on 8 of 14 instruments", "6 of 14 instruments in the
# walk-forward sweep lost money", "8 of 14 instruments came out net-positive".
# The first rule needed the word "profitable" glued to the numbers, so three of
# those four shapes were rewritten by no pass and checked by none — and a count
# the guard cannot see is a count that ships stale. The denominator is what makes
# a pair of numbers a sweep ratio; the word nearest to it says which side it counts.
RATIO = re.compile(r"\b(?P<count>\d{1,2})\s*(?:/|of)\s*(?P<total>\d{1,2})\b")
WIN_WORDS = (r"(?<!un)profitable", r"net[\s\-]?positive", r"came out positive",
             r"finished positive", r"made money")
LOSS_WORDS = (r"lost money", r"loses money", r"losses", r"\blost\b", r"unprofitable",
              r"not profitable", r"came out negative", r"net[\s\-]?negative")
RATIO_WINDOW = 60
# The pack also counts the losers on its own, in four phrasings ("the 6
# unprofitable ones", "the other 6 are listed", "the 6 instruments where the
# strategy sweep lost money"). A sweep run moves that number along with the
# ratio, and a stale loser count flatters the table in exactly the direction the
# disclosure exists to correct.
LOSERS = re.compile(r"\b(?:the\s+)?(?:other\s+)?(\d{1,2})\s+(?:(?:instruments\s+)?"
                    r"(?:that\s+lost|where\s+the\s+(?:strategy\s+)?sweep\s+lost)"
                    r"|(?:that\s+)?are\s+listed|reported\s+as\s+losses"
                    r"|unprofitable|not\s+profitable|losers)",
                    re.I)
MEAN = re.compile(r"(?P<pre>[+\u2212-]?\d{1,3}(?:\.\d+)?)%\s*(?:mean|average)"
                  r"|(?:mean|average)(?:\s+net\s+return)?\s+"
                  r"(?P<post>[+\u2212-]?\d{1,3}(?:\.\d+)?)%"
                  # and the form that started it: "+13.6% across the global book",
                  # where the venue, not the word "mean", says which figure this is.
                  # A deck can also print the number and the book on two lines.
                  r"|(?P<near>[+\u2212-]?\d{1,3}(?:\.\d+)?)%\s+(?:mean\s+|average\s+)?"
                  r"(?:across\s+the\s+)?(?P<venue>psx\s+equity|global|equity|psx)\s+book",
                  re.I)
# The venue phrase is the only thing that tells a per-venue mean from any other
# percentage, so it is read as a name and looked up, not pattern-matched away.
VENUE_WORDS = {"global": "Global", "equity": "PSX", "psx": "PSX",
               "psx equity": "PSX"}
PCT = re.compile(r"([+\u2212-])(\d{1,3}(?:\.\d+)?)%")
SHARPE = re.compile(r"Sharpe[^\d\n]{0,8}(\d\.\d{1,2})")
# A Sharpe in a table column has no word to lean on, and the pack has six of them
# per layout. The two-decimal shape is what makes it safe: the ledger's own row
# prints hit rate with three decimals and every return with a %, so nothing else
# in these tables is a bare d.dd.
SHARPE_CELL = re.compile(r"(?<=<td>)([+\u2212-]?\d\.\d{2})(?=</td>)"
                         r"|(?<=\| )\*{0,2}([+\u2212-]?\d\.\d{2})\*{0,2}(?= \|)")
# A verdict word is a figure only inside a table cell, where it names a row; in
# prose it is a sentence, and prose is what an investor argues with.
VERDICT = re.compile(r"(?:(?<=<td>)|(?<=\| ))(flat|profitable|unprofitable)(?=</td>|\s*\|)")
TDCLASS = re.compile(r'class="(up|dn)"[^>]*>\s*([+\u2212-])')
# The landing page colours a return green or red with a class, and a class that
# survives a sign flip is a loss printed in the winner's colour.
VENUE_MEAN = re.compile(r"\*\*(\w+)\*\*: mean net return ([+\-]?[\d.]+)%")
LOSE_LIST = re.compile(r"\b([A-Z][\w\d]+(?:(?:, |\s+and\s+|\s+or\s+)[A-Z][\w\d]+)+)\s+"
                       r"(?:were|are)\s+not\s+profitable\b")
WINDOW = 90


def sweep() -> dict[str, dict[str, str]]:
    """Section 2 of the ledger, as the ledger printed it."""
    return _rows_from(LEDGER.read_text(encoding="utf-8", errors="replace"))


def _rows_from(body: str) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for line in body.splitlines():
        if (m := SWEEP_ROW.match(line.strip())):
            sym, verdict, ret, bench, sharpe, _hit, _trades, dd = m.groups()
            table[sym] = {"verdict": verdict, "return": ret, "benchmark": bench,
                          "sharpe": sharpe.lstrip("+"), "drawdown": dd}
    if not table:
        raise RuntimeError(f"{LEDGER.name} has no parseable sweep table — rebuild it with "
                           f"python tools/make_track_record.py")
    return table


def _published_ledger() -> str:
    """TRACK_RECORD.md as it stood in the last commit, or "" if there is none."""
    r = subprocess.run(["git", "show", f"HEAD:{LEDGER.relative_to(ROOT).as_posix()}"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    return "" if r.returncode else r.stdout


def _published_table() -> dict[str, dict[str, str]]:
    """Section 2 as it stood in the last commit: the table the pack was copied from.

    A hand-copied "+44.5%" says which column it came from only by matching the
    value it was copied at, so refreshing the copies needs the old table, not just
    the new one. This is the one place the pack's history is read from git rather
    than regenerated; an empty result simply leaves every site alone.
    """
    try:
        return _rows_from(_published_ledger())
    except RuntimeError:
        return {}


def _attributed(body: str, names: list[str], start: int) -> str:
    """The instrument a figure belongs to, or "" when a reader could not tell.

    The *closest* name before it, within the window — not merely one that occurs
    there. "SILVER +43.5% — GOLD +28.7%" has two names inside the window of the
    second figure, and reading it by name length charged GOLD's return to SILVER,
    which left the copy neither checked nor correct. Then no other name may sit in
    the gap: a figure reached through a second instrument's name is ambiguous, and
    ambiguous is left alone rather than guessed at.
    """
    before = body[:start]
    hits = [(i, s) for s in names if (i := before.rfind(s)) != -1 and start - i <= WINDOW]
    if not hits:
        return ""
    i, sym = max(hits)
    if any(s in before[i + len(sym):] for s in names if s != sym):
        return ""
    return sym


def _signed(v: str) -> tuple[str, str]:
    v = v.replace("\u2212", "-")
    return ("-" if v.startswith("-") else "+", v.lstrip("+-").replace(",", ""))


def _role(old_row: dict, sign: str, digits: str) -> str:
    """Which ledger column a copied figure was: the one it was copied from.

    Both sign and magnitude have to agree, and a magnitude that fits two columns
    fits neither — an ambiguous site is left for the checker to flag rather than
    rewritten on a guess.
    """
    hits = [k for k in ("return", "benchmark", "drawdown")
            if _signed(old_row[k]) == (("-" if sign == "\u2212" else sign), f"{digits}%")]
    return hits[0] if len(hits) == 1 else ""


def _reshaped(sign: str, want: str, html: bool = False, unsigned: bool = False) -> str:
    """The ledger's value, written the way this site writes it.

    `unsigned` is only ever set by the mean-return prose, because a sentence like
    "the equity book loses 2.8% mean" carries the direction in its own words and
    adding a minus would say "loses -2.2%". Nothing else gets that treatment: a
    Sharpe cell has no word around it, so a sign flip has to show up in the number.
    Otherwise this site keeps its own sign, and a positive never grows a plus the
    table does not use. The landing page types a minus as − and the markdown as -,
    so the document's own convention decides the glyph.
    """
    ws, wd = _signed(want)
    if ws == "-":
        if unsigned and not sign:
            return wd.rstrip("%")
        return ("\u2212" if sign == "\u2212" or html else "-") + wd
    return "+" + wd if sign == "+" else wd


def _swap(m: re.Match, old_sub: str, new_sub: str) -> str:
    """The site's own text with the copied figure replaced; everything else kept."""
    return m.group(0).replace(old_sub, new_sub, 1)


def _sw(text: str, old_sub: str, new_sub: str) -> str:
    return text.replace(old_sub, new_sub, 1)


def _fix_pct(m: re.Match, body: str, ctx: dict) -> str:
    sym = _attributed(body, ctx["names"], m.start())
    if not sym or sym not in ctx["old"]:
        return m.group(0)
    key = _role(ctx["old"][sym], m.group(1), m.group(2))
    if not key:
        return m.group(0)
    return _swap(m, f"{m.group(1)}{m.group(2)}%",
                 _reshaped(m.group(1), ctx["new"][sym][key], ctx.get("html", False)))


def _fix_sharpe(m: re.Match, body: str, ctx: dict) -> str:
    """Either Sharpe shape: the one with the word in front of it, the one in a column."""
    got = next(g for g in m.groups() if g)
    at = m.start() + m.group(0).find(got)
    sym = _attributed(body, ctx["names"], at)
    if not sym or sym not in ctx["new"]:
        return m.group(0)
    want = _reshaped(got[0] if got[0] in "+\u2212-" else "", ctx["new"][sym]["sharpe"],
                     ctx.get("html", False))
    return m.group(0) if _mag(got) == _mag(want) else _swap(m, got, want)


def _fix_verdict(m: re.Match, body: str, ctx: dict) -> str:
    """A table cell's verdict word, the only place it is a figure rather than prose.

    A run that flips an instrument's verdict leaves the return rewritten and the
    word "profitable" behind, and in the landing page's markup a green class
    travels with the row — so both are set from the ledger's own verdict column.
    """
    sym = _attributed(body, ctx["names"], m.start())
    if not sym or sym not in ctx["new"] or m.group(1) == ctx["new"][sym]["verdict"]:
        return m.group(0)
    return _swap(m, m.group(1), ctx["new"][sym]["verdict"])


def _ratio_sense(body: str, m: re.Match) -> str:
    """Which side of the sweep a ratio counts, from its nearest descriptor."""
    lo = max(0, m.start() - RATIO_WINDOW)
    hi = min(len(body), m.end() + RATIO_WINDOW)
    cut = m.start() - lo
    best, dist = "", RATIO_WINDOW + 1
    for words, tag in ((WIN_WORDS, "wins"), (LOSS_WORDS, "losers")):
        for w in words:
            for hit in re.finditer(w, body[lo:hi], re.I):
                d = abs(hit.start() - cut)
                if d < dist:
                    best, dist = tag, d
    return best


def _is_ratio_claim(m: re.Match, body: str, total: str) -> bool:
    """Is this pair of numbers the sweep count, or a clock and a RAM size?"""
    # A real sweep count can never exceed its own total, and that arithmetic fact
    # is what keeps "24/7" — which has the word "losses" on either side of it in
    # the deck — out of the rewrite. The cost is that "99 of 14" is read as a
    # malformed clock and left alone; mangling an uptime stamp is the worse error.
    if int(m["count"]) > int(m["total"]):
        return False
    if m["total"] == str(total):
        return True
    return bool(_ratio_sense(body, m))


def _fix_ratio(m: re.Match, body: str, ctx: dict) -> str:
    if not _is_ratio_claim(m, body, ctx["total"]):
        return m.group(0)
    want = ctx["losers"] if _ratio_sense(body, m) == "losers" else ctx["wins"]
    return _sw(_swap(m, m["count"], want), m["total"], ctx["total"])


def _fix_losers(m: re.Match, _body: str, ctx: dict) -> str:
    return _swap(m, m.group(1), ctx["losers"])


def _fix_mean(m: re.Match, body: str, ctx: dict) -> str:
    got = _mean_value(m)
    # the venue named in the sentence wins; only a mean with no venue beside it has
    # to be identified by which old figure it matches, which is guesswork
    venue = _mean_venue(body, m)
    if venue not in ctx["new_means"]:
        venue = next((k for k, v in ctx["old_means"].items() if _mag(v) == _mag(got)), "")
    if venue not in ctx["new_means"]:
        return m.group(0)
    sign = got[0] if got[0] in "+\u2212-" else ""
    return _swap(m, got, _reshaped(sign, ctx["new_means"][venue],
                                   ctx.get("html", False), unsigned=True))


def _fix_td_class(m: re.Match) -> str:
    """`class="up"` must follow the sign inside it, not the row it was born in."""
    return m.group(0).replace(f'class="{m.group(1)}"',
                              'class="up"' if m.group(2) == "+" else 'class="dn"', 1)


def _ledger_headline() -> tuple[str, str, str]:
    """Profitable, total, losing — counted off the parsed table, not off prose.

    The ledger's own summary line is generated from the same rows, so reading it
    back would be reading a sentence about the table instead of the table.
    """
    table = sweep()
    wins = sum(1 for r in table.values() if r["verdict"] == "profitable")
    loss = sum(1 for r in table.values() if r["verdict"] != "profitable")
    return str(wins), str(len(table)), str(loss)


def _mag(v: str) -> str:
    return v.replace("\u2212", "-").lstrip("+-").rstrip("%").replace("**", "")


def _venue_means(body: str) -> dict[str, str]:
    """The per-venue mean returns, keyed by the venue the ledger named."""
    return {m.group(1): m.group(2) for m in VENUE_MEAN.finditer(body)}


def _mean_value(m: re.Match) -> str:
    """The number this match is about, ignoring the venue word it also captured."""
    return m["pre"] or m["post"] or m["near"]


def _mean_venue(body: str, m: re.Match) -> str:
    """Which venue's mean this is, from the venue named in or just after the phrase.

    The window is short on purpose: a mean belongs to the book named beside it, and
    a sentence that walks two paragraphs to reach the word "book" is not naming one.
    """
    if m["venue"]:
        return VENUE_WORDS.get(m["venue"].lower(), "")
    tail = body[m.end():m.end() + 45].lower()
    hits = [(tail.find(f"{phrase} book"), venue) for phrase, venue in VENUE_WORDS.items()]
    hits = [h for h in hits if h[0] != -1]
    # the nearest book wins: a sentence can compare both venues, and the one that
    # sits beside this number is the one this number came from
    return min(hits)[1] if hits else ""


def _ledger_means() -> set[str]:
    body = LEDGER.read_text(encoding="utf-8", errors="replace")
    # compared on magnitude, because the pack writes "PSX equities lost money,
    # -2.8% mean" and "a 2.8% mean loss" and both are the same claim; a sign the
    # prose already carries is not a stale figure
    return {_mag(v) for v in _venue_means(body).values()}


def stale_sweep_quotes(body_by_file: dict[str, str] | None = None) -> list[str]:
    """Every instrument figure the pack quotes that the ledger no longer prints."""
    bodies = corpus() if body_by_file is None else body_by_file
    table = sweep()
    names = sorted(table, key=len, reverse=True)
    wins, total, losers = _ledger_headline()
    loser_syms = sorted(s for s, r in table.items() if r["verdict"] != "profitable")
    want_means = _ledger_means()
    live_means = _venue_means(LEDGER.read_text(encoding="utf-8", errors="replace"))
    out: list[str] = []
    for name, raw in bodies.items():
        if name.endswith(".md") and "/releases/" in name:
            continue          # frozen at publication; see the note above DENIALS
        body = re.sub(r"\s+", " ", raw.replace("\n>", " ").replace("**", " "))
        for m in PCT.finditer(body):
            sym = _attributed(body, names, m.start())
            if not sym:
                continue
            row = table[sym]
            got = f"{m.group(1)}{m.group(2)}%".replace("\u2212", "-")
            allowed = {row["return"], row["benchmark"], row["drawdown"],
                       row["drawdown"].replace("-", "\u2212")}
            if got.replace("+", "") not in {a.replace("+", "").replace("\u2212", "-")
                                            for a in allowed}:
                out.append(f'{name}: {sym} {got!r} is not a figure the ledger prints '
                           f'for it (net {row["return"]}, buy & hold {row["benchmark"]}, '
                           f'drawdown {row["drawdown"]})')
        for m in SHARPE.finditer(body):
            sym = _attributed(body, names, m.start())
            if sym and m.group(1) != table[sym]["sharpe"]:
                out.append(f'{name}: Sharpe {m.group(1)} for {sym} — the ledger says '
                           f'{table[sym]["sharpe"]}')
        for m in SHARPE_CELL.finditer(body):
            got = next(g for g in m.groups() if g)
            sym = _attributed(body, names, m.start() + m.group(0).find(got))
            if sym and _mag(got) != _mag(table[sym]["sharpe"]):
                out.append(f'{name}: Sharpe {got} for {sym} in a table cell — the '
                           f'ledger says {table[sym]["sharpe"]}')
        for m in VERDICT.finditer(body):
            sym = _attributed(body, names, m.start())
            if sym and m.group(1) != table[sym]["verdict"]:
                out.append(f'{name}: {sym} is called "{m.group(1)}" in a table cell — '
                           f'the ledger says {table[sym]["verdict"]}')
        for m in RATIO.finditer(body):
            if not _is_ratio_claim(m, body, total):
                continue
            sense = _ratio_sense(body, m)
            want = losers if sense == "losers" else wins
            if (m["count"], m["total"]) != (want, total):
                out.append(f'{name}: "{m.group(0)}" counts the '
                           f'{"unprofitable" if sense == "losers" else "profitable"} side '
                           f'of the sweep — the ledger table has {wins} profitable and '
                           f'{losers} not, of {total}')
        for m in LOSERS.finditer(body):
            if m.group(1) != losers:
                out.append(f'{name}: "{m.group(0)}" — the ledger table has '
                           f'{losers} of {total} not profitable')
        for m in MEAN.finditer(body):
            got = _mag(_mean_value(m))
            venue = _mean_venue(body, m)
            want = live_means.get(venue)
            if want:
                if got != _mag(want):
                    out.append(f'{name}: "{m.group(0)}" is the {venue} book mean and '
                               f'the ledger prints it as {want}%')
            elif got not in want_means:
                out.append(f'{name}: "{m.group(0)}" — the ledger prints '
                           f'{sorted(want_means)}')
        for m in LOSE_LIST.finditer(body):
            # "BTC, EURUSD and OGDC were not profitable" reads as the whole losing
            # set, so it is checked as one: no winner may be named in it, and no
            # loser may be left out of it.
            listed = [s for s in re.split(r", |\s+and\s+|\s+or\s+", m.group(1))
                      if s in table]
            if not listed:
                continue
            wrong = [s for s in listed if table[s]["verdict"] == "profitable"]
            if wrong:
                out.append(f'{name}: "{m.group(0)}" names '
                           f'{", ".join(sorted(wrong))} as a loser; the ledger has '
                           f'them profitable')
            missing = sorted(set(loser_syms) - set(listed))
            if missing and set(listed) & set(loser_syms):
                out.append(f'{name}: "{m.group(0)}" leaves out '
                           f'{", ".join(missing)} — the ledger has '
                           f'{len(loser_syms)} unprofitable instruments')
    return out


ALIGNERS = [(PCT, _fix_pct), (SHARPE, _fix_sharpe), (SHARPE_CELL, _fix_sharpe),
            (VERDICT, _fix_verdict),
            (RATIO, _fix_ratio), (LOSERS, _fix_losers), (MEAN, _fix_mean)]


def align_ctx() -> dict:
    """The old table, the new table and the two headline counts, for one pass."""
    new = sweep()
    wins, total, losers = _ledger_headline()
    return {"old": _published_table(), "new": new,
            "names": sorted(new, key=len, reverse=True),
            "old_means": _venue_means(_published_ledger()),
            "new_means": _venue_means(LEDGER.read_text(encoding="utf-8",
                                                       errors="replace")),
            "wins": wins, "total": total, "losers": losers}


def _align_body(src: str, ctx: dict) -> tuple[str, int]:
    """One document, every sweep figure in it pointed at the rebuilt ledger."""
    hits = []
    for rx, fn in ALIGNERS:
        scan = src

        def rep(m, f=fn, s=scan):
            out = f(m, s, ctx)
            if out != m.group(0):
                hits.append(out)
            return out

        src, _n = rx.subn(rep, src)
    scan = src

    def rep_class(m):
        out = _fix_td_class(m)
        if out != m.group(0):
            hits.append(out)
        return out

    src, _n = TDCLASS.subn(rep_class, src)
    return src, len(hits)


def _sweep_align(ctx: dict) -> dict[str, int]:
    """Rewrite the hand-copied sweep across the pack.

    `RULES` rewrites a count standing next to a figure word, and a return standing
    next to an instrument name is not that — which is how the sweep table came to
    be copied by hand into five layouts and left there. The first pass of this
    found 61 stale sites, including three instruments whose verdict had since
    flipped sign and a landing page that would have gone on printing them green.

    The committed ledger is used only to work out which column each site was
    quoting; every value written comes from the generated ledger and nothing else.
    """
    changed: dict[str, int] = {}
    for name in PACK_FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8", errors="replace")
        body, n = _align_body(original, {**ctx, "html": name.endswith(".html")})
        if n and body != original:
            path.write_text(body, encoding="utf-8", newline="\n")
            changed[name] = n
    return changed


def _package_version() -> str:
    from kairos import __version__
    return __version__


def _values(facts: dict) -> dict[str, str]:
    # the ledger reports evidence figures as the strings it printed, so they are
    # re-parsed rather than trusted to already be numeric
    def num(key: str, thousands: bool) -> str:
        v = int(str(facts[key]).replace(",", ""))
        return f"{v:,}" if thousands else str(v)

    return {"lines_comma": num("lines", True), "modules": num("modules", False),
            "tests": num("tests", False), "strategies": num("strategies", False),
            "journaled_comma": num("journaled", True),
            "matured_comma": num("matured", True),
            "non_trades_comma": num("non_trades", True),
            "cycles": num("cycles", False), "fills": num("fills", False),
            "grid": num("grid", True)}


def figures() -> dict[str, int]:
    """Every figure the pack is allowed to quote, as integers.

    tests/test_kit_figures.py checks the pack against this and nothing else, so
    the command that rewrites the figures and the test that guards them cannot
    hold two different ideas of what a figure is.
    """
    facts = {**codebase(), **ledger()}
    return {k: int(v.replace(",", "")) for k, v in _values(facts).items()}


def _utf8_stdout() -> None:
    """Let the reporter print what it found.

    The ledger writes a drawdown with a typographic minus (−9.4%), a stale-figure
    message quotes it back, and on a Windows console that inherits cp1252 the
    print raised before the line landed — so the command that exists to name
    contradictions died partway through naming them, with a traceback instead of a
    report. Errors are replaced rather than dropped: a mangled character in a
    diagnostic beats no diagnostic.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _rewrite(facts: dict) -> dict[str, int]:
    """Point every quoted figure in the pack at the values just recomputed."""
    vals = _values(facts)
    vals.update(stamps(facts))
    changed: dict[str, int] = {}
    for name in PACK_FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        body = original = path.read_text(encoding="utf-8", errors="replace")
        for pattern, key, keep in (*RULES, *STAMP_RULES):
            def sub(m, v=vals[key], k=keep):
                return "".join(m.group(i) or "" for i in range(1, k + 1)) + v

            body, n = re.subn(pattern, sub, body)
            if n:
                changed[name] = changed.get(name, 0) + n
        if body != original:
            path.write_text(body, encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="rewrite every figure in docs/investors/ from these values")
    args = ap.parse_args()
    _utf8_stdout()

    facts = {**codebase(), **ledger()}
    if args.write:
        for name, n in sorted(_rewrite(facts).items()):
            print(f"{name:34} {n} figure(s) updated")
        for name, n in sorted(_sweep_align(align_ctx()).items()):
            print(f"{name:34} {n} sweep figure(s) updated")
        facts = {**codebase(), **ledger()}   # re-read, the ledger is the source
    order = ["lines", "modules", "tests", "strategies", "grid", "journaled",
             "matured", "directional", "non_trades", "cycles", "fills", "realized_pnl",
             "ledger_generated", "pack_date", "version"]
    facts.update(stamps(facts))
    if args.markdown:
        print("| figure | value |")
        print("|---|---|")
        for k in order:
            v = facts.get(k)
            print(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
    else:
        for k in order:
            v = facts.get(k)
            print(f"{k:18} {v:,}" if isinstance(v, int) else f"{k:18} {v}")
    bad = denials() + method_claims() + stale_sweep_quotes()
    print("\npack claims:      consistent with the pack" if not bad else
          "\npack claims:      CONTRADICTED")
    for line in bad:
        print(f"  {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
