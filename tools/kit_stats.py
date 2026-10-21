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
            "tests": _collected_tests(), "strategies": _strategy_count()}


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
                       ("of which directional calls", "directional")):
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
]


def stamps(facts: dict | None = None) -> dict[str, str]:
    """The string figures, resolved from the ledger and the package."""
    facts = ledger() if facts is None else facts
    return {"pack_date": str(facts["ledger_generated"]).strip(),
            "version": _package_version()}


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
            "cycles": num("cycles", False), "fills": num("fills", False)}


def figures() -> dict[str, int]:
    """Every figure the pack is allowed to quote, as integers.

    tests/test_kit_figures.py checks the pack against this and nothing else, so
    the command that rewrites the figures and the test that guards them cannot
    hold two different ideas of what a figure is.
    """
    facts = {**codebase(), **ledger()}
    return {k: int(v.replace(",", "")) for k, v in _values(facts).items()}


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

    facts = {**codebase(), **ledger()}
    if args.write:
        for name, n in sorted(_rewrite(facts).items()):
            print(f"{name:34} {n} figure(s) updated")
        facts = {**codebase(), **ledger()}   # re-read, the ledger is the source
    order = ["lines", "modules", "tests", "strategies", "journaled", "matured",
             "directional", "cycles", "fills", "realized_pnl", "ledger_generated",
             "pack_date", "version"]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
