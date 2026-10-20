"""The investor pack's numbers must be the repository's numbers.

This exists because the published v2.6.0 deck quoted a line count, a module count
and a test total that were three passes old, under a footer that said the figures
were verified that day. Nothing in the suite noticed: the numbers were prose in a
markdown table, and the only generated document in the pack was the track record.

Two kinds of figure, checked differently:

* Codebase figures (Python lines, modules, tests) change only when code is
  committed, so they are compared against the working tree right now. Edit the
  code and the pack must be edited in the same commit.
* Evidence figures (predictions journaled, horizons matured, cycles, fills) move
  while the daemon runs, so they cannot be pinned to a live count. They are
  pinned instead to the generated ledger, TRACK_RECORD.md, which is rebuilt by
  one command. If the ledger is regenerated and the deck is not, this fails.

Either way the pack cannot drift silently, which is the point: an outside party
can check every number in it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from kairos.config import ROOT

PACK_DIR = ROOT / "docs" / "investors"
PACK = [PACK_DIR / name for name in
        ("DECK.md", "ONE_PAGER.md", "WHITEPAPER.md", "landing.html", "README.md",
         "OUTREACH.md")] + [ROOT / "README.md"]
LEDGER = PACK_DIR / "TRACK_RECORD.md"


def _text(p: Path) -> str:
    body = p.read_text(encoding="utf-8", errors="replace")
    # Figures get wrapped, and a blockquote continuation marker lands between the
    # number and the word it belongs to. Match on the flattened text so a line
    # break cannot hide a stale figure.
    return re.sub(r"\s+", " ", body.replace("\n>", " "))


def _codebase_figures() -> dict[str, int]:
    r = subprocess.run(["git", "ls-files", "kairos/*.py"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        pytest.skip("no git index available")
    files = [f for f in r.stdout.split() if f.endswith(".py")]
    if not files:
        pytest.skip("no tracked source to count")
    lines = sum(len((ROOT / f).read_text(encoding="utf-8",
                                         errors="replace").splitlines())
                for f in files)
    return {"lines": lines, "modules": len(files)}


@pytest.fixture
def suite(request) -> int:
    """Collected items for this session — parametrized cases included.

    Counting definitions with a regex would be wrong: parametrize adds cases, so
    the only honest number is the one pytest itself reports.
    """
    args = [str(a) for a in (_opt(request.config, "file_or_dir") or [])]
    if args and args not in (["tests"], ["."], [str(ROOT / "tests")]):
        pytest.skip("figures are only checked on a full-suite run")
    if _opt(request.config, "keyword") or _opt(request.config, "mark"):
        pytest.skip("figures are only checked on a full-suite run")
    return len(request.session.items)


def _opt(config, name):
    """Read an option a given pytest build may not register."""
    try:
        return config.getoption(name)
    except ValueError:
        return None


def _ledger_figures() -> dict[str, str]:
    body = _text(LEDGER)
    out: dict[str, str] = {}
    for label, key in (("predictions journaled", "journaled"),
                       ("horizons matured", "matured"),
                       ("of which directional calls", "directional")):
        m = re.search(rf"\| {re.escape(label)} \| ([\d,]+) \|", body)
        assert m, f"{LEDGER.name} no longer reports {label}"
        out[key] = m.group(1)
    m = re.search(r"cycles run: \*\*([\d,]+)\*\*", body)
    assert m, f"{LEDGER.name} no longer reports autopilot cycles"
    out["cycles"] = m.group(1)
    m = re.search(r"fills: (\d+)", body)
    assert m, f"{LEDGER.name} no longer reports fills"
    out["fills"] = m.group(1)
    return out


@pytest.fixture(scope="module")
def codebase():
    return _codebase_figures()


@pytest.fixture(scope="module")
def ledger():
    return _ledger_figures()


@pytest.mark.parametrize("path", PACK, ids=lambda p: p.name)
def test_no_stale_codebase_figures(path, codebase, suite):
    body = _text(path)
    bad: list[str] = []

    for m in re.finditer(r"([\d,]{4,9})\s+lines(?:\s+of)?\s+Python", body):
        if int(m.group(1).replace(",", "")) != codebase["lines"]:
            bad.append(f"{m.group(0)!r} — repo has {codebase['lines']:,}")
    for m in re.finditer(r"<b>([\d,]{4,9})</b><small>lines of Python", body):
        if int(m.group(1).replace(",", "")) != codebase["lines"]:
            bad.append(f"{m.group(0)!r} — repo has {codebase['lines']:,}")
    for m in re.finditer(r"(\d{2,3}) modules", body):
        if int(m.group(1)) != codebase["modules"]:
            bad.append(f"{m.group(0)!r} — repo has {codebase['modules']} modules")
    # only the totals: a defect row that says "9 tests" for its own fix is not
    # the suite size and must not be rewritten.
    for m in re.finditer(r"#\s*(\d{2,4}) tests", body):
        if int(m.group(1)) != suite:
            bad.append(f"{m.group(0)!r} — the suite runs {suite}")
    for m in re.finditer(r"(\d{2,4}) regression tests(?!, including)", body):
        if int(m.group(1)) != suite:
            bad.append(f"{m.group(0)!r} — the suite runs {suite}")
    assert not bad, f"{path.relative_to(ROOT)} contradicts the repository: {bad}"


@pytest.mark.parametrize("path", PACK, ids=lambda p: p.name)
def test_no_stale_evidence_figures(path, ledger):
    """Evidence moves while the daemon runs, so the pack is pinned to the
    generated ledger rather than to the live journal — the same numbers, but
    checkable from a file an investor can ask for."""
    body = _text(path)
    bad: list[str] = []

    def check(pattern: str, want: str, label: str) -> None:
        for m in re.finditer(pattern, body):
            if m.group(1).replace(",", "") != want.replace(",", ""):
                bad.append(f"{m.group(0)!r} — the ledger says {want} {label}")

    check(r"([\d,]{3,7}) predictions journaled", ledger["journaled"], "journaled")
    check(r"([\d,]{2,7}) horizons matured", ledger["matured"], "matured")
    check(r"(?<!every )(\d{2,6}) (?:autopilot )?cycles", ledger["cycles"], "cycles")
    check(r"(\d+) fills", ledger["fills"], "fills")
    assert not bad, f"{path.relative_to(ROOT)} contradicts TRACK_RECORD.md: {bad}"


def test_the_pack_cites_the_ledger_as_the_source():
    """A figure an investor can verify has to say where it comes from."""
    assert "make_track_record" in _text(PACK_DIR / "README.md")


