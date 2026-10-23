"""The desk is one <script> in one HTML file, and nothing read it.

A missing `)` in the JS Global order panel took the entire page down: nav,
health, the watchlist, every tab. The API kept answering, the HTML still
rendered, and the page looked perfectly healthy right up to the first click —
because a classic script that fails to parse leaves no trace but a window with
no functions on it. Nothing in the suite loaded the desk, so nothing in the
suite could tell it from a poster.

What follows is a bracket walk, not a JavaScript engine. It knows only what a
parser has to know about the one thing that matters here: which brackets must
balance, once strings, comments, template literals and regex literals have
taken out the brackets that are only text.
"""

from __future__ import annotations

import re
from pathlib import Path

UI_PATH = Path(__file__).resolve().parents[1] / "kairos" / "api" / "ui.html"

OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {v: k for k, v in OPEN.items()}
# a `/` starts a regex only where a value could begin. after `)` or a name it is
# division, and division hides nothing.
REGEX_OK = set("(,=:[!&|;{}+-*%~^?")


def _string_end(src: str, i: int, quote: str) -> tuple[int, bool]:
    j = i + 1
    while j < len(src):
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == quote:
            return j, True
        j += 1
    return j, False


def _regex_end(src: str, i: int) -> tuple[int, bool]:
    j, in_class = i + 1, False
    while j < len(src):
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            return j, True
        elif c == "\n":
            return j, False          # no regex literal crosses a line
        j += 1
    return j, False


def _scan(src: str) -> list[tuple[int, str]]:
    """Every way in which this script's brackets do not balance; empty is good.
    Each complaint is the line it was found on and what is wrong there.

    A frame is `[kind, line, open_brackets]`: `"code"` at the top level,
    `"tmpl"` while inside a template literal's text, and `"interp"` for the
    `${…}` code that a template literal can hold. Stopping at the first stray
    closer is deliberate — after that point the stack no longer describes the
    code, and a second complaint would be an artefact of the first.
    """
    frames: list[list] = [["code", 1, []]]
    i, line, prev, n = 0, 1, "", len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        kind = frames[-1][0]
        if kind == "tmpl":
            if c == "\\":
                i += 2
                continue
            if c == "`":
                frames.pop()
                i += 1
                prev = c
                continue
            if c == "$" and src[i + 1:i + 2] == "{":
                frames.append(["interp", line, []])
                i += 2
                continue
            i += 1
            continue
        brackets = frames[-1][2]
        if c == "/" and src[i + 1:i + 2] == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            j = src.find("*/", i + 2)
            if j < 0:
                return [(line, "a /* comment that never closes")]
            line += src.count("\n", i, j)
            i = j + 2
            continue
        if c in "\"'":
            j, ok = _string_end(src, i, c)
            if not ok:
                return [(line, f"a {c} string that never closes")]
            line += src.count("\n", i, j)
            i = j + 1
            prev = c
            continue
        if c == "`":
            frames.append(["tmpl", line, []])
            i += 1
            prev = c
            continue
        if c == "/" and (prev == "" or prev in REGEX_OK):
            j, ok = _regex_end(src, i)
            if ok:
                line += src.count("\n", i, j)
                i = j + 1
                prev = "/"
                continue
            i += 1                    # division after all; nothing to skip
            prev = c
            continue
        if c in OPEN:
            brackets.append((c, line))
            i += 1
            prev = c
            continue
        if c in CLOSE:
            if c == "}" and kind == "interp" and not brackets:
                frames.pop()          # the end of a ${…}, back into the text
                i += 1
                prev = c
                continue
            if not brackets:
                return [(line, f"a '{c}' with nothing to match")]
            opener, oline = brackets.pop()
            if opener != CLOSE[c]:
                return [(line, f"a '{c}' closing a '{opener}' opened on "
                              f"line {oline}")]
            i += 1
            prev = c
            continue
        prev = c
        i += 1
    problems: list[tuple[int, str]] = []
    for frame in frames:
        if frame[0] == "tmpl":
            problems.append((frame[1], "a template literal never closed"))
        for opener, oline in frame[2]:
            problems.append((oline, f"a '{opener}' that is never closed"))
    return problems


def _desk_script() -> tuple[str, int, list[str]]:
    html = UI_PATH.read_text(encoding="utf-8")
    m = re.search(r"<script[^>]*>(.*?)</script>", html, re.S)
    if m is None:
        raise AssertionError(f"no <script> in {UI_PATH} — the desk lost its code")
    return m.group(1), html[:m.start(1)].count("\n"), html.splitlines()


def test_the_desk_ships_exactly_one_script():
    html = UI_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert len(blocks) == 1, (
        f"{len(blocks)} <script> blocks in the desk — this test reads one and "
        "would silently skip a second page of code")
    assert len(blocks[0]) > 10_000, "the desk script went missing"


def test_the_desk_script_closes_every_bracket_it_opens():
    src, offset, lines = _desk_script()
    problems = _scan(src)
    if problems:
        detail = "\n".join(
            f"  {why}\n    html line {lineno + offset}: "
            f"{lines[lineno + offset - 1].strip()[:130]}"
            for lineno, why in problems)
        raise AssertionError(
            "the desk script would not parse, which makes the whole page inert "
            "while the API goes on answering:\n" + detail)


def test_the_guard_bites_on_the_defect_it_was_written_for():
    """Not a comment about the bug — the bug itself, replayed. One `)` short is
    how the desk shipped, and a test that can only pass on good code is the same
    untested component it was written to catch."""
    broken = ("$('#jopending').innerHTML=qband+(JO.length?`<table>${x}</table>`\n"
              "  :((q&&!q.ok)?'the queue could not be read'\n"
              "              :'no staged orders');\n")
    whole = ("$('#jopending').innerHTML=qband+(JO.length?`<table>${x}</table>`\n"
             "  :((q&&!q.ok)?'the queue could not be read'\n"
             "              :'no staged orders'));\n")
    assert _scan(broken), "an unbalanced paren walked straight through"
    assert _scan(whole) == []


def test_the_walk_survives_brackets_that_are_only_text():
    """Strings, comments, template literals and regexes all hold brackets a
    parser must ignore. A scanner that counted them would cry wolf until
    someone deleted it, which is exactly how this defect class comes back."""
    src = r"""
    const a = x.replace(/\)\(/g, '');      // a comment with ( and ) in it
    const b = "a string with ) and ( inside";
    const c = `a template (${v}) with ${ nested(`deeper (${w})`) } done`;
    const d = [1, 2].map(v => ({v: v}));
    if (a) alert('('); else alert(')');
    const e = 12 / 4; const f = (g) / 2;
    """
    assert _scan(src) == []
