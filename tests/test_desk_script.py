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

The desk asked for its access token once per request, and every endpoint in the
page — the watchlist, the health strip, the autopilot and live-book pollers — is
a request. So the first paint of a locked desk stacked ~14 modal dialogs, and
the timers kept adding more while the operator was still working through the
ones already on screen: an error path that multiplied itself, and one that hid
the page behind a dialog nobody had asked for. The last test in this file is not
a bracket walk. It runs the desk's real token funnel under Node against a stub
browser and counts the dialogs, because "at most one prompt" is a behaviour and
no amount of reading the text proves a behaviour.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------- token funnel
# The desk's request helper, copied out of the page it ships in and run against a
# stub browser. A test that carried its own retyped copy of `api()` would keep
# passing after the desk lost the guard, so the slice is taken from markers and
# both ends are asserted below.
_FUNNEL_START = "let TOKEN_ASK"
_FUNNEL_END = ("if(!r.ok&&!j.ok)throw new Error(j.error||j.detail||"
               "('HTTP '+r.status));return j}")

_HARNESS = r"""
const fs = require('fs');
const build = new Function('localStorage', 'fetch', 'prompt',
  fs.readFileSync(process.argv[2], 'utf8') +
  " ;return {api, askToken: (typeof askToken === 'function' ? askToken : null)};");

function stubs(good, stored, answer) {
  const table = new Map(stored ? [['kairos_token', stored]] : []);
  const localStorage = {
    getItem: k => (table.has(k) ? table.get(k) : null),
    setItem: (k, v) => { table.set(k, v); },
  };
  let prompts = 0, fetches = 0, refused = 0;
  const fetch = (path, opts) => {
    fetches++;
    const sent = (opts && opts.headers && opts.headers['X-KAIROS-Token']) || '';
    if (good && sent === good) {
      return Promise.resolve({status: 200, ok: true,
                              json: () => Promise.resolve({ok: true})});
    }
    refused++;
    return Promise.resolve({status: 401, ok: false,
                            json: () => Promise.resolve({ok: false,
                                                         error: 'unauthorized'})});
  };
  const prompt = () => { prompts++; return answer(prompts); };
  return {localStorage, fetch, prompt,
          stats: () => ({prompts, fetches, refused})};
}

async function scenario(good, stored, answer, first, ticks) {
  const s = stubs(good, stored, answer);
  const {api} = build(s.localStorage, s.fetch, s.prompt);
  const batch = k => Promise.allSettled(
    Array.from({length: k}, (_, i) => api('/api/e' + i)));
  const all = (await batch(first)).concat(ticks ? await batch(ticks) : []);
  return Object.assign(s.stats(), {
    results: all.map(r => r.status === 'fulfilled'
                       ? 'ok' : String(r.reason.message)),
  });
}

const cases = {
  // 14 endpoints on first paint, then two more poller ticks, answered once
  storm:          () => scenario('GOOD', null,    () => 'GOOD',  14, 4),
  dismissed:      () => scenario('GOOD', null,    () => null,    14, 4),
  wrong_stored:   () => scenario('GOOD', 'WRONG', () => 'GOOD',  6, 0),
  refused_again:  () => scenario('GOOD', 'WRONG', () => 'WRONG', 3, 0),
  // a host that refuses to open dialogs at all — automation, embedded webviews
  no_dialog_host: () => scenario('GOOD', null,
    () => { throw new TypeError('prompt() is not supported.'); }, 5, 0),
};

(async () => {
  const out = {};
  for (const [name, fn] of Object.entries(cases)) {
    try { out[name] = await fn(); }
    catch (e) { out[name] = {crashed: String((e && e.message) || e)}; }
  }
  process.stdout.write(JSON.stringify(out));
})();
"""

# How the desk shipped: one prompt per request, one per poller tick, forever.
_BROKEN_FUNNEL = r"""
let TOKEN_ASK = null;
async function api(path,opts){opts=opts||{};opts.headers=Object.assign({},opts.headers||{});let sent=localStorage.getItem('kairos_token')||'';if(sent)opts.headers['X-KAIROS-Token']=sent;
let r=await fetch(path,opts);let j=await r.json().catch(()=>({ok:false,error:'bad response'}));
if(r.status===401){const t=prompt('KAIROS access token:');if(t&&t.trim()){opts.headers['X-KAIROS-Token']=t.trim();localStorage.setItem('kairos_token',t.trim());r=await fetch(path,opts);j=await r.json().catch(()=>({ok:false,error:'bad response'}));}if(r.status===401)throw new Error('unauthorized — wrong or missing token');}
if(!r.ok&&!j.ok)throw new Error(j.error||('HTTP '+r.status));return j}
"""


def _token_funnel() -> str:
    src = UI_PATH.read_text(encoding="utf-8")
    i = src.find(_FUNNEL_START)
    assert i >= 0, (f"no {_FUNNEL_START!r} in the desk — the token funnel moved "
                    "or went away, and this test would be guarding nothing")
    j = src.find(_FUNNEL_END, i)
    assert j >= 0, "the desk's api() ended somewhere new — this slice is stale"
    return src[i:j + len(_FUNNEL_END)]


def _run_funnel(tmp_path: Path, code: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this box — the funnel's behaviour cannot be "
                    "watched here, only its text")
    src = tmp_path / "funnel.js"
    src.write_text(code, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(harness), str(src)],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, f"the stub browser died: {r.stderr[:400]}"
    return json.loads(r.stdout)


def test_the_funnel_under_test_is_the_one_the_desk_ships():
    """Runs with or without Node: the slice must still be a token funnel, and
    still only one — a second `prompt(` in here is the defect coming back."""
    src = _token_funnel()
    assert "async function api(" in src, "the slice stopped holding the desk's api()"
    assert "function askToken()" in src, "the desk no longer shares one prompt"
    assert src.count("prompt(") == 1, (
        f"{src.count('prompt(')} calls to prompt() in the request helper — "
        "the desk can open more than one dialog for one page load")


def test_one_locked_desk_asks_for_the_token_once_not_once_per_request(tmp_path):
    out = _run_funnel(tmp_path, _token_funnel())
    storm = out["storm"]
    assert "crashed" not in storm, storm
    # the stub has to actually lock the desk, or "one prompt" means nothing
    assert storm["refused"] >= 14, (
        f"only {storm['refused']} requests were refused — the stub answered an "
        "unauthorized desk, so this test proved nothing")
    assert storm["prompts"] == 1, (
        f"first paint plus two poller ticks opened {storm['prompts']} dialogs")
    assert set(storm["results"]) == {"ok"}, storm["results"]


def test_a_dismissed_prompt_is_not_asked_again_by_the_next_poller(tmp_path):
    out = _run_funnel(tmp_path, _token_funnel())
    gone = out["dismissed"]
    assert gone["prompts"] == 1, (
        f"the operator already said no, and the desk asked {gone['prompts']} times")
    assert all("no token entered" in m for m in gone["results"]), gone["results"]


def test_a_stored_token_the_server_refuses_is_replaced_by_one_prompt(tmp_path):
    out = _run_funnel(tmp_path, _token_funnel())
    bad = out["wrong_stored"]
    assert bad["prompts"] == 1, f"{bad['prompts']} dialogs to fix one bad token"
    assert set(bad["results"]) == {"ok"}, bad["results"]
    same = out["refused_again"]
    assert same["prompts"] == 1, same["prompts"]
    assert all("was refused" in m for m in same["results"]), same["results"]


def test_a_host_with_no_dialogs_says_so_instead_of_showing_an_engine_error(tmp_path):
    """`prompt()` throws outright under automation and inside embedded webviews.
    That engine message used to be the only thing the operator ever saw."""
    out = _run_funnel(tmp_path, _token_funnel())
    quiet = out["no_dialog_host"]
    assert "crashed" not in quiet, quiet
    assert all("no token entered" in m for m in quiet["results"]), quiet["results"]


def test_the_guard_bites_on_the_prompt_storm_it_was_written_for(tmp_path):
    """The shipped-before funnel, replayed: 14 concurrent 401s and two poller
    ticks opened one dialog each. A test that can only pass on fixed code is a
    test that cannot see the defect it names."""
    out = _run_funnel(tmp_path, _BROKEN_FUNNEL)
    storm = out["storm"]
    assert storm["prompts"] > 1, (
        "the old funnel opened fewer than two dialogs — the stub is not "
        "reproducing a locked desk")
    assert storm["prompts"] >= storm["refused"] - 1, (
        "one prompt per refused request is the claim being falsified, and it held "
        f"only {storm['prompts']} times for {storm['refused']} refusals")
