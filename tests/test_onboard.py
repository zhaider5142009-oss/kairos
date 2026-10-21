"""`kairos setup` must create a usable config, and `kairos doctor` must never
repeat a secret.

These are the two commands a stranger runs first, so both touch credential
files. The properties worth pinning down are:

* setup writes the example file, not a stripped re-serialisation of it — the
  comments are the warnings about region-bound keys and the two order keys, and
  losing them loses the safety;
* a key taken from the environment lands in *its own provider's* block. Four
  providers share the token `YOUR-KEY`, so a global replace would put one
  vendor's key into three other seats and every one of them would fail
  confusingly;
* the doctor's report — the thing a user pastes into an issue — carries counts
  and names only. No row may contain a key that is actually in the config.
"""

from __future__ import annotations

import json
from pathlib import Path

from kairos import onboard as O

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config.example.yaml"

SENTINEL = "sk-or-v1-SENTINEL-NOT-A-REAL-KEY"


# ------------------------------------------------------------------- the write
def test_fill_slot_reaches_only_its_own_provider_block():
    text = EXAMPLE.read_text(encoding="utf-8")
    out, done = O.fill_slot(text, "deepseek", SENTINEL)
    assert done
    assert f'api_key: "{SENTINEL}"' in out
    # YOUR-KEY is also gemini's, zai's and grok's — they must be untouched
    assert out.count("YOUR-KEY") == text.count("YOUR-KEY") - 1
    assert SENTINEL not in out.split("gemini:")[1]


def test_fill_slot_leaves_an_unknown_provider_alone():
    text = EXAMPLE.read_text(encoding="utf-8")
    out, done = O.fill_slot(text, "nosuchprovider", SENTINEL)
    assert not done and out == text


def test_setup_creates_the_config_and_fills_from_the_environment(tmp_path, monkeypatch):
    dest = tmp_path / "config.yaml"
    monkeypatch.setattr(O, "config_path", lambda: dest)
    monkeypatch.setattr(O, "is_ignored", lambda p: True)
    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    r = O.run_setup(interactive=False)
    assert r["created"] is True and dest.is_file()
    assert r["filled"] == ["openrouter"]
    text = dest.read_text(encoding="utf-8")
    assert SENTINEL in text
    # commentary is the point of copying the example rather than re-dumping YAML
    assert "Council entries are prefixed" in text
    assert "Put real credentials ONLY in the git-ignored config.yaml" in text
    assert r["remaining"] == O.placeholders(dest)


def test_setup_is_idempotent_and_never_clobbers_an_existing_file(tmp_path, monkeypatch):
    dest = tmp_path / "config.yaml"
    dest.write_text("apis:\n  openrouter:\n    keys: []\n", encoding="utf-8")
    monkeypatch.setattr(O, "config_path", lambda: dest)
    monkeypatch.setattr(O, "is_ignored", lambda p: True)
    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL)
    r = O.run_setup(interactive=False)
    assert r["created"] is False and r["filled"] == []
    assert dest.read_text(encoding="utf-8") == "apis:\n  openrouter:\n    keys: []\n"


# ------------------------------------------------------------------- the ignore
def test_the_ignore_check_says_unknown_rather_than_passing_on_nothing_to_read(
        tmp_path, monkeypatch):
    """A user who installed from a tarball has no .gitignore to consult. Saying
    "ok" there would be the one answer that must not be given about a file that
    is going to hold their keys — doctor treats None as a warning, not a pass."""
    monkeypatch.setattr(O, "example_path",
                        lambda: tmp_path / "config.example.yaml")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("app: {}\n", encoding="utf-8")
    assert O.is_ignored(cfg) is None
    (tmp_path / ".gitignore").write_text("config.yaml\nreports/\n", encoding="utf-8")
    assert O.is_ignored(cfg) is True
    assert O.is_ignored(tmp_path / "README.md") is False


# --------------------------------------------------------------------- reports
def _quiet_checks(monkeypatch):
    """Keep `checks()` hermetic: the feed and posture probes belong to other
    tests; here the point is what a report is allowed to contain."""
    import kairos.health as H
    monkeypatch.setattr(H, "system_health", lambda: {
        "data_online": True, "market_data": {"yahoo": "ok", "tradingview": "ok"},
        "universe_size": 108, "llm": {"ok": True, "provider": "openrouter",
                                      "models": 4, "unanswered": {}},
        "wsl": {"available": False, "models_ready": False}})
    import kairos.autopilot as A
    monkeypatch.setattr(A.AUTOPILOT, "status", lambda: {
        "mode": "paper", "armed_live": False, "kill_switch": False})


def test_the_doctor_never_reads_a_key_aloud(monkeypatch):
    from kairos.config import CFG

    _quiet_checks(monkeypatch)
    fake = {
        "openrouter": {"enabled": True, "keys": [SENTINEL, "sk-or-v1-YOUR-KEY-2"]},
        "moonshot": {"enabled": True, "api_key": SENTINEL},
        "psx_local": {"enabled": True, "base_url": "http://localhost:8000"},
    }
    monkeypatch.setattr(type(CFG), "apis", property(lambda self: fake))
    blob = json.dumps(O.checks())
    assert SENTINEL not in blob
    assert "SENTINEL" not in blob
    assert "YOUR-KEY-2" not in blob


def test_the_doctor_counts_a_half_filled_slot_as_a_warning(monkeypatch):
    from kairos.config import CFG

    _quiet_checks(monkeypatch)
    monkeypatch.setattr(type(CFG), "apis", property(lambda self: {
        "openrouter": {"enabled": True, "keys": [SENTINEL, "sk-or-v1-YOUR-KEY-2"]}}))
    inv = O.key_inventory()
    row = [r for r in inv if r["provider"] == "openrouter"][0]
    assert row["filled"] == 1 and row["placeholders"] == 1


def test_the_real_config_is_reported_as_ignored():
    """If this fails, a user following the README has credential files that
    `git add .` would pick up. The file to test is the one `setup` writes in a
    checkout — the repository root — not the temp path the test harness points
    `config_path()` at."""
    assert O.is_ignored(O.example_path().parent / "config.yaml") is True


def test_missing_optional_extras_are_a_note_not_a_failure(monkeypatch):
    _quiet_checks(monkeypatch)
    monkeypatch.setattr(O, "_dep_ok", lambda m: m in O.REQUIRED)
    rows = {r["area"]: r for r in O.checks()}
    assert rows["core dependencies"]["status"] == "ok"
    assert rows["optional extras"]["status"] == "note"
    assert "pip install" in rows["optional extras"]["fix"]


def test_a_blocking_problem_is_a_fail_status(monkeypatch):
    _quiet_checks(monkeypatch)
    monkeypatch.setattr(O, "_dep_ok", lambda m: False)
    rows = {r["area"]: r for r in O.checks()}
    assert rows["core dependencies"]["status"] == "fail"
    assert rows["core dependencies"]["fix"].startswith("pip install")


def test_the_doctor_prints_its_advice_without_losing_characters(monkeypatch,
                                                                capsys):
    """The doctor's job is the sentence a user copies. `pip install -e ".[all]"`
    is one of those sentences, and rich reads `[all]` as a style tag and eats
    it — leaving a command that installs nothing. So this pins the rendered
    table, not the dictionary behind it."""
    from types import SimpleNamespace

    import kairos.cli as K

    _quiet_checks(monkeypatch)
    monkeypatch.setattr(O, "_dep_ok", lambda m: m in O.REQUIRED)
    monkeypatch.setattr(O, "tower_running", lambda port: False)
    K.cmd_doctor(SimpleNamespace(port=8600))
    out = capsys.readouterr().out
    assert '.[all]' in out, "the extras advice lost its bracket while rendering"


def test_the_python_floor_is_the_one_the_packaging_declares():
    """Read the manifest with a regex, not tomllib: tomllib is 3.11+, and this
    suite runs on the 3.10 floor the package itself promises."""
    import re
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    wants = re.search(r'^requires-python = ">=([^"]+)"', text, re.M).group(1)
    got = ".".join(str(x) for x in O.MIN_PYTHON)
    assert wants == got, f"pyproject says >={wants} but onboard checks {got}"


def test_the_doctor_checks_every_dependency_the_packaging_requires():
    """`kairos doctor` is the last thing a new user runs before trusting it. If
    the packaging gains a required dependency the doctor never looks for, the
    doctor certifies a broken install — which is how a missing `requests` ends
    up looking like a disabled broker bridge."""
    import re
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = text.split("[project]")[1].split("[project.optional-dependencies]")[0]
    core = core.split("dependencies = [")[1].split("]")[0]
    # REQUIRED lists import names; the manifest lists distribution names
    dist_to_import = {"pyyaml": "yaml", "scikit-learn": "sklearn"}
    wanted = {dist_to_import.get(d, d).lower()
              for d in re.findall(r'"([A-Za-z0-9_.-]+)', core)}
    checked = {m.lower().replace("_", "-") for m in O.REQUIRED}
    missing = sorted(w for w in wanted if w not in checked)
    assert not missing, f"doctor never checks for {missing} that pyproject requires"


def test_a_config_outside_the_repository_is_not_claimed_to_be_ignored(tmp_path,
                                                                      monkeypatch):
    """The repository's own .gitignore says nothing about a file in someone's
    temp directory. Borrowing the rule anyway hands out a pass for a fact that
    was never checked — and this is the check that guards their keys."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / ".gitignore").write_text("config.yaml\n", encoding="utf-8")
    monkeypatch.setattr(O, "example_path", lambda: repo / "config.example.yaml")
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    cfg = foreign / "config.yaml"
    cfg.write_text("app: {}\n", encoding="utf-8")
    assert O.is_ignored(cfg) is None
    assert O.is_ignored(repo / "config.yaml") is True
    assert O.is_ignored(repo / "notes.md") is False


def test_a_placeholder_key_is_not_counted_as_filled(tmp_path, monkeypatch):
    """setup, doctor and health must agree that the example's template is not a
    credential — otherwise the first run reports a council that cannot answer."""
    from kairos.config import is_placeholder

    cfg = tmp_path / "config.yaml"
    cfg.write_text(O.example_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(O, "config_path", lambda: cfg)
    assert O.placeholders(cfg) > 0
    for token in ("sk-or-v1-YOUR-KEY-1", "YOUR-MOONSHOT-KEY", "YOUR-KEY",
                  "REPLACE_ME", "", None):
        assert is_placeholder(token), token
    # built at runtime: a literal 40-hex token here would trip the repository's
    # own secret-shape tripwire, which is exactly what it is for
    for real in ("sk-or-v1-" + "ab" * 20, "sk-" + "1" * 20):
        assert not is_placeholder(real), real
