"""Live credentials must never reach a tracked file.

The system holds a broker login, a portal password, a trading PIN, an API token
and a pile of LLM keys. Any one of them appearing in source — or in a test
fixture, where it looks fake and stays forever — turns a public repository into
a credential leak. This runs in CI and on every operator's machine, so the rule
is enforced instead of remembered.

Nothing here prints a secret: failures name the key and the file only.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml

from kairos.config import ROOT

SECRET_HINTS = ("pin", "password", "secret", "token", "key", "login",
                "credential", "access")
MIN_LEN = 4


def _harvest(node, path: str, out: dict[str, str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _harvest(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _harvest(v, f"{path}[{i}]", out)
    elif isinstance(node, (str, int)) and any(h in path.lower() for h in SECRET_HINTS):
        s = str(node).strip()
        # placeholders in the shipped example are not secrets; neither is ""
        if len(s) >= MIN_LEN and "${" not in s and not s.startswith("<"):
            out[path] = s


def _tracked_files() -> list[str]:
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                       text=True, check=False)
    if r.returncode != 0:
        pytest.skip("no git index available")
    return [f for f in r.stdout.split()
           if f.endswith((".py", ".yaml", ".yml", ".md", ".html", ".json",
                          ".bat", ".sh", ".txt", ".toml"))]


def _live_secrets() -> dict[str, str]:
    found: dict[str, str] = {}
    cfg_path = ROOT / "config.yaml"
    if cfg_path.is_file():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        _harvest(raw, "", found)
    token = ROOT / "data" / "kairos_token.txt"
    if token.is_file():
        t = token.read_text(encoding="utf-8").strip()
        if len(t) >= MIN_LEN:
            found["api_server.token"] = t
    return found


def test_config_yaml_is_git_ignored():
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        pytest.skip("no local config to protect")
    r = subprocess.run(["git", "check-ignore", "config.yaml"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, "config.yaml must stay ignored — it holds live keys"


def test_no_live_credential_appears_in_tracked_source():
    secrets = _live_secrets()
    if not secrets:
        pytest.skip("no local credentials to check")
    files = _tracked_files()
    blobs = {f: (ROOT / f).read_text(encoding="utf-8", errors="ignore")
             for f in files if (ROOT / f).is_file()}
    leaks = [(name, f) for name, val in secrets.items()
             for f, body in blobs.items() if val in body]
    assert not leaks, (f"{len(leaks)} tracked file(s) contain a live credential "
                       f"value: {sorted(set(leaks))[:5]}")


def test_tracked_source_has_no_generic_secret_shapes():
    """Catches the shape of a leak rather than a known value: a long provider
    key pasted straight into code."""
    import re
    patterns = [
        re.compile(r"sk-[A-Za-z0-9]{16,}"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"xai-[A-Za-z0-9]{24,}"),
        re.compile(r"sk-or-v1-[A-Za-z0-9]{24,}"),
    ]
    bad = []
    for f in _tracked_files():
        p = ROOT / f
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            if pat.search(body):
                bad.append((f, pat.pattern))
    assert not bad, f"provider-key-shaped strings in tracked source: {bad[:5]}"
