"""Offline smoke tests — no network, no heavy/optional deps required.

These run in CI to guarantee the package imports, the universe and strategy
registries are populated, and config loads with sensible defaults. They are
deliberately cheap: they never hit Yahoo, OpenRouter, MT5 or WSL.
"""

from __future__ import annotations

import json

import pytest

import kairos
from kairos import strategies, universe
from kairos.config import CFG


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Keep registry mutations out of the live data dir and undo them after,
    so no test can leak a fake ticker into the next one."""
    snapshot = (list(universe.ALL), dict(universe._BY_KEY),
                dict(universe._BY_YAHOO), dict(universe._BY_TV))
    path = tmp_path / "reg.json"
    monkeypatch.setattr(universe, "_registered_path", lambda: path)
    yield path
    universe.ALL[:] = snapshot[0]
    for name, data in (("_BY_KEY", snapshot[1]), ("_BY_YAHOO", snapshot[2]),
                       ("_BY_TV", snapshot[3])):
        getattr(universe, name).clear()
        getattr(universe, name).update(data)


def test_version_is_string():
    assert isinstance(kairos.__version__, str)
    assert kairos.__version__.count(".") >= 1


def test_the_version_declared_everywhere_is_the_one_the_package_carries():
    """Four files state a version and only one of them is read by code. This
    repository's recurring defect is a change in one place leaving a stale copy
    in another, and a release tag that disagrees with `pip show kairos` is the
    same bug wearing a suit."""
    import re
    from pathlib import Path

    root = Path(kairos.__file__).resolve().parent.parent
    want = kairos.__version__

    def read(name: str) -> str:
        return (root / name).read_text(encoding="utf-8")

    assert re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M).group(1) == want
    assert re.search(r'^Version: "([^"]+)"', read("config.example.yaml"), re.M).group(1) == want
    # the server must stop hardcoding it, so the API schema is never a fifth copy
    assert 'version="2.' not in read("kairos/api/server.py")
    assert re.search(r'^__version__ = "([^"]+)"', read("kairos/__init__.py"), re.M).group(1) == want
    from kairos.api.server import app

    assert app.version == want


def test_the_public_repository_urls_are_the_public_repo():
    """`origin` is the private working repository. A stranger who installs the
    package reads `[project.urls]` from `pip show kairos`, and pointing them at
    a repository they cannot see is worse than pointing them nowhere."""
    from pathlib import Path

    root = Path(kairos.__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    lines = []
    for ln in text.split("[project.urls]")[1].splitlines():
        if ln.startswith("["):          # next table — the urls block is over
            break
        if ln.strip():
            lines.append(ln)
    assert lines, "the package declares no project URLs"
    for line in lines:
        assert "jarvis-reborn" not in line, line
        assert "zhaider5142009-oss/kairos" in line, line


def test_every_module_level_import_is_declared_in_the_packaging():
    """A missing dependency must not be an accident of the developer's own
    environment. CI once ran red for five pushes because `jsgsync` imported
    `requests` at module level while no manifest declared it — the Docker image
    worked only because its own requirements file happened to carry the fix."""
    import ast
    import re
    import sys
    from pathlib import Path

    root = Path(kairos.__file__).resolve().parent
    text = (root.parent / "pyproject.toml").read_text(encoding="utf-8")
    core = text.split("[project]")[1].split("[project.optional-dependencies]")[0]
    core = core.split("dependencies = [")[1].split("]")[0]
    extras = text.split("[project.optional-dependencies]")[1]
    extras = "\n".join(re.findall(r"= \[(.*?)\]", extras, re.S))
    declared = {d.lower().replace("_", "-")
                for d in re.findall(r'"([A-Za-z0-9_.-]+)', core + extras)}
    # PyPI name -> import name, for the pairs that differ
    for want, given in (("pyyaml", "yaml"), ("scikit-learn", "sklearn")):
        if want in declared:
            declared.add(given)

    std = set(sys.stdlib_module_names) | {"kairos", "__future__"}
    missing = {}
    for path in sorted(root.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                tops = [node.module.split(".")[0]]
            else:
                continue
            for top in tops:
                if top not in std and top.lower().replace("_", "-") not in declared:
                    missing.setdefault(str(path.relative_to(root)), set()).add(top)
    assert not missing, f"undeclared module-level imports: {missing}"


def test_requirements_txt_is_not_weaker_than_the_package():
    """README offers both as the install path; if one silently lacks something
    the other installs, two users end up with two different products."""
    import re
    from pathlib import Path

    root = Path(kairos.__file__).resolve().parent.parent
    core = Path(root / "pyproject.toml").read_text(encoding="utf-8")
    core = core.split("dependencies = [")[1].split("]")[0]
    core = {c.lower().replace("_", "-")
            for c in re.findall(r'"([A-Za-z0-9_.-]+)', core)}
    reqs = {r.lower().replace("_", "-")
            for r in re.findall(r"^([A-Za-z0-9_.-]+)",
                                (root / "requirements.txt").read_text(encoding="utf-8"),
                                re.M)}
    assert core <= reqs, f"`pip install -r requirements.txt` skips {core - reqs}"


def test_a_missing_card_extra_is_a_sentence_not_a_traceback(monkeypatch, capsys):
    """The free image ships no matplotlib on purpose. Asking it for a card has
    to explain the extra it lacks; `ModuleNotFoundError` in a terminal is how a
    newcomer decides the project does not work."""
    from types import SimpleNamespace

    import kairos.cli as K
    import sys

    real_import = __import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_matplotlib)
    monkeypatch.delitem(sys.modules, "kairos.signalcard", raising=False)
    assert K.cmd_share(SimpleNamespace(symbol="LPG", out=None)) == 3
    assert "[card]" in capsys.readouterr().out


def test_universe_is_populated():
    # the curated seed is 100; scan-driven PSX rotation can add more
    assert len(universe.ALL) >= 100
    assert len(universe.CRYPTO) == 22


def test_register_psx_is_idempotent_and_collision_safe(isolated_registry):
    n = len(universe.ALL)
    assert universe.register_psx("ZZTEST", name="Test Mill") is not None
    assert len(universe.ALL) == n + 1
    assert universe.register_psx("ZZTEST") is not None   # already PSX: reuse
    assert len(universe.ALL) == n + 1
    assert universe.resolve("ZZTEST").yahoo == "ZZTEST.KA"
    assert universe.register_psx("BTC") is None          # crypto key collision
    assert len(universe.ALL) == n + 1


def test_registered_psx_survives_a_restart(isolated_registry):
    """A rotated PSX watchlist must still resolve after the daemon restarts,
    or the desk cannot analyse or exit the paper book's PSX lines."""
    path = isolated_registry
    seed = len(universe.ALL)
    universe.register_psx("ZZPERSIST", name="Persist Mill", sector="Test")
    assert len(universe.ALL) == seed + 1
    assert path.exists() and "ZZPERSIST" in path.read_text(encoding="utf-8")

    # emulate the fresh process: forget it in memory, then restore from disk
    universe.ALL.pop()
    universe._BY_KEY.pop("ZZPERSIST", None)
    universe._BY_YAHOO.pop("ZZPERSIST.KA", None)
    universe._BY_TV.pop("PSX:ZZPERSIST", None)
    assert universe.resolve("ZZPERSIST") is None

    universe._restore_registered()
    assert universe.resolve("ZZPERSIST").market == "PSX"


def test_registered_cache_is_bounded(isolated_registry):
    for i in range(60):
        universe.register_psx(f"ZZB{i}", name=f"Bound {i}")
    assert len(json.loads(isolated_registry.read_text(encoding="utf-8"))) == 40


def test_a_missing_cell_never_becomes_a_company_name(isolated_registry):
    """A PSX feed's blank sector/label arrives as float NaN, `str()` of which is
    "nan" — truthy, so it used to be admitted as the scrip's name and printed on
    the desk and in the watchlist log. A label that reads like a gap is a gap."""
    for i, bad in enumerate((float("nan"), "nan", "N/A", None, "  ", "None")):
        inst = universe.register_psx(f"ZZMISS{i}", name=bad, sector=bad)
        assert inst.name == inst.key == f"ZZMISS{i}"
        assert inst.sector == "Unregistered"
    assert universe._clean("United Ceramic") == "United Ceramic"
    assert universe._clean(0.0) == "0.0"         # a real zero is a real value


def test_universe_has_expected_crypto():
    keys = {i.key.upper() for i in universe.CRYPTO}
    for expected in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        assert expected in keys


def test_resolve_is_case_insensitive():
    assert universe.resolve("BTC").key == "BTC"
    assert universe.resolve("btc").key == "BTC"
    assert universe.resolve("nope-not-a-ticker") is None


def test_strategy_registry_populated():
    assert len(strategies.STRATEGIES) >= 100


def test_config_has_dirs():
    # config must load even without config.yaml present
    assert CFG.data_dir is not None
    assert CFG.root is not None
