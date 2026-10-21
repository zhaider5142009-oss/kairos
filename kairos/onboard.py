"""First-run onboarding — `kairos setup`, `kairos doctor`, `kairos up`.

A stranger cloning this repository should be useful in one command, and should
never have to trust a document to find out whether their install is sound. So
these three things are answered by the code itself:

* **setup** creates the git-ignored `config.yaml` from the example, filling key
  slots from environment variables where they exist and prompting (masked, never
  echoed) where they do not. Values are written by line substitution so the
  example's commentary survives — a config file that lost its warnings about
  region-bound keys is worse than no config file.
* **doctor** states what it actually checked and what to do about each gap:
  interpreter, dependencies, the config file *and whether it is git-ignored*,
  key slots filled versus still placeholders, writable state directories, the
  market feeds, and the live-trading posture. It prints no secret material at
  all — only counts, and which slots are still placeholders.
* **up** starts the tower and opens the desk, because "run python -m kairos
  serve --host 0.0.0.0 --port 8600" is not a first run.

Nothing here can arm, order or connect to a venue. The one security-relevant
write is copying the example config, which contains no credentials.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from .config import PLACEHOLDER_RE, is_placeholder

MIN_PYTHON = (3, 10)
REQUIRED = ["yaml", "numpy", "pandas", "scipy", "statsmodels", "sklearn",
            "rich", "fastapi", "uvicorn", "pydantic"]
OPTIONAL = {
    "torch": "neural battery (LSTM/TemporalFusion) — without it the stat+ML "
             "models still run",
    "qiskit": "quantum-assisted portfolio optimisation — the annealer falls "
              "back to scipy",
    "MetaTrader5": "MT5 bridge for FX/commodity live execution (Windows only)",
    "httpx": "the LLM council and the news radar need it",
    "matplotlib": "equity-curve charts in the reports",
}
# provider block -> (placeholder token, environment variable that can fill it).
# Kimi is reached through the `moonshot` block — council entries prefixed "kimi:"
# route there — so it has no slot of its own in the example file.
KEY_SLOTS = {
    "openrouter": ("sk-or-v1-YOUR-KEY-1", "OPENROUTER_API_KEY"),
    "moonshot": ("YOUR-MOONSHOT-KEY", "MOONSHOT_API_KEY"),
    "deepseek": ("YOUR-KEY", "DEEPSEEK_API_KEY"),
}


def config_path() -> Path:
    from .config import CFG
    return (CFG.path if CFG.path is not None
            else Path(os.environ.get("KAIROS_CONFIG", "config.yaml")))


def example_path() -> Path:
    from .config import ROOT
    return ROOT / "config.example.yaml"


def is_ignored(path: Path) -> bool | None:
    """True when a readable .gitignore matches it, False when one does not, and
    None only when there was no .gitignore to read.

    The third answer matters: a user who installs from a tarball has no
    .gitignore, and for a file about to hold their keys "not ignored" and
    "unknowable" are not the same thing. `doctor` treats None as a warning.
    """
    readable = False
    repo = example_path().parent
    bases = [path.parent]
    if path.parent != repo and _inside(path, repo):
        # the repository's own rule only speaks for a file inside the repository
        bases.append(repo)
    for base in bases:
        gi = base / ".gitignore"
        if not gi.is_file():
            continue
        readable = True
        try:
            patterns = [ln.strip() for ln in gi.read_text(encoding="utf-8").splitlines()
                        if ln.strip() and not ln.strip().startswith("#")]
        except OSError:
            continue
        name = path.name
        for pat in patterns:
            p = pat.rstrip("/")
            if p == name or p == f"/{name}" or p.endswith(f"/{name}"):
                return True
            if _glob_matches(p, name):
                return True
    return False if readable else None


def _inside(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _glob_matches(pattern: str, name: str) -> bool:
    import fnmatch
    return "/" not in pattern and fnmatch.fnmatch(name, pattern)


# ------------------------------------------------------------------- the files
def write_example_config(dest: Path) -> bool:
    """Copy the example verbatim. Returns False if it is already there."""
    src = example_path()
    if not src.is_file():
        raise FileNotFoundError(f"{src} is missing — the repository is incomplete")
    if dest.exists():
        return False
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        dest.chmod(0o600)          # keys live here; owner-only by default
    except OSError:
        pass
    return True


def fill_slot(text: str, provider: str, value: str) -> tuple[str, bool]:
    """Replace one provider's placeholder inside its own block, keeping comments.

    Block-scoped on purpose: `YOUR-KEY` appears under four different providers,
    and a global find-and-replace would write one key into all of them.
    """
    token = KEY_SLOTS.get(provider, ("", ""))[0]
    if not token or not value:
        return text, False
    out: list[str] = []
    in_block = False
    done = False
    block = re.compile(rf"^  {re.escape(provider)}:\s*(#.*)?$")
    sibling = re.compile(r"^  [A-Za-z_0-9-]+:\s*(#.*)?$")
    for line in text.splitlines(keepends=True):
        if block.match(line):
            in_block = True
        elif in_block and sibling.match(line):
            in_block = False
        if in_block and not done and token in line:
            line = line.replace(token, value, 1)
            done = True
        out.append(line)
    return "".join(out), done


def key_inventory() -> list[dict[str, Any]]:
    """How many credential slots are real, without ever reading one aloud."""
    rows: list[dict[str, Any]] = []
    try:
        from .config import CFG
        apis = CFG.apis or {}
    except Exception as exc:
        return [{"provider": "config", "enabled": False, "filled": 0,
                 "placeholders": 0, "note": f"unreadable: {type(exc).__name__}"}]
    for name, spec in apis.items():
        if not isinstance(spec, dict):
            continue
        keys = spec.get("keys")
        single = spec.get("api_key") or spec.get("token") or ""
        vals = list(keys) if isinstance(keys, list) else ([single] if single else [])
        filled = sum(1 for k in vals if not is_placeholder(k))
        placeholders = len(vals) - filled
        rows.append({"provider": name, "enabled": bool(spec.get("enabled")),
                     "filled": filled, "placeholders": placeholders,
                     "note": ""})
    rows.sort(key=lambda r: (not r["enabled"], r["provider"]))
    return rows


def placeholders(path: Path) -> int:
    """How many credential slots in the file are still the example's tokens."""
    if not path.is_file():
        return 0
    try:
        return len(PLACEHOLDER_RE.findall(
            path.read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return 0


# ---------------------------------------------------------------------- doctor
def _dep_ok(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def checks() -> list[dict[str, str]]:
    """Every first-run question, answered with what to do about it."""
    out: list[dict[str, str]] = []
    ver = sys.version_info
    out.append({
        "area": "python",
        "status": "ok" if ver >= MIN_PYTHON else "fail",
        "detail": f"{ver.major}.{ver.minor}.{ver.micro}",
        "fix": "" if ver >= MIN_PYTHON else
        f"install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ — KAIROS will not import below it"})

    missing = [m for m in REQUIRED if not _dep_ok(m)]
    out.append({"area": "core dependencies",
                "status": "ok" if not missing else "fail",
                "detail": f"{len(REQUIRED) - len(missing)}/{len(REQUIRED)} present",
                "fix": "" if not missing else f"pip install {' '.join(missing)}"})
    quiet = [m for m, why in OPTIONAL.items() if not _dep_ok(m)]
    out.append({"area": "optional extras",
                "status": "ok" if not quiet else "note",
                "detail": "all present" if not quiet else ", ".join(quiet),
                "fix": "" if not quiet else
                "pip install -e \".[all]\" — " + "; ".join(OPTIONAL[q] for q in quiet[:3])})

    cfg = config_path()
    ign = is_ignored(cfg)
    out.append({
        "area": "config.yaml",
        "status": "ok" if cfg.is_file() and ign else ("warn" if cfg.is_file() else "note"),
        "detail": (f"{cfg}" + ("" if not cfg.is_file() else
                               " — git-ignored" if ign else
                               " — NOT git-ignored" if ign is False else
                               " — ignore status unknown")) if cfg.is_file()
        else "absent — the engine is running on built-in defaults and public data",
        "fix": ("" if cfg.is_file() and ign else
                "run `kairos setup`" if not cfg.is_file() else
                "add config.yaml to .gitignore before you put a key in it")})

    inv = key_inventory()
    live = [r for r in inv if r["enabled"] and r["filled"]]
    out.append({
        "area": "provider keys",
        "status": "ok" if live else "note",
        "detail": (", ".join(f"{r['provider']} {r['filled']} key(s)" for r in live)
                   or "none filled — the engine predicts, but the LLM council sits empty"),
        "fix": "" if live else "run `kairos setup`, or export OPENROUTER_API_KEY"})
    half = [r["provider"] for r in inv
            if r["enabled"] and r["placeholders"] and r["filled"]]
    left = placeholders(cfg)
    if half or left:
        out.append({"area": "key slots", "status": "warn",
                    "detail": (f"{left} placeholder slot(s) in {cfg.name}"
                              + (f" — enabled providers still asking: {', '.join(half)}"
                                 if half else "" if left else ", ".join(half))),
                    "fix": "a placeholder is sent to the vendor as a key and rejected — "
                           "fill it in config.yaml or set the provider enabled: false"})

    from .config import CFG
    unwritable = []
    for d in (CFG.data_dir, CFG.cache_dir, CFG.reports_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".kairos-write-probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
        except OSError:
            unwritable.append(str(d))
    out.append({"area": "state directories",
                "status": "ok" if not unwritable else "fail",
                "detail": f"{CFG.data_dir.name}/, cache, reports",
                "fix": "" if not unwritable else f"not writable: {', '.join(unwritable)} — "
                                                 "point app.data_dir somewhere you own"})

    try:
        from .health import system_health
        h = system_health()
        md = h.get("market_data") or {}
        bad = [k for k, v in md.items() if v not in ("ok", "up", "degraded")]
        llm = h.get("llm") or {}
        out.append({"area": "market data",
                    "status": "ok" if h.get("data_online") and not bad else
                    ("warn" if h.get("data_online") else "fail"),
                    "detail": f"{len(md) - len(bad)}/{len(md)} feeds answer, "
                              f"universe {h.get('universe_size')}",
                    "fix": "" if not bad else f"not answering: {', '.join(bad)} — "
                                              "predictions fall back to what does"})
        unanswered = llm.get("unanswered") or {}
        live = bool(llm.get("ok"))
        out.append({"area": "LLM council",
                    "status": "ok" if live and not unanswered else
                    ("warn" if live else "note"),
                    # the seat count is configured models, not answering ones —
                    # printing it beside "down" without saying so reads as four
                    # experts that are merely shy
                    "detail": f"{llm.get('provider')} · {llm.get('models')} seats"
                              + ("" if live else " configured, none answering")
                              + (f" · configured but not answering: {', '.join(unanswered)}"
                                 if unanswered else ""),
                    "fix": ("" if unanswered or not live else
                            "a configured provider that does not answer is billed as dead "
                            "weight — check the key, the quota and the account status")
                           + ("" if live else
                              "predictions run without the council; fill a key in "
                              "config.yaml to add the expert vote")})
    except Exception as exc:
        out.append({"area": "market data", "status": "fail",
                    "detail": f"health check raised {type(exc).__name__}: {exc}",
                    "fix": "run `kairos predict GOLD` to see the first failure point"})

    try:
        from .autopilot import AUTOPILOT
        st = AUTOPILOT.status() or {}
        out.append({"area": "trading posture",
                    "status": "ok" if not st.get("armed_live") else "warn",
                    "detail": f"{st.get('mode')} mode, live arming "
                              f"{'ON' if st.get('armed_live') else 'OFF'}, kill switch "
                              f"{'ENGAGED' if st.get('kill_switch') else 'clear'}",
                    "fix": "" if not st.get("armed_live") else
                    "this process believes it can trade real money — confirm you meant it"})
    except Exception:
        pass
    return out


def tower_running(port: int = 8600) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ----------------------------------------------------------------------- setup
def run_setup(interactive: bool = True) -> dict[str, Any]:
    """Create config.yaml from the example and fill what can be filled."""
    cfg = config_path()
    result: dict[str, Any] = {"created": False, "filled": [], "path": str(cfg),
                              "ignored": is_ignored(cfg)}
    if not cfg.exists():
        result["created"] = write_example_config(cfg)
    text = cfg.read_text(encoding="utf-8")
    for provider, (token, env) in KEY_SLOTS.items():
        value = os.environ.get(env, "").strip()
        if not value and interactive and token in text and sys.stdin.isatty():
            value = _prompt(provider)
        if value and token in text:
            text, done = fill_slot(text, provider, value)
            if done:
                result["filled"].append(provider)
    if result["filled"]:
        cfg.write_text(text, encoding="utf-8")
    result["remaining"] = len(PLACEHOLDER_RE.findall(
        cfg.read_text(encoding="utf-8", errors="ignore")))
    return result


def _prompt(provider: str) -> str:
    import getpass
    try:
        return getpass.getpass(f"  {provider} API key (Enter to skip, never echoed): "
                               ).strip()
    except (EOFError, KeyboardInterrupt):
        return ""
