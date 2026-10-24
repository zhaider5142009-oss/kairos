"""The LEAN seam: write the data a cross-check needs, and never claim it ran.

QuantConnect LEAN is a .NET engine that runs behind Docker. It is not installed
on this desk, and the box that carries the paper loop has no headroom for a
multi-gigabyte image beside it, so a LEAN backtest is a deliberate operator
step rather than something the daemon performs. What this module owns is the
honest half: it exports KAIROS bars into a dated, digest-carrying dataset LEAN
can read, and `status()` names exactly which piece of the engine is absent.

The distinction matters because "Docker is installed" is true on this machine
while the engine is not running, and an export filed as a validation is the
same class of defect as a score that counts non-trades as hits: a figure that
looks like work that was done.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from .config import CFG
from .store import atomic_write, dumps_strict
from .universe import resolve

LEAN_IMAGE = "quantconnect/lean:latest"
PROBE_TIMEOUT_S = 4.0
COLUMNS = ("date", "open", "high", "low", "close", "volume")


class LeanError(ValueError):
    """The export cannot be made truthful, so it is not made."""


@dataclass(frozen=True)
class LeanStatus:
    cli: str | None
    docker: str | None
    server: str | None
    image: bool

    @property
    def ready(self) -> bool:
        return bool(self.server and self.image)

    def missing(self) -> list[str]:
        gaps: list[str] = []
        if not self.cli:
            gaps.append("the lean CLI")
        if not self.docker:
            gaps.append("the docker client")
        if self.docker and not self.server:
            gaps.append("a running docker engine")
        if self.server and not self.image:
            gaps.append(f"the {LEAN_IMAGE} image")
        return gaps


def _run(argv: Sequence[str]) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True,
                             timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or "").strip() if out.returncode == 0 else None


def _probe(docker: str) -> tuple[str | None, bool]:
    server = _run([docker, "info", "--format", "{{.ServerVersion}}"])
    if not server:
        return None, False
    return server, _run([docker, "image", "inspect", LEAN_IMAGE]) is not None


def status(which: Callable[[str], str | None] = shutil.which,
           probe: Callable[[str], tuple[str | None, bool]] = _probe,
           data_dir: Path | None = None) -> dict[str, Any]:
    cli = which("lean")
    docker = which("docker")
    server, image = probe(docker) if docker else (None, False)
    st = LeanStatus(cli=cli, docker=docker, server=server, image=image)
    return {"ready": st.ready, "cli": cli, "docker": docker,
            "engine": server, "image": image, "missing": st.missing(),
            "data_dir": str(data_dir or CFG.data_dir / "lean")}


def _clean(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        raise LeanError(f"no bars for {symbol} — nothing to export")
    absent = [c for c in COLUMNS if c not in frame.columns]
    if absent:
        raise LeanError(f"bars for {symbol} are missing {', '.join(absent)}")

    out = frame[list(COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)

    dupes = out["date"].duplicated()
    if dupes.any():
        when = out.loc[dupes, "date"].iloc[0].strftime("%Y-%m-%d")
        raise LeanError(f"bars for {symbol} repeat {when} — a cross-check "
                        f"cannot choose between two rows for one date")

    for col in COLUMNS[1:]:
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.isna() | ~values.map(lambda v: math.isfinite(float(v)))
        if bad.any():
            when = out.loc[bad, "date"].iloc[0].strftime("%Y-%m-%d")
            raise LeanError(f"bars for {symbol} carry a non-finite {col} on {when}")
        out[col] = values

    return out


def _csv_text(bars: pd.DataFrame) -> str:
    lines = [",".join(COLUMNS)]
    for row in bars.itertuples(index=False):
        day = row.date.strftime("%Y-%m-%d")
        nums = ",".join(f"{float(v):.12g}" for v in row[1:])
        lines.append(f"{day},{nums}")
    return "\n".join(lines) + "\n"


def export_bars(symbol: str, data_dir: Path | None = None,
                days: int | None = None,
                bars: pd.DataFrame | None = None) -> dict[str, Any]:
    """Write `<KEY>.csv` plus a manifest. The manifest carries the digest.

    `bars` is the seam for tests and for a caller that already holds history;
    without it the same `MarketData.history` door the engine uses is opened, so
    an exported bar is the bar the desk voted on, repaired range included.
    """
    inst = resolve(symbol)
    if inst is None:
        raise LeanError(f"unknown instrument {symbol!r}")

    if bars is None:
        from .data import MarketData
        bars = MarketData().history(inst, days=days)

    clean = _clean(bars, inst.key)
    text = _csv_text(clean)

    target = Path(data_dir or CFG.data_dir / "lean")
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / f"{inst.key}.csv"
    atomic_write(csv_path, text)
    # Hashed from the file, not the string: Windows writes CRLF, and a digest
    # that does not match the bytes on disk verifies nothing.
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    first = clean["date"].iloc[0].strftime("%Y-%m-%d")
    last = clean["date"].iloc[-1].strftime("%Y-%m-%d")
    manifest = {
        "symbol": inst.key,
        "name": inst.name,
        "market": inst.market,
        "asset_class": inst.asset_class,
        "currency": inst.currency,
        "rows": int(len(clean)),
        "first_date": first,
        "last_date": last,
        "columns": list(COLUMNS),
        "date_basis": "trading date as delivered by the provider; no intraday timestamps",
        "digest_sha256": digest,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "note": "bars only — this is not a backtest and says nothing about returns",
    }
    manifest_path = target / f"{inst.key}.manifest.json"
    atomic_write(manifest_path, dumps_strict(manifest, indent=2) + "\n")

    return {"symbol": inst.key, "rows": len(clean), "first": first, "last": last,
            "csv": str(csv_path), "manifest": str(manifest_path),
            "digest": digest}
