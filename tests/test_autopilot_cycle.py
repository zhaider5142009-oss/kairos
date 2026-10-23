"""The autopilot cycle, and the brake it applies to size.

This is the loop that trades. Until now nothing in the suite ran it: every test
stopped at `ENGINE.predict` or at `_execute`, so the twelve lines between them —
the gates, the sizing chain and the row that the desk renders — were only ever
exercised by the live paper daemon, where a mistake is witnessed as a shrunken
position with no explanation attached.

The defect that earned this file: a strategy member that warns "reduce size"
cuts `suggested_weight` by 30%, and the desk printed the cut weight with no
record of the cut. The reason existed only in `position_notes`, which is not one
of the keys this table reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import kairos.autopilot as AP
from kairos.ensemble import veto_size_multiplier
from kairos.execution import PaperBroker
from kairos.models_stat import ModelForecast
from test_risk_veto_channel import _vote, _weight_and_notes

UI = Path(__file__).resolve().parents[1] / "kairos" / "api" / "ui.html"


def _fresh_paper(root: Path):
    """A paper book that belongs to this test, not to the session's data dir.

    `PaperBroker` reads `CFG.data_dir` in its own constructor, so a plain
    subclass cannot redirect it in time: the shared book from an earlier test
    would be loaded, sized against, and then quietly overwritten.
    """
    class _Paper(PaperBroker):
        def __init__(self) -> None:
            super().__init__()
            self.state_path = root / "paper_account.json"
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.capital = self.cash = 1_000_000.0
            self.positions, self.fills = {}, []
            self.realized_pnl = self.day_pnl = 0.0
            self._day_stamp = ""

    return _Paper


def _run_cycle(monkeypatch, tmp_path, votes, name, symbol="GOLD", battery=()):
    """One real paper cycle over one real engine prediction.

    `res` comes from the engine itself (`test_risk_veto_channel._engine`), with
    only its data feed, its model battery and its strategy library stubbed, so
    the `signal` block the desk row is built from is the block the engine
    actually journals. The broker, the state file and the risk posture are the
    production ones.
    """
    res = _weight_and_notes(monkeypatch, votes, battery)[2]
    books: list[PaperBroker] = []

    class _Recording(_fresh_paper(tmp_path / name)):
        def __init__(self) -> None:
            super().__init__()
            books.append(self)

    ap = AP.Autopilot()
    ap.state_path = tmp_path / f"{name}.autopilot.json"
    ap.st = ap._defaults()
    ap.st.update({"symbols": [symbol], "last_rotation": AP._now()})
    from kairos.engine import ENGINE
    monkeypatch.setattr(ENGINE, "predict", lambda key, horizon=None, **kw: res)
    monkeypatch.setattr(ENGINE, "refresh_news", lambda watch: {})
    monkeypatch.setattr(AP, "PaperBroker", _Recording)
    out = ap.cycle()
    return ap, res, books[0], out


_BULL = [_vote(1, 0.9, "Bull")]
_WARN = _BULL + [_vote(0, 0.6, "Amihud Illiquidity Cost")]
# Three models that concur on a 3% five-day move: enough R:R for the FUGU
# conductor to approve the entry, so the cycle below reaches the order rather
# than stopping at the gate. The same battery is used on both sides of the
# comparison, which is what makes the quantities comparable.
_AGREE = [ModelForecast(f"Model {i}", 1, 0.03, 5, 0.8) for i in range(3)]


# ------------------------------------------------------------------ the desk row
def test_the_desk_row_carries_the_brake_that_cut_it(monkeypatch, tmp_path):
    plain, _, _, _ = _run_cycle(monkeypatch, tmp_path, _BULL, "plain")
    braked, _, _, _ = _run_cycle(monkeypatch, tmp_path, _WARN, "braked")
    prow, row = plain.st["signals"][0], braked.st["signals"][0]
    assert row["risk_veto"] == 0.6
    assert row["veto_by"] == "Amihud Illiquidity Cost"
    assert row["veto_x"] == pytest.approx(veto_size_multiplier(0.6))
    # the number the operator compares is the number the brake produced
    assert row["weight"] < prow["weight"]
    assert row["weight"] == pytest.approx(prow["weight"] * row["veto_x"], abs=2e-4)
    assert prow["risk_veto"] == 0.0


def test_an_unbraked_row_shows_no_phantom_brake(monkeypatch, tmp_path):
    """A default of `1.0` printed as "veto ×1.00" would train the operator to
    ignore the label, which is worse than the gap it fills."""
    ap, _, _, _ = _run_cycle(monkeypatch, tmp_path, _BULL, "clean")
    row = ap.st["signals"][0]
    assert row["risk_veto"] == 0.0 and row["veto_by"] == ""
    assert row["veto_x"] == 1.0


# -------------------------------------------------------------------- the order
def test_the_brake_reaches_the_order_not_only_the_display(monkeypatch, tmp_path):
    plain, pres, pbook, pout = _run_cycle(monkeypatch, tmp_path, _BULL, "o_plain",
                                          battery=_AGREE)
    braked, bres, bbox, bout = _run_cycle(monkeypatch, tmp_path, _WARN, "o_braked",
                                          battery=_AGREE)
    assert pout["trades"] == 1 and bout["trades"] == 1, (
        "no entry fired, so this compared two zeroes — "
        f"plain={plain.st['signals'][0]['action']} "
        f"braked={braked.st['signals'][0]['action']}")
    pq = pbook.positions["GOLD"]["qty"]
    bq = bbox.positions["GOLD"]["qty"]
    assert bq < pq
    # Both cycles price off the same frame and the same Rs 1,000,000, so the
    # only thing that may separate two quantities is the 0.7 brake.
    assert bq == pytest.approx(pq * veto_size_multiplier(0.6), rel=1e-3), (
        f"plain qty {pq}, braked qty {bq}: the disclosed factor does not "
        f"describe the order that was actually placed")
    assert pres["risk"]["suggested_weight"] != bres["risk"]["suggested_weight"]


# ------------------------------------------------------------------- the render
def _weight_cell() -> str:
    html = UI.read_text(encoding="utf-8")
    m = re.search(r'<td class="r mono">\$\{pct\(s\.weight,1\)\}(.*?)</td>',
                  html, re.S)
    assert m, "the Weight cell of the autopilot table was restructured; " \
              "the contract below cannot be checked until it names its keys again"
    return m.group(1)


def test_the_desk_template_reads_only_keys_the_row_sends(monkeypatch, tmp_path):
    """`ui.html` is a template string, so a renamed field in the row dict is not
    an error — it is a `undefined` that the conditional swallows and the brake
    stops appearing with no test failing anywhere."""
    cell = _weight_cell()
    read = set(re.findall(r"s\.([A-Za-z_][A-Za-z_0-9]*)", cell))
    assert read, "the Weight cell reads no fields at all"
    ap, _, _, _ = _run_cycle(monkeypatch, tmp_path, _WARN, "contract")
    sent = set(ap.st["signals"][0])
    missing = read - sent
    assert not missing, (
        f"the desk asks for {sorted(missing)}, which the cycle does not send")
    assert "veto_x" in cell and "risk_veto" in cell, (
        "the Weight cell no longer renders the brake it is sized by")
