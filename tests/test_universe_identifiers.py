"""A KAIROS key is a stable handle; the exchange ticker is what feeds answer to.

They are not always the same string. PSX renamed ENGRO to ENGROH, and the
board spells Bank of Punjab BOP and Nestle NESTLE. Deriving every venue symbol
from the key meant those names were fetched under symbols that do not exist,
so they never got a bar — and a ticket would have carried the wrong script.
"""

from __future__ import annotations

import pandas as pd

from kairos import data as D
from kairos import universe as U
from kairos.universe import resolve


def test_every_psx_instrument_agrees_with_its_declared_ticker():
    for i in U.psx_universe():
        assert i.psx_ticker, i.key
        assert i.tv == f"PSX:{i.psx_ticker}", i.key
        assert resolve(i.psx_ticker) is i, i.key
        if i.yahoo:
            assert i.yahoo == f"{i.psx_ticker}.KA", i.key


def test_no_two_psx_instruments_share_a_ticker():
    # a duplicate would silently hand one name's store file to another
    ticks = [i.psx_ticker for i in U.psx_universe()]
    assert len(ticks) == len(set(ticks))


def test_a_full_tv_symbol_resolves_the_way_the_docstring_promised():
    assert resolve("PSX:OGDC") is resolve("OGDC")
    assert resolve("psx:ogdc") is resolve("OGDC")


def test_renamed_and_respelled_names_point_at_the_exchange_symbol():
    assert (resolve("ENGRO").key, resolve("ENGRO").tv) == ("ENGRO", "PSX:ENGROH")
    assert resolve("ENGROH") is resolve("ENGRO")
    assert resolve("BOP") is resolve("BANK")
    assert resolve("NESTLE") is resolve("NESP")


def test_the_board_sweep_writes_into_the_file_history_opens(tmp_path):
    st = D.Store(tmp_path)
    df = pd.DataFrame({
        "name": ["ENGROH", "BOP", "NESTLE", "ZZZZ"],
        "close": [245.46, 27.3, 7044.78, 5.0],
        "change": [1.0, 1.0, 1.0, 1.0],
        "volume": [1000.0, 2000.0, 30.0, 400.0],
    })
    D._accumulate_bars(st, df, "2026-10-24")

    for key, board in (("ENGRO", "ENGROH"), ("BANK", "BOP"), ("NESP", "NESTLE")):
        bar = st.load(key)
        assert bar is not None, f"{board} never reached {key}'s store file"
        assert float(bar["close"].iloc[-1]) == float(
            df.loc[df["name"] == board, "close"].iloc[0])
        assert st.load(board) is None, f"{board} opened a second store file"
    # a board name outside the registry keeps the only key it has
    assert st.load("ZZZZ") is not None
