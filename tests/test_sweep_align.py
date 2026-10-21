"""Refreshing the copied sweep table is a rewrite, and a rewrite can lie.

`tools/kit_stats.py --write` used to maintain only the figures standing next to a
figure word ("259 passing tests"), which left the walk-forward sweep table — the
part an investor actually reads — hand-copied into five layouts. The first run of
the checker over a refreshed sweep found 61 stale sites, and fixing them by hand
would have been the same mistake with more effort.

So the sweep figures are rewritten from the generated ledger now. That moves the
risk from "nobody noticed" to "something was written wrongly", and a rewrite of
trading evidence deserves its own tests:

* a sign that flips has to arrive as a negative number, in the glyph the document
  already uses, with the landing page's green/red class following it;
* a Sharpe in a column has no word around it, so it cannot borrow the prose
  exemption for "loses 2.8% mean", which must stay unsigned to stay true;
* two instrument names in one window must not be charged to the wrong one;
* a magnitude that fits two columns is left alone rather than guessed at;
* running it twice must change nothing the first run did not need to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import kit_stats  # noqa: E402  (the aligner is the thing under test)

# GOLD is the instrument whose whole row turns over; SILVER moves a little; WHEAT
# is built so that its old net return and its old drawdown are the same digits,
# which is the case the aligner must refuse to guess about.
OLD = {
    "GOLD": {"verdict": "profitable", "return": "+10.0%", "benchmark": "+2.0%",
             "sharpe": "1.50", "drawdown": "-5.0%"},
    "SILVER": {"verdict": "profitable", "return": "+44.5%", "benchmark": "+34.5%",
               "sharpe": "1.46", "drawdown": "-21.8%"},
    "WHEAT": {"verdict": "unprofitable", "return": "-5.0%", "benchmark": "+3.0%",
              "sharpe": "0.20", "drawdown": "-5.0%"},
}
NEW = {
    "GOLD": {"verdict": "unprofitable", "return": "-9.4%", "benchmark": "+58.4%",
             "sharpe": "-0.69", "drawdown": "-15.1%"},
    "SILVER": {"verdict": "profitable", "return": "+43.5%", "benchmark": "+36.1%",
               "sharpe": "1.44", "drawdown": "-22.3%"},
    "WHEAT": {"verdict": "unprofitable", "return": "-12.0%", "benchmark": "+3.0%",
              "sharpe": "0.11", "drawdown": "-7.5%"},
}


def _ctx(html: bool = False) -> dict:
    return {"old": OLD, "new": NEW, "names": sorted(NEW, key=len, reverse=True),
            "old_means": {"Global": "+13.6", "PSX": "-2.8"},
            "new_means": {"Global": "+10.0", "PSX": "-2.2"},
            "wins": "1", "total": "3", "losers": "2", "html": html}


def _align(text: str, html: bool = False) -> tuple[str, int]:
    return kit_stats._align_body(text, _ctx(html))


def test_a_turned_over_row_arrives_whole():
    """Net, benchmark, Sharpe and the verdict word, in one markdown row."""
    out, n = _align("| GOLD | profitable | +10.0% | +2.0% | 1.50 |")
    assert out == "| GOLD | unprofitable | -9.4% | +58.4% | -0.69 |", out
    assert n == 4, n


def test_a_negative_written_into_a_winners_row_stays_negative():
    """The sign is the finding, so it cannot be dropped to keep the old shape.

    The site read "+10.0%" and the ledger now says "-9.4%": keeping the site's
    plus would have published a profit that no longer exists.
    """
    out, _n = _align("GOLD +10.0% on the year")
    assert out == "GOLD -9.4% on the year", out


def test_the_landing_page_gets_its_own_minus_and_its_own_colour():
    """A −9.4% sitting in a cell still marked `class="up"` is a loss in green."""
    out, _n = _align('<tr><td>GOLD</td><td class="up">+10.0%</td></tr>', html=True)
    assert out == '<tr><td>GOLD</td><td class="dn">\u22129.4%</td></tr>', out


def test_an_unsigned_prose_mean_stays_unsigned():
    """"The equity book loses 2.8% mean" says the direction in words.

    Rewriting it to "loses -2.2%" would double-state the sign, and rewriting the
    one with a sign in it ("−2.8% mean") must keep the sign.
    """
    assert _align("the PSX book loses 2.8% mean")[0] == "the PSX book loses 2.2% mean"
    assert _align("\u22122.8% mean")[0] == "\u22122.2% mean"


def test_a_positive_value_does_not_grow_a_plus_the_column_never_used():
    """SILVER's Sharpe is 1.44 in a column of bare numbers, not +1.44."""
    out, _n = _align("| SILVER | +44.5% | +34.5% | 1.46 |")
    assert out == "| SILVER | +43.5% | +36.1% | 1.44 |", out


def test_the_nearest_name_owns_the_figure():
    """Two instruments in one sentence, each with its own return.

    Reading attribution by name length charged GOLD's return to SILVER, which left
    the deck's busiest line neither checked nor correct — it sat on a stale
    +28.7% while the guard reported the pack clean.
    """
    out, n = _align("SILVER **+44.5%** · GOLD **+10.0%** (vs +2.0% buy-and-hold)")
    assert out == "SILVER **+43.5%** · GOLD **-9.4%** (vs +58.4% buy-and-hold)", out
    assert n == 3, n


def test_an_ambiguous_column_is_left_for_a_human():
    """Old return and old drawdown with the same digits: which one was copied?

    Rewriting on a coin flip would move a figure the pack never quoted, so the
    site is untouched and the checker is left to name it.
    """
    out, n = _align("| WHEAT | -5.0% | +3.0% | 0.20 |")
    assert out.startswith("| WHEAT | -5.0% |"), out
    assert n == 1, n          # only the Sharpe, which has one column to come from


def test_the_headline_and_the_loser_count_move_together():
    """"8 of 14 profitable" and "the other 6" are one fact stated twice."""
    out, _n = _align("8 of 14 instruments profitable, and the other 6 unprofitable.")
    assert out == "1 of 3 instruments profitable, and the other 2 unprofitable.", out


def test_a_ratio_counts_the_side_its_nearest_word_names():
    """The sweep count leaves the word "profitable" a sentence's worth of room.

    Three of the pack's own sentences state the headline without gluing the
    descriptor to the numbers — "profitable on 8 of 14 instruments", "6 of 14
    instruments in the walk-forward sweep lost money", "8 of 14 instruments came
    out net-positive". The first rule matched only the glued form, so those three
    were neither rewritten nor checked, and two of them shipped a winner count
    that the ledger no longer printed. A ratio is also not the only pair of
    numbers in a page: "24/7" and "5 / 8 GB" must survive the rewrite untouched.
    """
    for src, want in [
        ("profitable on 8 of 14 instruments.", "profitable on 1 of 3 instruments."),
        ("8 of 14 instruments came out net-positive; the PSX book negative.",
         "1 of 3 instruments came out net-positive; the PSX book negative."),
        ("6 of 14 instruments in the walk-forward sweep lost money.",
         "2 of 3 instruments in the walk-forward sweep lost money."),
        ("Runs 24/7 on a 5 / 8 GB machine.", "Runs 24/7 on a 5 / 8 GB machine."),
    ]:
        out, _n = _align(src)
        assert out == want, (src, out)


def test_a_second_pass_has_nothing_left_to_change():
    """Idempotence: the rewritten pack is the ledger's own text, not a new drift."""
    once, _n = _align("| GOLD | profitable | +10.0% | +2.0% | 1.50 |")
    twice, n2 = _align(once)
    assert twice == once and n2 == 0, (twice, n2)
