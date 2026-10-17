"""Economic calendar awareness — high-impact event windows that gate risk.

Ported from the MiniMax v1 calendar module: FOMC decision days, US NFP
(first Friday), and SBP Monetary Policy Committee windows. When a decision
lands within the blackout window, the engine reduces position multipliers
and flags event risk in every dossier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

FOMC_2026 = [  # decision days (second day of each meeting)
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
]
SBP_MPC_2026 = [  # approximate MPC announcement windows (bi-monthly)
    date(2026, 1, 26), date(2026, 3, 23), date(2026, 5, 25),
    date(2026, 7, 27), date(2026, 9, 21), date(2026, 11, 24),
]
CPI_US_2026 = [date(2026, m, 12) for m in range(1, 13)]
BLACKOUT_DAYS_BEFORE = 1
BLACKOUT_DAYS_AFTER = 1


@dataclass
class EventWindow:
    name: str
    event_date: date
    starts: date
    ends: date
    severity: str                    # critical | high


def _nfp(year: int) -> list[date]:
    out = []
    for m in range(1, 13):
        d = date(year, m, 1)
        first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
        out.append(first_friday)
    return out


def upcoming_events(today: date | None = None, horizon_days: int = 10) -> list[EventWindow]:
    today = today or datetime.now().date()
    events: list[EventWindow] = []
    sources = ([("FOMC Rate Decision", d, "critical") for d in FOMC_2026] +
               [("US Non-Farm Payrolls", d, "high") for d in _nfp(today.year)] +
               [("US CPI", d, "high") for d in CPI_US_2026] +
               [("SBP Monetary Policy", d, "critical") for d in SBP_MPC_2026])
    for name, d, sev in sources:
        w = EventWindow(name, d, d - timedelta(days=BLACKOUT_DAYS_BEFORE),
                        d + timedelta(days=BLACKOUT_DAYS_AFTER), sev)
        if today <= w.ends and w.starts <= today + timedelta(days=horizon_days):
            events.append(w)
    return sorted(events, key=lambda w: w.event_date)


def risk_calendar(today: date | None = None) -> dict[str, Any]:
    today = today or datetime.now().date()
    events = upcoming_events(today)
    active = [w for w in events if w.starts <= today <= w.ends]
    upcoming = [w for w in events if w.starts > today][:5]
    multiplier = 1.0
    if any(w.severity == "critical" for w in active):
        multiplier = 0.5
    elif active:
        multiplier = 0.7
    return {
        "today": str(today),
        "in_blackout": bool(active),
        "active_events": [{"name": w.name, "date": str(w.event_date),
                           "severity": w.severity} for w in active],
        "upcoming": [{"name": w.name, "date": str(w.event_date),
                      "severity": w.severity} for w in upcoming],
        "position_multiplier": multiplier,
    }
