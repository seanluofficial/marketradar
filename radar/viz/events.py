"""Reference events for annotating the systemic-risk chart.

These are *labels on a plot*, nothing more. They are not used to fit, select or tune
anything -- every metric in the artifact is computed with no knowledge of them. The point
of drawing them is to let a viewer check the index against events they remember, which is
a weaker claim than "the index predicts crises" and the only one the data supports.

Dates are the commonly cited marker for each episode, not the start or end of it. A
252-day trailing window means the index responds over the months *following* a shock,
not on the day.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    date: str
    label: str
    note: str = ""


EVENTS: tuple[Event, ...] = (
    Event("2000-03-10", "Dot-com peak", "Nasdaq tops out; unwind runs into 2002"),
    Event("2001-09-11", "9/11", "US markets closed for four sessions"),
    Event("2002-10-09", "Bear market low", "End of the dot-com drawdown"),
    Event("2007-08-09", "Quant quake", "BNP freezes funds; first credit seizure"),
    Event("2008-09-15", "Lehman", "The canonical correlation-collapse event"),
    Event("2010-05-06", "Flash crash", "Intraday; barely visible in daily windows"),
    Event("2011-08-05", "US downgrade", "S&P cuts the US rating; euro crisis"),
    Event("2015-08-24", "China devaluation", "Renminbi shock, global selloff"),
    Event("2018-02-05", "Volmageddon", "Short-vol complex unwinds"),
    Event("2020-03-23", "COVID trough", "Fastest bear market on record"),
    Event("2022-06-16", "2022 rate shock", "Inflation repricing; bonds fail to hedge"),
    Event("2023-03-10", "SVB", "Regional bank failures"),
)


def events_between(start, end) -> tuple[Event, ...]:
    """Events falling inside a date range, for annotating a plotted window."""
    import pandas as pd

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return tuple(e for e in EVENTS if lo <= pd.Timestamp(e.date) <= hi)
