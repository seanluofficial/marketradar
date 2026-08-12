"""Price panels and return construction.

Alignment is where correlation pipelines quietly go wrong: an unaligned join produces
NaNs, `df.corr()` drops them pairwise, and you end up with a correlation matrix whose
entries were each estimated on a different sample and which need not be positive
semi-definite. Nothing downstream would tell you.

So this module is strict and loud:
  * a ticker without history covering the whole requested range is *dropped*, by name,
    with a reason -- never silently carried with a ragged head;
  * sporadic gaps (halts, vendor misses) are forward-filled on *prices* up to a small
    limit, which is equivalent to assigning a zero return to the missing day, and the
    count is reported;
  * anything still missing is an error, not a shrug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from radar.data import cache
from radar.data.universe import Universe, get_universe

#: Sporadic gaps longer than this are treated as missing history, not a halt.
MAX_FFILL_DAYS = 3

#: A ticker must have data starting within this many calendar days of the requested
#: start to count as covering the range (absorbs holidays and listing-day slop).
START_TOLERANCE_DAYS = 7


@dataclass
class PanelReport:
    """What the panel actually contains, and what it silently would have hidden."""

    requested: tuple[str, ...]
    retained: tuple[str, ...]
    dropped: dict[str, str] = field(default_factory=dict)
    start: date | None = None
    end: date | None = None
    n_obs: int = 0
    filled_values: int = 0

    @property
    def n_assets(self) -> int:
        return len(self.retained)

    def q(self, window: int) -> float:
        """N/T for a given window -- the number that decides whether the sample
        correlation matrix is usable at all. q >= 1 means singular."""
        return self.n_assets / float(window)

    def summary(self) -> str:
        lines = [
            f"{self.n_assets}/{len(self.requested)} tickers retained, "
            f"{self.n_obs} observations {self.start} -> {self.end}",
        ]
        if self.filled_values:
            lines.append(f"  forward-filled {self.filled_values} missing price cells")
        if self.dropped:
            lines.append(f"  dropped {len(self.dropped)}:")
            for ticker, reason in sorted(self.dropped.items()):
                lines.append(f"    {ticker:<6} {reason}")
        return "\n".join(lines)


def _resolve(universe: Universe | str | None, tickers: list[str] | None) -> list[str]:
    if tickers is not None:
        return list(tickers)
    if universe is None:
        raise ValueError("Provide either `universe` or `tickers`.")
    uni = get_universe(universe) if isinstance(universe, str) else universe
    return list(uni.tickers)


def price_panel(
    universe: Universe | str | None = None,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    field_name: str = "adj_close",
) -> tuple[pd.DataFrame, PanelReport]:
    """Aligned price panel (dates x tickers) read entirely from the disk cache.

    `adj_close` is the default and the right choice: it is split- and dividend-adjusted,
    so returns are total returns and are not corrupted by a 2-for-1 split showing up as
    a -50% day.

    When `start` is omitted the required range is inferred from the earliest date any
    requested ticker has, so the deepest-history name sets the bar and everything
    younger is dropped. That is deliberate -- it makes the history/breadth trade-off
    visible rather than silent -- but it means you usually want to pass `start`
    explicitly. `coverage_frontier()` shows what each choice costs in assets.
    """
    names = _resolve(universe, tickers)
    report = PanelReport(requested=tuple(names), retained=())

    series: dict[str, pd.Series] = {}
    for ticker in names:
        df = cache.read_eod(ticker)
        if df is None:
            report.dropped[ticker] = "not cached (run `radar fetch`)"
            continue
        if field_name not in df.columns:
            report.dropped[ticker] = f"no {field_name} column in cache"
            continue
        s = df[field_name].dropna()
        if s.empty:
            report.dropped[ticker] = "cached but empty"
            continue
        series[ticker] = s

    if not series:
        return pd.DataFrame(), report

    panel = pd.DataFrame(series).sort_index()

    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    if panel.empty:
        return panel, report

    tolerance = pd.Timedelta(days=START_TOLERANCE_DAYS)

    # Drop names whose history begins meaningfully after the panel does. Doing this
    # before any filling is what keeps a 2013-listed name from being back-filled into
    # the 2008 crisis with zero returns.
    panel_start = panel.index[0]
    late = []
    for ticker in panel.columns:
        first_valid = panel[ticker].first_valid_index()
        if first_valid is None:
            report.dropped[ticker] = "no data in requested range"
            late.append(ticker)
        elif first_valid > panel_start + tolerance:
            report.dropped[ticker] = (
                f"history starts {first_valid.date()}, after range start "
                f"{panel_start.date()}"
            )
            late.append(ticker)
    panel = panel.drop(columns=late)

    # Re-derive the calendar from the *surviving* columns before checking the tail.
    # Without this, dropping a late-listed name leaves its dates in the index and the
    # long-history names it was compared against then look delisted.
    panel = panel.dropna(how="all")
    if panel.empty or panel.shape[1] == 0:
        return panel, report

    # Trailing gaps mean the name stopped trading (delisting, acquisition) -- also a
    # coverage failure, and one that ffill would happily paper over with flat prices.
    panel_end = panel.index[-1]
    stale = []
    for ticker in panel.columns:
        last_valid = panel[ticker].last_valid_index()
        if last_valid is not None and last_valid < panel_end - tolerance:
            report.dropped[ticker] = f"history ends {last_valid.date()}, before {panel_end.date()}"
            stale.append(ticker)
    panel = panel.drop(columns=stale)

    panel = panel.dropna(how="all")
    if panel.empty or panel.shape[1] == 0:
        return panel, report

    # Last genuinely traded session per name, captured before filling -- afterwards the
    # forward fill has moved each column's last valid index forward by up to MAX_FFILL_DAYS.
    last_traded = {t: panel[t].last_valid_index() for t in panel.columns}

    missing_before = int(panel.isna().sum().sum())
    panel = panel.ffill(limit=MAX_FFILL_DAYS)
    report.filled_values = missing_before - int(panel.isna().sum().sum())

    panel_end = panel.index[-1]
    residual = panel.columns[panel.isna().any()].tolist()
    for ticker in residual:
        col = panel[ticker]
        gaps = int(col.isna().sum())
        last_valid = col.last_valid_index()
        # Distinguish a name that went dark from one with holes in the middle. A trailing
        # NaN run that survived the fill is a delisting or a long halt, and calling that
        # "unfillable gaps" would bury the interesting fact. Requires the run to be
        # non-empty *and* to account for every remaining gap.
        trailing = 0 if last_valid is None else int((panel.index > last_valid).sum())
        if trailing > 0 and trailing == gaps:
            went_dark = last_traded[ticker] or last_valid
            report.dropped[ticker] = (
                f"no data after {went_dark.date()} (delisting or halt); "
                f"{gaps} missing sessions to {panel_end.date()}"
            )
        else:
            report.dropped[ticker] = f"{gaps} unfillable gaps (> {MAX_FFILL_DAYS}d)"
    panel = panel.drop(columns=residual)

    report.retained = tuple(panel.columns)
    report.n_obs = len(panel)
    if not panel.empty:
        report.start = panel.index[0].date()
        report.end = panel.index[-1].date()
    return panel, report


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, first row dropped.

    Log rather than simple returns: they are additive across time, closer to symmetric,
    and correlation of log returns is the quantity the Mantegna distance metric assumes.
    Over daily horizons the two are numerically almost identical anyway.
    """
    if prices.empty:
        return prices
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices in {bad}; cannot take log returns.")
    return np.log(prices).diff().iloc[1:]


def returns_panel(
    universe: Universe | str | None = None,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, PanelReport]:
    """The pipeline entry point: cache -> aligned prices -> log returns.

    The returned frame is guaranteed rectangular and NaN-free, so every downstream
    correlation is estimated on one common sample.
    """
    prices, report = price_panel(universe=universe, tickers=tickers, start=start, end=end)
    if prices.empty:
        return prices, report

    rets = log_returns(prices)
    report.n_obs = len(rets)
    if not rets.empty:
        report.start = rets.index[0].date()
        report.end = rets.index[-1].date()

    assert not rets.isna().any().any(), "returns panel must be NaN-free"
    return rets, report


def coverage_frontier(
    universe: Universe | str = "core_equity",
    candidate_starts: tuple[str, ...] = ("1998-01-01", "2000-01-01", "2005-01-01", "2010-01-01"),
) -> pd.DataFrame:
    """How many names survive each candidate start date.

    The history/breadth trade-off made explicit: an earlier start buys more crises but
    costs assets, and N is exactly what drives estimation error.
    """
    rows = []
    for s in candidate_starts:
        _, report = price_panel(universe=universe, start=pd.Timestamp(s).date())
        rows.append(
            {
                "start": s,
                "n_assets": report.n_assets,
                "n_obs": report.n_obs,
                "q_at_252": round(report.q(252), 3) if report.n_assets else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("start")
