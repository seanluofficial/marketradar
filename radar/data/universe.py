"""Fixed, hand-pinned universes.

Deliberately *fixed* rather than reconstructed from index membership: we are not
attempting point-in-time constituents, so the honest framing is "a fixed basket
of names that survived to today," not "the S&P 100 as it was in 2008."

Direction of the survivorship bias matters and is easy to state backwards:
because every member survived, the basket is biased toward *resilient* names.
Crisis-period correlation collapse measured here is therefore an **understatement**
of what the full cross-section experienced. See SURVIVORSHIP_NOTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

SURVIVORSHIP_NOTE = (
    "Universes are fixed baskets of currently-listed names, not point-in-time index "
    "constituents. Every member survived to the present, so the basket is biased toward "
    "resilient firms: failed and acquired names (Lehman, Bear Stearns, Wachovia, Enron, "
    "GM's pre-2009 listing) are absent. The practical consequence is that crisis-period "
    "correlation collapse shown here is an understatement, not an overstatement."
)


@dataclass(frozen=True)
class Member:
    ticker: str
    name: str
    group: str


@dataclass(frozen=True)
class Universe:
    key: str
    title: str
    description: str
    group_label: str
    members: tuple[Member, ...]
    caveats: str = ""

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(m.ticker for m in self.members)

    @property
    def groups(self) -> Mapping[str, str]:
        """ticker -> group label (GICS sector, or asset class for cross-asset)."""
        return {m.ticker: m.group for m in self.members}

    @property
    def names(self) -> Mapping[str, str]:
        return {m.ticker: m.name for m in self.members}

    def __len__(self) -> int:
        return len(self.members)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": m.ticker, "name": m.name, "group": m.group} for m in self.members]
        ).set_index("ticker")

    def subset(self, tickers: Sequence[str]) -> "Universe":
        """Restrict to `tickers`, preserving this universe's ordering and metadata."""
        keep = set(tickers)
        members = tuple(m for m in self.members if m.ticker in keep)
        return Universe(
            key=self.key,
            title=self.title,
            description=self.description,
            group_label=self.group_label,
            members=members,
            caveats=self.caveats,
        )


def _m(group: str, entries: Sequence[tuple[str, str]]) -> list[Member]:
    return [Member(ticker=t, name=n, group=group) for t, n in entries]


# --------------------------------------------------------------------------------------
# core_equity: 81 US large caps across all 11 GICS sectors.
#
# Sizing note (the N vs T decision, made once and stated everywhere):
#   N = 81. Default window T = 252 trading days  ->  q = N/T ~ 0.32.
# That q is squarely in the regime where the sample correlation matrix is badly noisy
# but not singular -- which is precisely the regime Marchenko-Pastur eigenvalue
# clipping and Ledoit-Wolf shrinkage are built for, so the estimator toggle visibly
# changes the network. A 90-day window gives q ~ 0.32*2.8 = 0.90, effectively rank
# deficient; it is available but the app warns loudly.
# --------------------------------------------------------------------------------------
_CORE_EQUITY: list[Member] = [
    *_m(
        "Information Technology",
        [
            ("AAPL", "Apple"),
            ("MSFT", "Microsoft"),
            ("NVDA", "NVIDIA"),
            ("INTC", "Intel"),
            ("CSCO", "Cisco Systems"),
            ("ORCL", "Oracle"),
            ("IBM", "IBM"),
            ("TXN", "Texas Instruments"),
            ("ADBE", "Adobe"),
            ("QCOM", "Qualcomm"),
            ("AMAT", "Applied Materials"),
            ("AMD", "Advanced Micro Devices"),
        ],
    ),
    *_m(
        "Health Care",
        [
            ("JNJ", "Johnson & Johnson"),
            ("PFE", "Pfizer"),
            ("MRK", "Merck"),
            ("ABT", "Abbott Laboratories"),
            ("BMY", "Bristol-Myers Squibb"),
            ("LLY", "Eli Lilly"),
            ("AMGN", "Amgen"),
            ("UNH", "UnitedHealth"),
            ("MDT", "Medtronic"),
        ],
    ),
    *_m(
        "Financials",
        [
            ("JPM", "JPMorgan Chase"),
            ("BAC", "Bank of America"),
            ("WFC", "Wells Fargo"),
            ("C", "Citigroup"),
            ("GS", "Goldman Sachs"),
            ("MS", "Morgan Stanley"),
            ("AXP", "American Express"),
            ("USB", "U.S. Bancorp"),
            ("PNC", "PNC Financial"),
        ],
    ),
    *_m(
        "Consumer Discretionary",
        [
            ("AMZN", "Amazon"),
            ("HD", "Home Depot"),
            ("MCD", "McDonald's"),
            ("NKE", "Nike"),
            ("LOW", "Lowe's"),
            ("TGT", "Target"),
            ("SBUX", "Starbucks"),
            ("TJX", "TJX Companies"),
        ],
    ),
    *_m(
        "Consumer Staples",
        [
            ("PG", "Procter & Gamble"),
            ("KO", "Coca-Cola"),
            ("PEP", "PepsiCo"),
            ("WMT", "Walmart"),
            ("CL", "Colgate-Palmolive"),
            ("KMB", "Kimberly-Clark"),
            ("COST", "Costco"),
        ],
    ),
    *_m(
        "Industrials",
        [
            ("CAT", "Caterpillar"),
            ("DE", "Deere"),
            ("HON", "Honeywell"),
            ("MMM", "3M"),
            ("GE", "GE Aerospace"),
            ("BA", "Boeing"),
            ("LMT", "Lockheed Martin"),
            ("UPS", "United Parcel Service"),
            ("UNP", "Union Pacific"),
        ],
    ),
    *_m(
        "Energy",
        [
            ("XOM", "Exxon Mobil"),
            ("CVX", "Chevron"),
            ("COP", "ConocoPhillips"),
            ("SLB", "SLB"),
            ("EOG", "EOG Resources"),
            ("OXY", "Occidental Petroleum"),
            ("HAL", "Halliburton"),
        ],
    ),
    *_m(
        "Materials",
        [
            ("APD", "Air Products"),
            ("SHW", "Sherwin-Williams"),
            ("NUE", "Nucor"),
            ("ECL", "Ecolab"),
            ("PPG", "PPG Industries"),
        ],
    ),
    *_m(
        "Utilities",
        [
            ("NEE", "NextEra Energy"),
            ("DUK", "Duke Energy"),
            ("SO", "Southern Company"),
            ("D", "Dominion Energy"),
            ("AEP", "American Electric Power"),
            ("EXC", "Exelon"),
        ],
    ),
    *_m(
        "Real Estate",
        [
            ("SPG", "Simon Property Group"),
            ("PSA", "Public Storage"),
            ("O", "Realty Income"),
            ("AVB", "AvalonBay Communities"),
        ],
    ),
    *_m(
        "Communication Services",
        [
            ("T", "AT&T"),
            ("VZ", "Verizon"),
            ("DIS", "Walt Disney"),
            ("CMCSA", "Comcast"),
            ("EA", "Electronic Arts"),
        ],
    ),
]


_CROSS_ASSET: list[Member] = [
    *_m(
        "Equity",
        [
            ("SPY", "S&P 500"),
            ("QQQ", "Nasdaq 100"),
            ("IWM", "US small cap"),
            ("EFA", "Developed ex-US"),
            ("EEM", "Emerging markets"),
            ("VGK", "Europe"),
            ("EWJ", "Japan"),
        ],
    ),
    *_m(
        "Rates",
        [
            ("TLT", "US Treasury 20y+"),
            ("IEF", "US Treasury 7-10y"),
            ("SHY", "US Treasury 1-3y"),
            ("TIP", "US TIPS"),
        ],
    ),
    *_m(
        "Credit",
        [
            ("LQD", "US investment grade"),
            ("HYG", "US high yield"),
            ("EMB", "EM sovereign USD"),
        ],
    ),
    *_m(
        "Commodities",
        [
            ("GLD", "Gold"),
            ("SLV", "Silver"),
            ("DBC", "Broad commodities"),
            ("USO", "Crude oil"),
        ],
    ),
    *_m("Real assets", [("VNQ", "US REITs")]),
    *_m("Currency", [("UUP", "US dollar index")]),
]


_SECTOR_ETFS: list[Member] = [
    *_m(
        "Sector ETF",
        [
            ("XLB", "Materials"),
            ("XLE", "Energy"),
            ("XLF", "Financials"),
            ("XLI", "Industrials"),
            ("XLK", "Technology"),
            ("XLP", "Consumer Staples"),
            ("XLU", "Utilities"),
            ("XLV", "Health Care"),
            ("XLY", "Consumer Discretionary"),
            ("XLRE", "Real Estate"),
            ("XLC", "Communication Services"),
        ],
    ),
]


UNIVERSES: dict[str, Universe] = {
    u.key: u
    for u in (
        Universe(
            key="core_equity",
            title="US large caps (81 names, 11 sectors)",
            description=(
                "Fixed basket of US large caps spanning all 11 GICS sectors. The primary "
                "universe: deep history through 2000, 2008, 2011, 2015-16, 2020 and 2022, "
                "and enough names for sector structure to emerge without labels."
            ),
            group_label="GICS sector",
            members=tuple(_CORE_EQUITY),
            caveats=SURVIVORSHIP_NOTE,
        ),
        Universe(
            key="cross_asset",
            title="Cross-asset (20 ETFs)",
            description=(
                "Equity, rates, credit, commodities, real assets and the dollar. Used for the "
                "flight-to-safety view: in calm periods rates sit far from equity on the tree; "
                "in stress the credit and equity branches fuse and Treasuries detach."
            ),
            group_label="Asset class",
            members=tuple(_CROSS_ASSET),
            caveats=(
                "History is limited by the youngest fund (EMB and UUP both launched 2007), so "
                "the usable common window starts around 2007-04 and excludes 2000-02. ETF "
                "returns are NAV-tracking proxies, not the underlying asset."
            ),
        ),
        Universe(
            key="sector_etfs",
            title="GICS sector ETFs (11)",
            description=(
                "One ETF per GICS sector. Small and fast: useful as a sanity check that the "
                "pipeline recovers known sector relationships, and cheap enough to iterate on."
            ),
            group_label="Sector ETF",
            members=tuple(_SECTOR_ETFS),
            caveats=(
                "XLRE launched 2015 and XLC 2018; before those dates the universe silently has "
                "9 members. N=11 is far below any usable window, so shrinkage barely matters "
                "here -- that is the point of including it as a control."
            ),
        ),
    )
}

DEFAULT_UNIVERSE = "core_equity"


def get_universe(key: str = DEFAULT_UNIVERSE) -> Universe:
    try:
        return UNIVERSES[key]
    except KeyError:
        raise KeyError(
            f"Unknown universe {key!r}. Available: {', '.join(sorted(UNIVERSES))}"
        ) from None


def all_tickers() -> tuple[str, ...]:
    """Every ticker across every universe, deduplicated -- what the fetcher pulls."""
    seen: dict[str, None] = {}
    for u in UNIVERSES.values():
        for t in u.tickers:
            seen.setdefault(t, None)
    return tuple(seen)
