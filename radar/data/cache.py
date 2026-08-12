"""On-disk price cache: one parquet per ticker, merged idempotently.

The cache is the contract between the network layer and everything above it. Nothing
in `structure/`, `metrics/` or `viz/` may touch the network -- they read from here, so
the whole pipeline is reproducible offline and the deployed app never needs an API key.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from radar.config import EOD_DIR

#: Columns every cached frame carries, in order. Index is a tz-naive DatetimeIndex.
EOD_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "div_cash",
    "split_factor",
]

_UNSAFE = re.compile(r"[^A-Z0-9]+")


def _filename(ticker: str) -> str:
    """Filesystem-safe stem. Ticker punctuation (BRK.B) collapses to underscores."""
    return _UNSAFE.sub("_", ticker.strip().upper())


def eod_path(ticker: str) -> Path:
    return EOD_DIR / f"{_filename(ticker)}.parquet"


def read_eod(ticker: str) -> pd.DataFrame | None:
    """Cached history for `ticker`, or None if nothing is cached."""
    path = eod_path(ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.sort_index()


def write_eod(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Merge `frame` into the cache for `ticker` and return the merged result.

    New rows win on date collisions -- a re-fetch of an overlapping range replaces the
    cached values rather than duplicating them, which matters because Tiingo restates
    adjusted prices whenever a split or dividend lands.
    """
    EOD_DIR.mkdir(parents=True, exist_ok=True)
    incoming = _normalize(frame)

    existing = read_eod(ticker)
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, incoming])
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = incoming

    merged = merged.sort_index()
    merged.to_parquet(eod_path(ticker))
    return merged


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    for col in EOD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[EOD_COLUMNS].astype("float64")
    return df[~df.index.duplicated(keep="last")].sort_index()


def coverage(ticker: str) -> tuple[date, date] | None:
    """(first, last) cached date, or None if uncached/empty."""
    df = read_eod(ticker)
    if df is None or df.empty:
        return None
    return df.index[0].date(), df.index[-1].date()


def cached_tickers() -> list[str]:
    if not EOD_DIR.exists():
        return []
    return sorted(p.stem for p in EOD_DIR.glob("*.parquet"))


def cache_summary(tickers: list[str] | None = None) -> pd.DataFrame:
    """One row per ticker: coverage and row count. Used by `radar status`."""
    names = tickers if tickers is not None else cached_tickers()
    rows = []
    for t in names:
        df = read_eod(t)
        if df is None or df.empty:
            rows.append({"ticker": t, "rows": 0, "start": pd.NaT, "end": pd.NaT})
            continue
        rows.append(
            {
                "ticker": t,
                "rows": len(df),
                "start": df.index[0],
                "end": df.index[-1],
            }
        )
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
