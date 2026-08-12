"""Tiingo end-of-day adapter.

The only module in the project allowed to make network calls. It fetches, it caches,
and it stops -- no return construction, no cleaning beyond column renaming, so the raw
vendor response stays auditable on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from radar.config import (
    INTER_REQUEST_SLEEP,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    TIINGO_BASE_URL,
    tiingo_api_key,
)
from radar.data import cache

#: Far enough back to cover the dot-com unwind for the names that traded through it.
DEFAULT_START = date(1998, 1, 1)

#: Crypto has no comparable history; anything earlier is illiquid or nonexistent.
CRYPTO_DEFAULT_START = date(2016, 1, 1)

#: When extending a cached series we re-fetch a few days of overlap. Tiingo restates
#: adjusted prices when a split or dividend lands, so the tail is not immutable.
REFETCH_OVERLAP_DAYS = 7

_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjClose": "adj_close",
    "volume": "volume",
    "divCash": "div_cash",
    "splitFactor": "split_factor",
}


class TiingoError(RuntimeError):
    pass


class RateLimitError(TiingoError):
    pass


class UnknownTickerError(TiingoError):
    pass


@dataclass
class TiingoClient:
    api_key: str = field(default_factory=tiingo_api_key)
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_key}",
            }
        )

    def daily_prices(
        self,
        ticker: str,
        start: date = DEFAULT_START,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Adjusted daily OHLCV for one ticker. Empty frame if the range has no data."""
        params = {"startDate": start.isoformat(), "format": "json"}
        if end is not None:
            params["endDate"] = end.isoformat()

        payload = self._get(f"/tiingo/daily/{ticker}/prices", params, ticker)
        if not payload:
            return pd.DataFrame(columns=cache.EOD_COLUMNS)

        df = pd.DataFrame(payload)
        df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
        df = df.set_index("date")
        df.index = df.index.tz_convert(None).normalize()

        present = {src: dst for src, dst in _FIELD_MAP.items() if src in df.columns}
        df = df[list(present)].rename(columns=present)
        return df.sort_index()

    def crypto_prices(
        self,
        ticker: str,
        start: date = CRYPTO_DEFAULT_START,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Daily OHLCV for one crypto pair, mapped onto the equity cache schema.

        Crypto has no splits or dividends, so `adj_close` is set equal to `close` and the
        corporate-action columns are zero/one. That is not a fudge -- it means the same
        return construction, alignment and structure code runs unchanged over both asset
        classes, which is the whole reason the cache has a single schema.
        """
        params = {
            "tickers": ticker.lower(),
            "startDate": start.isoformat(),
            "resampleFreq": "1day",
        }
        if end is not None:
            params["endDate"] = end.isoformat()

        payload = self._get("/tiingo/crypto/prices", params, ticker)
        rows = payload[0].get("priceData", []) if payload else []
        if not rows:
            return pd.DataFrame(columns=cache.EOD_COLUMNS)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
        df = df.set_index("date")
        df.index = df.index.tz_convert(None).normalize()

        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep].astype(float)
        df["adj_close"] = df["close"]
        df["div_cash"] = 0.0
        df["split_factor"] = 1.0
        df = df.reindex(columns=cache.EOD_COLUMNS)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _get(self, path: str, params: dict, ticker: str) -> list[dict]:
        url = f"{TIINGO_BASE_URL}{path}"
        last_error: Exception | None = None

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                raise UnknownTickerError(f"{ticker}: not found on Tiingo (404)")
            if response.status_code in (401, 403):
                raise TiingoError(
                    f"{ticker}: auth failed ({response.status_code}). Check TIINGO_API_KEY."
                )
            if response.status_code == 429:
                last_error = RateLimitError(
                    f"{ticker}: rate limited by Tiingo (429). "
                    "Free tier caps unique symbols per hour; the cache means a resumed "
                    "run picks up where this one stopped."
                )
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue

            last_error = TiingoError(
                f"{ticker}: HTTP {response.status_code} -- {response.text[:200]}"
            )
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

        raise last_error if last_error else TiingoError(f"{ticker}: request failed")


def _fetch_start(ticker: str, start: date, force: bool) -> date | None:
    """Where to begin the request, or None if the cache already covers the range.

    Returns `start` when we need the full history (nothing cached, or the cache begins
    later than requested); otherwise the cached tail minus an overlap buffer.
    """
    if force:
        return start
    covered = cache.coverage(ticker)
    if covered is None:
        return start
    cached_start, cached_end = covered
    if cached_start > start:
        return start
    return cached_end - timedelta(days=REFETCH_OVERLAP_DAYS)


def fetch_ticker(
    ticker: str,
    start: date = DEFAULT_START,
    end: date | None = None,
    client: TiingoClient | None = None,
    force: bool = False,
    asset_class: str = "equity",
) -> dict:
    """Bring one ticker's cache up to date. Returns a status row (never raises)."""
    row = {"ticker": ticker, "status": "", "rows_added": 0, "start": None, "end": None}
    target_end = end or date.today()

    request_start = _fetch_start(ticker, start, force)
    if request_start is not None and request_start >= target_end:
        request_start = None

    before = cache.read_eod(ticker)
    before_rows = 0 if before is None else len(before)

    if request_start is None:
        row["status"] = "cached"
        row["rows_added"] = 0
    else:
        client = client or TiingoClient()
        fetch = client.crypto_prices if asset_class == "crypto" else client.daily_prices
        try:
            fresh = fetch(ticker, start=request_start, end=end)
        except TiingoError as exc:
            row["status"] = f"error: {exc}"
            return row
        if fresh.empty:
            row["status"] = "no-data"
        else:
            cache.write_eod(ticker, fresh)
            row["status"] = "fetched"
        time.sleep(INTER_REQUEST_SLEEP)

    after = cache.read_eod(ticker)
    if after is not None and not after.empty:
        row["rows_added"] = len(after) - before_rows
        row["start"] = after.index[0].date()
        row["end"] = after.index[-1].date()
    return row


def fetch_universe(
    tickers: list[str],
    start: date = DEFAULT_START,
    end: date | None = None,
    force: bool = False,
    progress: bool = True,
    asset_class: str = "equity",
) -> pd.DataFrame:
    """Update the cache for many tickers.

    Per-ticker failures are recorded and skipped rather than aborting the run: a rate
    limit two thirds of the way through a universe should leave you with two thirds of
    the data cached and a resumable job, not nothing.
    """
    client = TiingoClient()
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        row = fetch_ticker(
            ticker, start=start, end=end, client=client, force=force,
            asset_class=asset_class,
        )
        rows.append(row)
        if progress:
            print(
                f"[{i:>3}/{len(tickers)}] {ticker:<6} {row['status']:<12} "
                f"(+{row['rows_added']} rows)",
                flush=True,
            )
    return pd.DataFrame(rows).set_index("ticker")


def datetime_utcnow() -> datetime:  # pragma: no cover - trivial, kept for artifact stamps
    return datetime.now()
