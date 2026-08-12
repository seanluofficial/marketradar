"""Adapter tests. No network: a fake session replays canned Tiingo responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import pytest
import requests

from radar.data import cache, tiingo


@dataclass
class FakeResponse:
    status_code: int
    payload: object = None
    text: str = ""

    def json(self):
        return self.payload


@dataclass
class FakeSession:
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    headers: dict = field(default_factory=dict)

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


SAMPLE = [
    {
        "date": "2020-01-02T00:00:00.000Z",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "adjClose": 50.25, "volume": 1000, "divCash": 0.0, "splitFactor": 1.0,
    },
    {
        "date": "2020-01-03T00:00:00.000Z",
        "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5,
        "adjClose": 50.75, "volume": 1200, "divCash": 0.0, "splitFactor": 1.0,
    },
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tiingo.time, "sleep", lambda _: None)


def client(*responses) -> tiingo.TiingoClient:
    return tiingo.TiingoClient(api_key="test-key", session=FakeSession(list(responses)))


def test_daily_prices_maps_vendor_fields_to_cache_schema():
    c = client(FakeResponse(200, SAMPLE))
    df = c.daily_prices("AAPL", start=date(2020, 1, 1))

    assert list(df.columns) == [
        "open", "high", "low", "close", "adj_close", "volume", "div_cash", "split_factor"
    ]
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2020-01-02")
    assert df["adj_close"].iloc[0] == 50.25
    assert df.index.is_monotonic_increasing


def test_auth_header_is_set():
    c = client(FakeResponse(200, SAMPLE))
    assert c.session.headers["Authorization"] == "Token test-key"


def test_empty_payload_yields_empty_frame_with_schema():
    c = client(FakeResponse(200, []))
    df = c.daily_prices("AAPL")
    assert df.empty
    assert list(df.columns) == cache.EOD_COLUMNS


def test_unknown_ticker_raises_immediately_without_retry():
    session = FakeSession([FakeResponse(404, text="not found")])
    c = tiingo.TiingoClient(api_key="k", session=session)
    with pytest.raises(tiingo.UnknownTickerError):
        c.daily_prices("BOGUS")
    assert len(session.calls) == 1


def test_bad_key_raises_config_style_error():
    c = client(FakeResponse(403, text="forbidden"))
    with pytest.raises(tiingo.TiingoError, match="TIINGO_API_KEY"):
        c.daily_prices("AAPL")


def test_rate_limit_retries_then_raises():
    c = client(*[FakeResponse(429, text="limit")] * tiingo.RETRY_ATTEMPTS)
    with pytest.raises(tiingo.RateLimitError):
        c.daily_prices("AAPL")


def test_transient_failure_is_retried_then_succeeds():
    session = FakeSession([requests.ConnectionError("boom"), FakeResponse(200, SAMPLE)])
    c = tiingo.TiingoClient(api_key="k", session=session)
    df = c.daily_prices("AAPL")
    assert len(df) == 2
    assert len(session.calls) == 2


# ---------------------------------------------------------------------------
# Incremental fetch logic -- the part that keeps us inside the free-tier caps.
# ---------------------------------------------------------------------------


def test_fetch_start_is_full_history_when_nothing_cached():
    start = date(2010, 1, 1)
    assert tiingo._fetch_start("AAA", start, force=False) == start


def test_fetch_start_only_refetches_the_tail_when_cache_covers_the_range(make_series):
    make_series("AAA", start="2020-01-01", periods=100)
    _, cached_end = cache.coverage("AAA")

    got = tiingo._fetch_start("AAA", date(2020, 1, 1), force=False)
    assert got == cached_end - timedelta(days=tiingo.REFETCH_OVERLAP_DAYS)
    assert got < cached_end, "overlap must re-fetch restated adjusted prices"


def test_fetch_start_takes_full_history_when_cache_starts_too_late(make_series):
    make_series("AAA", start="2020-01-01", periods=100)
    start = date(2015, 1, 1)
    assert tiingo._fetch_start("AAA", start, force=False) == start


def test_force_ignores_the_cache(make_series):
    make_series("AAA", start="2020-01-01", periods=100)
    start = date(2020, 1, 1)
    assert tiingo._fetch_start("AAA", start, force=True) == start


def test_fetch_ticker_writes_to_cache_and_reports_rows():
    c = client(FakeResponse(200, SAMPLE))
    row = tiingo.fetch_ticker("AAPL", start=date(2020, 1, 1), client=c)
    assert row["status"] == "fetched"
    assert row["rows_added"] == 2
    assert cache.read_eod("AAPL") is not None


def test_fetch_ticker_records_errors_without_raising():
    c = client(FakeResponse(404, text="nope"))
    row = tiingo.fetch_ticker("BOGUS", start=date(2020, 1, 1), client=c)
    assert row["status"].startswith("error")
    assert cache.read_eod("BOGUS") is None


CRYPTO_SAMPLE = [
    {
        "ticker": "btcusd",
        "baseCurrency": "btc",
        "quoteCurrency": "usd",
        "priceData": [
            {"date": "2021-09-02T00:00:00+00:00", "open": 48800.0, "high": 50500.0,
             "low": 48600.0, "close": 49300.0, "volume": 1234.5},
            {"date": "2021-09-03T00:00:00+00:00", "open": 49300.0, "high": 51000.0,
             "low": 49100.0, "close": 50000.0, "volume": 2345.6},
        ],
    }
]


def test_crypto_prices_map_onto_the_equity_cache_schema():
    """Crypto reuses the equity schema so return construction, alignment and the whole
    structure layer run unchanged over both asset classes."""
    c = client(FakeResponse(200, CRYPTO_SAMPLE))
    df = c.crypto_prices("BTCUSD", start=date(2021, 9, 1))

    assert list(df.columns) == cache.EOD_COLUMNS
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2021-09-02")
    assert df["close"].iloc[1] == 50000.0
    # No corporate actions in crypto, so adjusted == raw and the action columns are inert.
    assert (df["adj_close"] == df["close"]).all()
    assert (df["div_cash"] == 0.0).all()
    assert (df["split_factor"] == 1.0).all()


def test_crypto_request_uses_the_crypto_endpoint_and_daily_resampling():
    session = FakeSession([FakeResponse(200, CRYPTO_SAMPLE)])
    c = tiingo.TiingoClient(api_key="k", session=session)
    c.crypto_prices("BTCUSD", start=date(2021, 9, 1))

    url, params = session.calls[0]
    assert url.endswith("/tiingo/crypto/prices")
    assert params["tickers"] == "btcusd"
    assert params["resampleFreq"] == "1day"


def test_crypto_empty_payload_yields_schema_only_frame():
    assert client(FakeResponse(200, [])).crypto_prices("NOPEUSD").empty
    # A known-shaped response with no priceData must behave the same way.
    empty = [{"ticker": "nopeusd", "priceData": []}]
    assert client(FakeResponse(200, empty)).crypto_prices("NOPEUSD").empty


def test_fetch_ticker_dispatches_on_asset_class():
    c = client(FakeResponse(200, CRYPTO_SAMPLE))
    row = tiingo.fetch_ticker(
        "BTCUSD", start=date(2021, 9, 1), client=c, asset_class="crypto"
    )
    assert row["status"] == "fetched"
    assert row["rows_added"] == 2
    assert cache.read_eod("BTCUSD") is not None


def test_fetch_universe_continues_past_a_failure(monkeypatch):
    """A rate limit mid-run must leave the earlier tickers cached and resumable."""
    calls = []

    def fake_fetch_ticker(ticker, **kwargs):
        calls.append(ticker)
        status = "error: rate limited" if ticker == "BBB" else "fetched"
        return {"ticker": ticker, "status": status, "rows_added": 1,
                "start": None, "end": None}

    monkeypatch.setattr(tiingo, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(tiingo, "TiingoClient", lambda: object())

    report = tiingo.fetch_universe(["AAA", "BBB", "CCC"], progress=False)
    assert calls == ["AAA", "BBB", "CCC"]
    assert report.loc["BBB", "status"].startswith("error")
    assert report.loc["CCC", "status"] == "fetched"
