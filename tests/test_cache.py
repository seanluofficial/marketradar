from __future__ import annotations

import pandas as pd

from radar.data import cache


def test_roundtrip_preserves_values_and_index(make_series):
    written = make_series("AAA", periods=50)
    read = cache.read_eod("AAA")
    assert read is not None
    assert list(read.columns) == cache.EOD_COLUMNS
    pd.testing.assert_index_equal(read.index, written.index, check_names=False)
    assert read["adj_close"].iloc[-1] == written["adj_close"].iloc[-1]


def test_missing_ticker_reads_as_none():
    assert cache.read_eod("NOPE") is None
    assert cache.coverage("NOPE") is None


def test_merge_is_idempotent(make_series):
    original = make_series("BBB", periods=40)
    cache.write_eod("BBB", original)
    assert len(cache.read_eod("BBB")) == 40


def test_overlapping_refetch_replaces_rather_than_duplicates(make_series):
    make_series("CCC", periods=40, seed=1)
    before = cache.read_eod("CCC")

    # Simulate Tiingo restating the adjusted close after a dividend: same dates,
    # different values. The new values must win and the row count must not grow.
    restated = before.tail(10).copy()
    restated["adj_close"] = restated["adj_close"] * 0.5
    cache.write_eod("CCC", restated)

    after = cache.read_eod("CCC")
    assert len(after) == len(before)
    assert after["adj_close"].iloc[-1] == before["adj_close"].iloc[-1] * 0.5
    assert after.index.is_monotonic_increasing
    assert not after.index.has_duplicates


def test_extending_appends_new_dates(make_series):
    make_series("DDD", start="2020-01-01", periods=30, seed=2)
    existing = cache.read_eod("DDD")
    new_index = pd.bdate_range(existing.index[-1] + pd.Timedelta(days=1), periods=5)
    extension = pd.DataFrame(
        {col: 1.0 for col in cache.EOD_COLUMNS}, index=new_index
    )
    cache.write_eod("DDD", extension)

    merged = cache.read_eod("DDD")
    assert len(merged) == 35
    assert merged.index[-1] == new_index[-1]


def test_missing_columns_are_filled_as_nan(make_series):
    frame = make_series("EEE", periods=10)[["adj_close"]]
    cache.write_eod("FFF", frame)
    read = cache.read_eod("FFF")
    assert list(read.columns) == cache.EOD_COLUMNS
    assert read["volume"].isna().all()
    assert read["adj_close"].notna().all()


def test_coverage_reports_first_and_last_dates(make_series):
    frame = make_series("GGG", periods=20)
    first, last = cache.coverage("GGG")
    assert first == frame.index[0].date()
    assert last == frame.index[-1].date()


def test_filename_sanitises_punctuation():
    assert cache.eod_path("BRK.B").stem == "BRK_B"
    assert cache.eod_path("aapl").stem == "AAPL"


def test_cache_summary_flags_uncached_tickers(make_series):
    make_series("HHH", periods=12)
    summary = cache.cache_summary(["HHH", "III"])
    assert summary.loc["HHH", "rows"] == 12
    assert summary.loc["III", "rows"] == 0
