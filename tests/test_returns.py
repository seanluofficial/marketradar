from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.data import cache, returns


def test_panel_is_rectangular_and_nan_free(make_series):
    for i, t in enumerate(["AAA", "BBB", "CCC"]):
        make_series(t, periods=120, seed=i)
    rets, report = returns.returns_panel(tickers=["AAA", "BBB", "CCC"])
    assert report.n_assets == 3
    assert not rets.isna().any().any()
    assert len(rets) == 119  # first row consumed by the diff


def test_uncached_ticker_is_dropped_with_a_reason(make_series):
    make_series("AAA", periods=60)
    _, report = returns.returns_panel(tickers=["AAA", "MISSING"])
    assert report.retained == ("AAA",)
    assert "not cached" in report.dropped["MISSING"]


def test_late_listing_is_dropped_not_backfilled(make_series):
    """The bias this guards against: a 2013 IPO silently appearing in the 2008 crisis."""
    make_series("OLD", start="2020-01-01", periods=200, seed=1)
    make_series("NEW", start="2020-06-01", periods=100, seed=2)

    _, report = returns.returns_panel(tickers=["OLD", "NEW"])
    assert report.retained == ("OLD",)
    assert "history starts" in report.dropped["NEW"]


def test_delisted_ticker_is_dropped(make_series):
    make_series("LIVE", start="2020-01-01", periods=200, seed=1)
    make_series("DEAD", start="2020-01-01", periods=60, seed=2)

    _, report = returns.returns_panel(tickers=["LIVE", "DEAD"])
    assert report.retained == ("LIVE",)
    assert "history ends" in report.dropped["DEAD"]


def test_short_gap_is_forward_filled_and_counted(make_series):
    make_series("AAA", start="2020-01-01", periods=100, seed=1)
    make_series("HALT", start="2020-01-01", periods=100, seed=2, drop_dates=["2020-02-05"])

    rets, report = returns.returns_panel(tickers=["AAA", "HALT"])
    assert set(report.retained) == {"AAA", "HALT"}
    assert report.filled_values == 1
    # A forward-filled price means a zero return on the halted day.
    assert rets.loc[pd.Timestamp("2020-02-05"), "HALT"] == pytest.approx(0.0)


def test_long_gap_is_dropped_rather_than_filled(make_series):
    make_series("AAA", start="2020-01-01", periods=100, seed=1)
    gap = pd.bdate_range("2020-02-03", periods=6).strftime("%Y-%m-%d").tolist()
    make_series("GAPPY", start="2020-01-01", periods=100, seed=2, drop_dates=gap)

    _, report = returns.returns_panel(tickers=["AAA", "GAPPY"])
    assert report.retained == ("AAA",)
    assert "unfillable gaps" in report.dropped["GAPPY"]


def test_name_that_goes_dark_is_reported_as_a_delisting_not_a_gap(make_series):
    """A trailing NaN run just inside the staleness tolerance is a delisting, and the
    report must say so -- this is the EA 2026-08-04 case seen in real data."""
    make_series("AAA", start="2020-01-01", periods=100, seed=1)
    tail = pd.bdate_range("2020-05-13", periods=5).strftime("%Y-%m-%d").tolist()
    make_series("GONE", start="2020-01-01", periods=100, seed=2, drop_dates=tail)

    _, report = returns.returns_panel(tickers=["AAA", "GONE"])
    assert report.retained == ("AAA",)
    assert "delisting or halt" in report.dropped["GONE"]
    assert "unfillable gaps" not in report.dropped["GONE"]


def test_interior_gap_is_still_reported_as_a_gap(make_series):
    make_series("AAA", start="2020-01-01", periods=100, seed=1)
    mid = pd.bdate_range("2020-02-03", periods=6).strftime("%Y-%m-%d").tolist()
    make_series("HOLEY", start="2020-01-01", periods=100, seed=2, drop_dates=mid)

    _, report = returns.returns_panel(tickers=["AAA", "HOLEY"])
    assert "unfillable gaps" in report.dropped["HOLEY"]


def test_fill_limit_comes_from_the_universe(make_series):
    """A 2-day hole is a weekend for equities and an outage for crypto, so the limit is
    a property of the universe rather than a module-level constant."""
    make_series("AAA", start="2020-01-01", periods=100, seed=1)
    gap = ["2020-02-05", "2020-02-06"]
    make_series("GAPPY", start="2020-01-01", periods=100, seed=2, drop_dates=gap)

    lenient, rep_lenient = returns.returns_panel(tickers=["AAA", "GAPPY"], max_ffill_days=3)
    assert set(rep_lenient.retained) == {"AAA", "GAPPY"}
    assert rep_lenient.filled_values == 2

    strict, rep_strict = returns.returns_panel(tickers=["AAA", "GAPPY"], max_ffill_days=1)
    assert rep_strict.retained == ("AAA",)
    assert "unfillable gaps (> 1d)" in rep_strict.dropped["GAPPY"]


def test_log_returns_match_the_definition(make_series):
    prices = pd.DataFrame(
        {"X": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=3)
    )
    rets = returns.log_returns(prices)
    assert rets["X"].iloc[0] == pytest.approx(np.log(1.10))
    assert rets["X"].iloc[1] == pytest.approx(np.log(99 / 110))
    assert len(rets) == 2


def test_log_returns_reject_non_positive_prices():
    prices = pd.DataFrame(
        {"X": [100.0, 0.0]}, index=pd.bdate_range("2020-01-01", periods=2)
    )
    with pytest.raises(ValueError, match="Non-positive"):
        returns.log_returns(prices)


def test_split_would_corrupt_close_but_not_adj_close(make_series):
    """Why the panel defaults to adj_close: a 2-for-1 split halves `close`."""
    make_series("SPLIT", start="2020-01-01", periods=50, seed=3)
    frame = cache.read_eod("SPLIT")
    frame.loc[frame.index[25:], "close"] /= 2.0  # raw price after a 2-for-1
    cache.write_eod("SPLIT", frame)

    adj, _ = returns.price_panel(tickers=["SPLIT"])
    raw, _ = returns.price_panel(tickers=["SPLIT"], field_name="close")
    assert returns.log_returns(raw)["SPLIT"].min() < -0.6  # phantom -50% day
    assert returns.log_returns(adj)["SPLIT"].min() > -0.1  # unaffected


def test_date_range_filters_are_applied(make_series):
    make_series("AAA", start="2020-01-01", periods=200, seed=1)
    rets, report = returns.returns_panel(
        tickers=["AAA"], start=pd.Timestamp("2020-03-01").date(),
        end=pd.Timestamp("2020-06-01").date(),
    )
    assert report.start >= pd.Timestamp("2020-03-01").date()
    assert report.end <= pd.Timestamp("2020-06-01").date()
    assert len(rets) == report.n_obs


def test_report_q_flags_the_singular_regime(make_series):
    for i in range(3):
        make_series(f"T{i}", periods=60, seed=i)
    _, report = returns.returns_panel(tickers=[f"T{i}" for i in range(3)])
    assert report.q(252) == pytest.approx(3 / 252)
    assert report.q(2) == pytest.approx(1.5)  # N > T -> singular


def test_empty_request_returns_empty_panel():
    rets, report = returns.returns_panel(tickers=["NOPE1", "NOPE2"])
    assert rets.empty
    assert report.n_assets == 0
    assert len(report.dropped) == 2
