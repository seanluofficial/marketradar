from __future__ import annotations

import pytest

from radar.data import universe


def test_no_duplicate_tickers_within_a_universe():
    for key, uni in universe.UNIVERSES.items():
        tickers = list(uni.tickers)
        assert len(tickers) == len(set(tickers)), f"{key} has duplicate tickers"


def test_every_member_has_a_group_and_name():
    for uni in universe.UNIVERSES.values():
        for member in uni.members:
            assert member.group.strip()
            assert member.name.strip()
            assert member.ticker == member.ticker.upper()


def test_core_equity_spans_all_eleven_gics_sectors():
    uni = universe.get_universe("core_equity")
    assert uni.to_frame()["group"].nunique() == 11


def test_core_equity_q_is_workable_at_the_default_window():
    """The N vs T decision is a property of the universe, so pin it in a test."""
    n = len(universe.get_universe("core_equity"))
    assert 0.25 < n / 252 < 0.45, "core_equity sizing drifted away from the stated q"
    assert n / 90 < 1.0, "universe grew past the point where a 90d window is even defined"


def test_all_tickers_deduplicates_across_universes():
    combined = universe.all_tickers()
    assert len(combined) == len(set(combined))
    assert set(universe.get_universe("cross_asset").tickers) <= set(combined)


def test_subset_preserves_order_and_metadata():
    uni = universe.get_universe("core_equity")
    picked = ["MSFT", "XOM", "AAPL"]
    sub = uni.subset(picked)
    assert set(sub.tickers) == set(picked)
    # ordering follows the universe, not the argument
    assert sub.tickers == tuple(t for t in uni.tickers if t in set(picked))
    assert sub.groups["XOM"] == "Energy"


def test_unknown_universe_raises_with_available_keys():
    with pytest.raises(KeyError, match="core_equity"):
        universe.get_universe("nope")
