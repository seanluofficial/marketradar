from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.data import cache


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Redirect the price cache at a temp dir so tests never touch real data."""
    eod = tmp_path / "eod"
    eod.mkdir(parents=True)
    monkeypatch.setattr(cache, "EOD_DIR", eod)
    return eod


@pytest.fixture
def make_series():
    """Build a cached synthetic price series for one ticker.

    Business-day index, geometric random walk, so log returns are well-defined.
    """

    def _make(
        ticker: str,
        start: str = "2020-01-01",
        periods: int = 260,
        seed: int = 0,
        drop_dates: list[str] | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range(start, periods=periods)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=periods)))
        frame = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "adj_close": prices,
                "volume": 1_000_000.0,
                "div_cash": 0.0,
                "split_factor": 1.0,
            },
            index=index,
        )
        if drop_dates:
            frame = frame.drop(index=pd.to_datetime(drop_dates), errors="ignore")
        cache.write_eod(ticker, frame)
        return frame

    return _make
