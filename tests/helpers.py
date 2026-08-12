"""Synthetic data with known structure, shared across test modules.

Lives here rather than inside a test module so that importing it does not pull in an
unrelated module's tests. Generating returns from an explicit factor model is what lets
the structure tests assert something real: when the number of blocks and the loadings are
known in advance, "the MST recovered the blocks" is a checkable claim rather than an
impression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_returns(
    n_assets: int,
    n_obs: int,
    n_blocks: int = 1,
    market_loading: float = 0.0,
    block_loading: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Returns with a known factor structure.

        r_i = market_loading * F_market + block_loading * F_block(i) + idiosyncratic

    Assets are assigned to blocks round-robin, so asset i belongs to block i % n_blocks.
    """
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n_obs)
    blocks = rng.standard_normal((n_blocks, n_obs))

    data = {}
    for i in range(n_assets):
        data[f"A{i:03d}"] = (
            market_loading * market
            + block_loading * blocks[i % n_blocks]
            + rng.standard_normal(n_obs)
        )
    return pd.DataFrame(data, index=pd.bdate_range("2000-01-03", periods=n_obs))
