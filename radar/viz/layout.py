"""Graph layouts that stay still.

The single most important detail in an animated network view. A force-directed layout is
a stochastic optimisation with no canonical solution: run it twice on the *same* graph
and you get two different pictures. Recomputed independently per frame, the nodes would
scramble on every step of the scrubber, and a viewer would read that motion as the market
restructuring when it is nothing but the optimiser landing somewhere else.

So layouts are chained: frame *n* starts from frame *n-1*'s positions and takes only a
few relaxation steps. Nodes then move when the topology actually changes and stay put
when it does not, which is the only way the animation carries information.

Computed once at build time and stored in the artifact, so the app never runs an
optimiser and every viewer sees identical pictures.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

#: The first frame has no predecessor, so it gets a full-quality layout.
INITIAL_ITERATIONS = 300

#: Later frames start near the answer and only need to relax. Small on purpose: more
#: iterations would let the layout drift even when the tree is unchanged.
WARM_ITERATIONS = 12

LAYOUT_SEED = 20200323  # fixed so builds are reproducible


def chained_layouts(
    trees: dict, seed: int = LAYOUT_SEED
) -> pd.DataFrame:
    """Warm-started 2-D positions for a time-ordered mapping of {window_end: graph}.

    Returns a long frame of (window_end, ticker, x, y).
    """
    positions: dict | None = None
    rows: list[dict] = []

    for window_end, tree in trees.items():
        iterations = INITIAL_ITERATIONS if positions is None else WARM_ITERATIONS
        positions = nx.spring_layout(
            tree,
            pos=positions,
            iterations=iterations,
            seed=seed,
            weight="weight",
        )
        for ticker, (x, y) in positions.items():
            rows.append(
                {"window_end": window_end, "ticker": ticker, "x": float(x), "y": float(y)}
            )

    return pd.DataFrame(rows)


def layout_drift(layouts: pd.DataFrame) -> pd.Series:
    """Mean node displacement between consecutive frames.

    The companion to `edge_survival`: that measures topology churn, this measures how
    much the *picture* moved. Large drift with high edge survival means the layout is
    lying about how much changed.
    """
    wide = layouts.pivot(index="window_end", columns="ticker", values=["x", "y"])
    dx = wide["x"].diff()
    dy = wide["y"].diff()
    return np.sqrt(dx**2 + dy**2).mean(axis=1)
