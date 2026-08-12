"""Hierarchical clustering and quasi-diagonalisation.

Named `hierarchy` rather than `clustering` on purpose: the headline function is
`clustering()`, and a module of the same name would be shadowed by it in the package
namespace, so `from radar.structure import clustering` would silently bind the function
in one context and the module in another.


This module is the project's **export surface**. The allocator project imports exactly
one function from here -- `clustering()` -- and nothing else, so the boundary between
"describe the structure" and "build a portfolio" stays sharp: linkage and seriation are
pure functions of the correlation matrix and live here; recursive bisection and weight
construction are allocation decisions and live there.

That seam matters for a reason beyond tidiness. The allocator publishes a live paper
track record, so every logged trade has to map to a known structure implementation. This
module is versioned with the package and pinned by the allocator; changing the linkage
method here is a dated, deliberate event, not a silent rewrite of past strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from radar.structure.correlation import DEFAULT_ESTIMATOR, estimate_correlation
from radar.structure.distance import correlation_to_distance

#: 'average' (UPGMA) by default rather than the 'single' linkage of the original HRP
#: paper. Single linkage chains: one intermediate asset can weld two otherwise distinct
#: clusters into a chain, which produces unstable dendrograms on noisy correlation data.
#: The allocator can override this, but whatever it picks must be pinned.
DEFAULT_METHOD = "average"

LINKAGE_METHODS = ("single", "complete", "average", "ward")


@dataclass(frozen=True)
class ClusteringResult:
    """A hierarchical clustering plus the diagnostics needed to judge it."""

    linkage: np.ndarray
    order: list[str]
    labels: list[str]
    method: str
    cophenetic_correlation: float
    merge_heights: np.ndarray

    @property
    def n_assets(self) -> int:
        return len(self.labels)

    def flat_clusters(self, n_clusters: int) -> pd.Series:
        """Cut the tree into `n_clusters` groups. Returns ticker -> cluster id."""
        assignments = fcluster(self.linkage, t=n_clusters, criterion="maxclust")
        return pd.Series(assignments, index=self.labels, name="cluster")

    def depth_ratio(self) -> float:
        """How much of the tree's height is spent on the final merge.

        Near 1 means one late merge dominates: a shallow, single-hub structure where
        hierarchical methods have almost nothing to exploit and will behave like naive
        risk parity. Near 0 means genuine nested structure.
        """
        heights = np.sort(self.merge_heights)
        if len(heights) < 2 or heights[-1] <= 0:
            return float("nan")
        return float((heights[-1] - heights[-2]) / heights[-1])


def condensed_distance(distance: pd.DataFrame) -> np.ndarray:
    """Square distance matrix -> scipy's condensed vector, symmetry enforced."""
    values = distance.to_numpy(dtype=float)
    values = 0.5 * (values + values.T)
    np.fill_diagonal(values, 0.0)
    return squareform(values, checks=False)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Seriation: leaf order that places correlated assets adjacent.

    This is stage 2 of López de Prado's HRP. It recursively replaces each cluster in the
    linkage with its two children until only original leaves remain, which arranges the
    correlation matrix so that large entries sit near the diagonal.

    Implemented directly rather than deferring to scipy so the allocator's dependency is
    on an explicit, inspectable algorithm -- but it is tested to agree with scipy's
    `leaves_list`, which computes the same ordering.
    """
    link = np.asarray(link)
    if link.size == 0:
        return [0]

    merges = link[:, :2].astype(int)
    n_items = int(link[-1, 3])

    order = pd.Series([merges[-1, 0], merges[-1, 1]])
    while order.max() >= n_items:
        order.index = range(0, order.shape[0] * 2, 2)  # make room between entries
        clusters = order[order >= n_items]
        positions, ids = clusters.index, clusters.values - n_items
        order[positions] = merges[ids, 0]
        right = pd.Series(merges[ids, 1], index=positions + 1)
        order = pd.concat([order, right]).sort_index()
        order.index = range(order.shape[0])
    return [int(x) for x in order.tolist()]


def cluster_from_distance(
    distance: pd.DataFrame, method: str = DEFAULT_METHOD
) -> ClusteringResult:
    """Hierarchical clustering of a precomputed distance matrix."""
    if method not in LINKAGE_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {', '.join(LINKAGE_METHODS)}.")
    if distance.empty:
        raise ValueError("Distance matrix is empty.")
    if len(distance) < 2:
        raise ValueError("Need at least 2 assets to cluster.")
    if distance.isna().any().any():
        raise ValueError("Distance matrix contains NaNs.")

    labels = list(distance.columns)
    condensed = condensed_distance(distance)
    link = linkage(condensed, method=method)

    cophenetic, _ = cophenet(link, condensed)
    order_idx = quasi_diagonal_order(link)

    return ClusteringResult(
        linkage=link,
        order=[labels[i] for i in order_idx],
        labels=labels,
        method=method,
        cophenetic_correlation=float(cophenetic),
        merge_heights=link[:, 2].copy(),
    )


def clustering(
    returns: pd.DataFrame,
    asof=None,
    window: int | None = None,
    method: str = DEFAULT_METHOD,
    estimator: str = DEFAULT_ESTIMATOR,
) -> ClusteringResult:
    """**The allocator's entry point.** Returns -> hierarchical clustering.

    Parameters
    ----------
    returns : the return panel, dates x assets, NaN-free.
    asof : optional cut-off. Only data up to and including this date is used, which is
        what keeps a backtest free of lookahead.
    window : optional trailing window length in observations.
    method : linkage method; pin this in the caller.
    estimator : correlation estimator; pin this too.

    The result carries `.linkage` and `.order` -- the (linkage, quasi_diagonal_order)
    pair the allocator consumes -- plus the diagnostics needed to tell whether the tree
    is worth trusting at all.
    """
    panel = returns if asof is None else returns.loc[:pd.Timestamp(asof)]
    if window is not None:
        panel = panel.tail(window)
    if len(panel) < 2:
        raise ValueError(f"Only {len(panel)} observations available at asof={asof}.")

    corr = estimate_correlation(panel, estimator).matrix
    return cluster_from_distance(correlation_to_distance(corr), method=method)


def cluster_stability(a: pd.Series, b: pd.Series) -> float:
    """Adjusted Rand index between two flat clusterings, over their shared assets.

    1.0 means identical partitions; 0.0 means no better than chance. Adjusted, so it does
    not reward agreement that random labelling would have produced anyway -- which
    matters here, because with few assets and few clusters raw agreement looks high by
    construction.
    """
    shared = a.index.intersection(b.index)
    if len(shared) < 2:
        return float("nan")
    return float(adjusted_rand_score(a.loc[shared].to_numpy(), b.loc[shared].to_numpy()))


def leaf_order_via_scipy(link: np.ndarray) -> list[int]:
    """scipy's leaf ordering. Kept as the cross-check for `quasi_diagonal_order`."""
    return [int(i) for i in leaves_list(link)]
