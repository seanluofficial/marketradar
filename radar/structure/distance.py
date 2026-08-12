"""Correlation -> distance.

Mantegna (1999): d(i,j) = sqrt(2 * (1 - rho(i,j))).

This is not an arbitrary monotone rescaling of correlation -- it is a genuine metric,
which is what licenses everything downstream. It satisfies d >= 0, d(i,j) = 0 iff the
returns are perfectly correlated, symmetry, and the triangle inequality, because it is
the Euclidean distance between the standardised return vectors on the unit sphere:

    ||x - y||^2 = 2 - 2*cos(x, y) = 2 * (1 - rho)   for unit-norm centred x, y.

Two consequences the project relies on:
  * a minimum spanning tree of these distances is meaningful rather than decorative;
  * classical MDS is the *principled* 2-D embedding, not merely a convenient one -- and
    being deterministic, it stays stable as the time scrubber moves, which stochastic
    embeddings like t-SNE and UMAP do not.

Range is [0, 2]: 0 at rho = 1, sqrt(2) at rho = 0, 2 at rho = -1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: sqrt(2): the distance between two uncorrelated assets. Useful as a reference line --
#: edges shorter than this are positively correlated pairs.
INDEPENDENT_DISTANCE = float(np.sqrt(2.0))


def correlation_to_distance(corr: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Mantegna distance matrix. Symmetric, zero diagonal, entries in [0, 2]."""
    if isinstance(corr, pd.DataFrame):
        labels = list(corr.columns)
        values = corr.to_numpy(dtype=float)
    else:
        values = np.asarray(corr, dtype=float)
        labels = [str(i) for i in range(values.shape[0])]

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {values.shape}.")

    # Clip before the square root: estimator round-off can push an entry a hair past 1,
    # and a negative radicand would surface as a silent NaN edge in the MST.
    clipped = np.clip(values, -1.0, 1.0)
    distance = np.sqrt(2.0 * (1.0 - clipped))
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)

    return pd.DataFrame(distance, index=labels, columns=labels)


def is_metric(distance: pd.DataFrame | np.ndarray, tolerance: float = 1e-9) -> bool:
    """Check the metric axioms, including the triangle inequality.

    O(n^3) and therefore a test/debug tool, not something to call per scrubber frame.
    """
    d = distance.to_numpy(dtype=float) if isinstance(distance, pd.DataFrame) else np.asarray(distance)
    n = len(d)
    if not np.allclose(d, d.T, atol=tolerance):
        return False
    if not np.allclose(np.diag(d), 0.0, atol=tolerance):
        return False
    if (d < -tolerance).any():
        return False
    # d[i,k] <= d[i,j] + d[j,k] for all j, vectorised over i and k.
    for j in range(n):
        if (d > d[:, [j]] + d[[j], :] + tolerance).any():
            return False
    return True


def nearest_neighbours(
    distance: pd.DataFrame, ticker: str, k: int = 5
) -> pd.Series:
    """The `k` closest assets to `ticker`, nearest first. Backs the click-a-node view."""
    if ticker not in distance.index:
        raise KeyError(f"{ticker!r} not in the distance matrix.")
    row = distance.loc[ticker].drop(labels=[ticker])
    return row.nsmallest(k)
