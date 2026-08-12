"""Eigenvalue-based structure metrics.

The headline number is the **absorption ratio** (Kritzman, Page & Turkington 2010): the
share of total variance captured by the leading eigenvectors of the correlation matrix.
It rises when markets become tightly coupled and falls when they decouple, and it spikes
at every crisis in the sample.

Two caveats are built into this module rather than left to the reader:

1. It is close to a monotone transform of average pairwise correlation. For a perfectly
   equicorrelated matrix, lambda_1 = 1 + (N-1)*rho exactly, so lambda_1/N -> rho for
   large N. `mean_correlation` is therefore computed alongside it everywhere, so the two
   can be plotted together and the relationship is visible rather than concealed.
2. Computed on overlapping rolling windows it is autocorrelated by construction. A
   window-to-window move is not independent evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Kritzman et al. use the eigenvectors accounting for roughly a fifth of the assets.
DEFAULT_COMPONENT_FRACTION = 0.2


def _as_array(corr: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = corr.to_numpy(dtype=float) if isinstance(corr, pd.DataFrame) else np.asarray(corr, float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {values.shape}.")
    return values


def eigen_spectrum(corr: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Eigenvalues in descending order. For a correlation matrix they sum to N."""
    return np.linalg.eigvalsh(_as_array(corr))[::-1]


def n_components_for(n_assets: int, fraction: float = DEFAULT_COMPONENT_FRACTION) -> int:
    """How many eigenvectors the absorption ratio uses. At least one."""
    return max(1, int(round(n_assets * fraction)))


def absorption_ratio(
    corr: pd.DataFrame | np.ndarray,
    n_components: int | None = None,
    fraction: float = DEFAULT_COMPONENT_FRACTION,
) -> float:
    """Share of variance absorbed by the leading eigenvectors.

    Because a correlation matrix has trace N, this is simply the sum of the top
    eigenvalues divided by N -- bounded in (0, 1], rising toward 1 as the market
    collapses onto a few common factors.
    """
    spectrum = eigen_spectrum(corr)
    k = n_components if n_components is not None else n_components_for(len(spectrum), fraction)
    k = int(np.clip(k, 1, len(spectrum)))
    return float(spectrum[:k].sum() / spectrum.sum())


def pc1_share(corr: pd.DataFrame | np.ndarray) -> float:
    """Variance share of the single leading eigenvector -- the market mode."""
    spectrum = eigen_spectrum(corr)
    return float(spectrum[0] / spectrum.sum())


def mean_correlation(corr: pd.DataFrame | np.ndarray) -> float:
    """Mean off-diagonal correlation.

    Reported next to the absorption ratio on purpose: for a near-equicorrelated matrix
    the two carry almost the same information, and showing them together is what keeps
    the systemic-risk index from looking like a more novel construction than it is.
    """
    values = _as_array(corr)
    n = len(values)
    if n < 2:
        return float("nan")
    return float((values.sum() - np.trace(values)) / (n * (n - 1)))


def effective_dimension(corr: pd.DataFrame | np.ndarray) -> float:
    """Participation ratio (sum(lambda))^2 / sum(lambda^2): how many independent
    directions the market is really moving in.

    Equals N for the identity and falls toward 1 as a single mode dominates. A more
    intuitive companion to the absorption ratio -- "the market is behaving like 4
    independent assets today" reads better than "PC1 explains 62%".
    """
    spectrum = eigen_spectrum(corr)
    return float(spectrum.sum() ** 2 / np.square(spectrum).sum())
