"""Correlation estimators.

Three estimators, deliberately shipped side by side rather than one "correct" choice,
because the difference between them *is* the point. With N assets and T observations the
sample correlation matrix carries O(N^2) parameters estimated from N*T numbers; at
q = N/T ~ 0.32 most of its small eigenvalues are noise, and at q >= 1 it is singular.

  sample       what most people plot. Kept so the noise is visible, not hypothetical.
  ledoit_wolf  optimal linear shrinkage toward a scaled identity (Ledoit & Wolf 2004).
               Well-conditioned and invertible, at the cost of biasing every entry.
  rmt_clipped  Marchenko-Pastur eigenvalue clipping (Laloux et al. 1999). Keeps the
               eigenvalues that lie outside the noise band and replaces the bulk with
               its mean. Correlation-native: it cleans the object we actually plot, and
               the factors it keeps are the same market and sector modes the absorption
               ratio measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ESTIMATORS = ("sample", "ledoit_wolf", "rmt_clipped")

DEFAULT_ESTIMATOR = "rmt_clipped"


@dataclass(frozen=True)
class CorrelationResult:
    """A correlation matrix plus everything needed to defend it."""

    matrix: pd.DataFrame
    estimator: str
    n_assets: int
    n_obs: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def q(self) -> float:
        """N/T. The single number that says how much to trust the sample matrix."""
        return self.n_assets / float(self.n_obs)

    @property
    def values(self) -> np.ndarray:
        return self.matrix.to_numpy()

    @property
    def tickers(self) -> list[str]:
        return list(self.matrix.columns)

    def summary(self) -> str:
        bits = [
            f"{self.estimator}: N={self.n_assets} T={self.n_obs} q={self.q:.3f}",
            f"  condition number {self.diagnostics.get('condition_number', float('nan')):.1f}",
        ]
        if "shrinkage" in self.diagnostics:
            bits.append(f"  shrinkage intensity {self.diagnostics['shrinkage']:.3f}")
        if "n_factors" in self.diagnostics:
            bits.append(
                f"  {self.diagnostics['n_factors']} eigenvalues kept above "
                f"lambda+ = {self.diagnostics['lambda_plus']:.3f} "
                f"(sigma^2 = {self.diagnostics['sigma_squared']:.3f})"
            )
        return "\n".join(bits)


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Covariance -> correlation, with an exactly unit diagonal."""
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValueError("Covariance has a non-positive diagonal entry.")
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return _symmetrize(corr)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Kill floating-point asymmetry. Downstream eigendecomposition assumes symmetry."""
    return 0.5 * (matrix + matrix.T)


def _validate(returns: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if returns.empty:
        raise ValueError("Return panel is empty.")
    if returns.isna().any().any():
        raise ValueError(
            "Return panel contains NaNs. Correlation must be estimated on one common "
            "sample -- use radar.data.returns_panel, which guarantees this."
        )
    if len(returns) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(returns)}.")

    values = returns.to_numpy(dtype=float)
    # Testing `std == 0` does not work: pandas and numpy return ~1e-18 rather than an
    # exact zero for a genuinely constant column, so the check has to be scaled to the
    # column's own magnitude. Letting one through yields a NaN row out of corrcoef that
    # propagates into the distance matrix and then into the tree, unflagged.
    scale = np.maximum(np.abs(values).max(axis=0), 1.0)
    tolerance = 100.0 * np.finfo(float).eps * scale
    constant = [
        ticker
        for ticker, sd, tol in zip(returns.columns, values.std(axis=0, ddof=1), tolerance)
        if sd <= tol
    ]
    if constant:
        raise ValueError(f"Zero-variance columns, correlation undefined: {constant}")
    return values, list(returns.columns)


def _finish(
    corr: np.ndarray, tickers: list[str], estimator: str, n_obs: int, diagnostics: dict
) -> CorrelationResult:
    corr = _symmetrize(np.clip(corr, -1.0, 1.0))
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    diagnostics = {
        **diagnostics,
        "min_eigenvalue": float(eigvals[0]),
        "max_eigenvalue": float(eigvals[-1]),
        "condition_number": float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-12 else np.inf,
        "mean_offdiagonal": float(
            (corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1))
        )
        if len(corr) > 1
        else np.nan,
    }
    return CorrelationResult(
        matrix=pd.DataFrame(corr, index=tickers, columns=tickers),
        estimator=estimator,
        n_assets=len(tickers),
        n_obs=n_obs,
        diagnostics=diagnostics,
    )


def sample_correlation(returns: pd.DataFrame) -> CorrelationResult:
    """Plain Pearson correlation. Singular once N >= T; noisy well before that."""
    values, tickers = _validate(returns)
    corr = np.corrcoef(values, rowvar=False)
    return _finish(corr, tickers, "sample", len(returns), {})


def ledoit_wolf_correlation(returns: pd.DataFrame) -> CorrelationResult:
    """Ledoit-Wolf shrinkage toward a scaled identity.

    Returns are standardised before fitting so that the shrinkage target corresponds to
    the identity *correlation* matrix rather than an equal-variance covariance -- which
    is the right target when the object of interest is correlation.
    """
    values, tickers = _validate(returns)
    standardised = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)

    lw = LedoitWolf(assume_centered=False).fit(standardised)
    corr = cov_to_corr(lw.covariance_)
    return _finish(
        corr, tickers, "ledoit_wolf", len(returns), {"shrinkage": float(lw.shrinkage_)}
    )


def marchenko_pastur_edge(q: float, sigma_squared: float = 1.0) -> float:
    """Upper edge of the Marchenko-Pastur bulk: lambda+ = sigma^2 (1 + sqrt(q))^2.

    Eigenvalues below this are indistinguishable from what pure noise would produce.
    """
    return float(sigma_squared * (1.0 + np.sqrt(q)) ** 2)


def _fit_noise_variance(eigvals: np.ndarray, q: float, max_iter: int = 100) -> float:
    """Estimate the noise variance carried by the bulk.

    Naively taking sigma^2 = 1 (the trace of a correlation matrix divided by N) is
    wrong: the market mode alone can hold 30-40% of the trace, so the *remaining*
    eigenvalues sit in a bulk far narrower than sigma^2 = 1 implies, and the naive
    lambda+ discards genuine sector factors. So fit it: repeatedly take the mean of the
    eigenvalues currently inside the bulk as the new sigma^2 until it stops moving.
    """
    sigma_squared = 1.0
    for _ in range(max_iter):
        edge = marchenko_pastur_edge(q, sigma_squared)
        bulk = eigvals[eigvals <= edge]
        if bulk.size == 0:
            return sigma_squared
        updated = float(bulk.mean())
        if abs(updated - sigma_squared) < 1e-12:
            return updated
        sigma_squared = updated
    return sigma_squared


def rmt_clipped_correlation(
    returns: pd.DataFrame, fit_sigma: bool = True
) -> CorrelationResult:
    """Marchenko-Pastur eigenvalue clipping (Laloux, Cizeau, Bouchaud & Potters 1999).

    Eigenvalues above the noise-band edge are kept untouched; everything inside the bulk
    is replaced by the bulk mean, which preserves the trace contributed by the noise
    while destroying its arbitrary structure. The diagonal is then renormalised to one.

    The result is positive definite by construction (every eigenvalue is either a kept
    positive one or the strictly positive bulk mean), so unlike the raw sample matrix it
    is usable even when q >= 1.
    """
    values, tickers = _validate(returns)
    n_assets, n_obs = len(tickers), len(returns)
    q = n_assets / float(n_obs)

    corr = np.corrcoef(values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(corr))  # ascending

    sigma_squared = _fit_noise_variance(eigvals, q) if fit_sigma else 1.0
    edge = marchenko_pastur_edge(q, sigma_squared)

    noise = eigvals <= edge
    if noise.all():
        # No eigenvalue escapes the bulk: the data are indistinguishable from noise, and
        # the honest cleaned matrix is the identity.
        cleaned = np.eye(n_assets)
    else:
        replacement = float(eigvals[noise].mean()) if noise.any() else 0.0
        clipped = np.where(noise, replacement, eigvals)
        cleaned = _symmetrize(eigvecs @ np.diag(clipped) @ eigvecs.T)
        cleaned = cov_to_corr(cleaned)

    return _finish(
        cleaned,
        tickers,
        "rmt_clipped",
        n_obs,
        {
            "sigma_squared": float(sigma_squared),
            "lambda_plus": float(edge),
            "n_factors": int((~noise).sum()),
            "largest_eigenvalue": float(eigvals[-1]),
            "variance_discarded": float(eigvals[noise].sum() / n_assets),
        },
    )


_DISPATCH = {
    "sample": sample_correlation,
    "ledoit_wolf": ledoit_wolf_correlation,
    "rmt_clipped": rmt_clipped_correlation,
}


def estimate_correlation(
    returns: pd.DataFrame, estimator: str = DEFAULT_ESTIMATOR
) -> CorrelationResult:
    """Estimate a correlation matrix. `estimator` is one of ESTIMATORS."""
    try:
        fn = _DISPATCH[estimator]
    except KeyError:
        raise ValueError(
            f"Unknown estimator {estimator!r}. Choose from {', '.join(ESTIMATORS)}."
        ) from None
    return fn(returns)
