# Market Structure Radar

Unsupervised market-structure analytics — filtered correlation networks, minimum spanning
trees, hierarchical clustering and a PCA-based systemic-risk index — plus a pre-registered
research programme testing whether any of it predicts anything.

**It doesn't, and that's the finding.** This is descriptive risk analytics, not a trading
signal, and the repository is built so that claim can be checked rather than taken on
trust.

```bash
pip install -e ".[app,dev]"
radar fetch --universe all       # populate the cache from Tiingo
radar build --start 2000-01-01   # precompute 1288 rolling windows (~3 min)
streamlit run radar/app/main.py  # the app
```

---

## What it measures

80 US large caps across all 11 GICS sectors, 2001–2026, in 1288 weekly 252-day windows.

| | calm | peak stress |
|---|---|---|
| absorption ratio | 0.589 (2001-01) | **0.882 (2020-04-22)** |
| mean pairwise correlation | 0.114 | 0.615 |
| effective dimension | 20.8 independent directions | **2.5** |
| MST tree length | 1.063 | 0.612 |

At the COVID peak an 80-asset portfolio moved like **2.5 independent assets**. Absorption-
ratio peaks land on 2008-12, 2009-08, 2011-12, 2012-05, 2020-04, 2022-12 and 2023-02 —
the actual crises, with no event labels supplied to any metric.

**Sector structure emerges from returns alone.** No labels are used to build the tree; they
are used only to score it, at a median **8.0× the random baseline**:

```
JPM (Financials)  → BAC 0.844, WFC 0.876, C 0.882, MS 0.909     [all Financials]
XOM (Energy)      → COP 0.810, CVX 0.813, EOG 0.830, OXY 0.837  [all Energy]
DUK (Utilities)   → SO 0.800, EXC 0.862, D 0.882, AEP 0.902     [all Utilities]
```

## The research programme

Four pre-registered experiments, 25 declared cells, every hypothesis written down with its
decision rules and predicted outcome **before** the code to test it existed.

| experiment | best explore | holdout | outcome |
|---|---|---|---|
| HRP vs naive weighting (crypto) | declustering effect 0.04 vs 6.37 on equities | — | no clusters to decluster |
| 011 time-series momentum | 0.604 | 0.456 (benchmark 0.590) | **null** |
| 012 volatility management | 0.614 | 0.341 (benchmark 0.590) | **null** |
| 012 low-volatility anomaly | 0.644 | 0.338 (benchmark 0.590) | **null** |

**Zero of 25 cells survived.** Every explore result was encouraging enough to be tempting;
every holdout removed it. See [`research/`](research/) for the pre-registrations, the
amendments with their superseded numbers, and the results.

The backtester carries a **positive control** — a clairvoyant signal must score Sharpe > 5
— so a null result cannot be an artefact of broken plumbing. Signals earn from t+1, costs
are charged on turnover at the moment of trading, cash earns zero, and nothing is fitted.

## Design decisions worth knowing

**Estimation, not just computation.** With N assets and a T-day window the sample
correlation matrix is noisy in proportion to q = N/T, and singular once q ≥ 1. At N=81 and
T=252, q ≈ 0.32 — noisy enough that the estimator choice visibly changes the network. Three
ship side by side: raw sample, Ledoit-Wolf shrinkage, and Marchenko-Pastur eigenvalue
clipping with σ² **fitted from the bulk** rather than assumed to be 1 (the market mode holds
30–60% of the trace, so the naive noise edge sits too high and discards real sector factors).

**The systemic-risk index has a name and a caveat.** It is the absorption ratio (Kritzman,
Page & Turkington 2010). For a near-equicorrelated matrix λ₁/N ≈ ρ̄, and measured here the
two correlate at **0.85** — so mean pairwise ρ is plotted alongside it rather than hidden.
Absorption ratio and MST tree length correlate at **−0.99**: near-duplicates as scalar
indices, so they are not presented as independent confirmation.

**Instability is measured, not hidden.** Between windows five trading days apart — sharing
247 of 252 observations — median MST edge survival is **0.873**; between non-overlapping
annual windows it is **0.33–0.48**. Roughly 60% of the tree turns over year to year.

**Layouts are chained, and it's verifiable.** Each frame's force-directed layout warm-starts
from the previous frame's positions, computed once at build time. Node displacement
correlates **−0.73** with edge survival — 0.065 when the tree is unchanged, 0.202 when
survival drops below 0.70. Recomputed independently per frame, the nodes would scramble
every step and a viewer would read optimiser noise as a regime change.

**Classical MDS, not t-SNE/UMAP.** The Mantegna distance d = √(2(1−ρ)) is a proper metric,
so MDS is the principled embedding — and it is deterministic, which keeps it stable under
the time scrubber. Stochastic embeddings would scramble frame to frame.

**Alignment is strict and loud.** A ragged join plus pairwise-complete correlation yields a
matrix whose entries come from different samples and need not be PSD, with no warning. So
names without history covering the range are dropped **by name with a reason**, short gaps
are forward-filled on prices up to a per-universe limit and counted, and anything left is an
error. Delistings are distinguished from interior gaps.

**Survivorship, in the right direction.** Universes are fixed baskets of currently-listed
names. Because every member survived, the basket is biased toward resilient firms — so the
crisis-period correlation collapse shown here is an *understatement*, not an overstatement.

## Layout

```
radar/
  data/            Tiingo adapter (equity + crypto), disk cache, universes, return panels
  structure/       correlation estimators → Mantegna distance → MST → hierarchical clustering
  metrics/         absorption ratio, effective dimension, rolling artifacts, diagnostics
  research/        backtester, momentum and volatility hypotheses, experiment runners
  viz/             chained layouts, network / index / heatmap / spectrum figures
  app/             Streamlit
research/          pre-registrations and results
```

The network boundary is exactly one module (`radar/data/tiingo.py`). Everything above it
reads from the disk cache, so the pipeline is reproducible offline and the deployed app
needs no API key.

Artifacts hold per-window metrics, MST edges and layouts, but **not** correlation matrices:
80×80 across 1288 windows would be ~66 MB, while rebuilding one selected window costs about
a millisecond. The whole artifact is 1.4 MB.

## Export surface

The allocator project consumes one function, kept deliberately narrow and versioned:

```python
structure.clustering(returns, asof) -> (linkage, quasi_diagonal_order)
```

Pinned by version there, so a change to the clustering here cannot silently rewrite the
strategy that generated an existing track record.

## Tests

```bash
pytest -q     # 195 tests, no network and no API key required
```

The adapter runs against canned responses; panel logic, estimators and the backtester run
against synthetic series with known structure.

## References

- Mantegna (1999), *Hierarchical structure in financial markets*
- Laloux, Cizeau, Bouchaud & Potters (1999), *Noise dressing of financial correlation matrices*
- Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*
- Onnela et al. (2003), *Dynamics of market correlations*
- Kritzman, Page & Turkington (2010), *Principal components as a measure of systemic risk*
- López de Prado (2016), *Building diversified portfolios that outperform out of sample*
- Moskowitz, Ooi & Pedersen (2012), *Time series momentum*
- Moreira & Muir (2017), *Volatility-managed portfolios*; Cederburg et al. (2020), critique
