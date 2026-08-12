# Market Structure Radar

Unsupervised market-structure analytics: filtered correlation networks, minimum spanning
trees, hierarchical clustering, and an absorption-ratio systemic-risk index — with a time
scrubber so you can watch the structure reorganise through 2008, 2011, 2015-16, 2020 and 2022.

**This is descriptive risk analytics, not a trading signal.** Nothing here predicts returns.
The output is a picture of how assets are moving together right now, and how that has changed.

---

## Status

Phase 1 of 5 complete: data adapter, universes, cache, return construction.

| Phase | Scope | State |
|---|---|---|
| 1 | Tiingo adapter, universes, disk cache, aligned return panels | done |
| 2 | Correlation estimators (sample / Ledoit-Wolf / RMT), MST, precomputed window artifacts | next |
| 3 | Rolling windows, time scrubber, absorption ratio, edge-survival metric | |
| 4 | Hierarchical clustering, dendrogram, classical MDS embedding, node drill-down | |
| 5 | Cross-asset view, deploy | |

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows; use bin/activate elsewhere
pip install -e ".[dev]"

cp .env.example .env            # add your free Tiingo key
radar universes                 # what's available and why
radar fetch --universe all      # populate the cache (resumable)
radar status                    # what landed
radar panel --start 2005-01-01  # build the return panel, see what was dropped and why
radar coverage                  # the history-vs-breadth trade-off, quantified
```

`radar fetch` is incremental and resumable. It only ever requests the missing tail of a
series plus a 7-day overlap (Tiingo restates adjusted prices when a dividend or split
lands), and a rate limit mid-run leaves everything fetched so far cached.

## Design decisions worth knowing

**Estimation, not just computation.** With N assets and a T-day window, the sample
correlation matrix is noisy in proportion to q = N/T, and singular once q ≥ 1. The primary
universe is 81 names and the default window is 252 days, so q ≈ 0.32 — noisy enough that
the estimator choice visibly changes the network, which is the point. Phase 2 ships three
estimators side by side: raw sample, Ledoit-Wolf shrinkage, and Marchenko-Pastur eigenvalue
clipping (Laloux/Bouchaud). A 90-day window is available and reported at q ≈ 0.90, with a
warning; that regime is shown as a demonstration of what breaks, not as a default.

**The systemic-risk index has a name and a caveat.** It is the absorption ratio (Kritzman,
Page & Turkington 2010) — the share of variance captured by the leading eigenvectors. For a
roughly equicorrelated matrix λ₁/N ≈ ρ̄, so it is close to a monotone transform of average
pairwise correlation. Mean pairwise ρ is therefore plotted alongside it rather than hidden.
Overlapping windows also make the series autocorrelated by construction, so week-to-week
moves are not independent evidence.

**MST instability is measured, not hidden.** Minimum spanning tree edges churn under
estimation noise, and a re-laid-out graph per frame would make sampling error look like
structural change. Layouts warm-start from the previous frame, and edge survival ratio
(Onnela et al.) is plotted as a first-class metric.

**Classical MDS, not t-SNE/UMAP.** The Mantegna distance d = √(2(1−ρ)) is a proper metric,
so MDS is the principled embedding for it — and it is deterministic, which means it stays
stable as you scrub. Stochastic embeddings would scramble frame to frame and destroy the
one feature the tool exists for.

**Alignment is strict and loud.** A ragged join plus pairwise-complete correlation gives you
a matrix whose entries were each estimated on a different sample, and which need not be
positive semi-definite — with no warning. So: names without history covering the requested
range are dropped by name with a reason; short gaps are forward-filled on prices (a zero
return for a halted day) up to a 3-day limit and counted; anything left over is an error.
Every panel reports exactly what it dropped.

**Survivorship, in the right direction.** Universes are fixed baskets of currently-listed
names, not point-in-time constituents. Lehman, Bear Stearns, Wachovia and Enron are absent.
The consequence is specific: because every member survived, the basket is biased toward
resilient firms, so the crisis-period correlation collapse shown here is an *understatement*.

## Layout

```
radar/
  config.py        paths, key loading
  data/            Tiingo adapter, disk cache, universes, return panels
  structure/       (phase 2) correlation -> distance -> MST -> clustering -> PCA
  metrics/         (phase 3) absorption ratio, edge survival, structure over time
  viz/             (phase 4) network, dendrogram, heatmap, MDS embedding
  app/             (phase 5) Streamlit
tests/
```

The network boundary is exactly one module (`radar/data/tiingo.py`). Everything above it
reads from the disk cache, so the whole pipeline is reproducible offline and the deployed
app needs no API key.

## Export surface

The allocator project consumes one function, kept deliberately narrow and versioned:

```python
structure.clustering(returns, asof) -> (linkage, quasi_diagonal_order)
```

It is pinned by version there so that a change to the clustering here cannot silently
rewrite the strategy that generated an existing paper track record.

## Tests

```bash
pytest -q
```

44 tests, no network and no API key required — the adapter is exercised against canned
responses and the panel logic against synthetic series.

## References

- Mantegna (1999), *Hierarchical structure in financial markets*
- Laloux, Cizeau, Bouchaud & Potters (1999), *Noise dressing of financial correlation matrices*
- Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*
- Onnela et al. (2003), *Dynamics of market correlations*
- Kritzman, Page & Turkington (2010), *Principal components as a measure of systemic risk*
