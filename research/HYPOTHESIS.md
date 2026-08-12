# Experiment 011 — Time-series momentum across asset classes

**Pre-registered 2026-08-12, before any backtest was run.** Committed prior to results
existing. If the analysis below is amended after seeing outcomes, that amendment is
recorded as a dated edit rather than a rewrite.

Numbering continues from the `hindsight` project's experiments 001–010.

## Motivation

Hindsight experiment 010 tested *cross-sectional* momentum (12-1, quintile long/short) on
US equities. Result: explore-partition Sharpe 0.161 at 10bps costs, falling to **−0.181 at
25bps**. Costs alone killed it, before the holdout was ever opened.

This tests a **different** hypothesis, not a retry of that one:

- **Time-series** momentum (own past return predicts own future return) rather than
  cross-sectional (relative rank predicts relative return). These are distinct effects
  with distinct literatures; Moskowitz, Ooi & Pedersen (2012) document the time-series
  version across 58 instruments and four asset classes.
- **Long-flat**, not long/short. Roughly half the turnover of a quintile L/S book, which
  matters because turnover is precisely what defeated 010.

## Hypothesis

**H1.** For an asset with positive trailing return over a lookback of L trading days, the
subsequent 21-day return is positive in expectation, sufficiently so that a long-flat
portfolio rebalanced monthly beats buy-and-hold on a risk-adjusted basis after costs.

**H0.** It does not, after realistic costs.

## The full panel, declared in advance

Every cell below will be run. No cell will be dropped, and no cell added, after results
are seen.

| dimension | values | count |
|---|---|---|
| universe | `crypto_core`, `crypto_majors`, `sector_etfs`, `cross_asset`, `core_equity` | 5 |
| lookback L | 63d (3m), 126d (6m), 252d (12m) | 3 |

**15 primary tests.**

Cost levels 10 / 25 / 50 bps per side are reported for every cell, but **25bps is the
primary** — the others exist to show the breakeven cost, not to be selected from.

## Decision rules, fixed now

1. **Primary metric:** net-of-cost annualised Sharpe at 25bps, explore partition.
2. **Multiple testing:** 15 tests, so the Bonferroni-corrected threshold at α = 0.05 is
   **p < 0.0033**. A nominal p < 0.05 in one cell out of fifteen is the expected number of
   false positives and will be reported as such.
3. **Kill threshold:** a cell with explore Sharpe **< 0.30** net of 25bps is dead. It does
   not proceed to the holdout.
4. **Holdout discipline:** the holdout partition is not computed, plotted or looked at for
   any cell that fails rule 3. Cells that pass are run on the holdout exactly once.
5. **Benchmark:** every cell is reported against buy-and-hold equal-weight on the same
   universe, and for crypto additionally against hold-BTC. A strategy that fails to beat
   its benchmark has failed regardless of its absolute Sharpe.
6. **Cost sensitivity:** report the breakeven cost at which Sharpe reaches zero. A
   strategy whose breakeven is below 50bps is not tradeable at retail.

## Partitions

Split by date, fixed before running:

- **explore** — earliest 70% of each universe's usable history
- **holdout** — final 30%, untouched until rule 4 is satisfied

Because crypto's usable history is short, `crypto_core` (7 coins from 2018) exists
alongside `crypto_majors` (14 coins from 2021) specifically so the length/breadth
trade-off is visible rather than chosen after the fact.

## Costs

Charged on turnover, per side, at rebalance. 25bps is intended to cover commission plus
spread plus slippage for a retail participant. This is *optimistic* for crypto alts and
for small-cap equities, and roughly fair for large-cap ETFs.

No borrowing costs are modelled because the strategy is long-flat. Cash earns **zero**,
which is conservative in a positive-rate environment and is stated rather than tuned.

## Known limitations, acknowledged before results

- **Survivorship.** All universes are fixed baskets of instruments alive today. Crypto is
  the worst affected: LUNA, FTT and a long tail of dead tokens are absent, which flatters
  any long-biased strategy.
- **Short crypto history.** Even `crypto_core` gives ~8 years, of which the holdout is
  ~2.5. That is too few independent 21-day periods for a confident conclusion, and a
  positive result there should be treated as weaker evidence than the same number on
  equities.
- **Multiple testing across projects.** This is the eleventh hypothesis tested across
  hindsight and this repo. The relevant correction is arguably over all of them, not just
  the 15 cells here, which makes the effective bar higher still.
- **This is not a deployment.** A surviving cell would justify further out-of-sample
  observation, not capital.

## Amendments

### 2026-08-12 — `crypto_core` start date, and annualisation

Recorded after the first explore run, **before the holdout was opened for any cell.**
Both changes are disclosed rather than silently applied, and the superseded numbers are
stated so the reader can judge whether the amendment flattered the result.

**1. `crypto_core` start moves 2018-01-01 → 2019-01-01.** Tiingo's 2018 crypto coverage
has multi-day holes, and under the universe's 1-day forward-fill limit that dropped 6 of
the 7 coins, leaving a single-asset panel. The cell therefore tested BTC-only trend
following, not the 7-coin basket that was registered. Superseded explore result for the
mislabelled cell: Sharpe 0.245 / 0.736 / 0.596 at 3m / 6m / 12m. The amendment is made on
data-coverage grounds, visible from the panel's drop report and not from the performance
numbers, and it *reduces* the sample rather than extending it.

**2. Annualisation is now inferred from the data rather than fixed at 252.** Crypto trades
every calendar day, so annualising it at 252 periods understated crypto Sharpes by
sqrt(365/252) ≈ 20%. This was a bug in the first run and affects the crypto cells only. It
makes crypto results *better*, which is the uncomfortable direction, so it is flagged
explicitly: no crypto cell should be read as clearing a threshold it only reached because
of this correction.

Neither amendment changes the decision rules, the panel size, or the significance
threshold. The count remains 15 cells and the Bonferroni bar remains p < 0.0033.

## Expected outcome

Stated in advance so it cannot be revised afterwards: **H0 is not rejected in any cell.**
The most likely result is that explore Sharpes cluster near zero, that the 10bps column
looks mildly encouraging in two or three cells, and that 25bps removes it. Recording this
prediction is the point — if the result matches, it is evidence the process works, and if
it does not, the surprise is informative.
