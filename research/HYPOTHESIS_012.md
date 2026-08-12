# Experiment 012 — Risk-based effects: volatility management and the low-volatility anomaly

**Pre-registered 2026-08-12, before any backtest was run**, and committed before the
implementation existed.

## Stopping rule (declared first, because it is the point)

**This is the last experiment run against this data.** If both hypotheses below fail, the
search for an edge in daily end-of-day prices ends, and no twelfth or thirteenth
hypothesis will be proposed as a substitute. Continuing to generate hypotheses against a
dataset that has already refused eleven of them does not produce knowledge — it produces
a false positive with a good story attached.

If a hypothesis survives, the correct next step is further out-of-sample observation, not
capital.

## Why these two, and not something else

Every hypothesis tested so far (001–011) has been a **return-prediction** claim: some
observable predicts a future return. Those have failed consistently, which is what theory
predicts for widely-observed data on liquid instruments.

These two are different in kind. Neither claims returns are predictable. Both rest on the
better-supported claim that **risk is more forecastable than return** — volatility is
strongly autocorrelated in a way returns are not.

**H1a — Volatility management** (Moreira & Muir 2017). Scaling exposure inversely to
recent realised volatility improves risk-adjusted return, because volatility persists
while returns do not.

*Known challenge, stated in advance:* Cederburg, O'Doherty, Wang & Yan (2020) found the
effect does not survive real-time implementation across many strategies. This is a genuine
coin-flip, not a favourite, and that assessment is recorded before seeing the result.

**H1b — Low-volatility anomaly.** Low-volatility assets have historically delivered
better risk-adjusted returns than high-volatility assets, contradicting a simple
risk-return trade-off. Long-only, low turnover, decades of literature.

*Known challenge:* much of the documented premium concentrates in shorts on high-volatility
names, which are expensive or impossible to borrow. The long-only leg is the honest retail
version and is the primary here.

## The panel, declared in advance

| hypothesis | universes | second dimension | cells |
|---|---|---|---|
| H1a vol-managed | `core_equity`, `cross_asset`, `crypto_core` | vol lookback 21d, 63d | 6 |
| H1b low-volatility | `core_equity` | formation 63d, 252d × {long-only, long-minus-high} | 4 |

**10 cells.** No cell will be added or dropped after results are seen.

## Fixed parameters — declared, not fitted

- **Target volatility:** 15% annualised for `core_equity`, 10% for `cross_asset`, 50% for
  `crypto_core`. Chosen to be near each universe's historical level and **fixed before
  running**. They are not optimised, and the Sharpe of a volatility-scaled strategy is
  close to invariant to this constant anyway except through the leverage cap.
- **Leverage cap: 1.5×.** Moreira-Muir's construction requires levering up in calm
  periods; forbidding that entirely would test a different, weaker claim. 1.5× is
  achievable in a retail margin account. Financing cost is **not** modelled, which is
  optimistic and is flagged here rather than discovered later.
- **Quintiles** for H1b: 80 assets → 16 per leg.
- Rebalance monthly (21 trading days), as in 011.

## Decision rules

Identical in structure to 011, with the correction updated for this panel:

1. **Primary metric:** net-of-cost annualised Sharpe at 25bps, explore partition.
2. **Kill threshold:** explore Sharpe < 0.30 → dead, does not reach the holdout.
3. **Must beat its benchmark** — equal-weight buy-and-hold on the same universe, and the
   single-asset benchmark where one exists. A vol-managed portfolio that fails to beat the
   portfolio it manages has failed by definition.
4. **Significance, within this experiment:** 10 cells → Bonferroni p < 0.005, |t| > 2.81.
5. **Significance, across the programme:** 25 cells across 011 and 012 → p < 0.002,
   |t| > 3.09. **This is the headline bar.** Correcting only within an experiment while
   running experiment after experiment on the same data is the loophole that makes
   multiple-testing corrections theatre.
6. **Holdout** computed once, only for cells clearing rules 2 and 3.

Partitions are the same chronological 70/30 split, and the holdout is the same data left
untouched in 011 for every universe except `core_equity`, whose holdout has now been used
once. That prior use is a real cost and is noted: `core_equity` holdout evidence in this
experiment is weaker than it would have been, because the partition is no longer pristine.

## Costs

25bps per side, as in 011. H1b's long-minus-high variant does not model borrow cost, which
makes it optimistic; the long-only variant is the one to believe.

## Expected outcome

Recorded in advance: **H0 not rejected in either hypothesis.** Somewhat less confident
than in 011 — vol management has a real mechanism behind it and the low-vol effect is
robustly documented — but the retail cost structure, the leverage cap, and the
Cederburg critique together make survival unlikely. Roughly 20% on either.
