# Experiment 011 — Results

**H0 is not rejected in any cell.** Time-series momentum, tested across 15 pre-registered
cells spanning crypto, sector ETFs, cross-asset ETFs and US large-cap equities, does not
produce a risk-adjusted edge that survives realistic costs and an untouched holdout.

This matches the outcome predicted in `HYPOTHESIS.md` before any backtest was run.

## Explore partition (net of 25bps per side)

| universe | 3m | 6m | 12m | EW benchmark | single-asset |
|---|---|---|---|---|---|
| crypto_core (6) | 0.160 | 0.394 | 0.444 | 0.406 | BTC **0.774** |
| crypto_majors (14) | −0.124 | 0.010 | 0.201 | −0.231 | BTC **0.347** |
| sector_etfs (11) | 0.121 | 0.262 | 0.180 | **0.501** | — |
| cross_asset (20) | 0.264 | 0.381 | 0.361 | 0.278 | SPY **0.444** |
| core_equity (80) | 0.406 | **0.538** | **0.604** | 0.478 | — |

Seven cells cleared the kill threshold (Sharpe ≥ 0.30). Applying rule 5 — a cell must
also beat its benchmark — left two: `core_equity` at 6m and 12m. Every crypto and
cross-asset cell that cleared the threshold lost to simply holding BTC or SPY.

## Holdout partition — run once, on those two cells only

| cell | explore | **holdout** | EW benchmark | t | verdict |
|---|---|---|---|---|---|
| core_equity 6m | 0.538 | **0.310** | 0.590 | 0.88 | FAILS |
| core_equity 12m | 0.604 | **0.456** | 0.590 | 1.29 | FAILS |

Both degraded out of sample, and both lost to equal-weight buy-and-hold on the same
universe. Neither approaches the Bonferroni-corrected bar of |t| > 2.94; neither would
clear an uncorrected |t| > 1.96 either.

## What this means

**The degradation pattern repeats.** Hindsight experiment 005 (PEAD) went from explore
Sharpe 0.141 to holdout −0.032. Experiment 010 (cross-sectional momentum) died on costs
before the holdout opened. Here, the two best cells fell 42% and 25% from explore to
holdout. Three independent hypotheses, the same shape of result.

**Trend following is a risk tool, not a return tool.** Average exposure across cells was
37–67%, so the strategy is often partly in cash, and holdout max drawdown for
`core_equity` 12m was −25.7%. It takes less risk and earns proportionally less. Sharpe
accounts for that, and it still loses to buy-and-hold — there is no free lunch here, only
a different risk level available more cheaply by holding less of the benchmark.

**Crypto is the weakest case, not the strongest.** Every crypto momentum variant tested
underperformed simply holding bitcoin (explore Sharpe 0.774). The 14-coin basket was worse
than the 6-coin one, consistent with the structure diagnostic finding an effective
dimension of 1.9 — there is little to diversify across and little for a cross-sectional
overlay to exploit.

**One nominal hit is the expected number.** With 15 tests at α = 0.05, roughly 0.75 cells
should look significant by chance. Two cells cleared the uncorrected explore bar. That is
what noise looks like, and it is why the holdout and the correction were fixed in advance.

## Costs, and what would have to be true

Breakeven costs for the two holdout cells were 192bps and 391bps per side, which sounds
generous until you note the Sharpe is below the benchmark's at *any* cost level — the
strategy does not lose to fees, it loses to the benchmark. Reducing costs would not save
it.

For this to become tradeable, the holdout Sharpe would need to exceed the benchmark's
0.590 with |t| > 2.94. It reached 0.456 with t = 1.29.

## Process notes

Two amendments were made after the first explore run and before the holdout was opened,
both recorded in `HYPOTHESIS.md` with the superseded numbers: the `crypto_core` start date
moved to 2019 because Tiingo's 2018 coverage collapsed the panel to a single asset, and
annualisation was corrected to infer periods per year from the data rather than assume 252
(which had understated crypto Sharpes by ~20%). The second amendment *improved* crypto
results and is flagged for that reason.

The backtester carries a positive control — a clairvoyant signal must score Sharpe > 5 —
so a null result cannot be an artefact of a broken harness.

## Conclusion

Eleven hypotheses have now been tested across `hindsight` and this repository. None has
produced an edge that survived costs and an out-of-sample holdout. That is the honest
state of the evidence, and it is recorded here rather than filed away.
