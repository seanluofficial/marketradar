# Experiment 012 — Results

**H0 is not rejected for either hypothesis.** Neither volatility management nor the
low-volatility anomaly survives a holdout at retail costs.

Per the stopping rule declared in `HYPOTHESIS_012.md`, **this concludes the search for an
edge in this data.** No further hypothesis will be proposed against it.

## Explore (net of 25bps)

| hypothesis | variant | Sharpe | benchmark | t |
|---|---|---|---|---|
| H1a | core_equity volmanaged 21d | 0.534 | 0.478 | 2.30 |
| H1a | core_equity volmanaged 63d | 0.614 | 0.478 | 2.65 |
| H1a | cross_asset volmanaged 21d | 0.273 | SPY 0.444 | 0.99 |
| H1a | cross_asset volmanaged 63d | 0.182 | SPY 0.444 | 0.66 |
| H1a | crypto_core volmanaged 21d | 0.742 | BTC **0.774** | 1.71 |
| H1a | crypto_core volmanaged 63d | 0.467 | BTC **0.774** | 1.08 |
| H1b | lowvol 63d long-only | 0.568 | 0.478 | 2.45 |
| H1b | lowvol 63d long/short | **−0.036** | 0.478 | −0.15 |
| H1b | lowvol 252d long-only | 0.644 | 0.478 | 2.78 |
| H1b | lowvol 252d long/short | **−0.069** | 0.478 | −0.30 |

Four cells cleared both the kill threshold and the benchmark rule, all on `core_equity`.

## Holdout — run once, on those four

| variant | explore | **holdout** | benchmark | t | verdict |
|---|---|---|---|---|---|
| volmanaged 21d | 0.534 | **0.368** | 0.590 | 1.04 | FAILS |
| volmanaged 63d | 0.614 | **0.341** | 0.590 | 0.96 | FAILS |
| lowvol 63d long | 0.568 | **0.426** | 0.590 | 1.20 | FAILS |
| lowvol 252d long | 0.644 | **0.338** | 0.590 | 0.96 | FAILS |

Every cell degraded. Every cell lost to equal-weight buy-and-hold. The largest |t| was
1.20 against a required 3.09 — and against an uncorrected 1.96, still nothing.

## What each hypothesis actually showed

**H1a, volatility management.** The overlay does what it mechanically claims: it steadies
the risk level. It does not improve risk-adjusted return out of sample. This reproduces
the Cederburg et al. critique on independent data — the effect is visible in-sample and
does not survive real-time implementation. Note also that the crypto cells lost to simply
holding bitcoin, and that financing cost for the 1.5× leverage was **not** charged, so the
real result is worse than shown.

**H1b, the low-volatility anomaly.** The long/short construction was *negative* in explore
(−0.036, −0.069), confirming that in this universe the premium lives entirely in the long
leg. The long-only leg beat the benchmark in explore and lost to it in the holdout. A
long-only low-volatility book is a defensive equity portfolio with a 95% average exposure;
it is not a source of alpha.

## The pattern across the programme

| experiment | best explore | best holdout | outcome |
|---|---|---|---|
| 005 PEAD (hindsight) | 0.141 | −0.032 | null |
| 010 cross-sectional momentum | 0.161 @10bps | dead on costs | null |
| 011 time-series momentum | 0.604 | 0.456 (bench 0.590) | null |
| 012 vol management / low vol | 0.644 | 0.426 (bench 0.590) | null |

**Twenty-five pre-registered cells across 011 and 012, plus ten prior hindsight
experiments. Zero survivors.** In every case the explore result was encouraging enough to
be tempting, and in every case the holdout removed it. That consistency is itself the
finding: it is what a market with no exploitable daily-frequency edge looks like from the
outside.

## Conclusion

The honest reading is that **daily end-of-day prices on liquid instruments do not contain
an edge that a retail participant can capture after costs.** This is the expected result —
it is the most heavily mined dataset in finance — and it has now been confirmed rather
than assumed, with pre-registered rules, untouched holdouts, a validated harness, and
predictions recorded before each run.

Improving the odds requires different information, not different strategies on this
information: intraday data, options, or genuinely alternative sources. Those are new
projects with new data, and the methodology built here transfers to them intact.
