# Research programme

Pre-registered hypothesis tests run against the data in this repository. Each
pre-registration was committed **before** the code to test it existed, with decision rules,
significance thresholds and a predicted outcome fixed in advance.

| file | what it covers |
|---|---|
| [`HYPOTHESIS.md`](HYPOTHESIS.md) | Experiment 011 — time-series momentum, 15 cells |
| [`RESULTS.md`](RESULTS.md) | Experiment 011 results |
| [`HYPOTHESIS_012.md`](HYPOTHESIS_012.md) | Experiment 012 — volatility management, low-volatility anomaly, 10 cells |
| [`RESULTS_012.md`](RESULTS_012.md) | Experiment 012 results, and the concluding summary |

Numbering continues from the [`hindsight`](https://github.com/seanluofficial/hindsight)
project's experiments 001–010.

## Outcome

**Zero of 25 pre-registered cells survived.** Combined with the ten prior hindsight
experiments, that is 12 hypotheses tested and none producing an edge that survives
realistic costs and an untouched out-of-sample holdout.

The consistency is the result. In every case the explore-partition number was encouraging
enough to be tempting, and in every case the holdout removed it:

```
005 PEAD                        0.141 → −0.032
010 cross-sectional momentum    0.161 @10bps → dead on costs
011 time-series momentum        0.604 → 0.456   (benchmark 0.590)
012 volatility management       0.614 → 0.341   (benchmark 0.590)
012 low-volatility anomaly      0.644 → 0.338   (benchmark 0.590)
```

## Method

- **Pre-registration before implementation.** Decision rules, kill thresholds and predicted
  outcomes are committed in git before the test exists, so they cannot be revised
  afterwards. Amendments are dated, justified, and published with the superseded numbers.
- **Multiple-testing correction across the programme**, not within an experiment.
  Correcting within an experiment while running experiment after experiment on the same
  data is the loophole that makes the correction theatre. The headline bar is
  0.05 / 25 cells → |t| > 3.09.
- **Holdouts computed once**, only for cells that clear the explore rules. Looking at a
  holdout and then declining to use it is not the same as not looking, so the runners
  restrict which cells are computed at all.
- **A positive control on the harness.** A clairvoyant signal must score Sharpe > 5. A
  backtester that cannot detect a real edge would manufacture null results silently.
- **Pessimistic defaults.** Signals earn from t+1, costs are charged on turnover at the
  moment of trading, cash earns zero, nothing is fitted, and optimistic omissions (unmodelled
  borrow and financing costs) are flagged where they occur.
- **A declared stopping rule.** Experiment 012 was pre-committed as the last test against
  this data. It failed, and the search ended rather than continuing to a thirteenth
  hypothesis.

## Conclusion

Daily end-of-day prices on liquid instruments do not contain an edge a retail participant
can capture after costs. This is the expected result for the most heavily mined dataset in
finance — the contribution is that it was confirmed under pre-registered conditions rather
than assumed, and that the negative results are published rather than filed away.

Improving the odds requires different information, not different strategies on this
information.
