# Evaluation

How well does each approach estimate the effect of a change it has never seen? Every estimate is
scored against the ground truth: the true outcome process re-run on the same transactions. All
numbers are simulation output on synthetic data.

Reproduce with:

```bash
python scripts/generate_data.py
python scripts/evaluate.py        # about 15 minutes; writes experiments/harness/
```

Full tables are in `experiments/harness/results.md`, the summary data in `results.json`, and
one row per estimate in `scenario_results.csv`.

## Setup

- **Split.** Models train on 2026-01-05 to 2026-01-27 (775,802 first attempts). They are
  evaluated on the held-out last 7 days, which they never see.
- **Windows.** Counterfactuals run on two unseen normal weekdays, 2026-01-28 and 2026-02-03.
- **Grid** (`evaluation/grid.py`): 42 scenarios per window.
  - Issuer degradations from 5pp to 40pp, on three issuers.
  - Outages of 15 and 60 minutes on every gateway.
  - Traffic changes from −30% to +120%.
  - Five method shifts and six reroutes.
  - Two combined scenarios: an outage with traffic rerouted away, and a degradation during a
    surge.
- **Range label, from the data.** A scenario is out of range when more than 5% of the payments it
  affects have issuer health, gateway health or utilisation outside the 0.1%–99.9% range seen in
  training. Across both windows, 49 runs are in range and 35 are out of range.
- **Errors.** MAE with a 95% bootstrap interval, plus RMSE. Median relative error and sign
  accuracy are computed only over scenarios with a material true effect, e.g. at least 0.1pp for
  success rate.

## Predictors

| Name | What it is |
|---|---|
| `rule_baseline` | Historical success rate per (issuer, method, gateway, 4-hour block), scaled by current ÷ normal health ratios and a load-band factor (`models/rule_baseline.py`) |
| `independent_ml` | Three separate gradient-boosted tree models for success, failure reason and log latency, on transaction fields, observed state and entity attributes (`models/independent.py`) |
| `true_probability` | The outcome process's own success probability. A reference for the best achievable transaction-level score, never a model input |

## Results

### Transaction level (test period, 223,821 first attempts)

| Predictor | AUC | Brier | Log loss |
|---|---|---|---|
| rule_baseline | 0.5159 | 0.03807 | 0.1894 |
| independent_ml | 0.5905 | 0.03653 | 0.1599 |
| true_probability | 0.6009 | 0.03640 | 0.1591 |

Most payment failures here are irreducible chance: even the true probabilities only reach AUC
0.60. The independent models get almost all of the way to that ceiling. The rule baseline scores
worse than a constant prediction per transaction (Brier 0.0381 against 0.0367 for the constant),
because its noisy health ratios add variance. That variance averages out over a scenario.

### Counterfactual success-rate error (pp)

| Scenario type | rule_baseline MAE [95% CI] | independent_ml MAE [95% CI] | rule / ML median rel. error |
|---|---|---|---|
| issuer_degradation (28) | 0.024 [0.013, 0.040] | 0.681 [0.286, 1.184] | 0.04 / 1.00 |
| gateway_outage (16) | 0.007 [0.004, 0.010] | 1.060 [0.656, 1.502] | 0.00 / 0.93 |
| method_shift (10) | 0.083 [0.049, 0.125] | 0.020 [0.012, 0.029] | 0.52 / 0.10 |
| routing_change (12) | 0.569 [0.170, 1.045] | 0.555 [0.155, 1.042] | 0.55 / 0.52 |
| traffic_change (14) | 1.804 [0.217, 3.964] | 1.933 [0.224, 4.288] | 0.54 / 0.54 |
| combined (4) | 1.331 [0.043, 2.619] | 2.143 [1.509, 2.784] | 0.36 / 0.78 |

By range, success-rate MAE for rules vs ML is 0.206 vs 0.337 pp in range, and 0.827 vs 1.772 pp
out of range.

## What the results say

**1. Predicting transactions well is not the same as predicting interventions well.** The
independent models are near the transaction-level ceiling. Yet for issuer degradations and
outages they predict almost no change: median relative error 1.00 and 0.93. They even get the
sign of a degradation's effect wrong 30% of the time.

This is not a tuning problem. I checked it directly:
- Lowering observed issuer health on HDFC UPI payments by 15pp moves the model's mean predicted
  success from 0.9675 to 0.9666.
- Gateway health is flat below 0.8.
- Adding monotonic constraints, so success cannot rise as health falls or load rises, did not
  change this. It moved the −15pp response by 0.2pp.

The cause is in the data. Degraded states are rare: about 1,500 of 775,802 training rows have
HDFC UPI health below 0.9. The observed health feature is also a noisy 15-minute trailing rate,
so most dips in it are noise, and don't predict failure. A model that fits observed data learns
exactly that weak association. An intervention, by contrast, states that reliability really has
dropped.

**2. The rule baseline wins on direct interventions because it assumes the structure.** Its
ratio rule, success scales with current health ÷ normal health, is a slope-of-one assumption. It
happens to match how degradations and outages act, so the rules are accurate even far out of
range, with median relative error 0.03. That's domain knowledge, not learning. The first version
of the rule baseline fitted that slope from data, and it failed the same way the ML models do.

**3. Nobody handles nonlinear load.** For traffic changes and reroutes, both predictors capture
only about half of the true effect, with median relative error of about 0.54. Saturation beyond
what history shows is exactly where both a band table and trees stop.

**4. Latency under failure is missed by both.** Out of range, P95 latency error is about 1.3 s
for both predictors. Neither ties latency to timeouts, so an outage's jump in P95 is invisible to
them.

## Implications for the shared representation (Phase 4)

A Set Transformer trained on the same observed data will face the same identification problem on
health-driven interventions. Its extra capacity does not create signal that the data lacks. The
honest test of the Vulcan-style representation is therefore:
- whether it improves on load, routing and combined scenarios, where the effect *is* in the
  data;
- whether it matches the trees at the transaction level;
- how it behaves on cold start.

It should not be expected to beat a structural rule on degradations. How much natural incident
variation training data needs before learned models catch up is a question worth answering
directly, as an experiment.
