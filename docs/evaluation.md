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
| `rule_baseline` | Historical success rate per (issuer, method, gateway, 4-hour block), scaled by current / normal health ratios and a load-band factor (`models/rule_baseline.py`) |
| `independent_ml` | Three separate gradient-boosted tree models for success, failure reason and log latency, on transaction fields, observed state and entity attributes (`models/independent.py`) |
| `shared_representation` | Ensemble of five Set Transformer encoders, masked-field pretrained then fine-tuned jointly on the three outcomes (`models/shared.py`, trained by `scripts/train.py`) |
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

A Set Transformer trained on the same observed data faces the same identification problem on
health-driven interventions. Its extra capacity does not create signal the data lacks. The honest
test of the Vulcan-style representation is therefore:
- whether it improves on load, routing and combined scenarios, where the effect *is* in the
  data;
- whether it matches the trees at the transaction level;
- how it behaves on cold start.

It should not be expected to beat a structural rule on degradations. How much natural incident
variation training data needs before learned models catch up is a question worth answering
directly, as an experiment.

## The shared representation, measured

Trained by `python scripts/train.py`: five members, `d_model` 64, two set-attention blocks, masked-
field pretraining then four fine-tuning epochs. 84 minutes on 8 CPU threads.

**Pretraining works.** Masked-field accuracy is 0.528 against a 0.386 frequency baseline, and
pretraining loss falls within two epochs (1.393 to 1.269). The encoder learns the field
co-occurrences before it sees an outcome.

The per-field breakdown is the interesting part, because it shows where the structure is:

| Field | Masked acc. | Frequency | Gain |
|---|---|---|---|
| merchant_category | 0.591 | 0.152 | +0.439 |
| log_merchant_aov | 0.401 | 0.064 | +0.337 |
| customer_segment | 0.873 | 0.528 | +0.345 |
| bank_type | 0.887 | 0.585 | +0.302 |
| is_peak | 0.950 | 0.700 | +0.251 |
| gateway_utilization | 0.241 | 0.067 | +0.174 |
| online_merchant | 0.954 | 0.764 | +0.190 |
| hour | 0.253 | 0.102 | +0.151 |
| issuer_id | 0.356 | 0.206 | +0.150 |
| issuer_health | 0.162 | 0.065 | +0.097 |
| is_new_device | 1.000 | 1.000 | +0.000 |

Category, amount, segment and bank type are largely predictable from the other fields, as
expected. Note `gateway_utilization` at 0.241 against 0.067: the encoder *can* infer load, and it
does. What it cannot do is turn a health number into a success probability, because in the
observed data the two are barely associated — which is the whole of the Phase 3 finding, visible
here in the pretraining loss. `is_new_device` adds nothing over its own base rate; that field is
a deterministic function of its neighbours.

**At the transaction level it ties the trees and edges past them.** AUC 0.5944 against 0.5905 for
the independent models and a 0.6009 ceiling; Brier and log loss are equal to four decimal places;
reason log loss is the best of the three (1.718 against 1.730). So the representation costs
nothing in per-payment accuracy.

**On counterfactuals it wins where the effect is in the data, and not otherwise.** Success-rate
MAE in pp:

| Scenario type | rules | independent ML | shared |
|---|---|---|---|
| routing_change | 0.569 | 0.555 | **0.444** |
| traffic_change | 1.804 | 1.933 | **1.641** |
| method_shift | 0.083 | **0.020** | 0.030 |
| combined | **1.331** | 2.143 | 2.045 |
| gateway_outage | **0.007** | 1.060 | 0.826 |
| issuer_degradation | **0.024** | 0.681 | 0.679 |

This is the pattern Phase 3 predicted. On routing and traffic — where the true response runs
through nonlinear load saturation and the data contains the relevant variation — the shared
representation is the best of the three, and it keeps its advantage out of range. On degradation
and outage it is barely better than the trees, and both remain an order of magnitude worse than
the rule baseline, for the same reason as before: degraded states are too rare in training for any
model to learn the health-to-success slope. It gets the *sign* right on every degradation where
the trees get it wrong 30% of the time, which is a real gain, but it is not a slope.

**P95 latency improves out of range**, from 1,307 ms to 1,140 ms, the best of the three. Neither
learned model ties latency to timeouts, so an outage's latency jump stays hard to predict.

**Net effect on the headline numbers**, success-rate MAE: in range 0.206 (rules), 0.288 (shared),
0.337 (independent ML); out of range 0.827, 1.568, 1.772. The shared representation sits between
the rules and the trees overall, and it is the best learned model everywhere.

**Cold start is untested here.** The plan predicted the shared representation would win there,
via attribute-only fallbacks for unseen entities. That experiment is Phase 5, and these numbers
say nothing about it.

### What this does not show

The shared model trains on the same observations as the trees, so it inherits the same ceiling on
health-driven interventions. More capacity did not buy identification. If the shared
representation is to beat a structural rule on degradations, the fix is in the data — more
natural incident variation, or explicit supervision for the health-to-success slope — not in the
model.

## Uncertainty

`simulate` reports a 90% interval per impact field for ensemble predictors, from two independent
sources whose variances add (`simulation/metrics.py`, `impact_interval`):

- **model uncertainty**: the spread of the five members' own impact estimates;
- **sampling uncertainty**: a bootstrap that resamples five-minute blocks, not rows, since
  transactions in the same block share load and issuer health.

Both frames are resampled on the same drawn blocks, so baseline and counterfactual stay paired in
time. Bootstrap draws are computed from per-block sums, so a resample is a matrix product rather
than a re-scoring of 30,000 rows; that plus predicting the members once instead of twice brings
the interval's cost to about 3 seconds on a 31,000-transaction day, against 54 seconds for the
two ensemble passes it rides on.

Whether these intervals are honest is Phase 5's calibration experiment: it checks how often the
true impact lands inside the predicted interval. Do not read a narrow interval as a validated one
until then.
