# Counterfactual engine

The engine answers "what happens if X changes?" for one evaluation window. It then checks the
answer against the ground truth, which is the true outcome process re-run on exactly the same
transactions. All results are simulation output on synthetic data.

```text
scenario ──► apply_scenario ──┬─► engine view (observed data only) ──► predictor ──► metrics ─┐
                              │                                                                ├─► compare
                              └─► ground-truth view (+ latent episodes) ──► true process ──► metrics ─┘
```

## Scenarios (`simulation/scenarios.py`)

A scenario is one or more interventions applied together. Every intervention accepts
`start_hour` (0–23, default 0) and `duration_minutes` (default: until the end of the window).

| Type | Fields | Meaning |
|---|---|---|
| `issuer_degradation` | `issuer`, optional `method`, `success_rate_delta` | Issuer success rate drops by an **absolute** amount: −0.15 means 15 percentage points |
| `gateway_outage` | `gateway` | The gateway fails every payment routed to it |
| `traffic_change` | optional `segment` (a method), `volume_delta` | Demand changes by that fraction. New demand is resampled from existing demand in the same segment and time span |
| `method_shift` | `from`, `to`, `percentage` | That share of `from` payments switch method. If the gateway can't take the new method, the payment is rerouted by the usual routing shares |
| `routing_change` | `source_gateway`, `target_gateway`, `traffic_percentage` | That share of the source gateway's traffic moves to the target. Only payments the target supports are moved |

Unknown issuers or gateways, out-of-range values and unknown fields are rejected at validation.

Interventions that change which transactions exist or where they are routed are applied first,
in scenario order. Degradations and outages are applied last. That way, "gateway_a is down, so
reroute 30% to gateway_b" gives the rerouted payments gateway_b's state.

Random choices, such as which card payments switch to UPI, are seeded from the dataset seed and
the intervention's position. The same scenario always gives the same result.

## Evaluation window

By default the window is one representative normal day: the non-festival weekday with the median
first-attempt success rate. A scenario can set `window_date` instead.

Models are fitted on every other day, never on the window (`models/splits.py`).

## What the engine may see

The engine gets only observable data. `interventions.observe` builds it from the transactions
table:
- the trailing health tables for every (issuer, method, minute) and (gateway, minute);
- observed load per gateway and utilisation bucket;
- gateway capacity, recovered as observed load ÷ observed utilisation.

The engine never reads the latent parameters. For an unchanged window its features reproduce the
observed features exactly, which is tested.

For a changed context, the engine recomputes features as follows:
- **Health:** looked up at each payment's new (issuer, method, gateway, minute).
- **Utilisation:** observed load, plus the change in first-attempt load, ÷ capacity. Retries the
  scenario would add or remove are not predicted. The engine has no behaviour model yet.
- **Degradation:** lowers `issuer_health` by the stated amount from the **first minute** of the
  span.
- **Outage:** sets `gateway_health` to 0.

The observed health feature lags, because it is a 15-minute trailing rate. The scenario states
the new state directly, though, so applying it at once gives a better estimate than simulating
the lag. With a lag, a short outage would be mostly invisible to the model.

## Ground truth

The ground truth runs `true_outcomes` twice on the same transactions:
- the window plus a 30-minute warm-up, unchanged;
- the same, with the scenario's context changes and its interventions added as latent episodes.

Draws are keyed by transaction id, so the difference between the runs is the effect of the
scenario, not sampling noise. The warm-up makes the unchanged run reproduce the full dataset
inside the window exactly, including retries and load. This is tested.

## Metrics (`simulation/metrics.py`)

Every metric is an expectation under a `Predictions` object. A model supplies probabilities; the
ground truth supplies realised outcomes as probability 1 or 0. Both go through the same code.

- **Scope: first attempts in the window.** The engine predicts each order's first attempt. The
  ground truth still simulates retries and their load, but they are not counted in the metrics.
- **Success, failure and timeout rates.** The timeout rate is P(fail) × P(timeout reason | fail).
- **Latency.** The average is the lognormal mean. P95 is the 95th percentile of the latency
  mixture across rows, sampled with a fixed seed.
- **GMV, successful GMV and failed GMV.**
- **GMV at risk:** counterfactual failed GMV minus baseline failed GMV scaled to the scenario's
  volume. A pure volume change with unchanged reliability therefore has no GMV at risk. A
  negative value means reliability improved.
- **Transactions affected:** rows whose method, gateway, health or utilisation changed, plus new
  rows.
- **Segments.** Impact per issuer, gateway, method, merchant category and city. Each side is
  grouped by its own values, so a rerouted payment counts under its old gateway in the baseline
  and its new one in the counterfactual. "Most affected" lists segments whose success rate moved
  at least 0.5 pp, ranked by GMV at risk.

## Rule baseline (`models/rule_baseline.py`)

This is the hand-authored comparison point. Everything in it is estimated from observed history:
- **Success:** the historical rate of the (issuer, method, gateway, 4-hour block) cell, shrunk
  toward coarser rates, then multiplied by three factors:
  - current ÷ normal issuer health;
  - current ÷ normal gateway health;
  - a historical load factor for the current utilisation band.
- **Failure reason:** the historical mix per method and low-health flags.
- **Latency:** the historical log-latency per method, gateway and utilisation band.

A first version fitted linear slopes on the health features instead. Observed health is noisy,
with about 10 attempts per (issuer, method) per 15 minutes, so the fitted slope was attenuated
to almost zero. That version predicted no effect at all from an issuer degradation. The ratio
rules have no slope to attenuate.

## Results: rule baseline vs ground truth

Reproduce with `python scripts/generate_data.py && python scripts/simulate.py`. The window is
2026-01-30, with 31,223 first attempts. Baseline success rate is 96.15% for the rules against
96.25% true.

| Scenario | Success rate change (rules / truth) | Extra failures (rules / truth) | GMV at risk, ₹ lakh (rules / truth) | P95 latency change, ms (rules / truth) |
|---|---|---|---|---|
| HDFC UPI success rate drops 15pp | -1.67 / -1.72 pp | +523 / +536 | +11.15 / +10.99 | +0 / +323 |
| gateway_a down for 60 minutes at the evening peak | -2.80 / -2.80 pp | +874 / +873 | +28.89 / +28.32 | +0 / +1,244 |
| UPI traffic up 30% | +0.04 / +0.05 pp | +232 / +219 | -0.92 / -2.15 | -99 / -130 |
| 20% of card payments move to UPI | +0.05 / +0.02 pp | -15 / -7 | -0.99 / +0.05 | -33 / -49 |
| Reroute 25% of gateway_a traffic to gateway_b | +0.01 / -0.02 pp | -3 / +6 | +0.00 / +0.18 | +89 / +114 |
| Festival-scale surge: all traffic up 80% | -1.81 / -3.92 pp | +1,977 / +3,137 | +31.77 / +54.46 | +1,320 / +5,000 |

What this shows:
- **Direct interventions are easy for rules.** A degradation or outage maps straight onto a
  health feature, and a ratio rule turns that into the right success-rate change.
- **Nonlinear load is where the rules fail.** The 80% surge pushes gateways past saturation. The
  rules capture less than half the drop, and far less of the latency blow-up. Their load factor
  is a coarse band table from a history where saturation was rare.
- **Rules miss the latency cost of failures.** Their latency depends on method, gateway and load,
  not on whether a payment times out. So an outage shows no P95 change even though every timed-out
  payment waits for the full timeout.

These are single runs on one window, so there are no confidence intervals yet. Those come with
the evaluation harness in Phase 3.
