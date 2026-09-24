# Synthetic ecosystem and true data-generating process

This document describes every assumption behind the synthetic data. All numbers are synthetic.
Bank, gateway and city names are labels; nothing here describes the real institutions or real
Indian payment infrastructure.

Constants are referenced by name rather than repeated here, so this file does not drift from the
code. The module that defines each one is given in brackets.

## Pipeline

```text
config ──► entities ──► traffic ──► first attempts (context only)
                                          │
              episodes + latent params ───┤
                                          ▼
                          outcome process (stages + retries)
                                          │
                                          ▼
                     observed state features ──► transactions.parquet
```

Each stage draws from its own child of `SeedSequence(config.seed)`
(`generator.generate_dataset`). Changing one stage therefore does not reshuffle the others. The
same seed produces byte-identical Parquet.

## Entities (`catalog.py`, `entities.py`)

| Entity | Source | Notes |
|---|---|---|
| Cities | 20 fixed, weighted by size | |
| Issuers | 12 fixed, with bank type and market share | Customers hold one bank, drawn by market share |
| Gateways | 4 fixed, with supported methods and routing share | Every gateway supports UPI; netbanking only on `gateway_a` and `gateway_b` |
| Merchant categories | 10 fixed, with average order value, method mix and online flag | |
| Customers | `n_customers`, with a segment (new, regular or power) | Segment sets activity and purchase history |
| Merchants | `n_merchants`, with category, size, city and primary gateway | Size sets volume; each merchant's method mix is a Dirichlet draw around its category's |

## Traffic (`traffic.py`)

Expected volume per minute is the product of three factors:
- the hour-of-day curve (`HOURLY_PROFILE`), which is quiet overnight, has a lunch bump and peaks
  in the evening;
- a weekend bump (`WEEKEND_MULTIPLIER`);
- a festival factor (`festival_traffic_multiplier`) on `n_festival_days` random days.

Counts per minute are Poisson, scaled so the expected total is `target_transactions`.
`PEAK_HOURS` defines `is_peak`.

## First attempts (`generator.py`)

Each transaction's context is built in this order, which is what gives the data its correlations:
1. **Merchant**, in proportion to its volume weight.
2. **Customer.** For offline merchants, `offline_local_share` of customers come from the
   merchant's city. Online merchants draw from everyone. Both are weighted by activity.
3. **Method**, from the customer's preference (`PREFERRED_METHOD_WEIGHT`) multiplied by the
   merchant's method mix.
4. **Amount**, lognormal around merchant average order value × customer spend level × a method
   factor. The method factor (`METHOD_AMOUNT_FACTOR`) makes UPI skew small and netbanking large.
5. **Device.** Usually the customer's own. `new_device_rate` of payments use a different device.
6. **Gateway.** The merchant's primary gateway, unless it doesn't support the method or the
   payment spills over (`gateway_spillover_rate`). Otherwise a supporting gateway is chosen by
   routing share.

## Episodes (`episodes.py`)

Scheduled disruptions give the training data natural variation in ecosystem state:

| Kind | Effect | Magnitude range | Duration range |
|---|---|---|---|
| `issuer_degradation` | Subtracts from issuer reliability. `ISSUER_DEGRADATION_UPI_ONLY_SHARE` of episodes hit UPI only | `ISSUER_DEGRADATION_MAGNITUDE` | `ISSUER_DEGRADATION_MINUTES` |
| `gateway_brownout` | Multiplies gateway pass-through by (1 − magnitude) | `GATEWAY_BROWNOUT_MAGNITUDE` | `GATEWAY_BROWNOUT_MINUTES` |
| `gateway_outage` | Gateway pass-through is 0 | 1.0 | `GATEWAY_OUTAGE_MINUTES` |

**These ranges define what counts as in-distribution.** A counterfactual outside them, such as an
issuer degradation larger than the top of `ISSUER_DEGRADATION_MAGNITUDE`, is extrapolation. The
experiments test that deliberately.

## Latent parameters (`latent.py`)

These are hidden ground truth. They are saved under `latent/` next to the observed tables, and
**must never be used as model features**.
- **Issuer reliability** per (issuer, method): a bank-type base, plus a method offset, plus
  per-issuer jitter.
- **Issuer latency factor** and **issuer UPI load sensitivity**, both by bank type. Public banks
  are the most sensitive.
- **Gateway capacity**: `GATEWAY_CAPACITY_HEADROOM` × the gateway's mean first-attempt load per
  `UTILIZATION_BUCKET_MINUTES` bucket. `gateway_a` has the least headroom, so it saturates first
  on festival evenings. Capacity is fixed when the dataset is generated. Counterfactuals that move
  traffic do not change it.

## Outcome process (`dgp.py`)

An attempt passes through the stages below in order. Each stage has a failure probability, and
the first stage whose draw falls below its probability sets the failure reason.

| Stage | Failure probability grows with | Reasons |
|---|---|---|
| risk | new device, new customer, high-risk category, amount (a logistic score) | `FRAUD_REJECTION` |
| auth | method (card and netbanking above UPI); ×2 on a new device | `AUTHENTICATION_FAILED` |
| funds | amount | `INSUFFICIENT_FUNDS` |
| limit | UPI with an amount over `UPI_LIMIT_AMOUNT` | `LIMIT_EXCEEDED` |
| gateway | gateway health (from episodes); utilisation above `GATEWAY_SATURATION_START`, quadratically | `NETWORK_ERROR`, `GATEWAY_TIMEOUT` |
| issuer | issuer reliability minus degradation; for UPI also peak hours, and utilisation above `ISSUER_LOAD_START` scaled by load sensitivity | `UPI_TIMEOUT` (UPI only, `UPI_TIMEOUT_SHARE` of issuer failures), `ISSUER_DECLINED` |
| residual | constant | `UNKNOWN` |

Utilisation is the gateway's attempts in the current `UTILIZATION_BUCKET_MINUTES` bucket divided
by its capacity. It includes retries.

**Latency** is lognormal. Its median is set by the method, then scaled by the issuer and gateway
latency factors, by utilisation above `LATENCY_LOAD_START`, and slightly by amount. Timeout
failures take a fixed ceiling instead (`TIMEOUT_LATENCY_MS`).

### Keyed randomness (`keyed_random.py`)

Every draw is a pure function of (seed, transaction id, stream). A transaction gets the same
draws however many other transactions exist, which gives two guarantees:
- **Replay is exact.** Re-running the process with no change reproduces the dataset.
- **Counterfactuals are paired.** A transaction present in both runs sees identical draws.

Each stage compares a fixed draw against a probability. So a change that only raises failure
probabilities can only turn successes into failures. The change between two runs is the effect
of the intervention, not sampling noise.

### Retries

A failed first attempt is retried once, with probability `RETRY_PROBABILITY[reason]`.
- The retry lands 1–2 minutes later.
- A non-UPI payment switches to UPI with probability `RETRY_SWITCH_TO_UPI`. The gateway is kept,
  since every gateway supports UPI.
- A retry's id is its parent's id + `RETRY_ID_OFFSET`, so it has its own draws.
- Retries are never retried themselves.

Retries add load, load raises failure rates, and those failures create more retries. `solve`
therefore iterates until the set of retries stops changing. The set starts empty and can only
grow, so this converges to a well-defined fixed point. It raises an error rather than returning
a partial answer if `MAX_FIXED_POINT_ITERATIONS` is exceeded.

## Observed state features (`state.py`)

These are the only view of ecosystem state a model gets:

| Feature | Definition |
|---|---|
| `issuer_health` | Success rate of the (issuer, method) over the previous `HEALTH_WINDOW_MINUTES`, **excluding the current minute**, smoothed toward the group mean with `HEALTH_PRIOR_WEIGHT` pseudo-attempts |
| `gateway_health` | The same, per gateway |
| `gateway_utilization` | Current bucket load ÷ capacity, the same quantity the outcome process uses. Treated as observable system telemetry |
| `hour`, `is_peak`, `is_festival` | From the timestamp and festival calendar |

## Ground truth (`simulation/oracle.py`)

`true_outcomes(dataset, context=..., episodes=...)` re-runs the process under a changed world:
- a modified first-attempt context: rerouted gateways, shifted methods, added transactions;
- a modified episode schedule: extra degradations or outages.

Added transactions need new ids that are unique and below `RETRY_ID_OFFSET`.

## Known simplifications

- **Load has little effect at a normal peak.** At normal evening peaks the gateways sit well
  below saturation, so stress comes mainly from festival evenings and episodes. Peak-hour effects
  on a normal day are small by design.
- **Retry load feeds back within a bucket.** A retry that lands in the same utilisation bucket as
  its parent adds to the load that parent saw. The fixed point is still unique and deterministic,
  but it is not strictly causal at sub-bucket resolution.
- **One bank per customer.** Card and UPI payments from the same customer go through the same
  issuer.
- **No behavioural learning.** Customers don't abandon a merchant or switch methods permanently
  after a failure; the only reaction is the single retry.
- **Health priors use the whole period.** The smoothing prior in `issuer_health` and
  `gateway_health` is the group's success rate over the full dataset. This is a mild look-ahead
  in the feature; the training split in Phase 3 should recompute priors from the training period
  only.
- **Merchant "unusualness" is not modelled.** The risk stage uses a high-risk-category flag, not
  whether the merchant is unusual for that customer.
