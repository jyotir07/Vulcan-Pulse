# Architecture

Vulcan Counterfactual answers "what if?" questions about a synthetic payment ecosystem and checks
every answer against the synthetic world's own ground truth. The whole system serves one loop:

```text
ask ──► simulate ──► predict ──► explain ──► mitigate ──► simulate again
 │                                                             │
 └──────────────── validated against ground truth ─────────────┘
```

## Components

```text
                        configs/ecosystem.yaml
                                 │
                                 ▼
   ┌──────────────────────── backend/data ─────────────────────────┐
   │ entities ─► traffic ─► first attempts ─► true outcome process │──► transactions.parquet
   │                         (dgp.py: stages, load, retries)       │    + latent/ (ground truth)
   └───────────────────────────────────────────────────────────────┘
                                 │ observed columns only
                                 ▼
   ┌──────────────────────── backend/models ───────────────────────┐
   │ rule_baseline       historical rates × health/load ratios     │
   │ independent         3 gradient-boosted trees                  │
   │ encoder + pretrain  Set Transformer, masked-field pretraining │
   │ heads + shared      success / reason / latency, 5-seed ensemble│
   └───────────────────────────────────────────────────────────────┘
                                 │ Predictor protocol: predict(frame) -> Predictions
                                 ▼
   ┌────────────────────── backend/simulation ─────────────────────┐
   │ scenarios      Pydantic schema, names checked against catalog │
   │ interventions  scenario ─► changed context + state features   │
   │ engine         baseline vs counterfactual predictions, cached │
   │ oracle         true process re-run with the same random draws │
   │ metrics        rates, GMV at risk, P95, segments, intervals   │
   └───────────────────────────────────────────────────────────────┘
            │                      │                       │
            ▼                      ▼                       ▼
   backend/services        backend/evaluation        backend/api (FastAPI)
   attribution             harness, experiments      /overview /ecosystem/graph
   explanation             scoring                   /simulate /mitigate
   mitigation                                        /parse /experiments /health
   nl_parser (LLM)                                          │
   workspace                                                ▼
                                                   frontend (Next.js, Recharts)
```

## Key design decisions

1. **Interventions act through state features.** Every transaction carries the observed state of
   the ecosystem when it happened: trailing issuer and gateway health, gateway utilisation, peak
   and festival flags. An intervention rewrites those features (and gateway or method
   assignments), and a model predicts outcomes under the new state. Without state features,
   "HDFC −15 pp" would have no way into any model.
2. **The engine only sees observable data.** Capacity is recovered from observed load ÷ observed
   utilisation; load after a reroute is recomputed by bookkeeping. The engine never reads the
   latent parameters, so a model can't win by peeking at the generator.
3. **Ground truth is paired.** The oracle re-runs the true process on the same transactions with
   the same keyed random draws, once without and once with the scenario. The difference is the
   scenario's effect, not sampling noise.
4. **Every predictor is scored the same way.** The rule baseline, the independent trees and the
   shared ensemble all implement one `Predictor` protocol; the ground truth passes through the
   same metric code as degenerate 0/1 predictions.
5. **The LLM never predicts.** It only translates text into a scenario, which the same Pydantic
   schema validates; the numbers always come from the engine.
6. **Storage is files.** Parquet for data, JSON + `.pt` for models, an in-process LRU for API
   responses. Simulations are deterministic, so caching by request is sound without invalidation.

## Request path: `POST /simulate`

1. The request's scenario is validated (unknown issuers, out-of-range values → 422).
2. `Workspace.with_window` fills in the default day: the last normal weekday of the held-out test
   period, which no predictor trained on.
3. `apply_scenario` builds the baseline and counterfactual frames for that day.
4. `predict_scenario` predicts the baseline once per day (cached) and only the changed rows of the
   counterfactual.
5. `build_result` computes metrics, impact and, for the ensemble, a 90% interval.
6. Attribution, explanation, optional Shapley factors and the per-5-minute timeline are added.
7. With `ground_truth: true`, the oracle runs on the same frames for the validation panel.

One simulation runs at a time; each already uses all the CPU threads the model is configured for.

## Where to read more

- `docs/dgp.md`: every assumption in the synthetic world.
- `docs/engine.md`: scenario semantics, what the engine may see, metrics.
- `docs/evaluation.md`: the harness and the model comparison.
- `docs/results.md`: the Phase 5 experiments.
- `docs/limitations.md`: what this is not.
