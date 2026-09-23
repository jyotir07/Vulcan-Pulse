# Vulcan Counterfactual — Implementation Plan

Source spec: `proj.md`. This file is the *how*; `proj.md` is the *what*.

## 1. Context

We are building a Vulcan-inspired research prototype. It answers "what if?" questions about a **synthetic** Indian payment ecosystem and checks every answer against known ground truth. The spec's guiding loop is **ask → simulate → predict → explain → mitigate → simulate again**. We build that loop end to end first, then make each stage better. We do not build all the stages to full depth before any of them connect.

## 2. Key design decisions

These are the decisions the spec leaves open. Everything below depends on them.

1. **Interventions act through ecosystem-state features.** Each transaction carries per-transaction fields (amount, method, issuer, merchant category, city, device, hour, gateway). It *also* carries state features describing the ecosystem at that moment:
   - `issuer_health`: rolling observed success rate of the issuer × method
   - `gateway_health`
   - `gateway_utilization`: load ÷ capacity
   - `traffic_multiplier`
   - `is_peak`, `is_festival`

   An intervention rewrites these state features, and assignments such as gateway or method. The model then predicts outcomes under the new state. Without state features, "HDFC −15%" has no way to enter the model.
2. **Training data must contain state variation.** The generator injects natural **episodes**: issuer degradations, gateway brownouts and outages, traffic peaks and festival days. So the model sees degraded states. Episode magnitudes are bounded, e.g. issuer degradation up to 10%. Interventions beyond that range test extrapolation on purpose.
3. **Ground truth comes from paired seeded runs.** The oracle re-runs the true data-generating process (DGP) on the *same* transactions with the same random seeds, once with the intervention and once without. The difference is the intervention's effect, not sampling noise.
4. **The evaluation avoids a circular result.** The model is learning our own DGP, so the headline numbers come from **held-out interventions**:
   - magnitudes outside the training range
   - unseen combinations of interventions
   - intervention types absent from training episodes
   - held-out merchants and issuers
5. **The rule baseline is not the DGP.** It uses historical success rates per segment plus linear adjustments. It never calls the generator's formulas.
6. **Load is recomputed deterministically.** When an intervention moves traffic (reroute, method shift, volume change), the engine recomputes `gateway_utilization` from the modified transaction set, using the accounting formula load = volume ÷ capacity per time bucket. This is observable bookkeeping, not the outcome model, so it does not leak the DGP.
7. **The LLM never predicts.** It only turns natural language into a scenario JSON, which Pydantic validates. The numbers always come from the engine.
8. **Storage is simple.** Parquet via pyarrow, no Postgres or Redis. Simulation results are cached in-process, keyed by (scenario hash, model version, seed). The runs are deterministic, so the cache is sound. Add a database only if a real need appears.

## 3. Tech stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2, NumPy, Pandas, pyarrow, PyTorch, scikit-learn (`HistGradientBoosting*` for the independent baselines, so no new gradient-boosting dependency).
- **Tooling:** pytest, ruff.
- **Frontend:** Next.js (App Router), TypeScript, Tailwind, Recharts.
- **Packaging:** Docker plus `docker-compose.yml` with two services: api and web.
- **LLM (Phase 8):** a provider-agnostic interface. The key stays server-side, and structured output is validated by Pydantic.

## 4. Repository layout

```text
rzp-ai/
├── proj.md, plan.md, README.md, LICENSE
├── pyproject.toml, docker-compose.yml
├── configs/                  # ecosystem.yaml, train.yaml, experiments/*.yaml
├── backend/
│   ├── data/                 # entities, episodes, generator, dgp (true process), io
│   ├── simulation/           # oracle (ground truth), scenarios (schema), interventions, engine, metrics
│   ├── models/               # rule_baseline, independent (sklearn), encoder (set transformer), heads, pretrain, train
│   ├── evaluation/           # harness, metrics (MAE/RMSE/calibration/CI), reports
│   ├── services/             # attribution, explanation, mitigation, nl_parser
│   └── api/                  # FastAPI app, routes, response schemas
├── frontend/                 # app/, components/, charts/, lib/
├── experiments/              # runnable scripts + result JSON/MD per experiment
├── data/generated/           # parquet (gitignored), data/schemas/
├── scripts/                  # generate_data.py, train.py, run_experiments.py
├── tests/
└── docs/                     # architecture.md, dgp.md, results.md, limitations.md
```

## 5. Phases

Each phase ends with passing tests and lint, a verified exit criterion, and a commit message handed to you. I do not commit.

The spec's phases map onto these as follows:

| Spec phase | Plan phase(s) |
|---|---|
| 1 Synthetic ecosystem | 1 |
| 2 Baseline models | 2 (rules), 3 (independent ML) |
| 3 Shared representation | 4 |
| 4 Counterfactual engine | 2 (built early as the vertical slice), then extended |
| 5 Validation | 3 (harness), 5 (experiments) |
| 6 Dashboard | 6 (explain/mitigate), 7 (API + UI) |
| 7 Natural language interface | 8 |

### Phase 0 — Scaffolding
- **Build:**
  - `pyproject.toml` with ruff and pytest configured
  - package skeleton
  - `configs/ecosystem.yaml`
  - `.gitignore` covering data and model artifacts
  - one smoke test
- **Exit:**
  - `pytest` passes
  - `ruff check .` is clean

### Phase 1 — Synthetic ecosystem and ground-truth oracle
- **Entities** (default sizes, all configurable):
  - ~50k customers
  - ~2k merchants across ~10 categories and 3 sizes
  - ~12 issuers
  - 4 gateways, each with supported methods, capacity and latency profile
  - ~20 cities
  - Customers have segment, city, device, preferred method and amount distribution. Merchants have category, size, AOV, volume and preferred methods.
- **Transaction sampling:** time-of-day and day-of-week intensity, then merchant (by volume), then customer (by city and segment affinity), then method (customer × merchant preferences), then issuer (customer), then gateway (routing policy + method support), then amount (lognormal by merchant AOV and customer).
- **Episodes:** a schedule of issuer degradations, gateway brownouts and outages, peak and festival traffic, with bounded magnitudes and durations.
- **True DGP (`backend/data/dgp.py`):**
  - Failure is a logistic model over context: issuer × method reliability, gateway health, a `gateway_utilization` saturation term (nonlinear above ~0.8), peak, amount, and customer tenure and device novelty (for risk).
  - Failure *reason* is a softmax over the taxonomy in the spec, conditioned on the dominant cause.
  - Latency is lognormal and grows with utilization and issuer latency profile.
  - A simple retry and method-switch behaviour applies after a failure.
  - Each transaction has its own seeded random stream, so the draws are reproducible and can be paired.
- **Oracle (`simulation/oracle.py`):** `true_outcomes(transactions, state_overrides, seed)`. It re-runs the DGP under an intervention using the same random streams.
- **Output:** ≥1M transactions (~30 simulated days) as Parquet, plus entity tables, plus the state-feature columns.
- **Tests:**
  - Determinism: same seed gives identical bytes.
  - Designed correlations hold: failure rate rises with utilization, UPI plus a degraded issuer at peak gives more timeouts, the travel category on gateway A at peak has higher latency.
  - The oracle with no intervention reproduces the observed outcomes exactly.
- **Exit:** `scripts/generate_data.py` produces 1M+ rows reproducibly. `docs/dgp.md` describes every assumption.

### Phase 2 — Scenario schema, counterfactual engine v0, rule baseline (the vertical slice)
- **Scenario schema (`simulation/scenarios.py`):** a Pydantic discriminated union.
  - Types: `issuer_degradation`, `gateway_outage`, `traffic_change`, `method_shift`, `routing_change`.
  - Optional scope fields: method, time window, duration.
  - A composite scenario is a list of these.
- **Interventions (`simulation/interventions.py`):** pure functions of (transactions, state) → (modified transactions, state).
  - Rewrite health features.
  - Reassign gateway or method, respecting method support.
  - Resample rows for volume changes.
  - Recompute utilization.
- **Engine (`simulation/engine.py`):** `simulate(scenario, predictor)`.
  1. Select the evaluation window (default: one representative day).
  2. Predict on the baseline.
  3. Apply the intervention.
  4. Predict on the counterfactual.
  5. Compute the metric difference.

  The `predictor` is an interface (`predict_success`, `predict_failure_reason`, `predict_latency`), so any of the three model families can plug in.
- **Metrics (`simulation/metrics.py`):**
  - Payment: success, failure and timeout rates; average and P95 latency.
  - Business: volume, GMV, GMV at risk, successful and failed GMV.
  - Ecosystem: affected issuers, gateways, merchants, cities and methods.
  - All metrics are computed from *expected* values: the sum of p(success) × amount.
- **Rule baseline (`models/rule_baseline.py`):** historical rates per (issuer, method, gateway, hour bucket) with linear state adjustments.
- **Exit:** a CLI prints the full before/after/impact block for each of the five scenario types, using the rule baseline and the oracle side by side.

### Phase 3 — Evaluation harness and independent ML baselines
- **Splits:**
  - time-based train/val/test
  - held-out merchants (~10%) and issuers (1–2)
  - normal-only training subset for the shift experiment
- **Independent models (`models/independent.py`):** three separate sklearn models — success, failure reason, latency — trained on field and state features. They conform to the predictor interface.
- **Harness (`evaluation/harness.py`):**
  - Takes a grid of scenarios (type × magnitude × scope) and a list of predictors.
  - Computes the predicted impact and the oracle's true impact.
  - Reports MAE, RMSE and relative error on Δsuccess-rate, GMV at risk and Δfailures.
  - Reports transaction-level metrics: AUC, Brier score, reliability curve, latency MAE.
  - Writes JSON and a Markdown table.
- **Exit:** a harness report comparing rules vs independent ML on in-range and out-of-range scenarios.

### Phase 4 — Shared payment representation
- **Encoder (`models/encoder.py`):**
  - Field embeddings: categorical lookups plus numeric embeddings via binned or MLP projection, each added to a field-identity embedding.
  - A Set Transformer (ISAB/SAB blocks, PMA pooling) or an equivalent permutation-invariant transformer.
  - Configurable `d_model`, default 128.
- **Pretraining (`models/pretrain.py`):** masked-field prediction over categorical fields and binned numerics, masking 15–30% of fields per row.
- **Heads (`models/heads.py`):** success (BCE), failure reason (CE), latency (regression on log latency). These are fine-tuned jointly on the shared representation.
- **Entity handling for cold-start:** entities go in as *attribute* fields (category, size, city, bank type) plus a hashed ID with dropout. Unseen entities then fall back to their attributes instead of an untrained ID embedding.
- **Uncertainty:** an ensemble of K=5 seeds, which gives predictive spread. The engine turns it into confidence intervals together with a bootstrap over transactions.
- **Scale check:** 1M rows × ~12 fields with a small model should train on CPU or a laptop GPU. The epoch count is set in `configs/train.yaml`.
- **Exit:**
  - Pretraining loss falls and masked-field accuracy beats the frequency baseline.
  - The shared model plugs into the engine and the harness unchanged.

### Phase 5 — Experiments (the scientific core)
All experiments are run from `scripts/run_experiments.py` and are reproducible from configs and seeds. Results go to `experiments/*/results.{json,md}` and `docs/results.md`.
1. **Main experiment:** does the shared representation improve counterfactual prediction? Test all three predictors on:
   - normal traffic
   - peak traffic
   - issuer degradation
   - gateway outage
   - new merchant
   - new customer
   - method shift
   - distribution shift

   Split each into in-range and out-of-range magnitudes.
2. **Cold-start:** known vs held-out merchants and issuers, comparing transaction-level and counterfactual error.
3. **Distribution shift:** train on normal-only traffic, test on peak and festival traffic, and report how much each predictor degrades.
4. **Calibration of uncertainty:** check how often the true impact falls inside the predicted 90% interval.
- **Exit:** results tables exist for all four. Findings are reported **as they are**, including where the shared model does not win.

### Phase 6 — Explain, attribute, mitigate
- **Attribution (`services/attribution.py`):**
  - Exact additive decomposition of Δmetric by segment: issuer, method, gateway, merchant category, hour bucket, city.
  - Optional factor attribution: Shapley values over up to 3 factors (intervention, peak, gateway congestion), computed by re-running the engine with each factor neutralised.
- **Explanation (`services/explanation.py`):** template text filled *only* from computed quantities, e.g. traffic share of the affected issuer, concentration in high-volume merchants, peak share, utilization. No LLM is involved.
- **Mitigation (`services/mitigation.py`):**
  1. Generate candidate reroutes: shift 10/20/30/50% of affected traffic to each compatible gateway.
  2. Evaluate each with the engine. Capacity saturation makes over-rerouting backfire, and that is intended.
  3. Rank by recovered success rate and GMV.
  4. Label every result "simulation output".
- **Exit:** the CLI runs the full loop for the HDFC demo scenario: impact, attribution, explanation, ranked mitigations, then a re-simulation with the chosen mitigation.

### Phase 7 — API and dashboard
- **API (`backend/api`):**
  - `GET /overview`: ecosystem health
  - `GET /ecosystem/graph`: nodes and edges with volume and success
  - `POST /simulate`: the request and response shapes from spec §29, plus confidence intervals, attribution and explanation
  - `POST /mitigate`
  - `GET /experiments`
  - `GET /health`
  - Validation: invalid scenarios return 422 with Pydantic detail.
  - Result caching as in design decision 8.
  - Target latency: `/simulate` in <3s on CPU for a one-day window. This is a target, not a measured figure.
- **Frontend:** one page with the sections from spec §24:
  - Overview tiles
  - Ecosystem graph
  - What-if builder: scenario dropdown, magnitude, duration
  - Impact: before/after bars, GMV-at-risk waterfall, affected segments, explanation
  - Mitigation comparison
  - Validation panel: predicted vs true for the current scenario
  - Outage timeline, bucketed per minute
  - A persistent "synthetic data" banner
- **Exit:**
  - `docker compose up` brings up both services.
  - The demo story in spec §32 runs end to end in the browser.

### Phase 8 — Natural-language scenario input
- **Parser:** `services/nl_parser.py` behind a provider-agnostic interface (`ScenarioParser.parse(text) -> Scenario`).
  - Structured output validated against the scenario schema.
  - Entity names resolved against the ecosystem registry.
  - Timeout and one retry. On failure, it asks the user to clarify rather than guessing.
- **Key handling:** the key is server-side only (env var), with a fallback to the form builder when no key is configured.
- **Endpoint:** `POST /parse`. The UI shows the parsed scenario for confirmation before simulating.
- **Exit:** a test set of ~30 phrasings parses to the expected scenarios, with the LLM mocked in unit tests plus one live smoke run.

### Phase 9 — Documentation and demo
- **README:**
  - pitch
  - the Vulcan acknowledgement using spec §34 wording
  - architecture diagram
  - quickstart
  - reproduction commands
  - benchmark tables
  - limitations from spec §36
- **Docs:** `docs/architecture.md`, `dgp.md`, `results.md` and `limitations.md`.
- **Demo:** a 3–5 minute demo script following spec §32. You record the video.

## 6. Verification (applies at every phase)

- `pytest` and `ruff check .` pass. From Phase 7, the frontend also passes `npm run lint` and `npm run build`.
- Reproducibility: `scripts/generate_data.py --seed 42` twice gives identical Parquet hashes, and experiments rerun to the same numbers.
- End to end: `docker compose up`, then run the HDFC scenario, check that the impact, attribution and mitigation render, and that the validation panel shows oracle vs predicted.
- Each phase report says exactly what was run and its output.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The model looks good only because it learned our own DGP | Held-out magnitudes, combinations and entities; results reported honestly |
| The shared representation does not beat independent GBMs | This is a valid finding; report it and analyse where each model wins (cold-start is the likely win) |
| The DGP is too simple, so everything is easy | Nonlinear saturation, interactions and episode noise; check difficulty in Phase 3 before building Phase 4 |
| Scope creep across 10 phases | The vertical slice (Phases 1–2) is usable on its own; later phases deepen it without rewriting it |
| Synthetic numbers mistaken for real ones | Disclaimer in the README, a UI banner and the API response metadata |
