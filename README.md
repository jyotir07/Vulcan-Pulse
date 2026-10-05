# Vulcan Counterfactual

> **Ask what happens before you change the payment network.**

A Vulcan-inspired payment ecosystem simulator that uses learned representations and
counterfactual modeling to estimate the impact of issuer degradation, gateway outages, traffic
shifts, payment-method changes and routing decisions — and checks every estimate against known
ground truth.

**A Vulcan-inspired research and engineering prototype built on synthetic payment data.** Every
number in this repository is simulation output from a synthetic ecosystem. None of it describes
real payment performance, real banks or gateways, or Razorpay.

## Why

Dashboards say what happened; prediction models say what is likely to happen. This project
explores a third question: **what is likely to happen if the system changes?**

```text
ask ──► simulate ──► predict ──► explain ──► mitigate ──► simulate again
 │                                                             │
 └──────────────── validated against ground truth ─────────────┘
```

Because the data is synthetic, the simulator knows the true generating process. Every
counterfactual is re-run through that process on the same transactions with the same random
draws, so predictions are scored against the true effect instead of a plausible-looking guess.

## Relation to Razorpay's Vulcan

Razorpay's Vulcan explores the idea of a shared foundation model for India's payment ecosystem,
where a common representation learned from structured payment data can support multiple
downstream decisions.

This project explores a different question on top of that idea:

**Can a learned payment representation be used to estimate the impact of changes to the payment
ecosystem before those changes happen?**

Vulcan Counterfactual is an independent research prototype built entirely on synthetic data. It
does not reproduce or use Razorpay's proprietary model, data, infrastructure, or algorithms.

## Architecture

```text
configs ─► synthetic ecosystem (backend/data) ─► 1M transactions + latent ground truth
                                   │ observed columns only
                                   ▼
            predictors (backend/models), one Predictor protocol:
              rule baseline · independent gradient-boosted trees ·
              shared representation (Set Transformer, masked-field pretraining,
                                     joint heads, 5-seed ensemble)
                                   │
                                   ▼
   counterfactual engine (backend/simulation) ◄──► oracle: true process, same draws
     scenario schema · interventions · cached prediction · metrics · intervals
                                   │
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
   services: attribution,   evaluation: harness,      FastAPI (backend/api)
   explanation, mitigation, experiments                      │
   NL parser (LLM → schema)                                  ▼
                                                   Next.js dashboard (frontend)
```

More in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

Requirements: Python 3.11+, Node 20+ (for the dashboard), about 8 GB of RAM.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/generate_data.py        # 1M+ synthetic transactions, deterministic (seed 42)
python scripts/loop.py                 # the full loop on the HDFC demo, with ground truth
```

`loop.py` uses the rule baseline by default and needs no training. For the learned models:

```bash
python scripts/train.py                # shared-representation ensemble; ~1.5 h on a laptop CPU
python scripts/loop.py --predictor shared_representation
```

### Dashboard

```bash
uvicorn backend.api.app:app --port 8000      # loads data and fits baselines: a few minutes
cd frontend && npm install && npm run build && npm start
# open http://localhost:3000
```

Or with Docker, after generating data and training on the host (the containers mount `data/` and
`artifacts/`):

```bash
docker compose up --build
```

Set `VULCAN_PREDICTORS=rule_baseline,independent_ml` to run without a trained ensemble.

### Natural-language input (optional)

Set `VULCAN_LLM_API_KEY` (and optionally `VULCAN_LLM_MODEL`) for the API process. The LLM only
turns a question into a structured scenario, validated by the same schema as the form, which the
user confirms before anything runs. Without a key the form builder is the only input.
`python scripts/nl_smoke.py` runs the 30-phrasing test set against the live model.

## Reproducing the results

```bash
python scripts/generate_data.py                       # identical Parquet for the same seed
python scripts/train.py                               # main ensemble
python scripts/train.py --suite cold_start            # without held-out merchants/customers/issuers
python scripts/train.py --suite normal_only           # without peak hours and festival days
python scripts/evaluate.py                            # harness: experiments/harness/
python scripts/run_experiments.py                     # Phase 5: experiments/{main,cold_start,
                                                      #   distribution_shift,calibration}/
pytest && ruff check .                                # backend tests and lint
cd frontend && npm run lint && npm run build          # frontend checks
```

## Results

BENCHMARK_TABLES

## Limitations

1. The data is synthetic.
2. Results do not represent real Razorpay payment performance.
3. The simulator has known assumptions, documented in [`docs/dgp.md`](docs/dgp.md).
4. Counterfactual estimates are only as good as the underlying data-generating process and model.
5. Real-world payment systems contain factors that may not be represented in the simulation.
6. The project is inspired by publicly described concepts around Vulcan and is not an
   implementation of Razorpay's proprietary system.

What the evaluation found wrong with the models themselves is in
[`docs/limitations.md`](docs/limitations.md).

## Repository map

| Path | What |
|---|---|
| `backend/data` | Entities, traffic, episodes, the true outcome process |
| `backend/simulation` | Scenario schema, interventions, engine, oracle, metrics |
| `backend/models` | Rule baseline, independent trees, Set Transformer encoder, pretraining, heads |
| `backend/evaluation` | Harness, experiment grids, scoring |
| `backend/services` | Attribution, explanation, mitigation, NL parser, workspace |
| `backend/api` | FastAPI app |
| `frontend` | Next.js dashboard |
| `scripts` | Data generation, training, evaluation, experiments, CLI loop |
| `experiments` | Result tables and JSON, regenerated by the scripts |
| `docs` | Architecture, DGP, engine, evaluation, results, limitations, demo script |
