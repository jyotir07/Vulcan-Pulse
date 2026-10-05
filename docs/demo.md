# Demo script (3–5 minutes)

Follows the spec's demo story. Numbers below are from the default dataset (seed 42) on the
default window, 2026-02-03; they are simulation output on synthetic data. Re-check them after
regenerating anything.

**Before recording:** `docker compose up --build` (or `uvicorn backend.api.app:app` plus
`npm run start` in `frontend/`), open http://localhost:3000, and run the HDFC scenario once so the
day's baseline is cached. Pick the predictor in the builder: `rule_baseline` answers in seconds,
`shared_representation` in tens of seconds on a laptop CPU.

## 1. The ecosystem (30 s)

Scroll through the health tiles and the ecosystem map.

> "This is a synthetic payment network: two thousand merchants, fifty thousand customers, twelve
> issuers, four gateways, a million transactions over thirty days. Every number on this page is
> generated. The banner says so, and so does every API response."

Point out: 96.1% success on a held-out day no model trained on; UPI is two thirds of volume;
netbanking is the least reliable method.

## 2. The learned representation (30 s)

> "Instead of training one model per question, we learn a shared representation of payments. Each
> transaction is a set of fields: amount, method, issuer, merchant category, gateway, the state of
> the ecosystem at that moment. A Set Transformer is pretrained by hiding fields and predicting
> them, then fine-tuned on success, failure reason and latency together. Five seeds make an
> ensemble, so we also get uncertainty."

Optional: show `docs/evaluation.md`'s masked-field table: 0.53 accuracy against a 0.39 frequency
baseline, before the model ever sees an outcome.

## 3. Ask (20 s)

Type in the question box: *"What if HDFC UPI success rate drops by 15% for two hours at 6pm?"*
With an LLM key configured, press Parse, show the structured scenario and confirm it:

> "The language model only translates the question. It never produces a number."

Without a key, pick the preset "HDFC UPI degradation (−15 pp, evening)" in the builder.

## 4. Simulate (40 s)

Press Simulate. Read the impact tiles and the before/after bars.

> "HDFC carries about 17% of UPI payments in that window. The day's success rate drops by about a
> quarter of a point, with ₹1.6–2.5 lakh of GMV at risk, concentrated in education, electronics
> and travel, because those are high-ticket categories."

## 5. Explain (30 s)

Show the explanation box, the waterfall and the Shapley split.

> "Every sentence here is a template filled from computed quantities: traffic share, where the
> GMV sits, how loaded the gateways are. The factor split says the whole effect is the
> degradation itself; gateways are only at 61% of capacity, so congestion adds nothing."

## 6. Mitigate (40 s)

Press "Find mitigations", then Apply on the top option.

> "Rerouting can't fix a bank. In this world an issuer's reliability doesn't depend on the gateway
> in front of it, and the search finds that: every reroute ranks below steering HDFC's UPI users
> to cards. Steering half of them recovers about 0.1 to 0.15 points."

For a gateway story, switch to the gateway_a outage preset: rerouting 50% to gateway_b
recovers about 1.5 of the 3 points lost, and pushing more traffic starts to saturate the target.

## 7. Validate (40 s)

Scroll to "Validation against ground truth" and the timeline.

> "Because we control the synthetic world, we know the true outcome. The true process was re-run on
> exactly these payments with the same random draws. The rule baseline predicted −0.24 points;
> the truth is −0.28. The mitigation's predicted recovery was +0.11, the true one +0.15."

Close on the honest part (`docs/results.md`):

> "The shared representation matches the best models on individual payments and wins on routing
> and load scenarios, but on issuer degradations the hand-written rule is still far more
> accurate, because degraded states are too rare in history for any learned model to see. And
> its confidence intervals are too narrow. Those are findings, not footnotes."
