"""The full loop on one scenario: simulate, attribute, explain, mitigate, simulate again.

Each simulation is shown next to the ground truth for the same transactions.

Usage:
    python scripts/loop.py                                  # the HDFC UPI demo scenario
    python scripts/loop.py --scenario my.json --predictor shared_representation --choose 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

from backend.config import REPO_ROOT
from backend.services.attribution import factor_attribution, segment_attribution, top_contributors
from backend.services.explanation import explain
from backend.services.mitigation import mitigate, with_action
from backend.services.workspace import PREDICTORS, Workspace
from backend.simulation.engine import build_result, ground_truth, predict_scenario
from backend.simulation.scenarios import Scenario

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"
DEMO_SCENARIO = {
    "interventions": [
        {
            "type": "issuer_degradation",
            "issuer": "HDFC",
            "method": "UPI",
            "success_rate_delta": -0.15,
            "start_hour": 18,
            "duration_minutes": 120,
        }
    ]
}


def _rupees(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}₹{value / 1e7:,.2f} Cr" if value >= 1e7 else f"{sign}₹{value / 1e5:,.2f} L"


def _impact_lines(label: str, predicted, truth) -> list[str]:
    p, t = predicted, truth
    return [
        f"{label:28}{p.source:>22}{'ground truth':>18}",
        f"{'Success rate':28}{p.baseline.success_rate:>11.2%} → {p.counterfactual.success_rate:.2%}"
        f"{t.baseline.success_rate:>9.2%} → {t.counterfactual.success_rate:.2%}",
        f"{'Success rate change':28}{p.impact.success_rate_delta_pp:>+20.2f} pp"
        f"{t.impact.success_rate_delta_pp:>+15.2f} pp",
        f"{'Extra failures':28}{p.impact.failures_delta:>+23,.0f}"
        f"{t.impact.failures_delta:>+18,.0f}",
        f"{'GMV at risk':28}{_rupees(p.impact.gmv_at_risk):>23}"
        f"{_rupees(t.impact.gmv_at_risk):>18}",
        f"{'Payments affected':28}{p.impact.transactions_affected:>23,}",
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scenario", type=Path, help="A scenario JSON file.")
    parser.add_argument("--predictor", choices=PREDICTORS, default="rule_baseline")
    parser.add_argument("--choose", type=int, default=1, help="Which ranked mitigation to apply.")
    args = parser.parse_args()

    started = time.perf_counter()
    raw = json.loads(args.scenario.read_text(encoding="utf-8")) if args.scenario else DEMO_SCENARIO
    ws = Workspace.load(args.data, predictors=(args.predictor,))
    predictor = ws.predictors[args.predictor]
    scenario = ws.with_window(Scenario.model_validate(raw))

    cf = ws.counterfactual(scenario)
    preds = predict_scenario(cf, predictor, cache=ws.cache)
    result = build_result(predictor.name, cf, ws.dataset, preds)
    truth = ground_truth(ws.dataset, cf)
    lines = [f"SCENARIO  (window {cf.window.date})", json.dumps(raw["interventions"]), ""]
    lines += _impact_lines("IMPACT", result, truth)
    if result.interval is not None:
        low, high = result.interval.success_rate_delta_pp
        lines.append(f"{'90% interval':28}{f'[{low:+.2f}, {high:+.2f}] pp':>23}")

    contributions = segment_attribution(ws.dataset, cf, preds)
    lines += ["", "WHERE THE GMV AT RISK COMES FROM"]
    for dimension in ("issuer", "merchant_category", "gateway", "hour_bucket"):
        top = top_contributors(contributions, dimension)
        lines.append(
            f"  {dimension:18}" + ", ".join(f"{c.segment} {_rupees(c.gmv_at_risk)}" for c in top)
        )
    lines.append("  by factor (Shapley):")
    for f in factor_attribution(cf, predictor):
        lines.append(
            f"    {f.factor:22}{f.success_rate_delta_pp:+.2f} pp  {_rupees(f.gmv_at_risk)}"
        )

    explanation = explain(ws.dataset, scenario, cf, result, contributions)
    lines += ["", "WHY", f"  {explanation.headline}"] + [f"  - {f}" for f in explanation.facts]

    report = mitigate(ws.dataset, ws.observed, scenario, predictor, cache=ws.cache)
    lines += ["", f"MITIGATIONS ({report.label}, ranked by recovered success rate)"]
    for rank, m in enumerate(report.candidates, 1):
        lines.append(
            f"  {rank}. {m.name:52}{m.success_rate_recovered_pp:+.2f} pp  "
            f"{_rupees(m.gmv_recovered)} recovered"
        )
    if not report.candidates:
        lines.append("  none found")
        print("\n".join(lines))
        return

    chosen = report.candidates[min(max(args.choose, 1), len(report.candidates)) - 1]
    mitigated = with_action(scenario, chosen.action)
    mcf = ws.counterfactual(mitigated)
    mresult = build_result(
        predictor.name, mcf, ws.dataset, predict_scenario(mcf, predictor, cache=ws.cache)
    )
    mtruth = ground_truth(ws.dataset, mcf)
    lines += ["", f"SIMULATE AGAIN WITH: {chosen.name}"]
    lines += _impact_lines("MITIGATED IMPACT", mresult, mtruth)
    recovered = mresult.counterfactual.success_rate - result.counterfactual.success_rate
    true_recovered = mtruth.counterfactual.success_rate - truth.counterfactual.success_rate
    lines.append(
        f"{'Recovered vs no action':28}{100 * recovered:>+20.2f} pp"
        f"{100 * true_recovered:>+15.2f} pp"
    )
    lines.append(
        f"\nSynthetic data; simulation output, not real payment performance. "
        f"({time.perf_counter() - started:.1f}s)"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
