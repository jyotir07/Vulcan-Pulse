"""Run what-if scenarios with the rule baseline and compare against ground truth.

Usage:
    python scripts/simulate.py                      # every scenario in configs/scenarios.json
    python scripts/simulate.py --scenario my.json   # one {"interventions": [...]} object
    python scripts/simulate.py --json out.json      # also write the full results
"""

import argparse
import json
import sys
import time
from pathlib import Path

from backend.config import REPO_ROOT
from backend.data.io import load_dataset
from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import first_attempts_excluding
from backend.simulation.engine import SimulationResult, ground_truth, simulate
from backend.simulation.interventions import apply_scenario, observe
from backend.simulation.metrics import most_affected
from backend.simulation.scenarios import Scenario

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"
DEFAULT_SCENARIOS = REPO_ROOT / "configs" / "scenarios.json"


def _rupees(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}₹{value / 1e7:,.2f} Cr" if value >= 1e7 else f"{sign}₹{value / 1e5:,.2f} L"


def _report(name: str, predicted: SimulationResult, truth: SimulationResult) -> str:
    p, t = predicted, truth
    rows = [
        ("Transactions", f"{p.baseline.transactions:,}", f"{p.counterfactual.transactions:,}", ""),
        (
            "Success rate",
            f"{p.baseline.success_rate:.2%}",
            f"{p.counterfactual.success_rate:.2%}",
            f"{t.baseline.success_rate:.2%} → {t.counterfactual.success_rate:.2%}",
        ),
        (
            "GMV",
            _rupees(p.baseline.gmv),
            _rupees(p.counterfactual.gmv),
            "",
        ),
        (
            "P95 latency",
            f"{p.baseline.p95_latency_ms:,.0f} ms",
            f"{p.counterfactual.p95_latency_ms:,.0f} ms",
            f"{t.baseline.p95_latency_ms:,.0f} → {t.counterfactual.p95_latency_ms:,.0f} ms",
        ),
    ]
    impact = [
        (
            "Success rate change",
            p.impact.success_rate_delta_pp,
            t.impact.success_rate_delta_pp,
            "{:+.2f} pp",
        ),
        (
            "Timeout rate change",
            p.impact.timeout_rate_delta_pp,
            t.impact.timeout_rate_delta_pp,
            "{:+.2f} pp",
        ),
        ("Extra failures", p.impact.failures_delta, t.impact.failures_delta, "{:+,.0f}"),
        ("GMV at risk", p.impact.gmv_at_risk, t.impact.gmv_at_risk, None),
    ]
    lines = [
        f"\n{'=' * 78}\n{name}   (window {p.window_date}, "
        f"{p.impact.transactions_affected:,} transactions affected)\n{'=' * 78}",
        f"{'':22}{'baseline':>14}{'scenario':>14}   ground truth",
    ]
    lines += [f"{label:22}{b:>14}{c:>14}   {truth_text}" for label, b, c, truth_text in rows]
    lines.append(f"\n{'IMPACT':22}{p.source:>14}{'ground truth':>17}")
    for label, predicted_value, true_value, fmt in impact:
        if fmt is None:
            lines.append(f"{label:22}{_rupees(predicted_value):>14}{_rupees(true_value):>17}")
        else:
            lines.append(f"{label:22}{fmt.format(predicted_value):>14}{fmt.format(true_value):>17}")
    for label, result in ((p.source, p), ("ground truth", t)):
        top = most_affected(result.segments, "merchant_category")
        names = ", ".join(f"{s.segment} ({s.success_rate_delta_pp:+.1f} pp)" for s in top) or "none"
        lines.append(f"Most affected categories [{label}]: {names}")
    return "\n".join(lines)


def main() -> None:
    # Windows consoles default to a code page without the rupee sign.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scenario", type=Path, help="A single scenario JSON file.")
    parser.add_argument("--json", type=Path, help="Write all results to this JSON file.")
    args = parser.parse_args()

    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"No dataset at {args.data}; run scripts/generate_data.py first.")
    if args.scenario:
        named = [(args.scenario.stem, json.loads(args.scenario.read_text(encoding="utf-8")))]
    else:
        entries = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))
        named = [(e["name"], e["scenario"]) for e in entries]

    started = time.perf_counter()
    dataset = load_dataset(args.data)
    observed = observe(dataset)
    models: dict = {}
    results = []
    for name, raw in named:
        scenario = Scenario.model_validate(raw)
        cf = apply_scenario(dataset, scenario, observed)
        # Fit once per window, never on the window itself.
        if cf.window.date not in models:
            models[cf.window.date] = RuleBaseline.fit(
                first_attempts_excluding(dataset, cf.window.date)
            )
        predicted = simulate(dataset, cf, models[cf.window.date])
        truth = ground_truth(dataset, cf)
        print(_report(name, predicted, truth))
        results.append(
            {
                "name": name,
                "scenario": scenario.model_dump(mode="json", by_alias=True),
                "predicted": predicted.model_dump(mode="json"),
                "ground_truth": truth.model_dump(mode="json"),
            }
        )

    print(
        f"\nSynthetic data; simulation output, not real payment performance. "
        f"({time.perf_counter() - started:.1f}s)"
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
