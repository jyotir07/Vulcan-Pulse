"""The Phase 5 experiments: scenario grids per suite, subset impacts, and their summaries.

A suite is a set of predictors trained on one slice of the training days (`models/splits.py`):
- main: every training row; answers the main experiment and the uncertainty calibration;
- cold_start: no held-out merchant, customer or issuer; its errors on those entities' payments
  are the "new merchant" and "new customer" categories and the cold-start experiment;
- normal_only: no peak hours or festival days; its errors on a festival day are the
  distribution-shift category.

Every run records the impact on the whole window, plus, for the cold-start suite, on the
payments of held-out and known entities separately, so one simulation answers several questions.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.data.generator import Dataset
from backend.evaluation.harness import OUT_OF_RANGE_SHARE, Support
from backend.evaluation.scoring import TARGETS, summarize_errors
from backend.models.splits import EntityHoldout
from backend.simulation.engine import (
    BaselineCache,
    Predictor,
    ScenarioPredictions,
    build_result,
    predict_scenario,
    true_predictions,
)
from backend.simulation.interventions import Counterfactual, ObservedEcosystem, apply_scenario
from backend.simulation.metrics import Impact, compute_impact, compute_metrics
from backend.simulation.scenarios import Scenario

ALL_ROWS = "all"

# A subset impact is only scored when the subset has enough payments to have a rate.
MIN_SUBSET_ROWS = 200


def _traffic(delta: float, segment: str | None = None, **timing) -> dict:
    return {"type": "traffic_change", "segment": segment, "volume_delta": delta, **timing}


def _degradation(issuer: str, delta: float, method: str | None = "UPI", **timing) -> dict:
    return {
        "type": "issuer_degradation",
        "issuer": issuer,
        "method": method,
        "success_rate_delta": delta,
        **timing,
    }


def _outage(gateway: str, minutes: int, start_hour: int = 19) -> dict:
    return {
        "type": "gateway_outage",
        "gateway": gateway,
        "start_hour": start_hour,
        "duration_minutes": minutes,
    }


def _shift(source: str, target: str, share: float) -> dict:
    return {"type": "method_shift", "from": source, "to": target, "percentage": share}


def _route(source: str, target: str, share: float) -> dict:
    return {
        "type": "routing_change",
        "source_gateway": source,
        "target_gateway": target,
        "traffic_percentage": share,
    }


OFF_PEAK = {"start_hour": 10, "duration_minutes": 240}
PEAK = {"start_hour": 18, "duration_minutes": 240}
EVENING = {"start_hour": 18, "duration_minutes": 120}

GridEntry = tuple[str, str, list[dict]]  # (category, scenario name, interventions)


def main_grid() -> list[GridEntry]:
    """The spec's scenario categories that the main suite answers directly."""
    grid: list[GridEntry] = []
    for label, timing in (("normal_traffic", OFF_PEAK), ("peak_traffic", PEAK)):
        when = "10:00-14:00" if label == "normal_traffic" else "18:00-22:00"
        for delta in (0.3, 0.8, 1.5):
            grid.append((label, f"all traffic {delta:+.0%} {when}", [_traffic(delta, **timing)]))
        grid.append((label, f"UPI traffic +80% {when}", [_traffic(0.8, "UPI", **timing)]))
    for issuer in ("HDFC", "SBI"):
        for delta in (-0.05, -0.10, -0.20, -0.40):
            grid.append(
                (
                    "issuer_degradation",
                    f"{issuer} UPI {delta * 100:+.0f}pp 18:00-20:00",
                    [_degradation(issuer, delta, **EVENING)],
                )
            )
    grid.append(
        ("issuer_degradation", "HDFC all methods -30pp all day", [_degradation("HDFC", -0.3, None)])
    )
    for gateway in ("gateway_a", "gateway_b", "gateway_c", "gateway_d"):
        for minutes in (15, 60):
            grid.append(
                (
                    "gateway_outage",
                    f"{gateway} outage {minutes}m at 19:00",
                    [_outage(gateway, minutes)],
                )
            )
    for source, target, share in (
        ("CARD", "UPI", 0.2),
        ("CARD", "UPI", 0.6),
        ("UPI", "CARD", 0.2),
        ("NETBANKING", "UPI", 0.5),
        ("UPI", "NETBANKING", 0.3),
    ):
        grid.append(
            ("method_shift", f"{share:.0%} {source} -> {target}", [_shift(source, target, share)])
        )
    return grid


def cold_start_grid() -> list[GridEntry]:
    """A mix of every intervention type, with degradations on both held-out and known issuers."""
    grid: list[GridEntry] = []
    for issuer, method in (("HDFC", "UPI"), ("Yes Bank", "UPI"), ("Canara", None)):
        category = "issuer_degradation" if issuer == "HDFC" else "held_out_issuer_degradation"
        for delta in (-0.10, -0.30):
            scope = method or "all methods"
            grid.append(
                (
                    category,
                    f"{issuer} {scope} {delta * 100:+.0f}pp 18:00-20:00",
                    [_degradation(issuer, delta, method, **EVENING)],
                )
            )
    grid += [
        ("gateway_outage", "gateway_a outage 60m at 19:00", [_outage("gateway_a", 60)]),
        ("traffic_change", "all traffic +80% 18:00-22:00", [_traffic(0.8, **PEAK)]),
        ("method_shift", "30% CARD -> UPI", [_shift("CARD", "UPI", 0.3)]),
        (
            "routing_change",
            "route 50% gateway_a -> gateway_b",
            [_route("gateway_a", "gateway_b", 0.5)],
        ),
    ]
    return grid


def shift_grid() -> list[GridEntry]:
    """Scenarios run on a normal day and on an unseen festival day by the normal-only suite."""
    return [
        (
            "issuer_degradation",
            "HDFC UPI -10pp 18:00-20:00",
            [_degradation("HDFC", -0.1, **EVENING)],
        ),
        (
            "issuer_degradation",
            "HDFC UPI -30pp 18:00-20:00",
            [_degradation("HDFC", -0.3, **EVENING)],
        ),
        ("gateway_outage", "gateway_a outage 60m at 19:00", [_outage("gateway_a", 60)]),
        ("gateway_outage", "gateway_b outage 60m at 19:00", [_outage("gateway_b", 60)]),
        ("traffic_change", "all traffic +30% 18:00-22:00", [_traffic(0.3, **PEAK)]),
        ("traffic_change", "all traffic +80% 18:00-22:00", [_traffic(0.8, **PEAK)]),
        ("traffic_change", "UPI traffic +50% 10:00-14:00", [_traffic(0.5, "UPI", **OFF_PEAK)]),
        ("method_shift", "20% CARD -> UPI", [_shift("CARD", "UPI", 0.2)]),
        (
            "routing_change",
            "route 50% gateway_a -> gateway_c",
            [_route("gateway_a", "gateway_c", 0.5)],
        ),
    ]


@dataclass(frozen=True)
class Suite:
    name: str
    predictors: list[Predictor]
    support: Support
    holdout: EntityHoldout | None = None  # set for the cold-start suite only


def subsets(suite: Suite, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row masks the suite's impacts are scored on, besides the whole window."""
    if suite.holdout is None:
        return {}
    h = suite.holdout
    return {
        "known": ~h.involved(frame),
        "held_out_merchant": h.merchant_rows(frame),
        "held_out_customer": h.customer_rows(frame),
        "held_out_issuer": h.issuer_rows(frame),
    }


def subset_impact(
    cf: Counterfactual, preds: ScenarioPredictions, base_rows: np.ndarray, cf_rows: np.ndarray
) -> Impact:
    before = compute_metrics(cf.baseline[base_rows], preds.baseline.take(np.flatnonzero(base_rows)))
    after = compute_metrics(
        cf.counterfactual[cf_rows], preds.counterfactual.take(np.flatnonzero(cf_rows))
    )
    return compute_impact(before, after, int(cf.affected[cf_rows].sum()))


def run_suite(
    dataset: Dataset,
    observed: ObservedEcosystem,
    suite: Suite,
    windows: dict[date, str],
    grid: list[GridEntry],
    progress=None,
) -> pd.DataFrame:
    """Long format: one row per (scenario, window, subset, predictor, target).

    `windows` maps each window date to its kind ("normal" or "festival").
    """
    rows = []
    cache = BaselineCache()
    for window, window_kind in windows.items():
        for category, name, interventions in grid:
            scenario = Scenario.model_validate(
                {"interventions": interventions, "window_date": window}
            )
            cf = apply_scenario(dataset, scenario, observed)
            affected = cf.counterfactual[cf.affected]
            outside = float(suite.support.outside(affected).mean()) if len(affected) else 0.0
            common = {
                "suite": suite.name,
                "category": category,
                "scenario": name,
                "window": window.isoformat(),
                "window_kind": window_kind,
                "range": "out_of_range" if outside > OUT_OF_RANGE_SHARE else "in_range",
                "outside_support_share": outside,
            }
            masks = {
                key: (base, subsets(suite, cf.counterfactual)[key])
                for key, base in subsets(suite, cf.baseline).items()
            }
            masks = {
                k: m for k, m in masks.items() if min(m[0].sum(), m[1].sum()) >= MIN_SUBSET_ROWS
            }

            truth_preds = true_predictions(dataset, cf)
            truth = {ALL_ROWS: build_result("ground_truth", cf, dataset, truth_preds).impact}
            for key, (b, c) in masks.items():
                truth[key] = subset_impact(cf, truth_preds, b, c)

            for predictor in suite.predictors:
                preds = predict_scenario(cf, predictor, cache=cache)
                result = build_result(predictor.name, cf, dataset, preds)
                predicted = {ALL_ROWS: result.impact}
                for key, (b, c) in masks.items():
                    predicted[key] = subset_impact(cf, preds, b, c)
                for subset, impact in predicted.items():
                    for target in TARGETS:
                        interval = (
                            getattr(result.interval, target)
                            if result.interval is not None and subset == ALL_ROWS
                            else (np.nan, np.nan)
                        )
                        rows.append(
                            {
                                **common,
                                "subset": subset,
                                "predictor": predictor.name,
                                "target": target,
                                "predicted": getattr(impact, target),
                                "truth": getattr(truth[subset], target),
                                "interval_low": interval[0],
                                "interval_high": interval[1],
                            }
                        )
            if progress:
                progress(f"[{suite.name}] {window} {name}: {common['range']}")
    return pd.DataFrame(rows)


def main_experiment_rows(results: pd.DataFrame) -> pd.DataFrame:
    """The main experiment's eight categories, each drawn from the suite that isolates it."""
    main = results[(results["suite"] == "main") & (results["subset"] == ALL_ROWS)]
    cold = results[results["suite"] == "cold_start"]
    shift = results[(results["suite"] == "normal_only") & (results["subset"] == ALL_ROWS)]
    festival = shift[shift["window_kind"] == "festival"]
    return pd.concat(
        [
            main,
            cold[cold["subset"] == "held_out_merchant"].assign(category="new_merchant"),
            cold[cold["subset"] == "held_out_customer"].assign(category="new_customer"),
            festival.assign(category="distribution_shift"),
        ],
        ignore_index=True,
    )


def main_summary(results: pd.DataFrame) -> pd.DataFrame:
    return summarize_errors(
        main_experiment_rows(results), ["target", "category", "range", "predictor"]
    )


def cold_start_summary(results: pd.DataFrame) -> pd.DataFrame:
    cold = results[results["suite"] == "cold_start"]
    return summarize_errors(cold, ["target", "subset", "predictor"])


def held_out_issuer_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Degradations of a held-out issuer against the same degradations of a known one."""
    cold = results[
        (results["suite"] == "cold_start")
        & (results["subset"] == ALL_ROWS)
        & results["category"].isin(["issuer_degradation", "held_out_issuer_degradation"])
    ]
    return summarize_errors(cold, ["target", "category", "predictor"])


def shift_summary(results: pd.DataFrame) -> pd.DataFrame:
    shift = results[(results["suite"] == "normal_only") & (results["subset"] == ALL_ROWS)]
    return summarize_errors(shift, ["target", "window_kind", "predictor"])


def calibration_summary(results: pd.DataFrame) -> pd.DataFrame:
    """How often the truth falls inside an ensemble's interval, per suite, target and range."""
    rows = results[(results["subset"] == ALL_ROWS) & results["interval_low"].notna()]
    out = []
    for (suite, target, range_), group in rows.groupby(["suite", "target", "range"], sort=True):
        inside = (group["truth"] >= group["interval_low"]) & (
            group["truth"] <= group["interval_high"]
        )
        width = group["interval_high"] - group["interval_low"]
        error = (group["predicted"] - group["truth"]).abs()
        out.append(
            {
                "suite": suite,
                "target": target,
                "range": range_,
                "predictor": group["predictor"].iloc[0],
                "n": len(group),
                "coverage": float(inside.mean()),
                "median_width": float(width.median()),
                "median_abs_error": float(error.median()),
            }
        )
    return pd.DataFrame(out)
