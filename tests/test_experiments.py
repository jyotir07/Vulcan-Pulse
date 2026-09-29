import numpy as np
import pytest

from backend.evaluation.experiments import (
    ALL_ROWS,
    Suite,
    calibration_summary,
    cold_start_grid,
    held_out_issuer_summary,
    main_grid,
    run_suite,
    shift_grid,
    subset_impact,
)
from backend.evaluation.harness import Support
from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import entity_holdout, suite_training_rows, time_split
from backend.simulation.engine import ground_truth, true_predictions
from backend.simulation.interventions import apply_scenario, observe
from backend.simulation.metrics import ensemble_mean
from backend.simulation.scenarios import Scenario


class _Ensemble:
    name = "test_ensemble"

    def __init__(self, members):
        self.members = members

    def predict(self, frame):
        return ensemble_mean(self.member_predictions(frame))

    def member_predictions(self, frame):
        return [m.predict(frame) for m in self.members]


@pytest.fixture(scope="module")
def split(dataset):
    return time_split(dataset, test_days=4)


@pytest.fixture(scope="module")
def observed(dataset):
    return observe(dataset)


@pytest.fixture(scope="module")
def cold_suite(dataset, split):
    train = suite_training_rows(dataset, split, "cold_start")
    members = [
        RuleBaseline.fit(train.sample(frac=1.0, replace=True, random_state=s)) for s in (0, 1)
    ]
    return Suite(
        "cold_start",
        [RuleBaseline.fit(train), _Ensemble(members)],
        Support.from_training(train),
        entity_holdout(dataset),
    )


def test_grids_validate_as_scenarios():
    for grid in (main_grid(), cold_start_grid(), shift_grid()):
        for _, _, interventions in grid:
            Scenario.model_validate({"interventions": interventions})
    categories = {c for c, _, _ in main_grid()}
    assert {"normal_traffic", "peak_traffic", "issuer_degradation", "gateway_outage"} <= categories


def test_whole_window_subset_matches_the_engine(dataset, observed):
    scenario = Scenario.model_validate({"interventions": [cold_start_grid()[0][2][0]]})
    cf = apply_scenario(dataset, scenario, observed)
    preds = true_predictions(dataset, cf)
    everything = subset_impact(
        cf, preds, np.ones(len(cf.baseline), bool), np.ones(len(cf.counterfactual), bool)
    )
    assert everything == ground_truth(dataset, cf).impact


def test_cold_start_suite_scores_subsets_and_intervals(dataset, observed, split, cold_suite):
    window = split.test_days[1]
    grid = [e for e in cold_start_grid() if "Yes Bank" in e[1] or "HDFC" in e[1]][:3]
    results = run_suite(dataset, observed, cold_suite, {window: "normal"}, grid)

    subsets = set(results["subset"])
    assert ALL_ROWS in subsets and "known" in subsets
    assert (results["window_kind"] == "normal").all()
    # Only the ensemble, and only on the whole window, has an interval.
    with_interval = results.dropna(subset=["interval_low"])
    assert set(with_interval["predictor"]) == {"test_ensemble"}
    assert set(with_interval["subset"]) == {ALL_ROWS}
    assert (with_interval["interval_low"] <= with_interval["interval_high"]).all()

    coverage = calibration_summary(results)
    assert coverage["coverage"].between(0, 1).all()
    issuers = held_out_issuer_summary(results)
    assert set(issuers["category"]) == {"issuer_degradation", "held_out_issuer_degradation"}
