import numpy as np
import pandas as pd
import pytest

from backend.data.dgp import FAILURE_REASONS
from backend.evaluation.grid import scenario_grid
from backend.evaluation.harness import Support, run_grid
from backend.evaluation.scoring import summarize_errors, transaction_scores
from backend.models.features import FEATURES, FeatureBuilder
from backend.models.independent import IndependentModels
from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import evaluation_windows, first_attempts_on, time_split
from backend.simulation.interventions import observe
from backend.simulation.metrics import Predictions
from backend.simulation.oracle import true_success_probability
from backend.simulation.scenarios import Scenario


@pytest.fixture(scope="module")
def split(dataset):
    return time_split(dataset, test_days=4)


@pytest.fixture(scope="module")
def train(dataset, split):
    return first_attempts_on(dataset, split.train_days)


@pytest.fixture(scope="module")
def independent(dataset, train):
    return IndependentModels.fit(train, dataset.customers, dataset.merchants, seed=1)


def test_time_split_is_disjoint_and_ordered(dataset, split, train):
    assert set(split.train_days).isdisjoint(split.test_days)
    assert max(split.train_days) < min(split.test_days)
    assert len(split.train_days) + len(split.test_days) == dataset.config.n_days
    assert set(train["timestamp"].dt.date) <= set(split.train_days)
    assert (train["attempt"] == 1).all()
    with pytest.raises(ValueError):
        time_split(dataset, test_days=dataset.config.n_days)


def test_evaluation_windows_are_normal_test_weekdays(dataset, split):
    for day in evaluation_windows(dataset, split):
        assert day in split.test_days
        assert day.weekday() < 5 and day not in dataset.festival_dates


def test_features_exclude_entity_ids_and_fix_categories(dataset, train):
    x = FeatureBuilder(dataset.customers, dataset.merchants)(train.head(500))
    assert list(x.columns) == list(FEATURES)
    assert "merchant_id" not in x.columns and "customer_id" not in x.columns
    # Categories come from the catalog, not from whatever values the slice happens to contain.
    assert len(x["city"].cat.categories) == 20
    assert not x.isna().any().any()


def test_independent_models_predict_valid_distributions(dataset, split, independent):
    test = first_attempts_on(dataset, split.test_days)
    preds = independent.predict(test)
    assert preds.p_success.shape == (len(test),)
    assert np.allclose(preds.reason_probs.sum(axis=1), 1.0)
    assert (preds.latency_median_ms > 0).all()
    assert (independent.latency_log_sigma > 0).all()
    observed = (test["transaction_status"] == "SUCCESS").mean()
    assert preds.p_success.mean() == pytest.approx(observed, abs=0.01)


def test_true_probability_is_consistent_with_outcomes(dataset):
    p = true_success_probability(dataset)
    first = dataset.transactions[dataset.transactions["attempt"] == 1]
    p_first = p.loc[first["transaction_id"]].to_numpy()
    assert ((p_first >= 0) & (p_first < 1)).all()
    # Zero chance means routed into a scheduled gateway outage: always a failure, and almost
    # always a gateway timeout (an earlier stage such as the risk check can still fail first).
    zero = p_first == 0
    assert (first.loc[zero, "transaction_status"] == "FAILED").all()
    assert (first.loc[zero, "failure_reason"] == "GATEWAY_TIMEOUT").mean() > 0.95
    observed = (first["transaction_status"] == "SUCCESS").mean()
    assert p_first.mean() == pytest.approx(observed, abs=0.005)


def test_summarize_errors_by_hand():
    results = pd.DataFrame(
        {
            "predictor": ["m"] * 3,
            "target": ["success_rate_delta_pp"] * 3,
            "predicted": [-1.0, -2.0, 0.05],
            "truth": [-2.0, -2.0, 0.0],
        }
    )
    row = summarize_errors(results, ["predictor", "target"]).iloc[0]
    assert row["n"] == 3
    assert row["mae"] == pytest.approx((1.0 + 0.0 + 0.05) / 3)
    assert row["rmse"] == pytest.approx(np.sqrt((1.0 + 0.0 + 0.0025) / 3))
    # Only the two scenarios with a material true effect count toward relative error and sign.
    assert row["n_material"] == 2
    assert row["median_relative_error"] == pytest.approx(0.25)
    assert row["sign_accuracy"] == pytest.approx(1.0)
    assert row["mae_ci_low"] <= row["mae"] <= row["mae_ci_high"]


def test_transaction_scores_perfect_and_constant():
    success = np.array([True, True, False, True, False, True, True, True, True, False])
    n = len(success)
    reasons = np.full(n, FAILURE_REASONS.index("ISSUER_DECLINED"))
    reason_probs = np.zeros((n, len(FAILURE_REASONS)))
    reason_probs[:, FAILURE_REASONS.index("ISSUER_DECLINED")] = 1.0
    latency = np.full(n, 1000.0)

    def preds(p):
        return Predictions(
            p_success=np.asarray(p, dtype=float),
            reason_probs=reason_probs,
            latency_median_ms=latency,
            latency_log_sigma=np.zeros(n),
        )

    perfect = transaction_scores(success, latency, reasons, preds(success.astype(float)))
    assert perfect["auc"] == pytest.approx(1.0)
    assert perfect["brier"] == pytest.approx(0.0)
    assert perfect["reason_top1_accuracy"] == pytest.approx(1.0)
    assert perfect["latency_log_mae"] == pytest.approx(0.0)
    constant = transaction_scores(success, latency, reasons, preds(np.full(n, 0.7)))
    assert constant["auc"] == pytest.approx(0.5)


def test_support_flags_values_outside_training(train):
    support = Support.from_training(train)
    frame = train.head(3).assign(
        issuer_health=[0.95, 0.0, 0.95], gateway_utilization=[0.4, 0.4, 9.0]
    )
    np.testing.assert_array_equal(support.outside(frame), [False, True, True])


def test_grid_scenarios_are_valid():
    grid = scenario_grid()
    assert len({name for name, _ in grid}) == len(grid)
    types = set()
    for _, interventions in grid:
        scenario = Scenario.model_validate({"interventions": interventions})
        types |= {iv.type for iv in scenario.interventions}
    assert types == {
        "issuer_degradation",
        "gateway_outage",
        "traffic_change",
        "method_shift",
        "routing_change",
    }


def test_run_grid_long_format(dataset, split, train):
    grid = [
        ("outage", [{"type": "gateway_outage", "gateway": "gateway_a", "duration_minutes": 60}]),
        ("shift", [{"type": "method_shift", "from": "CARD", "to": "UPI", "percentage": 0.2}]),
    ]
    windows = evaluation_windows(dataset, split)[:1]
    results = run_grid(
        dataset,
        observe(dataset),
        windows,
        [RuleBaseline.fit(train)],
        grid,
        Support.from_training(train),
    )
    assert len(results) == 2 * 5
    assert set(results["range"]) <= {"in_range", "out_of_range"}
    outage = results[
        (results["scenario"] == "outage") & (results["target"] == "success_rate_delta_pp")
    ]
    assert outage["truth"].iloc[0] < 0
    # An outage drives gateway health to 0, far outside anything seen in training.
    assert outage["range"].iloc[0] == "out_of_range"
