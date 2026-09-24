from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import GATEWAYS, METHODS
from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import first_attempts_excluding
from backend.simulation.engine import ground_truth, simulate
from backend.simulation.interventions import (
    WARMUP_MINUTES,
    apply_scenario,
    default_window_date,
    observe,
)
from backend.simulation.oracle import true_outcomes
from backend.simulation.scenarios import Scenario

SUPPORTS = {g.name: set(g.supported_methods) for g in GATEWAYS}


@pytest.fixture(scope="module")
def observed(dataset):
    return observe(dataset)


@pytest.fixture(scope="module")
def rules(dataset):
    return RuleBaseline.fit(first_attempts_excluding(dataset, default_window_date(dataset)))


def _apply(dataset, observed, *interventions, **scenario_fields):
    scenario = Scenario.model_validate({"interventions": list(interventions), **scenario_fields})
    return apply_scenario(dataset, scenario, observed)


def _degradation(delta=-0.15, **extra):
    return {
        "type": "issuer_degradation",
        "issuer": "HDFC",
        "method": "UPI",
        "success_rate_delta": delta,
        **extra,
    }


def test_default_window_is_a_normal_weekday(dataset):
    day = default_window_date(dataset)
    assert day.weekday() < 5 and day not in dataset.festival_dates


def test_engine_baseline_features_match_observed(dataset, observed):
    cf = _apply(dataset, observed, _degradation())
    observed_rows = dataset.transactions.set_index("transaction_id").loc[
        cf.baseline["transaction_id"]
    ]
    for column in ("issuer_health", "gateway_health", "gateway_utilization", "is_peak"):
        np.testing.assert_allclose(
            cf.baseline[column].to_numpy(dtype=float),
            observed_rows[column].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )


def test_ground_truth_baseline_matches_dataset(dataset, observed):
    """The warm-up makes a window-only replay reproduce the full run inside the window."""
    cf = _apply(dataset, observed, _degradation())
    replay = true_outcomes(dataset, context=cf.oracle_baseline_context)
    replay = replay[replay["attempt"] == 1].set_index("transaction_id")
    original = dataset.transactions.set_index("transaction_id")
    ids = cf.baseline["transaction_id"]
    for column in ("transaction_status", "failure_reason", "latency_ms"):
        pd.testing.assert_series_equal(
            replay.loc[ids, column], original.loc[ids, column], check_categorical=False
        )
    warmup = cf.oracle_baseline_context
    assert warmup["timestamp"].min() >= pd.Timestamp(cf.window.date) - timedelta(
        minutes=WARMUP_MINUTES
    )


def test_issuer_degradation(dataset, observed, rules):
    cf = _apply(dataset, observed, _degradation(start_hour=10, duration_minutes=120))
    hit = (
        (cf.baseline["issuer_id"] == 0)
        & (cf.baseline["payment_method"] == "UPI")
        & cf.baseline["timestamp"].dt.hour.between(10, 11)
    ).to_numpy()
    assert hit.any()
    np.testing.assert_array_equal(cf.affected, hit)
    np.testing.assert_allclose(
        cf.counterfactual["issuer_health"].to_numpy()[hit],
        np.clip(cf.baseline["issuer_health"].to_numpy()[hit] - 0.15, 0.0, 1.0),
    )
    episode = cf.extra_episodes.iloc[0]
    assert episode["kind"] == "issuer_degradation" and episode["magnitude"] == pytest.approx(0.15)
    assert episode["end_minute"] - episode["start_minute"] == 120
    pd.testing.assert_frame_equal(
        cf.oracle_baseline_context, cf.oracle_counterfactual_context, check_dtype=False
    )

    truth = ground_truth(dataset, cf)
    predicted = simulate(dataset, cf, rules)
    assert truth.impact.success_rate_delta_pp < 0
    assert predicted.impact.success_rate_delta_pp < 0
    assert truth.impact.transactions_affected == hit.sum()


def test_gateway_outage(dataset, observed, rules):
    cf = _apply(
        dataset,
        observed,
        {
            "type": "gateway_outage",
            "gateway": "gateway_a",
            "start_hour": 18,
            "duration_minutes": 30,
        },
    )
    hit = cf.affected
    rows = cf.counterfactual[hit]
    assert (rows["gateway_id"] == 0).all()
    assert (rows["gateway_health"] == 0.0).all()
    assert (rows["timestamp"].dt.hour == 18).all() and (rows["timestamp"].dt.minute < 30).all()

    truth = ground_truth(dataset, cf)
    predicted = simulate(dataset, cf, rules)
    # Every payment on the dead gateway fails, so the loss is at least the affected share.
    assert -truth.impact.success_rate_delta_pp >= 100 * hit.mean() * 0.9
    assert predicted.impact.success_rate_delta_pp < 0


def test_traffic_increase_adds_new_transactions(dataset, observed):
    cf = _apply(
        dataset, observed, {"type": "traffic_change", "segment": "UPI", "volume_delta": 0.3}
    )
    upi = (cf.baseline["payment_method"] == "UPI").sum()
    added = len(cf.counterfactual) - len(cf.baseline)
    assert added == round(0.3 * upi)
    new = cf.counterfactual.iloc[len(cf.baseline) :]
    assert (new["payment_method"] == "UPI").all()
    assert new["transaction_id"].is_unique
    first_attempts = dataset.transactions[dataset.transactions["attempt"] == 1]
    assert new["transaction_id"].min() > first_attempts["transaction_id"].max()
    assert (
        cf.counterfactual["gateway_utilization"].mean() > cf.baseline["gateway_utilization"].mean()
    )
    assert ground_truth(dataset, cf).counterfactual.transactions == len(cf.counterfactual)


def test_traffic_decrease_drops_transactions(dataset, observed):
    cf = _apply(dataset, observed, {"type": "traffic_change", "volume_delta": -0.5})
    assert len(cf.counterfactual) == len(cf.baseline) - round(0.5 * len(cf.baseline))
    assert set(cf.counterfactual["transaction_id"]) <= set(cf.baseline["transaction_id"])


def test_method_shift(dataset, observed):
    cf = _apply(
        dataset,
        observed,
        {"type": "method_shift", "from": "UPI", "to": "NETBANKING", "percentage": 0.2},
    )
    before = cf.baseline.set_index("transaction_id")["payment_method"].astype(str)
    after = cf.counterfactual.set_index("transaction_id")
    moved = (before == "UPI") & (after["payment_method"].astype(str) == "NETBANKING")
    assert moved.sum() == round(0.2 * (before == "UPI").sum())
    gateway_names = np.array([g.name for g in GATEWAYS])[after["gateway_id"].to_numpy()]
    methods = after["payment_method"].astype(str).to_numpy()
    assert all(m in SUPPORTS[g] for g, m in zip(gateway_names, methods, strict=True))


def test_routing_change_respects_method_support(dataset, observed):
    cf = _apply(
        dataset,
        observed,
        {
            "type": "routing_change",
            "source_gateway": "gateway_a",
            "target_gateway": "gateway_c",
            "traffic_percentage": 0.5,
        },
    )
    before = cf.baseline.set_index("transaction_id")
    after = cf.counterfactual.set_index("transaction_id")
    moved = (before["gateway_id"] == 0) & (after["gateway_id"] == 2)
    eligible = (before["gateway_id"] == 0) & (before["payment_method"].astype(str) != "NETBANKING")
    assert moved.sum() == round(0.5 * eligible.sum())
    assert (after.loc[moved, "payment_method"].astype(str) != "NETBANKING").all()


def test_same_scenario_same_result(dataset, observed):
    iv = {"type": "traffic_change", "segment": "CARD", "volume_delta": 0.4}
    first, second = _apply(dataset, observed, iv), _apply(dataset, observed, iv)
    pd.testing.assert_frame_equal(first.counterfactual, second.counterfactual)


def test_window_outside_dataset_rejected(dataset, observed):
    with pytest.raises(ValueError):
        _apply(dataset, observed, _degradation(), window_date="2030-01-01")


def test_rule_baseline_calibrated_on_held_out_day(dataset, rules, observed):
    cf = _apply(dataset, observed, _degradation())
    predicted = rules.predict(cf.baseline)
    actual = (
        dataset.transactions.set_index("transaction_id")
        .loc[cf.baseline["transaction_id"], "transaction_status"]
        .eq("SUCCESS")
        .mean()
    )
    assert predicted.p_success.mean() == pytest.approx(actual, abs=0.01)
    assert np.allclose(predicted.reason_probs.sum(axis=1), 1.0)
    assert set(np.unique(cf.baseline["payment_method"].astype(str))) <= set(METHODS)
