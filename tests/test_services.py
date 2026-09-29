import numpy as np
import pytest

from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import first_attempts_excluding
from backend.services.attribution import (
    ATTRIBUTION_DIMENSIONS,
    factor_attribution,
    segment_attribution,
)
from backend.services.explanation import explain
from backend.services.mitigation import LABEL, candidate_actions, mitigate
from backend.simulation.engine import build_result, predict_scenario
from backend.simulation.interventions import apply_scenario, default_window_date, observe
from backend.simulation.scenarios import Scenario

HDFC_UPI = {
    "type": "issuer_degradation",
    "issuer": "HDFC",
    "method": "UPI",
    "success_rate_delta": -0.15,
    "start_hour": 18,
    "duration_minutes": 120,
}
OUTAGE = {
    "type": "gateway_outage",
    "gateway": "gateway_a",
    "start_hour": 19,
    "duration_minutes": 60,
}


@pytest.fixture(scope="module")
def observed(dataset):
    return observe(dataset)


@pytest.fixture(scope="module")
def rules(dataset):
    return RuleBaseline.fit(first_attempts_excluding(dataset, default_window_date(dataset)))


def _run(dataset, observed, rules, *interventions):
    scenario = Scenario.model_validate({"interventions": list(interventions)})
    cf = apply_scenario(dataset, scenario, observed)
    preds = predict_scenario(cf, rules)
    return scenario, cf, preds, build_result(rules.name, cf, dataset, preds)


@pytest.mark.parametrize("iv", [HDFC_UPI, OUTAGE, {"type": "traffic_change", "volume_delta": 0.5}])
def test_segment_contributions_add_up_to_the_impact(dataset, observed, rules, iv):
    _, cf, preds, result = _run(dataset, observed, rules, iv)
    contributions = segment_attribution(dataset, cf, preds)
    for dimension in ATTRIBUTION_DIMENSIONS:
        rows = [c for c in contributions if c.dimension == dimension]
        assert sum(c.success_rate_delta_pp for c in rows) == pytest.approx(
            result.impact.success_rate_delta_pp, abs=1e-9
        )
        assert sum(c.failures_delta for c in rows) == pytest.approx(
            result.impact.failures_delta, abs=1e-6
        )
        assert sum(c.gmv_at_risk for c in rows) == pytest.approx(
            result.impact.gmv_at_risk, rel=1e-9, abs=1e-3
        )


def test_degradation_is_attributed_to_the_degraded_issuer(dataset, observed, rules):
    _, cf, preds, _ = _run(dataset, observed, rules, HDFC_UPI)
    by_issuer = [c for c in segment_attribution(dataset, cf, preds) if c.dimension == "issuer"]
    worst = max(by_issuer, key=lambda c: c.gmv_at_risk)
    assert worst.segment == "HDFC"
    others = [c for c in by_issuer if c.segment != "HDFC"]
    assert all(abs(c.success_rate_delta_pp) < 1e-9 for c in others)


def test_factor_shares_add_up_and_ignored_factors_get_nothing(dataset, observed, rules):
    _, cf, _, result = _run(dataset, observed, rules, HDFC_UPI)
    factors = {f.factor: f for f in factor_attribution(cf, rules)}
    total = sum(f.success_rate_delta_pp for f in factors.values())
    assert total == pytest.approx(result.impact.success_rate_delta_pp, abs=1e-9)
    # The rule baseline never reads the peak flag, so peak hours cannot earn credit.
    assert factors["peak_hours"].success_rate_delta_pp == pytest.approx(0.0, abs=1e-12)
    assert factors["intervention"].success_rate_delta_pp < 0


def test_explanation_states_the_issuer_share(dataset, observed, rules):
    scenario, cf, preds, result = _run(dataset, observed, rules, HDFC_UPI)
    text = explain(dataset, scenario, cf, result, segment_attribution(dataset, cf, preds))
    assert f"{result.impact.success_rate_delta_pp:+.2f} pp" in text.headline
    base = cf.baseline
    hour = base["timestamp"].dt.hour
    pool = base[(base["payment_method"] == "UPI") & hour.between(18, 19)]
    share = (pool["issuer_id"] == 0).mean()
    assert f"HDFC carries {share:.0%} of UPI payments 18:00-20:00" in text.facts[0]


def test_outage_mitigations_are_ranked_and_labelled(dataset, observed, rules):
    scenario = Scenario.model_validate({"interventions": [OUTAGE]})
    report = mitigate(dataset, observed, scenario, rules, top=100)
    recovered = [m.success_rate_recovered_pp for m in report.candidates]
    assert recovered == sorted(recovered, reverse=True)
    assert recovered[0] > 0
    assert report.label == LABEL and all(m.label == LABEL for m in report.candidates)
    assert all(m.action["source_gateway"] == "gateway_a" for m in report.candidates)


def test_degradation_offers_method_steering(dataset, observed, rules):
    scenario = Scenario.model_validate({"interventions": [HDFC_UPI]})
    names = [name for name, _ in candidate_actions(scenario, "gateway_a")]
    assert any(name.startswith("Steer") for name in names)
    report = mitigate(dataset, observed, scenario, rules, top=3)
    assert report.candidates[0].name.startswith("Steer")
    assert report.candidates[0].success_rate_recovered_pp > 0
    assert np.isfinite(report.candidates[0].gmv_recovered)
