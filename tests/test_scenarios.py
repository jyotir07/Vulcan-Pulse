import pytest
from pydantic import ValidationError

from backend.simulation.scenarios import (
    GatewayOutage,
    IssuerDegradation,
    MethodShift,
    RoutingChange,
    Scenario,
)


def test_parses_every_intervention_type():
    scenario = Scenario.model_validate(
        {
            "interventions": [
                {"type": "issuer_degradation", "issuer": "hdfc", "success_rate_delta": -0.15},
                {"type": "gateway_outage", "gateway": "GATEWAY_A", "duration_minutes": 20},
                {"type": "traffic_change", "segment": "UPI", "volume_delta": 0.3},
                {"type": "method_shift", "from": "CARD", "to": "UPI", "percentage": 0.2},
                {
                    "type": "routing_change",
                    "source_gateway": "gateway_a",
                    "target_gateway": "gateway_b",
                    "traffic_percentage": 0.25,
                },
            ]
        }
    )
    degradation, outage, _, shift, _ = scenario.interventions
    assert isinstance(degradation, IssuerDegradation) and degradation.issuer == "HDFC"
    assert isinstance(outage, GatewayOutage) and outage.gateway == "gateway_a"
    assert isinstance(shift, MethodShift) and (shift.from_method, shift.to_method) == (
        "CARD",
        "UPI",
    )


@pytest.mark.parametrize(
    "intervention",
    [
        {"type": "issuer_degradation", "issuer": "Not A Bank", "success_rate_delta": -0.1},
        {"type": "issuer_degradation", "issuer": "HDFC", "success_rate_delta": 0.1},
        {"type": "gateway_outage", "gateway": "gateway_z"},
        {"type": "traffic_change", "volume_delta": -1.0},
        {"type": "method_shift", "from": "UPI", "to": "UPI", "percentage": 0.2},
        {"type": "method_shift", "from": "CARD", "to": "UPI", "percentage": 1.5},
        {
            "type": "routing_change",
            "source_gateway": "gateway_a",
            "target_gateway": "gateway_a",
            "traffic_percentage": 0.2,
        },
        {"type": "gateway_outage", "gateway": "gateway_a", "start_hour": 24},
        {"type": "gateway_outage", "gateway": "gateway_a", "surprise": True},
        {"type": "unknown_kind"},
    ],
)
def test_rejects_invalid_interventions(intervention):
    with pytest.raises(ValidationError):
        Scenario.model_validate({"interventions": [intervention]})


def test_rejects_empty_scenario():
    with pytest.raises(ValidationError):
        Scenario.model_validate({"interventions": []})


def test_round_trips_through_json():
    original = Scenario(
        interventions=[
            RoutingChange(
                source_gateway="gateway_a", target_gateway="gateway_b", traffic_percentage=0.25
            ),
            MethodShift(from_method="CARD", to_method="UPI", percentage=0.2),
        ]
    )
    dumped = original.model_dump(mode="json", by_alias=True)
    assert Scenario.model_validate(dumped) == original
