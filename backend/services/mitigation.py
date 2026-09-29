"""Search for actions that recover a scenario's losses, by simulating each candidate.

Candidates are added to the scenario's own interventions and re-run through the engine:
- reroutes: move 10-100% of the most-affected gateway's traffic to each other gateway, over
  the scenario's time span. Moving too much onto a gateway saturates it, and the engine's load
  accounting makes that backfire; the search reports it rather than hiding it.
- method steering, for an issuer degradation scoped to one method: move that issuer's payments on
  the degraded method to each other method. Rerouting cannot help there, because an issuer's
  reliability does not depend on the gateway in front of it.

Every result is simulation output under the chosen predictor, not a guaranteed real-world effect.
"""

from pydantic import BaseModel

from backend.data.catalog import GATEWAYS, METHODS
from backend.data.generator import Dataset
from backend.services.attribution import segment_attribution
from backend.simulation.engine import (
    BaselineCache,
    Predictor,
    SimulationResult,
    build_result,
    predict_scenario,
)
from backend.simulation.interventions import ObservedEcosystem, apply_scenario
from backend.simulation.scenarios import GatewayOutage, IssuerDegradation, Scenario

SHARES = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
LABEL = "simulation output"


class Mitigation(BaseModel):
    name: str
    action: dict  # the intervention added to the scenario
    success_rate: float
    success_rate_recovered_pp: float
    gmv_at_risk: float
    gmv_recovered: float
    label: str = LABEL


class MitigationReport(BaseModel):
    scenario_success_rate: float
    scenario_gmv_at_risk: float
    baseline_success_rate: float
    candidates: list[Mitigation]  # best first
    label: str = LABEL


def _timing(iv) -> dict:
    timing = {"start_hour": iv.start_hour}
    if iv.duration_minutes is not None:
        timing["duration_minutes"] = iv.duration_minutes
    return timing


def candidate_actions(scenario: Scenario, source_gateway: str) -> list[tuple[str, dict]]:
    """(name, intervention) pairs to try on top of the scenario."""
    trigger = scenario.interventions[0]
    for iv in scenario.interventions:
        if isinstance(iv, GatewayOutage | IssuerDegradation):
            trigger = iv
            break
    if isinstance(trigger, GatewayOutage):
        source_gateway = trigger.gateway
    timing = _timing(trigger)

    actions = []
    for target in (g.name for g in GATEWAYS if g.name != source_gateway):
        for share in SHARES:
            actions.append(
                (
                    f"Reroute {share:.0%} of {source_gateway} traffic to {target}",
                    {
                        "type": "routing_change",
                        "source_gateway": source_gateway,
                        "target_gateway": target,
                        "traffic_percentage": share,
                        **timing,
                    },
                )
            )
    if isinstance(trigger, IssuerDegradation) and trigger.method is not None:
        for to in (m for m in METHODS if m != trigger.method):
            for share in SHARES:
                actions.append(
                    (
                        f"Steer {share:.0%} of {trigger.issuer} {trigger.method} payments to {to}",
                        {
                            "type": "method_shift",
                            "from": trigger.method,
                            "to": to,
                            "percentage": share,
                            "issuer": trigger.issuer,
                            **timing,
                        },
                    )
                )
    return actions


def _most_affected_gateway(dataset, cf, preds) -> str:
    by_gateway = [c for c in segment_attribution(dataset, cf, preds) if c.dimension == "gateway"]
    return max(by_gateway, key=lambda c: c.gmv_at_risk).segment


def with_action(scenario: Scenario, action: dict) -> Scenario:
    raw = scenario.model_dump(mode="json", by_alias=True)
    return Scenario.model_validate({**raw, "interventions": [*raw["interventions"], action]})


def mitigate(
    dataset: Dataset,
    observed: ObservedEcosystem,
    scenario: Scenario,
    predictor: Predictor,
    cache: BaselineCache | None = None,
    top: int = 5,
) -> MitigationReport:
    cache = cache or BaselineCache()
    cf = apply_scenario(dataset, scenario, observed)
    preds = predict_scenario(cf, predictor, with_interval=False, cache=cache)
    unmitigated: SimulationResult = build_result(predictor.name, cf, dataset, preds)

    candidates = []
    for name, action in candidate_actions(scenario, _most_affected_gateway(dataset, cf, preds)):
        mitigated_cf = apply_scenario(dataset, with_action(scenario, action), observed)
        result = build_result(
            predictor.name,
            mitigated_cf,
            dataset,
            predict_scenario(mitigated_cf, predictor, with_interval=False, cache=cache),
        )
        candidates.append(
            Mitigation(
                name=name,
                action=action,
                success_rate=result.counterfactual.success_rate,
                success_rate_recovered_pp=100.0
                * (result.counterfactual.success_rate - unmitigated.counterfactual.success_rate),
                gmv_at_risk=result.impact.gmv_at_risk,
                gmv_recovered=unmitigated.impact.gmv_at_risk - result.impact.gmv_at_risk,
            )
        )
    candidates.sort(key=lambda m: (m.success_rate_recovered_pp, m.gmv_recovered), reverse=True)
    return MitigationReport(
        scenario_success_rate=unmitigated.counterfactual.success_rate,
        scenario_gmv_at_risk=unmitigated.impact.gmv_at_risk,
        baseline_success_rate=unmitigated.baseline.success_rate,
        candidates=candidates[:top],
    )
