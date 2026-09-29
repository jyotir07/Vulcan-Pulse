import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.models.rule_baseline import RuleBaseline
from backend.models.splits import evaluation_windows, first_attempts_on, time_split
from backend.services.nl_parser import ScenarioParser
from backend.services.workspace import Workspace
from backend.simulation.interventions import observe

HDFC = {
    "type": "issuer_degradation",
    "issuer": "HDFC",
    "method": "UPI",
    "success_rate_delta": -0.15,
    "start_hour": 18,
    "duration_minutes": 120,
}


@pytest.fixture(scope="module")
def client(dataset):
    split = time_split(dataset, test_days=4)

    def load() -> Workspace:
        return Workspace(
            dataset=dataset,
            observed=observe(dataset),
            split=split,
            default_window=evaluation_windows(dataset, split)[-1],
            predictors={
                "rule_baseline": RuleBaseline.fit(first_attempts_on(dataset, split.train_days))
            },
        )

    with TestClient(create_app(load, parser=ScenarioParser(None))) as c:
        yield c


def test_health_and_overview(client):
    assert client.get("/health").json()["predictors"] == ["rule_baseline"]
    overview = client.get("/overview").json()
    assert overview["synthetic"] is True
    assert 0.8 < overview["success_rate"] < 1.0
    assert sum(m["share"] for m in overview["methods"]) == pytest.approx(1.0)
    assert overview["default_predictor"] == "rule_baseline"


def test_graph_edges_join_known_nodes(client):
    graph = client.get("/ecosystem/graph").json()
    ids = {n["id"] for n in graph["nodes"]}
    assert graph["edges"] and all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    methods = [n for n in graph["nodes"] if n["kind"] == "method"]
    issuers = [n for n in graph["nodes"] if n["kind"] == "issuer"]
    assert sum(n["volume"] for n in methods) == sum(n["volume"] for n in issuers)


def test_simulate_returns_impact_truth_and_explanation(client):
    response = client.post(
        "/simulate", json={"scenario": {"interventions": [HDFC]}, "factors": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["synthetic"] is True and body["meta"]["cached"] is False
    assert body["impact"]["success_rate_delta_pp"] < 0
    assert body["ground_truth"]["impact"]["success_rate_delta_pp"] < 0
    assert body["attribution"]["issuer"][0]["segment"] == "HDFC"
    assert {f["factor"] for f in body["factors"]} == {
        "intervention",
        "peak_hours",
        "gateway_congestion",
    }
    assert body["explanation"]["facts"]
    evening = [p for p in body["timeline"] if 18 * 60 <= p["minute"] < 20 * 60]
    assert all(p["counterfactual"] <= p["baseline"] + 1e-12 for p in evening)

    again = client.post("/simulate", json={"scenario": {"interventions": [HDFC]}, "factors": True})
    assert again.json()["meta"]["cached"] is True
    assert again.json()["impact"] == body["impact"]


def test_invalid_scenarios_are_422(client):
    bad_issuer = {**HDFC, "issuer": "Nonexistent Bank"}
    assert (
        client.post("/simulate", json={"scenario": {"interventions": [bad_issuer]}}).status_code
        == 422
    )
    assert client.post("/simulate", json={"scenario": {"interventions": []}}).status_code == 422
    unknown = client.post(
        "/simulate",
        json={"scenario": {"interventions": [HDFC]}, "predictor": "shared_representation"},
    )
    assert unknown.status_code == 422


def test_mitigate_ranks_labelled_candidates(client):
    outage = {
        "type": "gateway_outage",
        "gateway": "gateway_a",
        "start_hour": 19,
        "duration_minutes": 60,
    }
    body = client.post("/mitigate", json={"scenario": {"interventions": [outage]}, "top": 3}).json()
    report = body["report"]
    assert report["label"] == "simulation output"
    assert len(report["candidates"]) == 3
    recovered = [c["success_rate_recovered_pp"] for c in report["candidates"]]
    assert recovered == sorted(recovered, reverse=True)


def test_experiments_lists_written_results(client):
    assert isinstance(client.get("/experiments").json(), dict)


def test_parse_without_a_key_points_to_the_builder(client):
    body = client.post("/parse", json={"text": "HDFC UPI drops 15%"}).json()
    assert body["status"] == "unavailable"
    assert client.post("/parse", json={"text": ""}).status_code == 422
