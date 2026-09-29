import json
from pathlib import Path

import httpx
import pytest

from backend.data.catalog import GATEWAYS, ISSUERS
from backend.services import nl_parser
from backend.services.nl_parser import (
    TOOL_NAME,
    AnthropicClient,
    LLMError,
    ScenarioParser,
    tool_schema,
)
from backend.simulation.scenarios import Scenario

PHRASINGS = json.loads((Path(__file__).parent / "data" / "nl_phrasings.json").read_text("utf-8"))


class FakeClient:
    """Replays canned outputs (or raises canned errors) and records every prompt."""

    provider = "fake"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def structured(self, system, user, schema):
        self.prompts.append(user)
        output = self.outputs[min(len(self.prompts), len(self.outputs)) - 1]
        if isinstance(output, Exception):
            raise output
        return output


def test_there_are_thirty_phrasings():
    assert len(PHRASINGS) == 30


@pytest.mark.parametrize("case", PHRASINGS, ids=[c["text"] for c in PHRASINGS])
def test_phrasing_parses_to_expected_scenario(case):
    result = ScenarioParser(FakeClient(case["llm"])).parse(case["text"])
    if case["expected"] == "clarify":
        assert result.status == "clarify" and result.message
        assert result.scenario is None
        return
    assert result.status == "parsed", result.message
    assert result.scenario == Scenario.model_validate({"interventions": case["expected"]})


def test_no_client_means_unavailable_not_a_guess():
    result = ScenarioParser(None).parse("HDFC drops 15%")
    assert result.status == "unavailable" and "builder" in result.message


def test_rejected_output_is_retried_once_with_the_error():
    bad = {
        "status": "ok",
        "interventions": [
            {"type": "issuer_degradation", "issuer": "HDFC", "success_rate_delta": 0.15}
        ],
    }
    good = {
        "status": "ok",
        "interventions": [
            {"type": "issuer_degradation", "issuer": "HDFC", "success_rate_delta": -0.15}
        ],
    }
    client = FakeClient(bad, good)
    result = ScenarioParser(client).parse("HDFC drops 15%")
    assert result.status == "parsed"
    assert len(client.prompts) == 2 and "rejected" in client.prompts[1]


def test_transport_failure_is_retried_then_reported():
    good = {"status": "ok", "interventions": [{"type": "gateway_outage", "gateway": "gateway_a"}]}
    assert ScenarioParser(FakeClient(LLMError("timeout"), good)).parse("a down").status == "parsed"
    failing = FakeClient(LLMError("timeout"), LLMError("timeout"))
    result = ScenarioParser(failing).parse("a down")
    assert result.status == "clarify" and len(failing.prompts) == 2


def test_schema_only_offers_catalog_entities():
    props = tool_schema()["properties"]["interventions"]["items"]["properties"]
    assert set(props["issuer"]["enum"]) - {None} == {i.name for i in ISSUERS}
    assert set(props["gateway"]["enum"]) - {None} == {g.name for g in GATEWAYS}


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", AnthropicClient.url)
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self):
        return self._payload


def test_anthropic_client_forces_the_tool_and_reads_its_input(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(json=json, headers=headers, timeout=timeout)
        return _Response(
            200,
            {"content": [{"type": "tool_use", "name": TOOL_NAME, "input": {"status": "clarify"}}]},
        )

    monkeypatch.setattr(nl_parser.httpx, "post", fake_post)
    out = AnthropicClient("key", "model-x", timeout=3.0).structured("sys", "hi", {"type": "object"})
    assert out == {"status": "clarify"}
    assert sent["json"]["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert sent["headers"]["x-api-key"] == "key" and sent["timeout"] == 3.0


def test_anthropic_client_turns_failures_into_llm_errors(monkeypatch):
    monkeypatch.setattr(nl_parser.httpx, "post", lambda *a, **k: _Response(500, {}))
    with pytest.raises(LLMError):
        AnthropicClient("key", "m").structured("s", "u", {})

    def timeout(*a, **k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(nl_parser.httpx, "post", timeout)
    with pytest.raises(LLMError):
        AnthropicClient("key", "m").structured("s", "u", {})

    monkeypatch.setattr(nl_parser.httpx, "post", lambda *a, **k: _Response(200, {"content": []}))
    with pytest.raises(LLMError):
        AnthropicClient("key", "m").structured("s", "u", {})
