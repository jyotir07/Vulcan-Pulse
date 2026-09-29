"""Natural language to a validated scenario. The LLM translates; it never predicts.

The model is asked for one structured object through a forced tool call whose schema enumerates
the catalog's issuer and gateway names, so it can only pick entities that exist. Its output is
then validated by the same Pydantic `Scenario` the form builder uses. Anything that does not
validate is sent back once with the error; if it still fails, or the model says the request is
ambiguous, the user is asked to clarify rather than given a guess.

The client is an interface: `AnthropicClient` is one implementation over plain HTTP, and any
provider that can return a JSON object for a schema fits. The API key stays server-side.
"""

import os
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.simulation.scenarios import Scenario

TIMEOUT_SECONDS = 20.0
MAX_ATTEMPTS = 2  # the first try and one retry
TOOL_NAME = "submit_scenario"

# Fields each intervention type accepts; anything else the model fills in is dropped before
# validation, since the schema is one flat object for every type.
FIELDS = {
    "issuer_degradation": ("issuer", "method", "success_rate_delta"),
    "gateway_outage": ("gateway",),
    "traffic_change": ("segment", "volume_delta"),
    "method_shift": ("from", "to", "percentage", "issuer"),
    "routing_change": ("source_gateway", "target_gateway", "traffic_percentage"),
}
TIMING = ("start_hour", "duration_minutes")

SYSTEM_PROMPT = f"""You translate a payment-operations "what if" question into a structured \
scenario for a simulator of a synthetic Indian payment ecosystem. You never estimate outcomes.

Intervention types:
- issuer_degradation: an issuer's success rate drops. success_rate_delta is NEGATIVE and in \
absolute terms: "drops by 15%" or "falls 15 points" means -0.15. method is null for all methods.
- gateway_outage: a gateway is fully down.
- traffic_change: volume changes. volume_delta is relative: "+30%" is 0.3, "halves" is -0.5. \
segment is a payment method or null for all traffic.
- method_shift: a share of payments moves from one method to another. percentage in (0, 1]. \
Optional issuer limits it to that bank's customers.
- routing_change: a share of one gateway's traffic moves to another. traffic_percentage in (0, 1].

Timing, for every type: start_hour is the hour of day (0-23) the change starts, default 0. \
duration_minutes is how long it lasts; null means until the end of the day. "for half an hour" \
is 30. "at the evening peak" starts at 18.

Issuers: {", ".join(i.name for i in ISSUERS)}. Gateways: {", ".join(g.name for g in GATEWAYS)} \
("gateway A" is gateway_a). Methods: {", ".join(METHODS)} ("cards" is CARD, "net banking" is \
NETBANKING).

Several changes in one question are several interventions. If the question names something \
that is not in these lists, asks for a prediction instead of a change, or leaves a required \
number out, set status to "clarify" and ask one short question instead of guessing."""


def tool_schema() -> dict:
    method = {"type": ["string", "null"], "enum": [*METHODS, None]}
    issuer = {"type": ["string", "null"], "enum": [*(i.name for i in ISSUERS), None]}
    gateway = {"type": ["string", "null"], "enum": [*(g.name for g in GATEWAYS), None]}
    number = {"type": ["number", "null"]}
    intervention = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(FIELDS)},
            "issuer": issuer,
            "method": method,
            "success_rate_delta": number,
            "gateway": gateway,
            "segment": method,
            "volume_delta": number,
            "from": method,
            "to": method,
            "percentage": number,
            "source_gateway": gateway,
            "target_gateway": gateway,
            "traffic_percentage": number,
            "start_hour": {"type": ["integer", "null"]},
            "duration_minutes": {"type": ["integer", "null"]},
        },
        "required": ["type"],
    }
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "clarify"]},
            "clarification": {"type": "string"},
            "interventions": {"type": "array", "items": intervention},
        },
        "required": ["status"],
    }


class LLMError(Exception):
    """The provider failed, timed out, or returned something that is not the tool call."""


class LLMClient(Protocol):
    provider: str

    def structured(self, system: str, user: str, schema: dict) -> dict: ...


class AnthropicClient:
    provider = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, timeout: float = TIMEOUT_SECONDS):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def structured(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self._model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [
                {"name": TOOL_NAME, "description": "Submit the scenario.", "input_schema": schema}
            ],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = httpx.post(self.url, json=body, headers=headers, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"{self.provider} request failed: {e}") from e
        for block in response.json().get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
                return block["input"]
        raise LLMError(f"{self.provider} returned no {TOOL_NAME} call")


class ParseResult(BaseModel):
    status: Literal["parsed", "clarify", "unavailable"]
    scenario: Scenario | None = None
    message: str | None = None
    provider: str | None = None


def to_scenario(output: dict) -> Scenario:
    """The model's flat interventions, trimmed to each type's fields, validated as a Scenario."""
    interventions = []
    for raw in output.get("interventions") or []:
        kind = raw.get("type")
        if kind not in FIELDS:
            raise ValueError(f"unknown intervention type {kind!r}")
        keep = {"type": kind}
        for name in (*FIELDS[kind], *TIMING):
            if raw.get(name) is not None:
                keep[name] = raw[name]
        interventions.append(keep)
    return Scenario.model_validate({"interventions": interventions})


class ScenarioParser:
    def __init__(self, client: LLMClient | None):
        self.client = client

    def parse(self, text: str) -> ParseResult:
        if self.client is None:
            return ParseResult(
                status="unavailable",
                message="Natural-language input is not configured on this server "
                "(set VULCAN_LLM_API_KEY). Use the scenario builder instead.",
            )
        prompt = text.strip()
        error = ""
        for _ in range(MAX_ATTEMPTS):
            try:
                output = self.client.structured(SYSTEM_PROMPT, prompt, tool_schema())
            except LLMError as e:
                error = str(e)
                continue
            if output.get("status") == "clarify":
                return ParseResult(
                    status="clarify",
                    message=output.get("clarification") or "Could you rephrase the scenario?",
                    provider=self.client.provider,
                )
            try:
                return ParseResult(
                    status="parsed", scenario=to_scenario(output), provider=self.client.provider
                )
            except (ValidationError, ValueError) as e:
                error = str(e)
                prompt = (
                    f"{text.strip()}\n\nYour previous answer was rejected by the validator:\n"
                    f"{error}\nCorrect it, or ask for clarification."
                )
        return ParseResult(
            status="clarify",
            message=f"I couldn't turn that into a valid scenario ({error.splitlines()[0]}). "
            "Try rephrasing, or use the scenario builder.",
            provider=self.client.provider,
        )


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def parser_from_env() -> ScenarioParser:
    key = os.environ.get("VULCAN_LLM_API_KEY")
    if not key:
        return ScenarioParser(None)
    return ScenarioParser(AnthropicClient(key, os.environ.get("VULCAN_LLM_MODEL", DEFAULT_MODEL)))
