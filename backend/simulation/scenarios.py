"""Structured what-if scenarios.

Names are resolved against the catalog at validation time, so an unknown issuer or gateway fails
with a clear error instead of silently matching nothing. Times are relative to the evaluation
window: `start_hour` is the hour of the window day, and a missing `duration_minutes` means "until
the end of the window".
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.data.catalog import GATEWAYS, ISSUERS

MethodName = Literal["UPI", "CARD", "NETBANKING"]


def _issuer_name(value: str) -> str:
    for issuer in ISSUERS:
        if issuer.name.lower() == value.strip().lower():
            return issuer.name
    raise ValueError(f"unknown issuer {value!r}; expected one of {[i.name for i in ISSUERS]}")


def _gateway_name(value: str) -> str:
    for gateway in GATEWAYS:
        if gateway.name == value.strip().lower():
            return gateway.name
    raise ValueError(f"unknown gateway {value!r}; expected one of {[g.name for g in GATEWAYS]}")


class _Intervention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    start_hour: int = Field(default=0, ge=0, le=23)
    duration_minutes: int | None = Field(default=None, gt=0)


class IssuerDegradation(_Intervention):
    type: Literal["issuer_degradation"] = "issuer_degradation"
    issuer: str
    method: MethodName | None = None
    # Absolute change in success rate: -0.15 means 15 percentage points lower.
    success_rate_delta: float = Field(lt=0.0, ge=-1.0)

    @field_validator("issuer")
    @classmethod
    def _known_issuer(cls, value: str) -> str:
        return _issuer_name(value)


class GatewayOutage(_Intervention):
    type: Literal["gateway_outage"] = "gateway_outage"
    gateway: str

    @field_validator("gateway")
    @classmethod
    def _known_gateway(cls, value: str) -> str:
        return _gateway_name(value)


class TrafficChange(_Intervention):
    type: Literal["traffic_change"] = "traffic_change"
    # A payment method, or None for all traffic.
    segment: MethodName | None = None
    volume_delta: float = Field(gt=-1.0, le=5.0)


class MethodShift(_Intervention):
    type: Literal["method_shift"] = "method_shift"
    from_method: MethodName = Field(alias="from")
    to_method: MethodName = Field(alias="to")
    percentage: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _distinct(self) -> "MethodShift":
        if self.from_method == self.to_method:
            raise ValueError("method_shift needs different 'from' and 'to' methods")
        return self


class RoutingChange(_Intervention):
    type: Literal["routing_change"] = "routing_change"
    source_gateway: str
    target_gateway: str
    traffic_percentage: float = Field(gt=0.0, le=1.0)

    @field_validator("source_gateway", "target_gateway")
    @classmethod
    def _known_gateway(cls, value: str) -> str:
        return _gateway_name(value)

    @model_validator(mode="after")
    def _distinct(self) -> "RoutingChange":
        if self.source_gateway == self.target_gateway:
            raise ValueError("routing_change needs different source and target gateways")
        return self


Intervention = Annotated[
    IssuerDegradation | GatewayOutage | TrafficChange | MethodShift | RoutingChange,
    Field(discriminator="type"),
]


class Scenario(BaseModel):
    """One or more interventions applied together to one evaluation window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interventions: list[Intervention] = Field(min_length=1)
    # The day to evaluate; defaults to a representative normal day of the dataset.
    window_date: date | None = None
