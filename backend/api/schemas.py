"""Request and response shapes of the HTTP API."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from backend.services.attribution import FactorContribution, SegmentContribution
from backend.services.explanation import Explanation
from backend.services.mitigation import MitigationReport
from backend.simulation.metrics import Impact, ImpactInterval, Metrics, SegmentImpact
from backend.simulation.scenarios import Scenario

DISCLAIMER = "Synthetic data; simulation output, not real payment performance."


class Meta(BaseModel):
    synthetic: bool = True
    disclaimer: str = DISCLAIMER
    predictor: str
    model_version: str
    window_date: date
    cached: bool = False
    elapsed_ms: float


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: Scenario
    predictor: str | None = None
    # Also run the true process on the same transactions, for the validation panel.
    ground_truth: bool = True
    # Shapley split over intervention / peak / congestion; re-predicts the window several times.
    factors: bool = False


class TimelinePoint(BaseModel):
    minute: int  # minutes after the window's midnight
    baseline: float  # expected success rate of payments in this bucket
    counterfactual: float
    true_baseline: float | None = None  # realized, when ground truth was requested
    true_counterfactual: float | None = None


class TruthBlock(BaseModel):
    baseline: Metrics
    counterfactual: Metrics
    impact: Impact


class SimulateResponse(BaseModel):
    meta: Meta
    baseline: Metrics
    counterfactual: Metrics
    impact: Impact
    interval: ImpactInterval | None
    most_affected: dict[str, list[SegmentImpact]]
    attribution: dict[str, list[SegmentContribution]]
    factors: list[FactorContribution] | None
    explanation: Explanation
    ground_truth: TruthBlock | None
    timeline: list[TimelinePoint]


class MitigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: Scenario
    predictor: str | None = None
    top: int = Field(default=5, ge=1, le=50)


class MitigateResponse(BaseModel):
    meta: Meta
    report: MitigationReport


class MethodHealth(BaseModel):
    method: str
    share: float
    success_rate: float


class Overview(BaseModel):
    synthetic: bool = True
    disclaimer: str = DISCLAIMER
    window_date: date
    available_dates: list[date]
    transactions: int
    success_rate: float
    gmv: float
    gmv_per_hour: float
    avg_latency_ms: float
    p95_latency_ms: float
    methods: list[MethodHealth]
    issuers: list[str]
    gateways: list[str]
    predictors: list[str]
    default_predictor: str


class GraphNode(BaseModel):
    id: str
    kind: str  # merchant_category | method | issuer | gateway
    label: str
    volume: int
    success_rate: float


class GraphEdge(BaseModel):
    source: str
    target: str
    volume: int
    success_rate: float


class EcosystemGraph(BaseModel):
    window_date: date
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
