"""HTTP API over the counterfactual engine.

Data and predictors load once at startup into a `Workspace`. Simulations are deterministic, so
responses are cached in-process by (request, predictor, model version); the cache is sound
without invalidation because nothing changes while the process runs.

One simulation runs at a time: each already uses every CPU thread the model is configured for,
and interleaving two would only make both slower.

Configuration (environment):
- VULCAN_DATA_DIR: generated dataset, default data/generated/default
- VULCAN_PREDICTORS: comma-separated predictors to load, default all three
- VULCAN_DEFAULT_PREDICTOR: used when a request names none
- VULCAN_LLM_API_KEY, VULCAN_LLM_MODEL: natural-language parsing; without a key /parse reports
  itself unavailable and the UI falls back to the form builder
"""

import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    EcosystemGraph,
    GraphEdge,
    GraphNode,
    Meta,
    MethodHealth,
    MitigateRequest,
    MitigateResponse,
    Overview,
    ParseRequest,
    SimulateRequest,
    SimulateResponse,
    TimelinePoint,
    TruthBlock,
)
from backend.config import REPO_ROOT
from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.services.attribution import (
    ATTRIBUTION_DIMENSIONS,
    factor_attribution,
    segment_attribution,
    top_contributors,
)
from backend.services.explanation import explain
from backend.services.mitigation import mitigate
from backend.services.nl_parser import ParseResult, ScenarioParser, parser_from_env
from backend.services.workspace import PREDICTORS, Workspace
from backend.simulation.engine import (
    ScenarioPredictions,
    build_result,
    predict_scenario,
    true_predictions,
)
from backend.simulation.interventions import Counterfactual
from backend.simulation.metrics import SEGMENT_DIMENSIONS, most_affected

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CACHE_SIZE = 256
TIMELINE_BUCKET_MINUTES = 5  # per-minute realized rates rest on ~20 payments and are mostly noise


DateParam = Annotated[date | None, Query(alias="date")]


class _ResultCache:
    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str):
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def put(self, key: str, value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._size:
            self._items.popitem(last=False)


def _model_version(ws: Workspace, predictor: str) -> str:
    days = ws.split.train_days
    return f"{predictor}@seed{ws.dataset.config.seed}:{days[0]}..{days[-1]}"


def _bucket_rates(frame: pd.DataFrame, p: np.ndarray) -> pd.Series:
    minutes = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    bucket = (minutes // TIMELINE_BUCKET_MINUTES * TIMELINE_BUCKET_MINUTES).to_numpy()
    return pd.Series(p).groupby(bucket).mean()


def _timeline(
    cf: Counterfactual, preds: ScenarioPredictions, truth: ScenarioPredictions | None
) -> list[TimelinePoint]:
    series = {
        "baseline": _bucket_rates(cf.baseline, preds.baseline.p_success),
        "counterfactual": _bucket_rates(cf.counterfactual, preds.counterfactual.p_success),
    }
    if truth is not None:
        series["true_baseline"] = _bucket_rates(cf.baseline, truth.baseline.p_success)
        series["true_counterfactual"] = _bucket_rates(
            cf.counterfactual, truth.counterfactual.p_success
        )
    table = pd.DataFrame(series).sort_index()
    return [
        TimelinePoint(
            minute=int(minute),
            **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        )
        for minute, row in table.iterrows()
    ]


def create_app(
    load_workspace: Callable[[], Workspace] | None = None,
    parser: ScenarioParser | None = None,
) -> FastAPI:
    state: dict = {}

    def default_loader() -> Workspace:
        names = os.environ.get("VULCAN_PREDICTORS", ",".join(PREDICTORS))
        return Workspace.load(
            Path(os.environ.get("VULCAN_DATA_DIR", DEFAULT_DATA)),
            predictors=tuple(n.strip() for n in names.split(",") if n.strip()),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["ws"] = (load_workspace or default_loader)()
        state["cache"] = _ResultCache(CACHE_SIZE)
        state["lock"] = threading.Lock()
        state["parser"] = parser or parser_from_env()
        yield

    app = FastAPI(title="Vulcan Counterfactual", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("VULCAN_CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def ws() -> Workspace:
        return state["ws"]

    def default_predictor() -> str:
        loaded = list(ws().predictors)
        wanted = os.environ.get("VULCAN_DEFAULT_PREDICTOR", "shared_representation")
        return wanted if wanted in loaded else loaded[0]

    def resolve(name: str | None) -> str:
        name = name or default_predictor()
        if name not in ws().predictors:
            raise HTTPException(
                status_code=422,
                detail=f"predictor {name!r} is not loaded; available: {list(ws().predictors)}",
            )
        return name

    def cached(kind: str, request, predictor: str, compute):
        key = json.dumps(
            [kind, predictor, request.model_dump(mode="json", by_alias=True)], sort_keys=True
        )
        hit = state["cache"].get(key)
        if hit is not None:
            return hit.model_copy(update={"meta": hit.meta.model_copy(update={"cached": True})})
        with state["lock"]:
            response = compute()
        state["cache"].put(key, response)
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "predictors": list(ws().predictors), "synthetic": True}

    @app.get("/overview", response_model=Overview)
    def overview(window_date: DateParam = None) -> Overview:
        w = ws()
        day = window_date or w.default_window
        txns = w.dataset.transactions
        first = txns[(txns["attempt"] == 1) & (txns["timestamp"].dt.date == day)]
        if first.empty:
            raise HTTPException(status_code=404, detail=f"no transactions on {day}")
        success = first["transaction_status"] == "SUCCESS"
        methods = [
            MethodHealth(
                method=m,
                share=float((first["payment_method"] == m).mean()),
                success_rate=float(success[first["payment_method"] == m].mean()),
            )
            for m in METHODS
        ]
        return Overview(
            window_date=day,
            available_dates=w.split.test_days,
            transactions=len(first),
            success_rate=float(success.mean()),
            gmv=float(first["amount"].sum()),
            gmv_per_hour=float(first["amount"].sum() / 24),
            avg_latency_ms=float(first["latency_ms"].mean()),
            p95_latency_ms=float(first["latency_ms"].quantile(0.95)),
            methods=methods,
            issuers=[i.name for i in ISSUERS],
            gateways=[g.name for g in GATEWAYS],
            predictors=list(w.predictors),
            default_predictor=default_predictor(),
        )

    @app.get("/ecosystem/graph", response_model=EcosystemGraph)
    def graph(window_date: DateParam = None) -> EcosystemGraph:
        w = ws()
        day = window_date or w.default_window
        txns = w.dataset.transactions
        first = txns[(txns["attempt"] == 1) & (txns["timestamp"].dt.date == day)]
        frame = pd.DataFrame(
            {
                "merchant_category": "category:"
                + w.dataset.merchants["merchant_category"].to_numpy()[first["merchant_id"]],
                "method": "method:" + first["payment_method"].astype(str).to_numpy(),
                "issuer": "issuer:"
                + np.array([i.name for i in ISSUERS])[first["issuer_id"].to_numpy()],
                "gateway": "gateway:"
                + np.array([g.name for g in GATEWAYS])[first["gateway_id"].to_numpy()],
                "success": (first["transaction_status"] == "SUCCESS").to_numpy(),
            }
        )
        nodes = []
        for kind in ("merchant_category", "method", "issuer", "gateway"):
            for node_id, g in frame.groupby(kind)["success"]:
                nodes.append(
                    GraphNode(
                        id=node_id,
                        kind=kind,
                        label=node_id.split(":", 1)[1],
                        volume=len(g),
                        success_rate=float(g.mean()),
                    )
                )
        edges = []
        for a, b in (("merchant_category", "method"), ("method", "issuer"), ("issuer", "gateway")):
            for (source, target), g in frame.groupby([a, b])["success"]:
                edges.append(
                    GraphEdge(
                        source=source, target=target, volume=len(g), success_rate=float(g.mean())
                    )
                )
        return EcosystemGraph(window_date=day, nodes=nodes, edges=edges)

    @app.post("/simulate", response_model=SimulateResponse)
    def simulate(request: SimulateRequest) -> SimulateResponse:
        name = resolve(request.predictor)

        def compute() -> SimulateResponse:
            started = time.perf_counter()
            w = ws()
            predictor = w.predictors[name]
            scenario = w.with_window(request.scenario)
            cf = w.counterfactual(scenario)
            preds = predict_scenario(cf, predictor, cache=w.cache)
            result = build_result(name, cf, w.dataset, preds)
            contributions = segment_attribution(w.dataset, cf, preds)
            truth_preds = true_predictions(w.dataset, cf) if request.ground_truth else None
            truth = (
                build_result("ground_truth", cf, w.dataset, truth_preds) if truth_preds else None
            )
            return SimulateResponse(
                meta=Meta(
                    predictor=name,
                    model_version=_model_version(w, name),
                    window_date=cf.window.date,
                    elapsed_ms=1000 * (time.perf_counter() - started),
                ),
                baseline=result.baseline,
                counterfactual=result.counterfactual,
                impact=result.impact,
                interval=result.interval,
                most_affected={d: most_affected(result.segments, d) for d in SEGMENT_DIMENSIONS},
                attribution={
                    d: top_contributors(contributions, d, top=5) for d in ATTRIBUTION_DIMENSIONS
                },
                factors=factor_attribution(cf, predictor) if request.factors else None,
                explanation=explain(w.dataset, scenario, cf, result, contributions),
                ground_truth=(
                    TruthBlock(
                        baseline=truth.baseline,
                        counterfactual=truth.counterfactual,
                        impact=truth.impact,
                    )
                    if truth
                    else None
                ),
                timeline=_timeline(cf, preds, truth_preds),
            )

        return cached("simulate", request, name, compute)

    @app.post("/mitigate", response_model=MitigateResponse)
    def mitigate_route(request: MitigateRequest) -> MitigateResponse:
        name = resolve(request.predictor)

        def compute() -> MitigateResponse:
            started = time.perf_counter()
            w = ws()
            scenario = w.with_window(request.scenario)
            report = mitigate(
                w.dataset, w.observed, scenario, w.predictors[name], cache=w.cache, top=request.top
            )
            return MitigateResponse(
                meta=Meta(
                    predictor=name,
                    model_version=_model_version(w, name),
                    window_date=scenario.window_date,
                    elapsed_ms=1000 * (time.perf_counter() - started),
                ),
                report=report,
            )

        return cached("mitigate", request, name, compute)

    @app.post("/parse", response_model=ParseResult)
    def parse(request: ParseRequest) -> ParseResult:
        """Text to a validated scenario for the user to confirm; never runs a simulation."""
        return state["parser"].parse(request.text)

    @app.get("/experiments")
    def experiments() -> dict:
        """Every experiment's summary, as written by the evaluation scripts."""
        out = {}
        for path in sorted(EXPERIMENTS_DIR.glob("*/results.json")):
            summary = path.parent / "results.md"
            out[path.parent.name] = {
                # NaN is not valid JSON; a missing statistic is sent as null.
                "summary": json.loads(
                    path.read_text(encoding="utf-8"), parse_constant=lambda _: None
                ),
                "markdown": summary.read_text(encoding="utf-8") if summary.exists() else None,
            }
        return out

    return app


app = create_app()
