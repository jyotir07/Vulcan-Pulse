"""Counterfactual engine: baseline prediction, intervention, counterfactual prediction, impact.

`simulate` runs any predictor on the engine's observed view of the scenario. `ground_truth` runs
the true process on the same transactions. Both return the same result type, so predictions and
truth can be compared field by field.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from backend.data.catalog import GATEWAYS, ISSUERS
from backend.data.dgp import FAILURE_REASONS
from backend.data.generator import Dataset
from backend.simulation.interventions import Counterfactual
from backend.simulation.metrics import (
    Impact,
    ImpactInterval,
    Metrics,
    Predictions,
    SegmentImpact,
    compute_impact,
    compute_metrics,
    ensemble_mean,
    impact_interval,
    segment_impacts,
)
from backend.simulation.oracle import true_outcomes

GROUND_TRUTH = "ground_truth"


class Predictor(Protocol):
    name: str

    def predict(self, frame: pd.DataFrame) -> Predictions: ...


@runtime_checkable
class EnsemblePredictor(Predictor, Protocol):
    """A predictor whose members disagree, so its spread measures model uncertainty."""

    def member_predictions(self, frame: pd.DataFrame) -> list[Predictions]: ...


class SimulationResult(BaseModel):
    source: str
    window_date: date
    baseline: Metrics
    counterfactual: Metrics
    impact: Impact
    segments: list[SegmentImpact]
    # None unless the predictor is an ensemble; ground truth has no sampling uncertainty.
    interval: ImpactInterval | None = None


def labelled(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    categories = dataset.merchants["merchant_category"].to_numpy()
    return frame.assign(
        issuer=np.array([i.name for i in ISSUERS])[frame["issuer_id"].to_numpy()],
        gateway=np.array([g.name for g in GATEWAYS])[frame["gateway_id"].to_numpy()],
        merchant_category=categories[frame["merchant_id"].to_numpy()],
    )


@dataclass(frozen=True)
class ScenarioPredictions:
    baseline: Predictions
    counterfactual: Predictions
    # Per-member (baseline, counterfactual) pairs; empty unless the predictor is an ensemble.
    members: list[tuple[Predictions, Predictions]] = field(default_factory=list)


class BaselineCache:
    """Baseline predictions per (predictor, window).

    A window's baseline frame is the same for every scenario, so a grid or a session of API calls
    predicts it once. The predictor object is kept alongside its entry so a recycled `id` can
    never match a different model.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, date, bool], tuple[Predictor, list[Predictions]]] = {}

    def get(self, predictor: Predictor, window: date, members: bool, compute) -> list[Predictions]:
        key = (id(predictor), window, members)
        entry = self._entries.get(key)
        if entry is None or entry[0] is not predictor:
            entry = (predictor, compute())
            self._entries[key] = entry
        return entry[1]


def _splice(base: Predictions, position: np.ndarray, changed: np.ndarray, fresh) -> Predictions:
    """Counterfactual predictions: baseline rows reused where inputs are unchanged."""
    out = base.take(np.where(position >= 0, position, 0))
    if fresh is None:
        return out
    arrays = {}
    for name in ("p_success", "reason_probs", "latency_median_ms", "latency_log_sigma"):
        values = getattr(out, name).copy()
        values[changed] = getattr(fresh, name)
        arrays[name] = values
    return Predictions(**arrays)


def predict_scenario(
    cf: Counterfactual,
    predictor: Predictor,
    with_interval: bool = True,
    cache: BaselineCache | None = None,
) -> ScenarioPredictions:
    """Predictions for both sides of a counterfactual.

    Only counterfactual rows whose inputs changed are sent through the model; every other row has
    exactly its baseline inputs, and every predictor scores rows independently, so its baseline
    prediction is reused. An ensemble's members are predicted once and averaged here, rather than
    paying for a second pass to get the interval.
    """
    ensemble = with_interval and isinstance(predictor, EnsemblePredictor)

    def predict(frame: pd.DataFrame) -> list[Predictions]:
        return predictor.member_predictions(frame) if ensemble else [predictor.predict(frame)]

    if cache is None:
        before = predict(cf.baseline)
    else:
        before = cache.get(predictor, cf.window.date, ensemble, lambda: predict(cf.baseline))

    position = pd.Index(cf.baseline["transaction_id"]).get_indexer(
        cf.counterfactual["transaction_id"]
    )
    changed = np.flatnonzero(cf.affected)
    fresh = predict(cf.counterfactual.iloc[changed]) if len(changed) else [None] * len(before)
    after = [_splice(b, position, changed, f) for b, f in zip(before, fresh, strict=True)]

    if not ensemble:
        return ScenarioPredictions(baseline=before[0], counterfactual=after[0])
    return ScenarioPredictions(
        baseline=ensemble_mean(before),
        counterfactual=ensemble_mean(after),
        members=list(zip(before, after, strict=True)),
    )


def build_result(
    source: str, cf: Counterfactual, dataset: Dataset, preds: ScenarioPredictions
) -> SimulationResult:
    baseline = labelled(cf.baseline, dataset)
    counterfactual = labelled(cf.counterfactual, dataset)
    before = compute_metrics(baseline, preds.baseline)
    after = compute_metrics(counterfactual, preds.counterfactual)
    affected = int(cf.affected.sum())
    result = SimulationResult(
        source=source,
        window_date=cf.window.date,
        baseline=before,
        counterfactual=after,
        impact=compute_impact(before, after, affected),
        segments=segment_impacts(
            baseline, preds.baseline, counterfactual, preds.counterfactual
        ),
    )
    if not preds.members:
        return result

    # Each member's own impact gives model uncertainty; the interval's bootstrap term accounts
    # for which transactions the window happened to contain.
    member_impacts = [
        compute_impact(
            compute_metrics(cf.baseline, b), compute_metrics(cf.counterfactual, a), affected
        )
        for b, a in preds.members
    ]
    return result.model_copy(
        update={
            "interval": impact_interval(
                cf.baseline,
                preds.baseline,
                cf.counterfactual,
                preds.counterfactual,
                affected,
                member_impacts,
            )
        }
    )


def simulate(
    dataset: Dataset,
    cf: Counterfactual,
    predictor: Predictor,
    with_interval: bool = True,
    cache: BaselineCache | None = None,
) -> SimulationResult:
    preds = predict_scenario(cf, predictor, with_interval, cache)
    return build_result(predictor.name, cf, dataset, preds)


def _realized(frame: pd.DataFrame, outcomes: pd.DataFrame) -> Predictions:
    """Outcomes of the true process for `frame`'s transactions, as degenerate predictions."""
    first = outcomes[outcomes["attempt"] == 1].set_index("transaction_id")
    rows = first.loc[frame["transaction_id"]]
    failed = (rows["transaction_status"] == "FAILED").to_numpy()
    reasons = np.zeros((len(rows), len(FAILURE_REASONS)))
    codes = rows["failure_reason"].cat.codes.to_numpy()
    reasons[np.flatnonzero(failed), codes[failed]] = 1.0
    return Predictions(
        p_success=(~failed).astype(float),
        reason_probs=reasons,
        latency_median_ms=rows["latency_ms"].to_numpy().astype(float),
        latency_log_sigma=np.zeros(len(rows)),
    )


def true_predictions(dataset: Dataset, cf: Counterfactual) -> ScenarioPredictions:
    """The true process's realized outcomes on both sides, as degenerate predictions."""
    baseline = true_outcomes(dataset, context=cf.oracle_baseline_context)
    episodes = dataset.latent.episodes
    if len(cf.extra_episodes):
        episodes = pd.concat([episodes, cf.extra_episodes], ignore_index=True)
    counterfactual = true_outcomes(
        dataset, context=cf.oracle_counterfactual_context, episodes=episodes
    )
    return ScenarioPredictions(
        baseline=_realized(cf.baseline, baseline),
        counterfactual=_realized(cf.counterfactual, counterfactual),
    )


def ground_truth(dataset: Dataset, cf: Counterfactual) -> SimulationResult:
    return build_result(GROUND_TRUTH, cf, dataset, true_predictions(dataset, cf))
