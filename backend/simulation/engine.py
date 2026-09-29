"""Counterfactual engine: baseline prediction, intervention, counterfactual prediction, impact.

`simulate` runs any predictor on the engine's observed view of the scenario. `ground_truth` runs
the true process on the same transactions. Both return the same result type, so predictions and
truth can be compared field by field.
"""

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


def _labelled(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    categories = dataset.merchants["merchant_category"].to_numpy()
    return frame.assign(
        issuer=np.array([i.name for i in ISSUERS])[frame["issuer_id"].to_numpy()],
        gateway=np.array([g.name for g in GATEWAYS])[frame["gateway_id"].to_numpy()],
        merchant_category=categories[frame["merchant_id"].to_numpy()],
    )


def _result(
    source: str,
    cf: Counterfactual,
    dataset: Dataset,
    baseline_preds: Predictions,
    counterfactual_preds: Predictions,
) -> SimulationResult:
    baseline = _labelled(cf.baseline, dataset)
    counterfactual = _labelled(cf.counterfactual, dataset)
    before = compute_metrics(baseline, baseline_preds)
    after = compute_metrics(counterfactual, counterfactual_preds)
    return SimulationResult(
        source=source,
        window_date=cf.window.date,
        baseline=before,
        counterfactual=after,
        impact=compute_impact(before, after, int(cf.affected.sum())),
        segments=segment_impacts(baseline, baseline_preds, counterfactual, counterfactual_preds),
    )


def simulate(
    dataset: Dataset, cf: Counterfactual, predictor: Predictor, with_interval: bool = True
) -> SimulationResult:
    # An ensemble's members are needed for the interval anyway, so they are predicted once and
    # averaged here, rather than paying for a second pass through the model.
    if with_interval and isinstance(predictor, EnsemblePredictor):
        before = predictor.member_predictions(cf.baseline)
        after = predictor.member_predictions(cf.counterfactual)
        baseline_preds = ensemble_mean(before)
        counterfactual_preds = ensemble_mean(after)
        members = list(zip(before, after, strict=True))
    else:
        baseline_preds = predictor.predict(cf.baseline)
        counterfactual_preds = predictor.predict(cf.counterfactual)
        members = []

    result = _result(predictor.name, cf, dataset, baseline_preds, counterfactual_preds)
    if not members:
        return result

    # Each member's own impact gives model uncertainty; the interval's bootstrap term accounts
    # for which transactions the window happened to contain.
    affected = int(cf.affected.sum())
    member_impacts = [
        compute_impact(
            compute_metrics(cf.baseline, b), compute_metrics(cf.counterfactual, a), affected
        )
        for b, a in members
    ]
    return result.model_copy(
        update={
            "interval": impact_interval(
                cf.baseline,
                baseline_preds,
                cf.counterfactual,
                counterfactual_preds,
                affected,
                member_impacts,
            )
        }
    )


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


def ground_truth(dataset: Dataset, cf: Counterfactual) -> SimulationResult:
    baseline = true_outcomes(dataset, context=cf.oracle_baseline_context)
    episodes = dataset.latent.episodes
    if len(cf.extra_episodes):
        episodes = pd.concat([episodes, cf.extra_episodes], ignore_index=True)
    counterfactual = true_outcomes(
        dataset, context=cf.oracle_counterfactual_context, episodes=episodes
    )
    return _result(
        GROUND_TRUTH,
        cf,
        dataset,
        _realized(cf.baseline, baseline),
        _realized(cf.counterfactual, counterfactual),
    )
