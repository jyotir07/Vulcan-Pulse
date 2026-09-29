"""Where a simulated impact comes from.

Segment attribution splits an impact exactly across the segments of one dimension: the
contributions sum to the total. Factor attribution asks how much of the impact exists because of
the intervention itself, peak-hour conditions and gateway congestion, by re-predicting with each
factor neutralised and sharing the result out by Shapley values.

Both describe the predictor's view of the scenario, not the true process: a factor the model
ignores gets no credit, even if it matters in reality.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from pydantic import BaseModel

from backend.data.generator import Dataset
from backend.simulation.engine import Predictor, ScenarioPredictions, labelled
from backend.simulation.interventions import Counterfactual
from backend.simulation.metrics import compute_impact, compute_metrics

ATTRIBUTION_DIMENSIONS = (
    "issuer",
    "payment_method",
    "gateway",
    "merchant_category",
    "hour_bucket",
    "city",
)
HOUR_BUCKET = 4
# Congestion is neutralised by capping utilization here, the level the rule baseline treats as
# unloaded and below which neither gateway nor issuer load terms act in the synthetic world.
UNCONGESTED_UTILIZATION = 0.7
FACTORS = ("intervention", "peak_hours", "gateway_congestion")


class SegmentContribution(BaseModel):
    dimension: str
    segment: str
    success_rate_delta_pp: float
    failures_delta: float
    gmv_at_risk: float


class FactorContribution(BaseModel):
    factor: str
    success_rate_delta_pp: float
    gmv_at_risk: float


def _with_buckets(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    labels = [f"{h:02d}:00-{h + HOUR_BUCKET:02d}:00" for h in range(0, 24, HOUR_BUCKET)]
    bucket = frame["timestamp"].dt.hour.to_numpy() // HOUR_BUCKET
    return labelled(frame, dataset).assign(hour_bucket=np.array(labels)[bucket])


def segment_attribution(
    dataset: Dataset, cf: Counterfactual, preds: ScenarioPredictions
) -> list[SegmentContribution]:
    """Exact additive split of the total impact over each dimension's segments.

    A segment's success-rate contribution is its expected successes over the whole window's volume,
    after minus before, so contributions add up to the total change in the overall rate. GMV at
    risk uses the window's overall volume ratio for the same reason; per-segment ratios (as in
    `segment_impacts`) would not add up.
    """
    before = _with_buckets(cf.baseline, dataset)
    after = _with_buckets(cf.counterfactual, dataset)
    p_b, p_c = preds.baseline.p_success, preds.counterfactual.p_success
    n_b, n_c = len(before), len(after)
    ratio = after["amount"].sum() / before["amount"].sum()

    def sums(frame: pd.DataFrame, p: np.ndarray, dimension: str) -> pd.DataFrame:
        amount = frame["amount"].to_numpy()
        return (
            pd.DataFrame(
                {
                    "segment": frame[dimension].astype(str).to_numpy(),
                    "success": p,
                    "failures": 1.0 - p,
                    "failed_gmv": amount * (1.0 - p),
                }
            )
            .groupby("segment")
            .sum()
        )

    out = []
    for dimension in ATTRIBUTION_DIMENSIONS:
        joined = (
            sums(before, p_b, dimension)
            .join(sums(after, p_c, dimension), how="outer", lsuffix="_b", rsuffix="_c")
            .fillna(0.0)
        )
        for segment, row in joined.iterrows():
            out.append(
                SegmentContribution(
                    dimension=dimension,
                    segment=str(segment),
                    success_rate_delta_pp=100.0 * (row["success_c"] / n_c - row["success_b"] / n_b),
                    failures_delta=row["failures_c"] - row["failures_b"],
                    gmv_at_risk=row["failed_gmv_c"] - row["failed_gmv_b"] * ratio,
                )
            )
    return out


def top_contributors(
    contributions: list[SegmentContribution], dimension: str, top: int = 3
) -> list[SegmentContribution]:
    """The segments adding the most GMV at risk; segments that add none are left out."""
    rows = [c for c in contributions if c.dimension == dimension and c.gmv_at_risk > 0]
    return sorted(rows, key=lambda c: c.gmv_at_risk, reverse=True)[:top]


def _neutralise(frame: pd.DataFrame, factors: frozenset[str]) -> pd.DataFrame:
    """The frame with every factor *not* in `factors` switched off."""
    out = frame
    if "peak_hours" not in factors:
        out = out.assign(is_peak=False)
    if "gateway_congestion" not in factors:
        out = out.assign(
            gateway_utilization=np.minimum(
                out["gateway_utilization"].to_numpy(), UNCONGESTED_UTILIZATION
            )
        )
    return out


def factor_attribution(cf: Counterfactual, predictor: Predictor) -> list[FactorContribution]:
    """Shapley split of the impact over intervention, peak hours and gateway congestion.

    The value of a coalition is the impact predicted when only its factors are active. Without
    the intervention the counterfactual is the baseline, so every such coalition is worth zero and
    the other factors earn credit only through how they amplify or dampen the intervention.
    Peak is neutralised through the `is_peak` flag only; the hour of day is left alone.
    """
    values: dict[frozenset[str], np.ndarray] = {}
    for size in range(len(FACTORS) + 1):
        for coalition in combinations(FACTORS, size):
            active = frozenset(coalition)
            if "intervention" not in active:
                values[active] = np.zeros(2)
                continue
            base = _neutralise(cf.baseline, active)
            after = _neutralise(cf.counterfactual, active)
            impact = compute_impact(
                compute_metrics(base, predictor.predict(base)),
                compute_metrics(after, predictor.predict(after)),
                int(cf.affected.sum()),
            )
            values[active] = np.array([impact.success_rate_delta_pp, impact.gmv_at_risk])

    n = len(FACTORS)
    out = []
    for factor in FACTORS:
        others = [f for f in FACTORS if f != factor]
        share = np.zeros(2)
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for coalition in combinations(others, size):
                s = frozenset(coalition)
                share += weight * (values[s | {factor}] - values[s])
        out.append(
            FactorContribution(
                factor=factor, success_rate_delta_pp=float(share[0]), gmv_at_risk=float(share[1])
            )
        )
    return out
