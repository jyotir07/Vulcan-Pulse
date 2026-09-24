"""Impact metrics over the first attempts of a window.

Every metric is an expectation under a `Predictions` object, so the same code scores a model's
probabilities and the ground truth's realized outcomes (probability 1 or 0). Success rate is the
first-attempt success rate; retries are part of the world (they add load) but not of the metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel

from backend.data.dgp import FAILURE_REASONS

TIMEOUT_REASONS = ("UPI_TIMEOUT", "GATEWAY_TIMEOUT")
_TIMEOUT_COLUMNS = [FAILURE_REASONS.index(r) for r in TIMEOUT_REASONS]
LATENCY_SAMPLES_PER_ROW = 16
LATENCY_SAMPLE_SEED = 20260101
# A segment counts as affected when its success rate moves at least this much.
AFFECTED_SEGMENT_THRESHOLD_PP = 0.5
SEGMENT_DIMENSIONS = ("issuer", "gateway", "payment_method", "merchant_category", "city")


@dataclass(frozen=True)
class Predictions:
    p_success: np.ndarray  # [row]
    reason_probs: np.ndarray  # [row, reason], distribution of the failure reason given failure
    latency_median_ms: np.ndarray  # [row]
    latency_log_sigma: np.ndarray  # [row], spread of log latency around the median

    def __post_init__(self) -> None:
        n = len(self.p_success)
        if self.reason_probs.shape != (n, len(FAILURE_REASONS)):
            raise ValueError(f"reason_probs must be ({n}, {len(FAILURE_REASONS)})")
        if len(self.latency_median_ms) != n or len(self.latency_log_sigma) != n:
            raise ValueError("latency arrays must have one value per row")
        if np.any((self.p_success < 0) | (self.p_success > 1)):
            raise ValueError("p_success must be within [0, 1]")


class Metrics(BaseModel):
    transactions: int
    success_rate: float
    failure_rate: float
    timeout_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    gmv: float
    successful_gmv: float
    failed_gmv: float


class Impact(BaseModel):
    transactions_affected: int
    success_rate_delta_pp: float
    timeout_rate_delta_pp: float
    failures_delta: float
    # Extra failed GMV beyond what baseline failure rates would give on the scenario's volume,
    # so a pure volume change with unchanged reliability has none.
    gmv_at_risk: float
    avg_latency_delta_ms: float
    p95_latency_delta_ms: float


class SegmentImpact(BaseModel):
    dimension: str
    segment: str
    baseline_transactions: int
    counterfactual_transactions: int
    baseline_success_rate: float
    counterfactual_success_rate: float
    success_rate_delta_pp: float
    gmv_at_risk: float


def compute_metrics(frame: pd.DataFrame, preds: Predictions) -> Metrics:
    amount = frame["amount"].to_numpy()
    p = preds.p_success
    n = len(frame)
    timeout = (1.0 - p) * preds.reason_probs[:, _TIMEOUT_COLUMNS].sum(axis=1)

    # Mean of a lognormal is median * exp(sigma^2 / 2); the p95 of the mixture over rows needs
    # samples. A fixed seed keeps repeated runs identical.
    sigma = preds.latency_log_sigma
    avg_latency = float(np.mean(preds.latency_median_ms * np.exp(sigma**2 / 2)))
    z = np.random.default_rng(LATENCY_SAMPLE_SEED).standard_normal((n, LATENCY_SAMPLES_PER_ROW))
    samples = preds.latency_median_ms[:, None] * np.exp(sigma[:, None] * z)

    return Metrics(
        transactions=n,
        success_rate=float(p.mean()),
        failure_rate=float(1.0 - p.mean()),
        timeout_rate=float(timeout.mean()),
        avg_latency_ms=avg_latency,
        p95_latency_ms=float(np.percentile(samples, 95)),
        gmv=float(amount.sum()),
        successful_gmv=float((amount * p).sum()),
        failed_gmv=float((amount * (1.0 - p)).sum()),
    )


def compute_impact(baseline: Metrics, counterfactual: Metrics, affected: int) -> Impact:
    volume_ratio = counterfactual.gmv / baseline.gmv
    return Impact(
        transactions_affected=affected,
        success_rate_delta_pp=100.0 * (counterfactual.success_rate - baseline.success_rate),
        timeout_rate_delta_pp=100.0 * (counterfactual.timeout_rate - baseline.timeout_rate),
        failures_delta=counterfactual.failure_rate * counterfactual.transactions
        - baseline.failure_rate * baseline.transactions,
        gmv_at_risk=counterfactual.failed_gmv - baseline.failed_gmv * volume_ratio,
        avg_latency_delta_ms=counterfactual.avg_latency_ms - baseline.avg_latency_ms,
        p95_latency_delta_ms=counterfactual.p95_latency_ms - baseline.p95_latency_ms,
    )


def _by_segment(frame: pd.DataFrame, preds: Predictions, dimension: str) -> pd.DataFrame:
    amount = frame["amount"].to_numpy()
    grouped = pd.DataFrame(
        {
            "segment": frame[dimension].astype(str).to_numpy(),
            "transactions": 1,
            "success": preds.p_success,
            "gmv": amount,
            "failed_gmv": amount * (1.0 - preds.p_success),
        }
    ).groupby("segment")
    return grouped.sum()


def segment_impacts(
    baseline: pd.DataFrame,
    baseline_preds: Predictions,
    counterfactual: pd.DataFrame,
    counterfactual_preds: Predictions,
) -> list[SegmentImpact]:
    """Per-segment change, each side grouped by its own values (a rerouted payment counts
    under its old gateway in the baseline and its new one in the counterfactual)."""
    out = []
    for dimension in SEGMENT_DIMENSIONS:
        before = _by_segment(baseline, baseline_preds, dimension)
        after = _by_segment(counterfactual, counterfactual_preds, dimension)
        joined = before.join(after, how="outer", lsuffix="_b", rsuffix="_c").fillna(0.0)
        for segment, row in joined.iterrows():
            rate_b = row["success_b"] / row["transactions_b"] if row["transactions_b"] else 0.0
            rate_c = row["success_c"] / row["transactions_c"] if row["transactions_c"] else 0.0
            ratio = row["gmv_c"] / row["gmv_b"] if row["gmv_b"] else 0.0
            out.append(
                SegmentImpact(
                    dimension=dimension,
                    segment=str(segment),
                    baseline_transactions=int(row["transactions_b"]),
                    counterfactual_transactions=int(row["transactions_c"]),
                    baseline_success_rate=rate_b,
                    counterfactual_success_rate=rate_c,
                    success_rate_delta_pp=100.0 * (rate_c - rate_b),
                    gmv_at_risk=row["failed_gmv_c"] - row["failed_gmv_b"] * ratio,
                )
            )
    return out


def most_affected(
    segments: list[SegmentImpact], dimension: str, top: int = 3
) -> list[SegmentImpact]:
    candidates = [
        s
        for s in segments
        if s.dimension == dimension
        and abs(s.success_rate_delta_pp) >= AFFECTED_SEGMENT_THRESHOLD_PP
    ]
    return sorted(candidates, key=lambda s: s.gmv_at_risk, reverse=True)[:top]
