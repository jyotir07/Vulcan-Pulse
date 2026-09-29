"""Impact metrics over the first attempts of a window.

Every metric is an expectation under a `Predictions` object, so the same code scores a model's
probabilities and the ground truth's realized outcomes (probability 1 or 0). Success rate is the
first-attempt success rate; retries are part of the world (they add load) but not of the metrics.

`impact_interval` turns an ensemble's predictive spread and a bootstrap over transactions into a
confidence interval on each impact field.
"""

from dataclasses import dataclass
from statistics import NormalDist

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

# Transactions inside one block share load and issuer health, so they are not independent draws.
# The bootstrap resamples blocks rather than rows.
BOOTSTRAP_BLOCK_MINUTES = 5
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 20260203
CI_LEVEL = 0.90
IMPACT_FIELDS = (
    "success_rate_delta_pp",
    "timeout_rate_delta_pp",
    "failures_delta",
    "gmv_at_risk",
    "avg_latency_delta_ms",
    "p95_latency_delta_ms",
)


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

    def take(self, rows: np.ndarray) -> "Predictions":
        return Predictions(
            p_success=self.p_success[rows],
            reason_probs=self.reason_probs[rows],
            latency_median_ms=self.latency_median_ms[rows],
            latency_log_sigma=self.latency_log_sigma[rows],
        )


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


class ImpactInterval(BaseModel):
    """A two-sided confidence interval per impact field, in the field's own units."""

    level: float
    success_rate_delta_pp: tuple[float, float]
    timeout_rate_delta_pp: tuple[float, float]
    failures_delta: tuple[float, float]
    gmv_at_risk: tuple[float, float]
    avg_latency_delta_ms: tuple[float, float]
    p95_latency_delta_ms: tuple[float, float]


def _impact_vector(impact: Impact) -> np.ndarray:
    return np.array([getattr(impact, f) for f in IMPACT_FIELDS], dtype=float)


def ensemble_mean(members: list[Predictions]) -> Predictions:
    """Combine ensemble members into one prediction.

    Success and reason probabilities average in probability space; latency medians average in log
    space, where a member's median is a lognormal location, and the spread is moment-matched to
    the members' mixture.
    """
    log_median = np.stack([np.log(m.latency_median_ms) for m in members])
    sigma = np.stack([m.latency_log_sigma for m in members])
    return Predictions(
        p_success=np.mean([m.p_success for m in members], axis=0),
        reason_probs=np.mean([m.reason_probs for m in members], axis=0),
        latency_median_ms=np.exp(log_median.mean(axis=0)),
        latency_log_sigma=np.sqrt((sigma**2).mean(axis=0) + log_median.var(axis=0)),
    )


def _interval(point: np.ndarray, spread: np.ndarray, level: float) -> ImpactInterval:
    half = NormalDist().inv_cdf(0.5 + level / 2) * spread
    bounds = zip(point - half, point + half, strict=True)
    return ImpactInterval(level=level, **dict(zip(IMPACT_FIELDS, bounds, strict=True)))


def impact_interval(
    baseline: pd.DataFrame,
    baseline_preds: Predictions,
    counterfactual: pd.DataFrame,
    counterfactual_preds: Predictions,
    affected: int,
    member_impacts: list[Impact],
    level: float = CI_LEVEL,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> ImpactInterval:
    """Confidence interval on each impact field from two independent sources.

    The ensemble members are alternative fits of the same data, so their spread on the full
    window is model uncertainty. The block bootstrap resamples the window, so its spread is
    sampling uncertainty. The two are independent, so their variances add.

    Blocks, not rows: transactions in the same five minutes share load and issuer health, so
    resampling rows independently would understate the spread.
    """
    point = _impact_vector(
        compute_impact(
            compute_metrics(baseline, baseline_preds),
            compute_metrics(counterfactual, counterfactual_preds),
            affected,
        )
    )
    variance = np.zeros(len(IMPACT_FIELDS))
    if len(member_impacts) > 1:
        members = np.stack([_impact_vector(i) for i in member_impacts])
        # ddof=1: the members sample the model's uncertainty, they are not the whole of it.
        variance = members.var(axis=0, ddof=1)

    start = baseline["timestamp"].min()
    base_groups, base_blocks = _block_rows(baseline["timestamp"], start)
    cf_groups, cf_blocks = _block_rows(counterfactual["timestamp"], start)
    # Both frames are padded to the same block count so one draw of indices resamples them on
    # matching times. Extra slots are empty and contribute nothing.
    n_blocks = max(len(base_groups), len(cf_groups))
    base_sums = _pad_blocks(_block_sums(baseline, baseline_preds, base_groups), n_blocks)
    cf_sums = _pad_blocks(_block_sums(counterfactual, counterfactual_preds, cf_groups), n_blocks)
    base_p95 = _LatencyQuantile.fit(baseline, baseline_preds, base_blocks, n_blocks)
    cf_p95 = _LatencyQuantile.fit(counterfactual, counterfactual_preds, cf_blocks, n_blocks)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
    # Both frames are resampled on the same drawn blocks, so baseline and counterfactual stay
    # paired in time and the difference is not inflated by unrelated sampling noise.
    counts = np.zeros((n_bootstrap, n_blocks), dtype=np.int64)
    np.add.at(counts, (np.arange(n_bootstrap)[:, None], picks), 1)
    draws = _impact_from_sums(
        counts @ base_sums, counts @ cf_sums, base_p95.p95(counts), cf_p95.p95(counts)
    )

    variance += draws.var(axis=0, ddof=1)
    return _interval(point, np.sqrt(variance), level)


def _pad_blocks(sums: np.ndarray, n_blocks: int) -> np.ndarray:
    if len(sums) >= n_blocks:
        return sums[:n_blocks]
    return np.vstack([sums, np.zeros((n_blocks - len(sums), sums.shape[1]))])


@dataclass(frozen=True)
class _LatencyQuantile:
    """P95 of a block resample, from a fixed set of per-row lognormal draws.

    `compute_metrics` draws each row's latency samples from one fixed seed, so a resample's
    distribution is the same samples with rows repeated. Sorting once and tracking each draw's
    block turns the P95 of any resample into a cumulative-weight lookup, instead of re-sorting
    half a million values per replicate.
    """

    sorted_samples: np.ndarray  # [row * draws], ascending
    block_of_sample: np.ndarray  # [row * draws], the block each draw came from

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, preds: Predictions, blocks: np.ndarray, n_blocks: int
    ) -> "_LatencyQuantile":
        n = len(frame)
        z = np.random.default_rng(LATENCY_SAMPLE_SEED).standard_normal(
            (n, LATENCY_SAMPLES_PER_ROW)
        )
        samples = preds.latency_median_ms[:, None] * np.exp(preds.latency_log_sigma[:, None] * z)
        order = np.argsort(samples, axis=None, kind="stable")
        # Both frames are padded to a common block count; a frame with fewer blocks never reaches
        # a slot above its own last one, so the clamp only guards a malformed input.
        clamped = np.minimum(blocks, n_blocks - 1)
        flat_blocks = np.repeat(clamped, LATENCY_SAMPLES_PER_ROW)
        return cls(
            sorted_samples=samples.ravel()[order],
            block_of_sample=flat_blocks[order],
        )

    def p95(self, counts: np.ndarray) -> np.ndarray:
        """P95 for each row of `counts`, which gives a block's multiplicity in a resample.

        `np.percentile`'s linear rule: take the order statistics at `floor(v)` and `floor(v) + 1`
        and interpolate. Duplicating a draw weights it, so the weighted array is a multiset whose
        order statistics are read off these sorted draws.
        """
        out = np.empty(len(counts))
        for i, draw in enumerate(counts):
            cumulative = np.cumsum(draw[self.block_of_sample])
            total = int(cumulative[-1])
            if total == 0:
                out[i] = np.nan
                continue
            v = 0.95 * (total - 1)
            lo = int(np.floor(v))
            fraction = v - lo
            out[i] = self._at(cumulative, lo) + fraction * (
                self._at(cumulative, min(lo + 1, total - 1)) - self._at(cumulative, lo)
            )
        return out

    def _at(self, cumulative: np.ndarray, rank: int) -> float:
        """The value at 0-based position `rank` in the weighted multiset."""
        index = int(np.searchsorted(cumulative, rank + 1, side="left"))
        return float(self.sorted_samples[min(index, len(self.sorted_samples) - 1)])


def _block_rows(timestamps: pd.Series, start: pd.Timestamp) -> tuple[list[np.ndarray], np.ndarray]:
    """Row indices grouped by time block, in block order, plus each row's block number.

    Blocks with no rows in this frame are kept as empty groups, so a block means the same thing in
    the baseline and the counterfactual.
    """
    minutes = ((timestamps - start).dt.total_seconds() // 60).to_numpy()
    # A counterfactual can carry rows a few minutes before the baseline's first, which floors to
    # block -1; shift so the earliest block is 0 and every row indexes a real block.
    blocks = (minutes // BOOTSTRAP_BLOCK_MINUTES).astype(np.int64)
    blocks -= blocks.min()
    order = np.argsort(blocks, kind="stable")
    sorted_blocks = blocks[order]
    edges = np.searchsorted(sorted_blocks, np.arange(int(sorted_blocks.max()) + 2))
    return [order[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)], blocks


def _block_sums(frame: pd.DataFrame, preds: Predictions, rows: list[np.ndarray]) -> np.ndarray:
    """Per-block sums of the quantities every additive metric is built from, as [block, 7].

    Working in sums rather than rows is what makes the bootstrap a single matrix product: a
    resample's totals are just its block counts weighted by these sums.
    """
    amount = frame["amount"].to_numpy()
    p = preds.p_success
    per_row = np.column_stack(
        [
            np.ones(len(p)),
            p,
            (1.0 - p) * preds.reason_probs[:, _TIMEOUT_COLUMNS].sum(axis=1),
            amount,
            amount * p,
            amount * (1.0 - p),
            preds.latency_median_ms * np.exp(preds.latency_log_sigma**2 / 2),
        ]
    )
    sums = np.zeros((len(rows), per_row.shape[1]))
    for block, idx in enumerate(rows):
        if len(idx):
            sums[block] = per_row[idx].sum(axis=0)
    return sums


def _impact_from_sums(
    base: np.ndarray, cf: np.ndarray, base_p95: np.ndarray, cf_p95: np.ndarray
) -> np.ndarray:
    """Impact fields from per-block totals, vectorized over `[n_bootstrap, 7]` inputs.

    Mirrors `compute_impact` exactly, so the bootstrap distribution is centred on the same
    quantity the point estimate reports.
    """
    n_b, n_c = base[:, 0], cf[:, 0]
    return np.column_stack(
        [
            100.0 * (cf[:, 1] / n_c - base[:, 1] / n_b),
            100.0 * (cf[:, 2] / n_c - base[:, 2] / n_b),
            (n_c - cf[:, 1]) - (n_b - base[:, 1]),
            cf[:, 5] - base[:, 5] * (cf[:, 3] / base[:, 3]),
            cf[:, 6] / n_c - base[:, 6] / n_b,
            cf_p95 - base_p95,
        ]
    )


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
