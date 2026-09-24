"""The true data-generating process for payment outcomes.

An attempt passes through ordered stages (risk, customer, gateway, issuer, residual). Each stage
has its own failure probability and its own keyed uniform draw; the first stage that fails sets
the failure reason. Because every stage compares a fixed draw against a probability, raising a
probability can only turn successes into failures, never the reverse, which keeps paired
counterfactuals low-noise.

Failed first attempts may be retried once. Retries add load, load raises failure rates, and those
failures create more retries, so outcomes and retries are solved together by fixed-point
iteration (see `solve`).
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from backend.data.catalog import GATEWAYS, METHODS
from backend.data.keyed_random import keyed_normal, keyed_uniform
from backend.data.latent import UTILIZATION_BUCKET_MINUTES, LatentParams, minute_index
from backend.data.traffic import PEAK_HOURS

FAILURE_REASONS = (
    "INSUFFICIENT_FUNDS",
    "ISSUER_DECLINED",
    "UPI_TIMEOUT",
    "GATEWAY_TIMEOUT",
    "NETWORK_ERROR",
    "AUTHENTICATION_FAILED",
    "FRAUD_REJECTION",
    "LIMIT_EXCEEDED",
    "UNKNOWN",
)
_R = {name: code for code, name in enumerate(FAILURE_REASONS)}
_UPI = METHODS.index("UPI")

# Stage order; also the column order of Attempts.stage_draws.
STAGES = ("risk", "auth", "funds", "limit", "gateway", "issuer", "unknown")
_U = {name: i for i, name in enumerate(STAGES)}
# Keyed-random stream ids. Stage draws use streams 1..len(STAGES); keyed_normal uses two streams.
S_LATENCY = 20
S_RETRY, S_RETRY_DELAY, S_RETRY_SWITCH, S_RETRY_SECOND = range(30, 34)

RETRY_ID_OFFSET = 10**9
MAX_FIXED_POINT_ITERATIONS = 100

HIGH_RISK_CATEGORIES = ("electronics", "travel", "gaming")

AUTH_FAILURE_BY_METHOD = (0.006, 0.012, 0.015)
NEW_DEVICE_AUTH_MULTIPLIER = 2.0
FUNDS_FAILURE_BASE = 0.004
UPI_LIMIT_AMOUNT = 50_000.0
UPI_LIMIT_FAILURE = 0.35

GATEWAY_SATURATION_START = 0.8
GATEWAY_BASE_TIMEOUT = 0.002
GATEWAY_NETWORK_ERROR = 0.0015
GATEWAY_SATURATION_COEF = 1.5
GATEWAY_MAX_TIMEOUT = 0.85

ISSUER_LOAD_START = 0.7
ISSUER_PEAK_TIMEOUT = 0.003
ISSUER_LOAD_COEF = 0.6
UPI_TIMEOUT_SHARE = 0.6

RESIDUAL_FAILURE = 0.0008

LATENCY_MEDIAN_MS = (1200.0, 1800.0, 3000.0)
LATENCY_SIGMA = 0.35
LATENCY_LOAD_START = 0.7
LATENCY_LOAD_COEF = 3.0
TIMEOUT_LATENCY_MS = {"GATEWAY_TIMEOUT": 20_000, "UPI_TIMEOUT": 30_000}

# Probability a customer retries, by why the first attempt failed.
RETRY_PROBABILITY = {
    "INSUFFICIENT_FUNDS": 0.10,
    "ISSUER_DECLINED": 0.35,
    "UPI_TIMEOUT": 0.50,
    "GATEWAY_TIMEOUT": 0.50,
    "NETWORK_ERROR": 0.50,
    "AUTHENTICATION_FAILED": 0.35,
    "FRAUD_REJECTION": 0.10,
    "LIMIT_EXCEEDED": 0.10,
    "UNKNOWN": 0.50,
}
RETRY_SWITCH_TO_UPI = 0.35

# The first-attempt fields the outcome process reads; everything else is derived from them.
CONTEXT_COLUMNS = (
    "transaction_id",
    "timestamp",
    "customer_id",
    "merchant_id",
    "issuer_id",
    "gateway_id",
    "amount",
    "payment_method",
    "device_type",
    "is_new_device",
    "city",
)


@dataclass
class Attempts:
    """Column arrays for a set of payment attempts, as the outcome process needs them.

    The keyed draws are carried along with each attempt so the fixed-point loop never recomputes
    them; they depend only on the attempt's key.
    """

    key: np.ndarray  # uint64 transaction id
    minute: np.ndarray
    issuer: np.ndarray
    gateway: np.ndarray
    method: np.ndarray
    amount: np.ndarray
    is_new_device: np.ndarray
    is_new_customer: np.ndarray
    is_high_risk_category: np.ndarray
    stage_draws: np.ndarray  # [attempt, stage] uniforms
    latency_noise: np.ndarray

    def concat(self, other: "Attempts") -> "Attempts":
        return Attempts(
            **{
                f.name: np.concatenate([getattr(self, f.name), getattr(other, f.name)])
                for f in fields(self)
            }
        )

    def take(self, index: np.ndarray) -> "Attempts":
        return Attempts(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


@dataclass
class Outcomes:
    reason: np.ndarray  # failure reason code, -1 on success
    latency_ms: np.ndarray


def _stage_draws(seed: int, keys: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [keyed_uniform(seed, keys, stream) for stream in range(1, len(STAGES) + 1)]
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def utilization(gateway: np.ndarray, minute: np.ndarray, latent: LatentParams) -> np.ndarray:
    """Load over capacity of each attempt's gateway in its utilization bucket."""
    flat = gateway * latent.n_buckets + minute // UTILIZATION_BUCKET_MINUTES
    load = np.bincount(flat, minlength=len(GATEWAYS) * latent.n_buckets)
    return load[flat] / latent.gateway_capacity[gateway]


def attempt_outcomes(
    att: Attempts,
    util: np.ndarray,
    latent: LatentParams,
    degradation: np.ndarray,
    health: np.ndarray,
) -> Outcomes:
    n = len(att.key)
    is_upi = att.method == _UPI
    is_peak = np.isin((att.minute // 60) % 24, PEAK_HOURS)
    reason = np.full(n, -1)

    def stage(name: str, p: np.ndarray, code) -> None:
        u = att.stage_draws[:, _U[name]]
        hit = (reason == -1) & (u < p)
        reason[hit] = code(u, p)[hit] if callable(code) else code

    risk_logit = (
        -6.2
        + 1.6 * att.is_new_device
        + 1.0 * att.is_new_customer
        + 0.5 * att.is_high_risk_category
        + 0.6 * np.log(att.amount / 1000.0)
    )
    stage("risk", _sigmoid(risk_logit), _R["FRAUD_REJECTION"])

    auth = np.asarray(AUTH_FAILURE_BY_METHOD)[att.method]
    auth = auth * np.where(att.is_new_device, NEW_DEVICE_AUTH_MULTIPLIER, 1.0)
    stage("auth", auth, _R["AUTHENTICATION_FAILED"])

    funds = np.minimum(FUNDS_FAILURE_BASE * (att.amount / 1000.0) ** 0.35, 0.1)
    stage("funds", funds, _R["INSUFFICIENT_FUNDS"])

    limit = np.where(is_upi & (att.amount > UPI_LIMIT_AMOUNT), UPI_LIMIT_FAILURE, 0.0)
    stage("limit", limit, _R["LIMIT_EXCEEDED"])

    saturation = np.maximum(util - GATEWAY_SATURATION_START, 0.0)
    timeout = GATEWAY_BASE_TIMEOUT + np.minimum(
        GATEWAY_SATURATION_COEF * saturation**2, GATEWAY_MAX_TIMEOUT
    )
    gw_health = health[att.gateway, att.minute]
    gw_fail = 1.0 - gw_health * (1.0 - timeout) * (1.0 - GATEWAY_NETWORK_ERROR)
    stage(
        "gateway",
        gw_fail,
        lambda u, p: np.where(
            u < GATEWAY_NETWORK_ERROR, _R["NETWORK_ERROR"], _R["GATEWAY_TIMEOUT"]
        ),
    )

    reliability = np.clip(
        latent.issuer_reliability[att.issuer, att.method]
        - degradation[att.issuer, att.method, att.minute],
        0.0,
        1.0,
    )
    load = np.maximum(util - ISSUER_LOAD_START, 0.0)
    upi_load = np.where(
        is_upi,
        latent.issuer_load_sensitivity[att.issuer]
        * (ISSUER_PEAK_TIMEOUT * is_peak + ISSUER_LOAD_COEF * load**2),
        0.0,
    )
    issuer_fail = 1.0 - reliability * (1.0 - np.minimum(upi_load, 1.0))
    stage(
        "issuer",
        issuer_fail,
        lambda u, p: np.where(
            is_upi & (u < UPI_TIMEOUT_SHARE * p), _R["UPI_TIMEOUT"], _R["ISSUER_DECLINED"]
        ),
    )

    stage("unknown", np.full(n, RESIDUAL_FAILURE), _R["UNKNOWN"])

    latency = (
        np.asarray(LATENCY_MEDIAN_MS)[att.method]
        * latent.issuer_latency_factor[att.issuer]
        * latent.gateway_latency_factor[att.gateway]
        * (1.0 + LATENCY_LOAD_COEF * np.maximum(util - LATENCY_LOAD_START, 0.0))
        * (1.0 + 0.08 * np.log10(np.maximum(att.amount, 100.0) / 1000.0))
        * np.exp(LATENCY_SIGMA * att.latency_noise)
    )
    for name, ms in TIMEOUT_LATENCY_MS.items():
        latency = np.where(reason == _R[name], ms, latency)

    return Outcomes(reason=reason, latency_ms=np.round(latency).astype(np.int64))


def _retry_candidates(first: Attempts, seed: int) -> Attempts:
    """The retry each first attempt would produce if it failed and the customer retried."""
    retry_key = first.key + np.uint64(RETRY_ID_OFFSET)
    delay = 1 + (keyed_uniform(seed, retry_key, S_RETRY_DELAY) < 0.5)
    switch = (first.method != _UPI) & (
        keyed_uniform(seed, retry_key, S_RETRY_SWITCH) < RETRY_SWITCH_TO_UPI
    )
    retries = first.take(np.arange(len(first.key)))
    retries.key = retry_key
    retries.minute = first.minute + delay
    # Every gateway supports UPI, so switching method never needs a new route.
    retries.method = np.where(switch, _UPI, first.method)
    retries.stage_draws = _stage_draws(seed, retry_key)
    retries.latency_noise = keyed_normal(seed, retry_key, S_LATENCY)
    return retries


def solve(
    first: Attempts, latent: LatentParams, seed: int
) -> tuple[Attempts, np.ndarray, Outcomes]:
    """Outcomes for first attempts plus the retries they trigger.

    Returns all attempts (first attempts, then retries), each retry's parent index, and outcomes.
    Retries only land after their parent, and the retry set starts empty and can only grow as load
    rises, so iterating until the retry set stops changing reaches a well-defined fixed point.
    """
    degradation = latent.issuer_degradation()
    health = latent.gateway_health()
    candidates = _retry_candidates(first, seed)
    in_period = candidates.minute < latent.n_minutes
    retry_p = np.array([RETRY_PROBABILITY[r] for r in FAILURE_REASONS])
    retry_draw = keyed_uniform(seed, first.key, S_RETRY)

    retry_mask = np.zeros(len(first.key), dtype=bool)
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        parent = np.flatnonzero(retry_mask)
        attempts = first.concat(candidates.take(parent))
        outcomes = attempt_outcomes(
            attempts,
            utilization(attempts.gateway, attempts.minute, latent),
            latent,
            degradation,
            health,
        )
        first_reason = outcomes.reason[: len(first.key)]
        failed = first_reason >= 0
        next_mask = failed & in_period & (retry_draw < retry_p[np.maximum(first_reason, 0)])
        if np.array_equal(next_mask, retry_mask):
            return attempts, parent, outcomes
        retry_mask = next_mask
    raise RuntimeError(
        f"Retry fixed point did not converge in {MAX_FIXED_POINT_ITERATIONS} iterations"
    )


def attempts_from_transactions(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    latent: LatentParams,
    seed: int,
) -> Attempts:
    customer = transactions["customer_id"].to_numpy()
    merchant = transactions["merchant_id"].to_numpy()
    key = transactions["transaction_id"].to_numpy().astype(np.uint64)
    method_index = {m: i for i, m in enumerate(METHODS)}
    return Attempts(
        key=key,
        minute=minute_index(transactions["timestamp"], latent.start),
        issuer=transactions["issuer_id"].to_numpy(),
        gateway=transactions["gateway_id"].to_numpy(),
        method=transactions["payment_method"].astype(str).map(method_index).to_numpy(),
        amount=transactions["amount"].to_numpy(),
        is_new_device=transactions["is_new_device"].to_numpy(),
        is_new_customer=(customers["customer_segment"].to_numpy() == "new")[customer],
        is_high_risk_category=merchants["merchant_category"]
        .isin(HIGH_RISK_CATEGORIES)
        .to_numpy()[merchant],
        stage_draws=_stage_draws(seed, key),
        latency_noise=keyed_normal(seed, key, S_LATENCY),
    )


def first_attempt_context(attempts: pd.DataFrame) -> pd.DataFrame:
    """Strip outcomes, retries and state features, leaving the input to `simulate_outcomes`."""
    first = attempts[attempts["attempt"] == 1]
    return first[list(CONTEXT_COLUMNS)].sort_values("transaction_id", ignore_index=True)


def simulate_outcomes(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    latent: LatentParams,
    seed: int,
) -> pd.DataFrame:
    """First-attempt transactions in, every attempt (with retries) and its outcome out."""
    first = attempts_from_transactions(transactions, customers, merchants, latent, seed)
    attempts, parent, outcomes = solve(first, latent, seed)

    retry = attempts.take(np.arange(len(first.key), len(attempts.key)))
    second_ms = np.floor(keyed_uniform(seed, retry.key, S_RETRY_SECOND) * 60_000)
    retry_rows = transactions.iloc[parent].assign(
        transaction_id=retry.key.astype(np.int64),
        timestamp=latent.start + (retry.minute * 60_000 + second_ms).astype("timedelta64[ms]"),
        payment_method=pd.Categorical.from_codes(retry.method, METHODS),
        attempt=2,
        parent_transaction_id=transactions["transaction_id"].to_numpy()[parent],
    )

    out = pd.concat(
        [transactions.assign(attempt=1, parent_transaction_id=-1), retry_rows],
        ignore_index=True,
    )
    out["transaction_status"] = pd.Categorical(
        np.where(outcomes.reason >= 0, "FAILED", "SUCCESS"), categories=["SUCCESS", "FAILED"]
    )
    out["failure_reason"] = pd.Categorical.from_codes(outcomes.reason, FAILURE_REASONS)
    out["latency_ms"] = outcomes.latency_ms
    return out.sort_values(["timestamp", "transaction_id"], ignore_index=True)
