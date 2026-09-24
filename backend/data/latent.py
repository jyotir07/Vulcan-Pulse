"""Ground-truth parameters of the outcome process.

Nothing in here is observable to the models. It is stored apart from the observed tables so it
cannot leak into features by accident.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backend.config import EcosystemConfig
from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.data.traffic import MINUTES_PER_DAY

UTILIZATION_BUCKET_MINUTES = 5

BANK_TYPE_RELIABILITY = {"private": 0.988, "public": 0.980, "small_finance": 0.975}
METHOD_RELIABILITY_OFFSET = (0.0, 0.004, -0.006)
RELIABILITY_JITTER_SIGMA = 0.004
BANK_TYPE_LATENCY = {"private": 1.0, "public": 1.15, "small_finance": 1.1}
# How strongly an issuer's UPI rails suffer under peak load.
BANK_TYPE_LOAD_SENSITIVITY = {"private": 1.0, "public": 1.5, "small_finance": 1.2}

GATEWAY_LATENCY_FACTOR = (1.0, 1.1, 0.95, 1.2)
# Capacity as a multiple of the gateway's mean load. gateway_a has the least headroom, so it
# saturates first on festival evenings.
GATEWAY_CAPACITY_HEADROOM = (3.4, 3.6, 3.9, 4.4)


@dataclass(frozen=True)
class LatentParams:
    start: np.datetime64
    n_minutes: int
    issuer_reliability: np.ndarray  # [issuer, method]
    issuer_latency_factor: np.ndarray  # [issuer]
    issuer_load_sensitivity: np.ndarray  # [issuer]
    gateway_latency_factor: np.ndarray  # [gateway]
    gateway_capacity: np.ndarray  # [gateway], attempts per utilization bucket
    episodes: pd.DataFrame

    @property
    def n_buckets(self) -> int:
        return -(-self.n_minutes // UTILIZATION_BUCKET_MINUTES)

    def with_episodes(self, episodes: pd.DataFrame) -> "LatentParams":
        return replace(self, episodes=episodes)

    def issuer_degradation(self) -> np.ndarray:
        """[issuer, method, minute] reliability reduction; overlapping episodes add up."""
        out = np.zeros((len(ISSUERS), len(METHODS), self.n_minutes))
        rows = self.episodes[self.episodes["kind"] == "issuer_degradation"]
        for row in rows.itertuples():
            if pd.isna(row.method):
                methods = list(range(len(METHODS)))
            else:
                methods = [METHODS.index(row.method)]
            out[row.target_id, methods, row.start_minute : row.end_minute] += row.magnitude
        return out

    def gateway_health(self) -> np.ndarray:
        """[gateway, minute] multiplier on gateway pass-through; 0 during an outage."""
        out = np.ones((len(GATEWAYS), self.n_minutes))
        rows = self.episodes[self.episodes["kind"].isin(["gateway_brownout", "gateway_outage"])]
        for row in rows.itertuples():
            out[row.target_id, row.start_minute : row.end_minute] *= 1.0 - row.magnitude
        return out


def minute_index(timestamps: pd.Series | np.ndarray, start: np.datetime64) -> np.ndarray:
    delta = np.asarray(timestamps, dtype="datetime64[ms]") - start
    return (delta // np.timedelta64(1, "m")).astype(np.int64)


def build_latent(
    config: EcosystemConfig,
    transactions: pd.DataFrame,
    episodes: pd.DataFrame,
    rng: np.random.Generator,
) -> LatentParams:
    bank_types = [i.bank_type for i in ISSUERS]
    base = np.array([BANK_TYPE_RELIABILITY[b] for b in bank_types])
    reliability = (
        base[:, None]
        + np.asarray(METHOD_RELIABILITY_OFFSET)[None, :]
        + rng.normal(0.0, RELIABILITY_JITTER_SIGMA, (len(ISSUERS), len(METHODS)))
    )
    latency = np.array([BANK_TYPE_LATENCY[b] for b in bank_types]) * rng.lognormal(
        0.0, 0.15, len(ISSUERS)
    )

    n_minutes = config.n_days * MINUTES_PER_DAY
    n_buckets = -(-n_minutes // UTILIZATION_BUCKET_MINUTES)
    mean_load = np.bincount(transactions["gateway_id"], minlength=len(GATEWAYS)) / n_buckets

    return LatentParams(
        start=np.datetime64(config.start_date, "ms"),
        n_minutes=n_minutes,
        issuer_reliability=np.clip(reliability, 0.9, 0.998),
        issuer_latency_factor=latency,
        issuer_load_sensitivity=np.array([BANK_TYPE_LOAD_SENSITIVITY[b] for b in bank_types]),
        gateway_latency_factor=np.asarray(GATEWAY_LATENCY_FACTOR),
        gateway_capacity=np.asarray(GATEWAY_CAPACITY_HEADROOM) * mean_load,
        episodes=episodes,
    )
