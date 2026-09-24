"""Scheduled disruptions that give the training data natural variation in ecosystem state.

The magnitude ranges here define what the models see during training. Counterfactuals outside
these ranges are extrapolation, and the experiments test that on purpose.
"""

import numpy as np
import pandas as pd

from backend.config import EcosystemConfig
from backend.data.catalog import GATEWAYS, ISSUERS
from backend.data.traffic import MINUTES_PER_DAY

ISSUER_DEGRADATION_PER_DAY = 0.08
ISSUER_DEGRADATION_MINUTES = (10, 120)
ISSUER_DEGRADATION_MAGNITUDE = (0.02, 0.10)
ISSUER_DEGRADATION_UPI_ONLY_SHARE = 0.6

GATEWAY_BROWNOUT_PER_DAY = 0.05
GATEWAY_BROWNOUT_MINUTES = (15, 90)
GATEWAY_BROWNOUT_MAGNITUDE = (0.05, 0.30)

GATEWAY_OUTAGE_PER_DAY = 0.02
GATEWAY_OUTAGE_MINUTES = (5, 30)

EPISODE_COLUMNS = ("kind", "target_id", "method", "start_minute", "end_minute", "magnitude")


def _draw(
    rng: np.random.Generator,
    kind: str,
    target_id: int,
    per_day: float,
    minutes: tuple[int, int],
    magnitude: tuple[float, float],
    n_minutes: int,
) -> list[dict]:
    rows = []
    for _ in range(rng.poisson(per_day * n_minutes / MINUTES_PER_DAY)):
        start = int(rng.integers(0, n_minutes))
        duration = int(rng.integers(minutes[0], minutes[1] + 1))
        rows.append(
            {
                "kind": kind,
                "target_id": target_id,
                "method": None,
                "start_minute": start,
                "end_minute": min(start + duration, n_minutes),
                "magnitude": float(rng.uniform(*magnitude)),
            }
        )
    return rows


def generate_episodes(config: EcosystemConfig, rng: np.random.Generator) -> pd.DataFrame:
    """One row per episode. `method` is None when an issuer degradation hits every method."""
    n_minutes = config.n_days * MINUTES_PER_DAY
    rows = []
    for issuer_id in range(len(ISSUERS)):
        degradations = _draw(
            rng,
            "issuer_degradation",
            issuer_id,
            ISSUER_DEGRADATION_PER_DAY,
            ISSUER_DEGRADATION_MINUTES,
            ISSUER_DEGRADATION_MAGNITUDE,
            n_minutes,
        )
        for row in degradations:
            if rng.random() < ISSUER_DEGRADATION_UPI_ONLY_SHARE:
                row["method"] = "UPI"
        rows += degradations
    for gateway_id in range(len(GATEWAYS)):
        rows += _draw(
            rng,
            "gateway_brownout",
            gateway_id,
            GATEWAY_BROWNOUT_PER_DAY,
            GATEWAY_BROWNOUT_MINUTES,
            GATEWAY_BROWNOUT_MAGNITUDE,
            n_minutes,
        )
        rows += _draw(
            rng,
            "gateway_outage",
            gateway_id,
            GATEWAY_OUTAGE_PER_DAY,
            GATEWAY_OUTAGE_MINUTES,
            (1.0, 1.0),
            n_minutes,
        )
    episodes = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    return episodes.sort_values(["start_minute", "kind", "target_id"], ignore_index=True)
