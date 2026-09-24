"""Observed ecosystem-state features attached to every attempt.

These are what a real payments team could see at the moment of a transaction: recent success
rates and current load. They are computed from outcomes, never from the latent parameters, and
the health windows end before the attempt's own minute so an outcome never leaks into its own
features.
"""

from datetime import date

import numpy as np
import pandas as pd

from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.data.dgp import utilization
from backend.data.latent import LatentParams, minute_index
from backend.data.traffic import PEAK_HOURS

HEALTH_WINDOW_MINUTES = 15
# Pseudo-attempts at the group's overall success rate, so quiet windows don't swing to 0 or 1.
HEALTH_PRIOR_WEIGHT = 5.0


def _trailing_success_rate(
    group: np.ndarray, n_groups: int, minute: np.ndarray, success: np.ndarray, n_minutes: int
) -> np.ndarray:
    flat = group * n_minutes + minute
    size = n_groups * n_minutes
    attempts = np.bincount(flat, minlength=size).reshape(n_groups, n_minutes)
    successes = np.bincount(flat, weights=success, minlength=size).reshape(n_groups, n_minutes)

    def window(counts: np.ndarray) -> np.ndarray:
        # Sum over minutes [t - W, t - 1] for every t, via a zero-padded cumulative sum.
        cum = np.zeros((n_groups, n_minutes + 1))
        np.cumsum(counts, axis=1, out=cum[:, 1:])
        end = np.arange(n_minutes)
        start = np.maximum(end - HEALTH_WINDOW_MINUTES, 0)
        return cum[:, end] - cum[:, start]

    prior = successes.sum(axis=1) / np.maximum(attempts.sum(axis=1), 1)
    rate = (window(successes) + HEALTH_PRIOR_WEIGHT * prior[:, None]) / (
        window(attempts) + HEALTH_PRIOR_WEIGHT
    )
    return rate[group, minute]


def add_state_features(
    attempts: pd.DataFrame, latent: LatentParams, festival_dates: list[date]
) -> pd.DataFrame:
    minute = minute_index(attempts["timestamp"], latent.start)
    method_index = {m: i for i, m in enumerate(METHODS)}
    method = attempts["payment_method"].astype(str).map(method_index).to_numpy()
    success = (attempts["transaction_status"] == "SUCCESS").to_numpy().astype(float)
    issuer = attempts["issuer_id"].to_numpy()
    gateway = attempts["gateway_id"].to_numpy()
    hour = attempts["timestamp"].dt.hour

    return attempts.assign(
        issuer_health=_trailing_success_rate(
            issuer * len(METHODS) + method,
            len(ISSUERS) * len(METHODS),
            minute,
            success,
            latent.n_minutes,
        ),
        gateway_health=_trailing_success_rate(
            gateway, len(GATEWAYS), minute, success, latent.n_minutes
        ),
        gateway_utilization=utilization(gateway, minute, latent),
        hour=hour,
        is_peak=hour.isin(PEAK_HOURS),
        is_festival=attempts["timestamp"].dt.date.isin(festival_dates),
    )
