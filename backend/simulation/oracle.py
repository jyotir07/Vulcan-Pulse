"""Ground truth: re-run the true outcome process under a changed world.

A counterfactual is expressed as a modified first-attempt context (rerouted, shifted or added
transactions) and/or a modified episode schedule (extra degradations or outages). Randomness is
keyed by transaction id, so every transaction present in both runs sees identical draws and the
difference between runs is the effect of the change, not sampling noise.
"""

import pandas as pd

from backend.data.dgp import RETRY_ID_OFFSET, first_attempt_context, simulate_outcomes
from backend.data.generator import Dataset
from backend.data.state import add_state_features


def true_outcomes(
    dataset: Dataset,
    context: pd.DataFrame | None = None,
    episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every attempt, with outcomes and observed state, in the shape of `dataset.transactions`.

    With no overrides this reproduces `dataset.transactions` exactly.
    """
    if context is None:
        context = first_attempt_context(dataset.transactions)
    ids = context["transaction_id"]
    # Retries take id + RETRY_ID_OFFSET, so a first attempt at or above it would collide.
    if not ids.is_unique or ids.max() >= RETRY_ID_OFFSET:
        raise ValueError(f"Context transaction ids must be unique and below {RETRY_ID_OFFSET}")
    latent = dataset.latent if episodes is None else dataset.latent.with_episodes(episodes)
    attempts = simulate_outcomes(
        context, dataset.customers, dataset.merchants, latent, dataset.config.seed
    )
    return add_state_features(attempts, latent, dataset.festival_dates)
