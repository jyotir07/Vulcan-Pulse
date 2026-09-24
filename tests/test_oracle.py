import pandas as pd
import pytest

from backend.data.dgp import RETRY_ID_OFFSET, first_attempt_context
from backend.simulation.oracle import true_outcomes


def _episode(kind: str, target_id: int, start: int, end: int, magnitude: float, method=None):
    return pd.DataFrame(
        [
            {
                "kind": kind,
                "target_id": target_id,
                "method": method,
                "start_minute": start,
                "end_minute": end,
                "magnitude": magnitude,
            }
        ]
    )


def _with_episode(dataset, episode: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dataset.latent.episodes, episode], ignore_index=True)


def _failed(frame: pd.DataFrame) -> pd.Series:
    return frame["transaction_status"] == "FAILED"


def test_no_change_reproduces_dataset(dataset):
    pd.testing.assert_frame_equal(true_outcomes(dataset), dataset.transactions)


def test_degradation_only_turns_successes_into_failures(dataset):
    """Paired draws: making an issuer worse can never rescue a payment that failed before."""
    extra = _episode("issuer_degradation", 0, 0, dataset.latent.n_minutes, 0.20)
    before = dataset.transactions.set_index("transaction_id")
    after = true_outcomes(dataset, episodes=_with_episode(dataset, extra))
    after = after.set_index("transaction_id")

    shared = before.index.intersection(after.index)
    failed_before = _failed(before.loc[shared])
    failed_after = _failed(after.loc[shared])
    assert (failed_after | ~failed_before).all()

    assert _failed(after[after["issuer_id"] == 0]).mean() > (
        _failed(before[before["issuer_id"] == 0]).mean() + 0.1
    )
    other = shared[before.loc[shared, "issuer_id"] != 0]
    # Other issuers only move through the extra retry load, which is small.
    assert (failed_after[other] != failed_before[other]).mean() < 0.005


def test_gateway_outage_fails_everything_routed_there(dataset):
    outage = _episode("gateway_outage", 1, 600, 660, 1.0)
    out = true_outcomes(dataset, episodes=_with_episode(dataset, outage))
    minute = (out["timestamp"] - pd.Timestamp(dataset.latent.start)) // pd.Timedelta(minutes=1)
    during = out[(out["gateway_id"] == 1) & minute.between(600, 659)]
    assert len(during) > 0
    assert _failed(during).all()
    assert (during["failure_reason"] == "GATEWAY_TIMEOUT").mean() > 0.9


def test_rerouting_moves_load(dataset):
    context = first_attempt_context(dataset.transactions)
    context.loc[context["gateway_id"] == 0, "gateway_id"] = 1
    out = true_outcomes(dataset, context=context)

    assert (out["gateway_id"] != 0).all()
    before = dataset.transactions.loc[dataset.transactions["gateway_id"] == 1]
    assert (
        out.loc[out["gateway_id"] == 1, "gateway_utilization"].mean()
        > before["gateway_utilization"].mean() * 1.5
    )


def test_added_transactions_keep_existing_draws(dataset):
    """Extra traffic changes load, but untouched transactions keep their own draws."""
    context = first_attempt_context(dataset.transactions)
    extra = context.sample(frac=0.3, random_state=0).assign(
        transaction_id=lambda f: range(len(context), len(context) + len(f))
    )
    out = true_outcomes(dataset, context=pd.concat([context, extra], ignore_index=True))

    assert len(out[out["attempt"] == 1]) == len(context) + len(extra)
    before = dataset.transactions.set_index("transaction_id")
    after = out.set_index("transaction_id")
    shared = before.index.intersection(after.index)
    # More load can only add failures to the original transactions.
    assert (_failed(after.loc[shared]) | ~_failed(before.loc[shared])).all()
    assert after.loc[shared, "gateway_utilization"].mean() > (
        before.loc[shared, "gateway_utilization"].mean()
    )


def test_colliding_transaction_ids_rejected(dataset):
    context = first_attempt_context(dataset.transactions)
    with pytest.raises(ValueError):
        true_outcomes(dataset, context=pd.concat([context, context.head(1)], ignore_index=True))
    with pytest.raises(ValueError):
        true_outcomes(
            dataset,
            context=context.assign(transaction_id=context["transaction_id"] + RETRY_ID_OFFSET),
        )
