import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import GATEWAYS, ISSUERS
from backend.data.keyed_random import keyed_uniform
from backend.data.state import _trailing_success_rate


def test_keyed_uniform_depends_only_on_key():
    keys = np.arange(1000, dtype=np.uint64)
    full = keyed_uniform(7, keys, stream=1)
    assert np.array_equal(full[500:], keyed_uniform(7, keys[500:], stream=1))
    assert not np.array_equal(full, keyed_uniform(7, keys, stream=2))
    assert not np.array_equal(full, keyed_uniform(8, keys, stream=1))
    assert full.min() >= 0.0 and full.max() < 1.0
    assert full.mean() == pytest.approx(0.5, abs=0.03)


def test_retries_follow_failed_parents(dataset):
    txns = dataset.transactions.set_index("transaction_id")
    retries = txns[txns["attempt"] == 2]
    assert len(retries) > 0
    parents = txns.loc[retries["parent_transaction_id"]]
    assert (parents["transaction_status"] == "FAILED").all()
    assert (retries["timestamp"].to_numpy() > parents["timestamp"].to_numpy()).all()
    assert (retries["customer_id"].to_numpy() == parents["customer_id"].to_numpy()).all()
    same_or_upi = (retries["payment_method"].to_numpy() == parents["payment_method"].to_numpy()) | (
        retries["payment_method"] == "UPI"
    ).to_numpy()
    assert same_or_upi.all()


def test_failure_rate_rises_with_utilization(dataset):
    txns = dataset.transactions
    failed = txns["transaction_status"] == "FAILED"
    low = failed[txns["gateway_utilization"] < 0.5].mean()
    high = failed[txns["gateway_utilization"] > 0.85].mean()
    assert high > low + 0.01


def test_upi_timeouts_concentrate_at_festival_peak(dataset):
    upi = dataset.transactions[dataset.transactions["payment_method"] == "UPI"]
    timeout = upi["failure_reason"] == "UPI_TIMEOUT"
    stressed = upi["is_peak"] & upi["is_festival"]
    assert timeout[stressed].mean() > 2 * timeout[~upi["is_peak"] & ~upi["is_festival"]].mean()


def test_travel_on_gateway_a_slower_at_festival_peak(dataset):
    txns = dataset.transactions.merge(
        dataset.merchants[["merchant_id", "merchant_category"]], on="merchant_id"
    )
    ok = (
        (txns["transaction_status"] == "SUCCESS")
        & (txns["merchant_category"] == "travel")
        & (txns["gateway_id"] == 0)
    )
    stressed = txns["is_peak"] & txns["is_festival"]
    assert (
        txns.loc[ok & stressed, "latency_ms"].median()
        > txns.loc[ok & ~txns["is_peak"], "latency_ms"].median()
    )


def test_observed_issuer_health_drops_during_degradation(dataset):
    txns = dataset.transactions
    rows = dataset.latent.episodes[dataset.latent.episodes["kind"] == "issuer_degradation"]
    minute = (txns["timestamp"] - pd.Timestamp(dataset.latent.start)) // pd.Timedelta(minutes=1)
    inside = np.zeros(len(txns), dtype=bool)
    for ep in rows.itertuples():
        # Skip the first window so the trailing health has had time to react.
        inside |= (
            (txns["issuer_id"] == ep.target_id)
            & minute.between(ep.start_minute + 15, ep.end_minute - 1)
            & (pd.isna(ep.method) | (txns["payment_method"] == ep.method))
        ).to_numpy()
    assert inside.sum() > 0
    assert txns.loc[inside, "issuer_health"].mean() < txns.loc[~inside, "issuer_health"].mean()


def test_state_features_in_range(dataset):
    txns = dataset.transactions
    for column in ("issuer_health", "gateway_health"):
        assert txns[column].between(0.0, 1.0).all()
    assert (txns["gateway_utilization"] > 0).all()
    assert set(txns["issuer_id"]) <= set(range(len(ISSUERS)))
    assert set(txns["gateway_id"]) <= set(range(len(GATEWAYS)))


def test_trailing_health_excludes_own_minute():
    group = np.zeros(3, dtype=int)
    minute = np.array([0, 1, 2])
    success = np.array([0.0, 1.0, 1.0])
    health = _trailing_success_rate(group, 1, minute, success, n_minutes=3)[0]
    prior = 2 / 3
    # Minute 0 has no history, so it sits at the prior regardless of its own failure.
    assert health[0] == pytest.approx(prior)
    assert health[1] == pytest.approx((0 + 5 * prior) / (1 + 5))
    assert health[2] == pytest.approx((1 + 5 * prior) / (2 + 5))
