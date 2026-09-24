import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import GATEWAYS
from backend.data.generator import generate_dataset
from backend.data.io import load_dataset, save_dataset


def _first_attempts(dataset) -> pd.DataFrame:
    return dataset.transactions[dataset.transactions["attempt"] == 1]


def test_same_seed_is_identical(small_config, dataset):
    again = generate_dataset(small_config)
    pd.testing.assert_frame_equal(dataset.transactions, again.transactions)
    pd.testing.assert_frame_equal(dataset.customers, again.customers)
    pd.testing.assert_frame_equal(dataset.latent.episodes, again.latent.episodes)
    assert dataset.festival_dates == again.festival_dates


def test_different_seed_differs(small_config, dataset):
    other = generate_dataset(small_config.model_copy(update={"seed": small_config.seed + 1}))
    assert not dataset.transactions["amount"].equals(other.transactions["amount"])


def test_volume_close_to_target(small_config, dataset):
    assert len(_first_attempts(dataset)) == pytest.approx(
        small_config.target_transactions, rel=0.02
    )


def test_gateway_supports_method(dataset):
    txns = dataset.transactions
    names = np.array([g.name for g in GATEWAYS])[txns["gateway_id"]]
    supported = {g.name: set(g.supported_methods) for g in GATEWAYS}
    assert all(m in supported[g] for g, m in zip(names, txns["payment_method"], strict=True))


def test_evening_peak_busier_than_night(dataset):
    hour = _first_attempts(dataset)["timestamp"].dt.hour
    assert (hour == 20).sum() > 5 * (hour == 3).sum()


def test_festival_days_busier(dataset):
    first = _first_attempts(dataset)
    per_day = first.groupby(first["timestamp"].dt.date).size()
    festival = per_day[per_day.index.isin(dataset.festival_dates)]
    normal = per_day[~per_day.index.isin(dataset.festival_dates)]
    assert festival.mean() > 1.3 * normal.mean()


def test_upi_tickets_smaller_than_netbanking(dataset):
    median = _first_attempts(dataset).groupby("payment_method", observed=True)["amount"].median()
    assert median["UPI"] < median["NETBANKING"]


def test_offline_merchants_mostly_local(small_config, dataset):
    txns = _first_attempts(dataset).merge(
        dataset.merchants[["merchant_id", "online", "city"]].rename(
            columns={"city": "merchant_city"}
        ),
        on="merchant_id",
    )
    offline = txns[~txns["online"]]
    local_share = (offline["city"] == offline["merchant_city"]).mean()
    assert local_share > small_config.offline_local_share - 0.02


def test_new_device_differs_from_usual(dataset):
    txns = dataset.transactions.merge(
        dataset.customers[["customer_id", "device_type"]].rename(
            columns={"device_type": "usual_device"}
        ),
        on="customer_id",
    )
    new = txns["is_new_device"]
    assert (txns.loc[new, "device_type"].astype(str) != txns.loc[new, "usual_device"]).all()
    assert (txns.loc[~new, "device_type"].astype(str) == txns.loc[~new, "usual_device"]).all()


def test_roundtrip(tmp_path, small_config, dataset):
    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path)
    pd.testing.assert_frame_equal(dataset.transactions, loaded.transactions)
    pd.testing.assert_frame_equal(dataset.latent.episodes, loaded.latent.episodes)
    np.testing.assert_array_equal(
        dataset.latent.issuer_reliability, loaded.latent.issuer_reliability
    )
    assert loaded.latent.start == dataset.latent.start
    assert loaded.festival_dates == dataset.festival_dates
    assert loaded.config == dataset.config
