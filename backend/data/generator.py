from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.config import EcosystemConfig
from backend.data.catalog import CITIES, DEVICES, GATEWAYS, METHODS
from backend.data.dgp import simulate_outcomes
from backend.data.entities import (
    cities_table,
    gateways_table,
    generate_customers,
    generate_merchants,
    issuers_table,
)
from backend.data.episodes import generate_episodes
from backend.data.latent import LatentParams, build_latent
from backend.data.state import add_state_features
from backend.data.traffic import pick_festival_dates, sample_timestamps

# Share of a customer's method choice that goes to their preferred method.
PREFERRED_METHOD_WEIGHT = 0.70
# UPI skews to small tickets; netbanking to large ones (bills, fees).
METHOD_AMOUNT_FACTOR = (0.8, 1.2, 1.5)
AMOUNT_NOISE_SIGMA = 0.6


@dataclass
class Dataset:
    config: EcosystemConfig
    customers: pd.DataFrame
    merchants: pd.DataFrame
    issuers: pd.DataFrame
    gateways: pd.DataFrame
    cities: pd.DataFrame
    # Every attempt, first attempts and retries, with outcomes and observed state features.
    transactions: pd.DataFrame
    festival_dates: list[date]
    latent: LatentParams


def _sample_rows(cum_weights: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling: row i picks the index where u[i] falls in its cumulative weights."""
    return (u[:, None] * cum_weights[:, -1:] > cum_weights).sum(axis=1)


def _sample_customers(
    config: EcosystemConfig,
    customers: pd.DataFrame,
    merchant_city: np.ndarray,
    merchant_online: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(merchant_city)
    weights = customers["activity_weight"].to_numpy()
    customer_city = customers["city_id"].to_numpy()

    global_cum = np.cumsum(weights)
    chosen = np.searchsorted(global_cum, rng.random(n) * global_cum[-1], side="right")

    local = ~merchant_online & (rng.random(n) < config.offline_local_share)
    u = rng.random(n)
    for city_id in range(len(CITIES)):
        rows = np.flatnonzero(local & (merchant_city == city_id))
        members = np.flatnonzero(customer_city == city_id)
        if len(rows) == 0 or len(members) == 0:
            continue
        city_cum = np.cumsum(weights[members])
        picks = np.searchsorted(city_cum, u[rows] * city_cum[-1], side="right")
        chosen[rows] = members[picks]
    return chosen


def _sample_methods(
    customer_pref: np.ndarray, merchant_weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = len(customer_pref)
    other = (1.0 - PREFERRED_METHOD_WEIGHT) / (len(METHODS) - 1)
    pref = np.full((n, len(METHODS)), other)
    pref[np.arange(n), customer_pref] = PREFERRED_METHOD_WEIGHT
    return _sample_rows(np.cumsum(pref * merchant_weights, axis=1), rng.random(n))


def _sample_gateways(
    config: EcosystemConfig,
    method: np.ndarray,
    primary: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(method)
    supports = np.array([[m in g.supported_methods for m in METHODS] for g in GATEWAYS])
    shares = np.array([g.routing_share for g in GATEWAYS])

    fallback_weights = shares[None, :] * supports[:, method].T
    fallback = _sample_rows(np.cumsum(fallback_weights, axis=1), rng.random(n))

    use_primary = supports[primary, method] & (rng.random(n) >= config.gateway_spillover_rate)
    return np.where(use_primary, primary, fallback)


def generate_transactions(
    config: EcosystemConfig,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    festival_dates: list[date],
    rng: np.random.Generator,
) -> pd.DataFrame:
    timestamps = sample_timestamps(config, festival_dates, rng)
    n = len(timestamps)

    volume = merchants["volume_weight"].to_numpy()
    merchant = rng.choice(len(merchants), n, p=volume / volume.sum())

    customer = _sample_customers(
        config,
        customers,
        merchants["city_id"].to_numpy()[merchant],
        merchants["online"].to_numpy()[merchant],
        rng,
    )

    method_index = {m: i for i, m in enumerate(METHODS)}
    customer_pref = customers["preferred_payment_method"].map(method_index).to_numpy()[customer]
    merchant_weights = merchants[["upi_weight", "card_weight", "netbanking_weight"]].to_numpy()
    method = _sample_methods(customer_pref, merchant_weights[merchant], rng)

    amount = (
        merchants["average_order_value"].to_numpy()[merchant]
        * customers["spend_multiplier"].to_numpy()[customer]
        * np.asarray(METHOD_AMOUNT_FACTOR)[method]
        * rng.lognormal(0.0, AMOUNT_NOISE_SIGMA, n)
    )

    device_index = {d: i for i, d in enumerate(DEVICES)}
    usual_device = customers["device_type"].map(device_index).to_numpy()[customer]
    is_new_device = rng.random(n) < config.new_device_rate
    # Shift by 1 or 2 so a "new" device is always different from the usual one.
    other_device = (usual_device + rng.integers(1, len(DEVICES), n)) % len(DEVICES)
    device = np.where(is_new_device, other_device, usual_device)

    gateway = _sample_gateways(
        config, method, merchants["primary_gateway_id"].to_numpy()[merchant], rng
    )

    return pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "timestamp": timestamps,
            "customer_id": customer,
            "merchant_id": merchant,
            "issuer_id": customers["issuer_id"].to_numpy()[customer],
            "gateway_id": gateway,
            "amount": np.maximum(np.round(amount, 2), 1.0),
            "payment_method": pd.Categorical.from_codes(method, METHODS),
            "device_type": pd.Categorical.from_codes(device, DEVICES),
            "is_new_device": is_new_device,
            "city": customers["city"].to_numpy()[customer],
        }
    )


def generate_dataset(config: EcosystemConfig) -> Dataset:
    # Independent child streams per stage, so changing one stage doesn't reshuffle the others.
    customer_rng, merchant_rng, calendar_rng, transaction_rng, episode_rng, latent_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(6)
    )
    customers = generate_customers(config, customer_rng)
    merchants = generate_merchants(config, merchant_rng)
    festival_dates = pick_festival_dates(config, calendar_rng)
    first_attempts = generate_transactions(
        config, customers, merchants, festival_dates, transaction_rng
    )
    episodes = generate_episodes(config, episode_rng)
    latent = build_latent(config, first_attempts, episodes, latent_rng)

    attempts = simulate_outcomes(first_attempts, customers, merchants, latent, config.seed)
    return Dataset(
        config=config,
        customers=customers,
        merchants=merchants,
        issuers=issuers_table(),
        gateways=gateways_table(),
        cities=cities_table(),
        transactions=add_state_features(attempts, latent, festival_dates),
        festival_dates=festival_dates,
        latent=latent,
    )
