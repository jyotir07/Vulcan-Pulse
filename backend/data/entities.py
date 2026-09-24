import numpy as np
import pandas as pd

from backend.config import EcosystemConfig
from backend.data.catalog import (
    CITIES,
    DEVICES,
    GATEWAYS,
    ISSUERS,
    MERCHANT_CATEGORIES,
    METHODS,
)

CUSTOMER_SEGMENTS = ("new", "regular", "power")
SEGMENT_SHARES = (0.20, 0.60, 0.20)
SEGMENT_ACTIVITY = (0.5, 1.0, 4.0)
SEGMENT_HISTORY_MEAN = (2, 40, 250)

MERCHANT_SIZES = ("small", "medium", "large")
SIZE_SHARES = (0.70, 0.25, 0.05)
SIZE_VOLUME = (1.0, 8.0, 60.0)

DEVICE_SHARES = (0.72, 0.18, 0.10)
PREFERRED_METHOD_SHARES = (0.65, 0.25, 0.10)


def _normalized(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def cities_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city_id": np.arange(len(CITIES)),
            "city": [c.name for c in CITIES],
            "state": [c.state for c in CITIES],
            "weight": [c.weight for c in CITIES],
        }
    )


def issuers_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_id": np.arange(len(ISSUERS)),
            "issuer_name": [i.name for i in ISSUERS],
            "bank_type": [i.bank_type for i in ISSUERS],
            "market_share": [i.market_share for i in ISSUERS],
        }
    )


def gateways_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gateway_id": np.arange(len(GATEWAYS)),
            "gateway_name": [g.name for g in GATEWAYS],
            "supported_methods": [",".join(g.supported_methods) for g in GATEWAYS],
            "routing_share": [g.routing_share for g in GATEWAYS],
        }
    )


def generate_customers(config: EcosystemConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_customers
    segment = rng.choice(len(CUSTOMER_SEGMENTS), n, p=_normalized(SEGMENT_SHARES))
    city_id = rng.choice(len(CITIES), n, p=_normalized([c.weight for c in CITIES]))
    issuer_id = rng.choice(len(ISSUERS), n, p=_normalized([i.market_share for i in ISSUERS]))

    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "customer_segment": np.array(CUSTOMER_SEGMENTS)[segment],
            "city_id": city_id,
            "city": np.array([c.name for c in CITIES])[city_id],
            "state": np.array([c.state for c in CITIES])[city_id],
            "device_type": np.array(DEVICES)[rng.choice(len(DEVICES), n, p=DEVICE_SHARES)],
            "issuer_id": issuer_id,
            "preferred_payment_method": np.array(METHODS)[
                rng.choice(len(METHODS), n, p=PREFERRED_METHOD_SHARES)
            ],
            "historical_transaction_count": rng.poisson(np.array(SEGMENT_HISTORY_MEAN)[segment]),
            # Latent generative parameters: relative purchase frequency and spend level.
            "activity_weight": np.array(SEGMENT_ACTIVITY)[segment] * rng.lognormal(0.0, 0.5, n),
            "spend_multiplier": rng.lognormal(0.0, 0.4, n),
        }
    )


def generate_merchants(config: EcosystemConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_merchants
    category = rng.choice(
        len(MERCHANT_CATEGORIES), n, p=_normalized([c.weight for c in MERCHANT_CATEGORIES])
    )
    size = rng.choice(len(MERCHANT_SIZES), n, p=_normalized(SIZE_SHARES))
    city_id = rng.choice(len(CITIES), n, p=_normalized([c.weight for c in CITIES]))

    base_aov = np.array([c.average_order_value for c in MERCHANT_CATEGORIES])[category]
    base_methods = np.array([c.method_weights for c in MERCHANT_CATEGORIES])[category]
    # Each merchant deviates from its category's method mix; concentration 20 keeps it close.
    method_weights = np.vstack([rng.dirichlet(20.0 * row) for row in base_methods])

    return pd.DataFrame(
        {
            "merchant_id": np.arange(n),
            "merchant_category": np.array([c.name for c in MERCHANT_CATEGORIES])[category],
            "merchant_size": np.array(MERCHANT_SIZES)[size],
            "online": np.array([c.online for c in MERCHANT_CATEGORIES])[category],
            "city_id": city_id,
            "city": np.array([c.name for c in CITIES])[city_id],
            "state": np.array([c.state for c in CITIES])[city_id],
            "average_order_value": np.round(base_aov * rng.lognormal(0.0, 0.3, n), 2),
            "volume_weight": np.array(SIZE_VOLUME)[size] * rng.lognormal(0.0, 0.5, n),
            "upi_weight": method_weights[:, 0],
            "card_weight": method_weights[:, 1],
            "netbanking_weight": method_weights[:, 2],
            "primary_gateway_id": rng.choice(
                len(GATEWAYS), n, p=_normalized([g.routing_share for g in GATEWAYS])
            ),
        }
    )
