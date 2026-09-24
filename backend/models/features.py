"""Model inputs: transaction fields, observed state, and entity attributes.

Entities enter through their attributes (merchant category and size, customer segment and
history), never through merchant or customer ids, so a model can score entities it has not seen.
Categories are fixed from the catalog so codes line up between training and prediction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.data.catalog import CITIES, DEVICES, GATEWAYS, ISSUERS, MERCHANT_CATEGORIES, METHODS
from backend.data.entities import CUSTOMER_SEGMENTS, MERCHANT_SIZES

CATEGORIES = {
    "payment_method": list(METHODS),
    "issuer_id": list(range(len(ISSUERS))),
    "bank_type": sorted({i.bank_type for i in ISSUERS}),
    "gateway_id": list(range(len(GATEWAYS))),
    "merchant_category": [c.name for c in MERCHANT_CATEGORIES],
    "merchant_size": list(MERCHANT_SIZES),
    "device_type": list(DEVICES),
    "customer_segment": list(CUSTOMER_SEGMENTS),
    "city": [c.name for c in CITIES],
}
NUMERIC = (
    "log_amount",
    "log_merchant_aov",
    "log_customer_history",
    "issuer_health",
    "gateway_health",
    "gateway_utilization",
    "hour",
    "is_peak",
    "is_festival",
    "is_new_device",
    "online_merchant",
)
FEATURES = (*CATEGORIES, *NUMERIC)


@dataclass(frozen=True)
class FeatureBuilder:
    customers: pd.DataFrame
    merchants: pd.DataFrame

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        customer = frame["customer_id"].to_numpy()
        merchant = frame["merchant_id"].to_numpy()
        issuer = frame["issuer_id"].to_numpy()
        raw = {
            "payment_method": frame["payment_method"].astype(str).to_numpy(),
            "issuer_id": issuer,
            "bank_type": np.array([i.bank_type for i in ISSUERS])[issuer],
            "gateway_id": frame["gateway_id"].to_numpy(),
            "merchant_category": self.merchants["merchant_category"].to_numpy()[merchant],
            "merchant_size": self.merchants["merchant_size"].to_numpy()[merchant],
            "device_type": frame["device_type"].astype(str).to_numpy(),
            "customer_segment": self.customers["customer_segment"].to_numpy()[customer],
            "city": frame["city"].astype(str).to_numpy(),
        }
        out = {name: pd.Categorical(raw[name], categories=CATEGORIES[name]) for name in raw}
        history = self.customers["historical_transaction_count"].to_numpy()[customer]
        out.update(
            {
                "log_amount": np.log(frame["amount"].to_numpy()),
                "log_merchant_aov": np.log(
                    self.merchants["average_order_value"].to_numpy()[merchant]
                ),
                "log_customer_history": np.log1p(history),
                "issuer_health": frame["issuer_health"].to_numpy(),
                "gateway_health": frame["gateway_health"].to_numpy(),
                "gateway_utilization": frame["gateway_utilization"].to_numpy(),
                "hour": frame["hour"].to_numpy(),
                "is_peak": frame["is_peak"].to_numpy().astype(float),
                "is_festival": frame["is_festival"].to_numpy().astype(float),
                "is_new_device": frame["is_new_device"].to_numpy().astype(float),
                "online_merchant": self.merchants["online"].to_numpy()[merchant].astype(float),
            }
        )
        return pd.DataFrame(out, columns=list(FEATURES))
