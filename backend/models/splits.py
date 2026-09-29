from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from backend.data.catalog import ISSUERS
from backend.data.generator import Dataset

TEST_DAYS = 7

# Cold start: entities whose transactions are removed from training. Two mid-sized issuers, one
# private and one public, so each still has siblings of its bank type in training.
HOLDOUT_ISSUERS = ("Yes Bank", "Canara")
HOLDOUT_FRACTION = 0.1
HOLDOUT_SEED_KEY = 7

Suite = Literal["main", "cold_start", "normal_only"]
SUITES: tuple[Suite, ...] = ("main", "cold_start", "normal_only")


@dataclass(frozen=True)
class TimeSplit:
    """Train on the earlier days, evaluate on the last `TEST_DAYS`; models never see test days."""

    train_days: list[date]
    test_days: list[date]


def time_split(dataset: Dataset, test_days: int = TEST_DAYS) -> TimeSplit:
    start, n_days = dataset.config.start_date, dataset.config.n_days
    if not 0 < test_days < n_days:
        raise ValueError(f"test_days must be between 1 and {n_days - 1}")
    days = [start + timedelta(days=d) for d in range(n_days)]
    return TimeSplit(train_days=days[:-test_days], test_days=days[-test_days:])


def evaluation_windows(dataset: Dataset, split: TimeSplit) -> list[date]:
    """The first and last normal weekday of the test period: two ordinary, unseen days."""
    normal = [d for d in split.test_days if d.weekday() < 5 and d not in dataset.festival_dates]
    if not normal:
        raise ValueError("the test period has no normal weekday to evaluate on")
    return sorted({normal[0], normal[-1]})


def first_attempts_on(dataset: Dataset, days: list[date]) -> pd.DataFrame:
    txns = dataset.transactions
    return txns[(txns["attempt"] == 1) & txns["timestamp"].dt.date.isin(days)]


def first_attempts_excluding(dataset: Dataset, window_date: date) -> pd.DataFrame:
    """Observed first attempts from every day except the evaluation window."""
    txns = dataset.transactions
    keep = (txns["attempt"] == 1) & (txns["timestamp"].dt.date != window_date)
    return txns[keep]


@dataclass(frozen=True)
class EntityHoldout:
    """Merchants, customers and issuers that a cold-start model never sees in training."""

    merchants: np.ndarray
    customers: np.ndarray
    issuers: np.ndarray

    def merchant_rows(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["merchant_id"].isin(self.merchants).to_numpy()

    def customer_rows(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["customer_id"].isin(self.customers).to_numpy()

    def issuer_rows(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["issuer_id"].isin(self.issuers).to_numpy()

    def involved(self, frame: pd.DataFrame) -> np.ndarray:
        return self.merchant_rows(frame) | self.customer_rows(frame) | self.issuer_rows(frame)


def entity_holdout(dataset: Dataset) -> EntityHoldout:
    rng = np.random.default_rng([dataset.config.seed, HOLDOUT_SEED_KEY])
    n_merchants, n_customers = len(dataset.merchants), len(dataset.customers)
    names = [i.name for i in ISSUERS]
    return EntityHoldout(
        merchants=np.sort(
            rng.choice(n_merchants, size=round(HOLDOUT_FRACTION * n_merchants), replace=False)
        ),
        customers=np.sort(
            rng.choice(n_customers, size=round(HOLDOUT_FRACTION * n_customers), replace=False)
        ),
        issuers=np.array([names.index(n) for n in HOLDOUT_ISSUERS]),
    )


def normal_traffic(frame: pd.DataFrame) -> np.ndarray:
    """Rows outside peak hours and festival days: the distribution-shift training regime."""
    return ~(frame["is_peak"].to_numpy() | frame["is_festival"].to_numpy())


def suite_training_rows(dataset: Dataset, split: TimeSplit, suite: Suite) -> pd.DataFrame:
    """The first attempts each experiment suite trains on; every suite uses the same days.

    - main: every first attempt on the training days;
    - cold_start: minus any row touching a held-out merchant, customer or issuer;
    - normal_only: minus peak hours and festival days.
    """
    train = first_attempts_on(dataset, split.train_days)
    if suite == "main":
        return train
    if suite == "cold_start":
        return train[~entity_holdout(dataset).involved(train)]
    if suite == "normal_only":
        return train[normal_traffic(train)]
    raise ValueError(f"unknown suite {suite!r}; expected one of {SUITES}")
