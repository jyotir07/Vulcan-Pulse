from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from backend.data.generator import Dataset

TEST_DAYS = 7


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
