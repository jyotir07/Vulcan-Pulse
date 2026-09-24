from datetime import date

import pandas as pd

from backend.data.generator import Dataset


def first_attempts_excluding(dataset: Dataset, window_date: date) -> pd.DataFrame:
    """Observed first attempts from every day except the evaluation window."""
    txns = dataset.transactions
    keep = (txns["attempt"] == 1) & (txns["timestamp"].dt.date != window_date)
    return txns[keep]
