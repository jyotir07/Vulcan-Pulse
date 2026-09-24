import pytest

from backend.config import load_ecosystem_config
from backend.data.generator import generate_dataset


@pytest.fixture(scope="session")
def small_config():
    return load_ecosystem_config().model_copy(
        update={
            "n_customers": 5000,
            "n_merchants": 300,
            "n_days": 14,
            "target_transactions": 100_000,
        }
    )


@pytest.fixture(scope="session")
def dataset(small_config):
    return generate_dataset(small_config)
