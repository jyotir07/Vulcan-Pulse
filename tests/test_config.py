import pytest
from pydantic import ValidationError

from backend.config import EcosystemConfig, load_ecosystem_config


def test_default_config_loads():
    config = load_ecosystem_config()
    assert config.target_transactions >= 1_000_000


def test_unknown_key_rejected():
    fields = load_ecosystem_config().model_dump()
    with pytest.raises(ValidationError):
        EcosystemConfig.model_validate({**fields, "n_merchnts": 10})


def test_non_positive_size_rejected():
    fields = load_ecosystem_config().model_dump()
    with pytest.raises(ValidationError):
        EcosystemConfig.model_validate({**fields, "n_issuers": 0})
