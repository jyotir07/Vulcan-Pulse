from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ECOSYSTEM_CONFIG = REPO_ROOT / "configs" / "ecosystem.yaml"


class EcosystemConfig(BaseModel):
    # Reject unknown keys so a typo in the YAML fails loudly instead of silently using a default.
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    n_customers: int = Field(gt=0)
    n_merchants: int = Field(gt=0)

    start_date: date
    n_days: int = Field(gt=0)
    target_transactions: int = Field(gt=0)

    n_festival_days: int = Field(ge=0)
    festival_traffic_multiplier: float = Field(ge=1.0)

    offline_local_share: float = Field(ge=0.0, le=1.0)
    new_device_rate: float = Field(ge=0.0, le=1.0)
    gateway_spillover_rate: float = Field(ge=0.0, le=1.0)


def load_ecosystem_config(path: Path = DEFAULT_ECOSYSTEM_CONFIG) -> EcosystemConfig:
    with open(path, encoding="utf-8") as f:
        return EcosystemConfig.model_validate(yaml.safe_load(f))
