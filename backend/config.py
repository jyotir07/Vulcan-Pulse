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
    n_issuers: int = Field(gt=0)
    n_gateways: int = Field(gt=0)
    n_cities: int = Field(gt=0)
    n_days: int = Field(gt=0)
    target_transactions: int = Field(gt=0)


def load_ecosystem_config(path: Path = DEFAULT_ECOSYSTEM_CONFIG) -> EcosystemConfig:
    with open(path, encoding="utf-8") as f:
        return EcosystemConfig.model_validate(yaml.safe_load(f))
