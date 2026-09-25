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


DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    ffn_multiplier: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)

    numeric_bins: int = Field(ge=2)
    mask_rate_min: float = Field(gt=0.0, lt=1.0)
    mask_rate_max: float = Field(gt=0.0, lt=1.0)
    pretrain_rows: int = Field(gt=0)
    pretrain_epochs: int = Field(ge=0)

    finetune_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    success_sample_rate: float = Field(gt=0.0, le=1.0)
    entity_dropout: float = Field(ge=0.0, lt=1.0)
    reason_loss_weight: float = Field(ge=0.0)
    latency_loss_weight: float = Field(ge=0.0)

    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    ensemble_size: int = Field(gt=0)
    seed: int
    threads: int = Field(gt=0)


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig.model_validate(yaml.safe_load(f))
