"""Shared-representation predictor: an ensemble of Set Transformer encoders with outcome heads.

Each member is pretrained with masked-field prediction and then fine-tuned jointly on success,
failure reason and latency, from its own seed. Members share the inputs of the independent
models (`FeatureBuilder`), so the comparison isolates the representation, not the features.

The ensemble prediction is the members' mean; `member_predictions` exposes the spread. Fine-tuning
keeps every failure and samples successes, reweighting them so probabilities stay calibrated.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.config import TrainConfig
from backend.data.dgp import FAILURE_REASONS
from backend.models.encoder import N_FIELDS, FieldTokenizer, PaymentEncoder
from backend.models.features import CATEGORIES, FeatureBuilder
from backend.models.heads import OutcomeHeads, outcome_loss
from backend.models.pretrain import PretrainReport, pretrain
from backend.simulation.metrics import Predictions

PREDICT_BATCH = 8192
ISSUER_FIELD = list(CATEGORIES).index("issuer_id")


@dataclass(frozen=True)
class _Tensors:
    codes: torch.Tensor
    numeric: torch.Tensor
    bins: torch.Tensor
    success: torch.Tensor
    reason: torch.Tensor  # failure reason code, -1 on success
    latency: torch.Tensor  # standardized log latency

    def __getitem__(self, idx) -> "_Tensors":
        return _Tensors(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    def __len__(self) -> int:
        return len(self.codes)


@dataclass(frozen=True)
class FinetuneReport:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int


@dataclass(frozen=True)
class MemberReport:
    seed: int
    pretrain: PretrainReport | None
    finetune: FinetuneReport


class _Member(torch.nn.Module):
    def __init__(self, config: TrainConfig, vocab_sizes: list[int]):
        super().__init__()
        self.encoder = PaymentEncoder(
            vocab_sizes,
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.ffn_multiplier,
            config.dropout,
        )
        self.heads = OutcomeHeads(config.d_model, len(FAILURE_REASONS))

    def forward(self, codes, numeric, mask=None):
        return self.heads(self.encoder(codes, numeric, mask))


def _loss(member: _Member, batch: _Tensors, weight, mask, config: TrainConfig) -> torch.Tensor:
    return outcome_loss(
        member(batch.codes, batch.numeric, mask),
        batch.success,
        batch.reason,
        batch.latency,
        weight,
        config.reason_loss_weight,
        config.latency_loss_weight,
    )


def _val_loss(member: _Member, val: _Tensors, config: TrainConfig) -> float:
    member.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(val), PREDICT_BATCH):
            batch = val[slice(start, start + PREDICT_BATCH)]
            weight = torch.ones(len(batch))
            total += _loss(member, batch, weight, None, config).item() * len(batch)
    return total / len(val)


def _finetune(
    member: _Member, train: _Tensors, val: _Tensors, config: TrainConfig, seed: int
) -> FinetuneReport:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        member.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    failed = torch.nonzero(train.success == 0).squeeze(1)
    succeeded = torch.nonzero(train.success == 1).squeeze(1)
    success_weight = 1.0 / config.success_sample_rate
    train_losses, val_losses = [], []
    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(config.finetune_epochs):
        member.train()
        keep = torch.rand(len(succeeded), generator=generator) < config.success_sample_rate
        rows = torch.cat([failed, succeeded[keep]])
        rows = rows[torch.randperm(len(rows), generator=generator)]
        total, batches = 0.0, 0
        for start in range(0, len(rows), config.batch_size):
            batch = train[rows[start : start + config.batch_size]]
            weight = torch.where(batch.success == 1, success_weight, 1.0)
            mask = torch.zeros(len(batch), N_FIELDS, dtype=torch.bool)
            mask[:, ISSUER_FIELD] = torch.rand(len(batch), generator=generator) < (
                config.entity_dropout
            )
            loss = _loss(member, batch, weight, mask, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        train_losses.append(total / batches)
        val_losses.append(_val_loss(member, val, config))
        if val_losses[-1] < best:
            best, best_epoch = val_losses[-1], epoch
            best_state = {k: v.clone() for k, v in member.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break
    member.load_state_dict(best_state)
    member.eval()
    return FinetuneReport(train_loss=train_losses, val_loss=val_losses, best_epoch=best_epoch)


@dataclass(frozen=True)
class SharedModel:
    features: FeatureBuilder
    tokenizer: FieldTokenizer
    latency_mean: float
    latency_std: float
    members: list[_Member]
    config: TrainConfig
    reports: list[MemberReport]
    train_days: list[str]
    name: str = "shared_representation"

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        customers: pd.DataFrame,
        merchants: pd.DataFrame,
        config: TrainConfig,
        progress=None,
    ) -> "SharedModel":
        """Fit on observed first attempts. `train` must not include any evaluation window."""
        features = FeatureBuilder(customers, merchants)
        x = features(train)
        tokenizer = FieldTokenizer.fit(x, config.numeric_bins)
        codes, numeric = tokenizer(x)
        log_latency = np.log(train["latency_ms"].to_numpy().astype(float))
        latency_mean, latency_std = float(log_latency.mean()), float(log_latency.std())
        data = _Tensors(
            codes=codes,
            numeric=numeric,
            bins=tokenizer.numeric_bins(x),
            success=torch.from_numpy(
                (train["transaction_status"] == "SUCCESS").to_numpy().astype(np.float32)
            ),
            reason=torch.from_numpy(train["failure_reason"].cat.codes.to_numpy().astype(np.int64)),
            latency=torch.from_numpy(
                ((log_latency - latency_mean) / latency_std).astype(np.float32)
            ),
        )
        rng = np.random.default_rng(config.seed)
        is_val = rng.random(len(data)) < config.validation_fraction
        fit_rows, val = data[torch.from_numpy(~is_val)], data[torch.from_numpy(is_val)]
        n_classes = tokenizer.vocab_sizes + tokenizer.numeric_bin_counts

        members, reports = [], []
        for k in range(config.ensemble_size):
            seed = config.seed + k
            torch.manual_seed(seed)
            member = _Member(config, tokenizer.vocab_sizes)
            # Start the success head at the base rate instead of 0.5.
            with torch.no_grad():
                member.heads.success.bias.fill_(float(torch.logit(data.success.mean())))
            pretrain_report = None
            if config.pretrain_epochs:
                rows = np.random.default_rng(seed).permutation(len(fit_rows))
                sample = fit_rows[torch.from_numpy(rows[: config.pretrain_rows])]
                pretrain_report = pretrain(
                    member.encoder,
                    sample.codes,
                    sample.numeric,
                    torch.cat([sample.codes, sample.bins], dim=1),
                    (val.codes, val.numeric, torch.cat([val.codes, val.bins], dim=1)),
                    n_classes,
                    config.pretrain_epochs,
                    config.batch_size,
                    config.learning_rate,
                    config.weight_decay,
                    (config.mask_rate_min, config.mask_rate_max),
                    torch.Generator().manual_seed(seed),
                )
                if progress:
                    progress(
                        f"member {k}: pretrain loss {pretrain_report.train_loss}, masked acc "
                        f"{pretrain_report.masked_accuracy:.3f} vs frequency "
                        f"{pretrain_report.frequency_baseline_accuracy:.3f}"
                    )
            finetune = _finetune(member, fit_rows, val, config, seed)
            if progress:
                progress(
                    f"member {k}: val loss {[round(v, 4) for v in finetune.val_loss]}, "
                    f"best epoch {finetune.best_epoch}"
                )
            members.append(member)
            reports.append(MemberReport(seed=seed, pretrain=pretrain_report, finetune=finetune))

        return cls(
            features=features,
            tokenizer=tokenizer,
            latency_mean=latency_mean,
            latency_std=latency_std,
            members=members,
            config=config,
            reports=reports,
            train_days=sorted({d.isoformat() for d in train["timestamp"].dt.date}),
        )

    def member_predictions(self, frame: pd.DataFrame) -> list[Predictions]:
        codes, numeric = self.tokenizer(self.features(frame))
        out = []
        for member in self.members:
            member.eval()
            parts = []
            with torch.inference_mode():
                for start in range(0, len(codes), PREDICT_BATCH):
                    sl = slice(start, start + PREDICT_BATCH)
                    parts.append(member(codes[sl], numeric[sl]))
            p_success = torch.sigmoid(torch.cat([p["success_logit"] for p in parts]))
            reason = torch.softmax(torch.cat([p["reason_logits"] for p in parts]), dim=-1)
            mu = torch.cat([p["latency_mu"] for p in parts]).double().numpy()
            log_sigma = torch.cat([p["latency_log_sigma"] for p in parts]).double().numpy()
            out.append(
                Predictions(
                    p_success=p_success.double().numpy(),
                    reason_probs=reason.double().numpy(),
                    latency_median_ms=np.exp(self.latency_mean + self.latency_std * mu),
                    latency_log_sigma=self.latency_std * np.exp(log_sigma),
                )
            )
        return out

    def predict(self, frame: pd.DataFrame) -> Predictions:
        members = self.member_predictions(frame)
        log_median = np.stack([np.log(m.latency_median_ms) for m in members])
        sigma = np.stack([m.latency_log_sigma for m in members])
        return Predictions(
            p_success=np.mean([m.p_success for m in members], axis=0),
            reason_probs=np.mean([m.reason_probs for m in members], axis=0),
            latency_median_ms=np.exp(log_median.mean(axis=0)),
            # Moment-matched spread of the members' log-latency mixture.
            latency_log_sigma=np.sqrt((sigma**2).mean(axis=0) + log_median.var(axis=0)),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": self.config.model_dump(),
            "tokenizer": {
                "numeric_mean": self.tokenizer.numeric_mean.tolist(),
                "numeric_std": self.tokenizer.numeric_std.tolist(),
                "numeric_bin_edges": [e.tolist() for e in self.tokenizer.numeric_bin_edges],
            },
            "latency_mean": self.latency_mean,
            "latency_std": self.latency_std,
            "train_days": self.train_days,
            "reports": [asdict(r) for r in self.reports],
        }
        (directory / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for k, member in enumerate(self.members):
            torch.save(member.state_dict(), directory / f"member_{k}.pt")

    @classmethod
    def load(
        cls, directory: Path, customers: pd.DataFrame, merchants: pd.DataFrame
    ) -> "SharedModel":
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        config = TrainConfig.model_validate(meta["config"])
        tok = meta["tokenizer"]
        tokenizer = FieldTokenizer(
            numeric_mean=np.array(tok["numeric_mean"]),
            numeric_std=np.array(tok["numeric_std"]),
            numeric_bin_edges=tuple(np.array(e) for e in tok["numeric_bin_edges"]),
        )
        members = []
        for k in range(config.ensemble_size):
            member = _Member(config, tokenizer.vocab_sizes)
            member.load_state_dict(torch.load(directory / f"member_{k}.pt", weights_only=True))
            member.eval()
            members.append(member)
        reports = [
            MemberReport(
                seed=r["seed"],
                pretrain=PretrainReport(**r["pretrain"]) if r["pretrain"] else None,
                finetune=FinetuneReport(**r["finetune"]),
            )
            for r in meta["reports"]
        ]
        return cls(
            features=FeatureBuilder(customers, merchants),
            tokenizer=tokenizer,
            latency_mean=meta["latency_mean"],
            latency_std=meta["latency_std"],
            members=members,
            config=config,
            reports=reports,
            train_days=meta["train_days"],
        )
