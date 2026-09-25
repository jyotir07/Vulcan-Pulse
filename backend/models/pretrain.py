"""Masked-field pretraining: hide a random 15–30% of a transaction's fields and predict them.

Categorical fields are predicted as their category, numeric fields as their quantile bin. The
encoder learns how fields co-occur (merchant category with method and amount, hour with load,
issuer with health) before it sees any outcome.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from backend.models.encoder import N_FIELDS, PaymentEncoder
from backend.models.features import FEATURES


class MaskedFieldHead(nn.Module):
    def __init__(self, d_model: int, n_classes: list[int]):
        super().__init__()
        self.fields = nn.ModuleList(nn.Linear(d_model, n) for n in n_classes)

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        return [head(tokens[:, f]) for f, head in enumerate(self.fields)]


def random_field_mask(
    n_rows: int, rate_min: float, rate_max: float, generator: torch.Generator
) -> torch.Tensor:
    """Each row hides each field with its own rate from [rate_min, rate_max], at least one."""
    rate = rate_min + (rate_max - rate_min) * torch.rand(n_rows, 1, generator=generator)
    mask = torch.rand(n_rows, N_FIELDS, generator=generator) < rate
    forced = torch.randint(N_FIELDS, (n_rows,), generator=generator)
    mask[torch.arange(n_rows), forced] = True
    return mask


def masked_field_loss(
    logits: list[torch.Tensor], targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    total = torch.zeros(())
    for f, field_logits in enumerate(logits):
        hidden = mask[:, f]
        if hidden.any():
            total = total + F.cross_entropy(
                field_logits[hidden], targets[hidden, f], reduction="sum"
            )
    return total / mask.sum()


@dataclass(frozen=True)
class PretrainReport:
    train_loss: list[float]  # mean loss per epoch
    # On held-out rows, with the same masking: the model against always guessing each field's
    # most common training value.
    masked_accuracy: float
    frequency_baseline_accuracy: float
    field_accuracy: dict[str, float]
    field_baseline_accuracy: dict[str, float]


def pretrain(
    encoder: PaymentEncoder,
    codes: torch.Tensor,
    numeric: torch.Tensor,
    targets: torch.Tensor,
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_classes: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mask_rate: tuple[float, float],
    generator: torch.Generator,
) -> PretrainReport:
    """`targets` [row, field] holds category codes then numeric bins, in `FEATURES` order."""
    head = MaskedFieldHead(encoder.norm.normalized_shape[0], n_classes)
    params = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        encoder.train()
        order = torch.randperm(len(codes), generator=generator)
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            mask = random_field_mask(len(idx), *mask_rate, generator)
            logits = head(encoder.tokens(codes[idx], numeric[idx], mask))
            loss = masked_field_loss(logits, targets[idx], mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        losses.append(total / batches)

    encoder.eval()
    val_codes, val_numeric, val_targets = val
    mask = random_field_mask(len(val_codes), *mask_rate, generator)
    majority = torch.stack([torch.bincount(targets[:, f]).argmax() for f in range(N_FIELDS)])
    hits = torch.zeros(N_FIELDS)
    baseline_hits = torch.zeros(N_FIELDS)
    with torch.inference_mode():
        for start in range(0, len(val_codes), 8192):
            sl = slice(start, start + 8192)
            logits = head(encoder.tokens(val_codes[sl], val_numeric[sl], mask[sl]))
            for f, field_logits in enumerate(logits):
                hidden = mask[sl, f]
                truth = val_targets[sl][hidden, f]
                hits[f] += (field_logits[hidden].argmax(-1) == truth).sum()
                baseline_hits[f] += (truth == majority[f]).sum()
    counts = mask.sum(dim=0).clamp(min=1)
    return PretrainReport(
        train_loss=losses,
        masked_accuracy=float(hits.sum() / mask.sum()),
        frequency_baseline_accuracy=float(baseline_hits.sum() / mask.sum()),
        field_accuracy=dict(zip(FEATURES, (hits / counts).tolist(), strict=True)),
        field_baseline_accuracy=dict(zip(FEATURES, (baseline_hits / counts).tolist(), strict=True)),
    )
