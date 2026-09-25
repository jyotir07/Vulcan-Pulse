"""Outcome heads on the shared representation, trained jointly with the encoder.

- success: one logit, binary cross-entropy;
- failure reason: logits over the taxonomy, cross-entropy on failed rows only, so it models the
  reason given a failure;
- latency: mean and log spread of standardized log latency, Gaussian negative log-likelihood, so
  each transaction gets its own lognormal spread rather than one per method.
"""

import torch
import torch.nn.functional as F
from torch import nn

LOG_SIGMA_BOUNDS = (-5.0, 3.0)


class OutcomeHeads(nn.Module):
    def __init__(self, d_model: int, n_reasons: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.success = nn.Linear(d_model, 1)
        self.reason = nn.Linear(d_model, n_reasons)
        self.latency = nn.Linear(d_model, 2)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.hidden(z)
        latency = self.latency(h)
        return {
            "success_logit": self.success(h).squeeze(-1),
            "reason_logits": self.reason(h),
            "latency_mu": latency[:, 0],
            "latency_log_sigma": latency[:, 1].clamp(*LOG_SIGMA_BOUNDS),
        }


def outcome_loss(
    out: dict[str, torch.Tensor],
    success: torch.Tensor,
    reason: torch.Tensor,
    latency: torch.Tensor,
    weight: torch.Tensor,
    reason_weight: float,
    latency_weight: float,
) -> torch.Tensor:
    """Weighted mean joint loss. `reason` is ignored where `success` is 1."""
    failed = 1.0 - success
    bce = F.binary_cross_entropy_with_logits(out["success_logit"], success, reduction="none")
    ce = F.cross_entropy(out["reason_logits"], reason.clamp(min=0), reduction="none") * failed
    z = (latency - out["latency_mu"]) * torch.exp(-out["latency_log_sigma"])
    nll = 0.5 * z**2 + out["latency_log_sigma"]
    per_row = bce + reason_weight * ce + latency_weight * nll
    return (weight * per_row).sum() / weight.sum()
