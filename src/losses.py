"""Loss functions for multi-label CXR classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Multi-label focal loss with per-class positive weights.

    The implementation deliberately uses BCEWithLogits internally for numerical
    stability. With gamma=0, it is exactly BCEWithLogitsLoss(pos_weight=...).

    Args:
        gamma: Focusing parameter. The project compares 0, 1, and 2.
        pos_weight: Optional tensor/list of shape [num_classes]. It is calculated
            from the training partition of the current fold only.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, gamma: float = 2.0, pos_weight=None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.gamma = float(gamma)
        self.reduction = reduction

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if weight.ndim != 1:
                raise ValueError("pos_weight must be a 1-D tensor with one value per class")
            if not torch.isfinite(weight).all() or (weight <= 0).any():
                raise ValueError("pos_weight must contain finite positive values")
            self.register_buffer("pos_weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape; got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )

        targets = targets.to(dtype=logits.dtype)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            if pos_weight.numel() != logits.shape[-1]:
                raise ValueError(
                    f"pos_weight length ({pos_weight.numel()}) does not match "
                    f"number of classes ({logits.shape[-1]})"
                )
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        # Avoid unnecessary operations in the gamma=0 baseline. This makes the
        # result numerically identical to weighted BCEWithLogitsLoss.
        if self.gamma == 0:
            loss = bce
        else:
            probabilities = torch.sigmoid(logits)
            p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
            focal_factor = (1.0 - p_t).pow(self.gamma)
            loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
