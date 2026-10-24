"""
models/losses.py — Phase 3: Focal Loss for Low-SNR Streak Detection
====================================================================
Mathematically rigorous, numerically stable Focal Loss implementation
designed specifically for satellite streak segmentation in astronomical FITS images.

Mathematical Formulation:
-------------------------
For ground-truth target y in {0, 1} and predicted probability p in [0.0, 1.0]:
    p_t = p if y == 1 else (1 - p)
    alpha_t = alpha if y == 1 else (1 - alpha)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t + eps)

Why gamma >= 2.0 is critical for SPHEREx:
------------------------------------------
In astronomical images, >99.5% of pixels are background sky noise (y = 0).
When gamma >= 2.0, well-classified background pixels (p < 0.05) generate virtually zero loss:
    (1 - 0.95)^2.0 = (0.05)^2 = 0.0025 (loss downscaled by 400x!)
This prevents the loss gradient from being swamped by dark sky pixels and forces the
network to stay vigilant on faint, low-SNR satellite tracks near the noise floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense binary streak detection with extreme class imbalance.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Must be >= 2.0 per technical requirements.
        Higher gamma decreases the relative loss for easy, well-classified examples.
    alpha : float, optional
        Weighting factor for the rare streak class (default: 0.25).
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    eps : float
        Small numerical stability epsilon to prevent log(0).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities in [0.0, 1.0] (or raw logits if is_logits=True).
        targets : torch.Tensor
            Ground truth continuous or binary target in [0.0, 1.0], matching prediction shape.
        is_logits : bool
            If True, predictions are treated as raw unactivated logits.
            If False (default), predictions are treated as post-Sigmoid probabilities.

        Returns
        -------
        torch.Tensor
            Calculated loss according to reduction mode.
        """
        if is_logits:
            probs = torch.sigmoid(predictions)
            bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        else:
            probs = torch.clamp(predictions, min=self.eps, max=1.0 - self.eps)
            bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # Compute p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # Compute focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # Compute alpha balancing factor: alpha_t
        if self.alpha >= 0.0:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction mode: '{self.reduction}'")

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, "
            f"reduction='{self.reduction}')"
        )
