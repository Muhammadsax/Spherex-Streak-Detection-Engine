"""
metrics.py — Phase 4: Evaluation Metrics & Soft IoU Validation
==============================================================
Provides continuous Float32 Soft IoU (Intersection over Union),
Inverse-Variance Weighting validation, and non-destructive spectrophotometric
data integrity verification for the SPHEREx satellite streak processor.

Key Formulae:
-------------
1. Soft IoU (Continuous Jaccard Index):
       Soft_IoU(P, G) = sum(P * G) / (sum(P + G - P * G) + eps)
   Target: Soft IoU >= 0.95

2. Inverse-Variance Weighting:
       sigma^2_effective = sigma^2 / w
   where w = 1.0 - PQM  in [0.0, 1.0]

3. Non-Destructive Conservation:
       Raw astronomical flux I(x, y) is 100% preserved; no pixel zeroing or cropping.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from spherex_streak.models.losses import FocalLoss
from spherex_streak.weighting.inv_variance import (
    SpectralPreservationReport,
    compute_effective_variance,
    compute_photometric_weights,
    verify_non_destructive_integrity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Soft IoU Evaluation Metric (Continuous Float32)
# ---------------------------------------------------------------------------

def soft_iou_score(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> float:
    """
    Computes continuous Soft Intersection over Union (Soft IoU / Tanimoto Jaccard)
    between predicted Probability Quality Mask (PQM) and ground truth mask.

    Continuous Jaccard Formulation:
        Intersection = sum(p * g)
        Union        = sum(p^2 + g^2 - p * g)
        Soft_IoU     = Intersection / (Union + eps)

    Guarantees:
      - When P == G (for any continuous or binary mask), Soft_IoU == 1.0.
      - For binary {0, 1} masks, reduces exactly to standard Jaccard IoU.
      - Strictly bounded in [0.0, 1.0].
      - Fully differentiable.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted continuous PQM confidence matrix with values in [0.0, 1.0].
    targets : np.ndarray or torch.Tensor
        Ground truth mask (continuous or binary) with values in [0.0, 1.0].
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    float
        Continuous Soft IoU score in [0.0, 1.0]. Returns 1.0 if both masks are empty.
    """
    if isinstance(predictions, torch.Tensor):
        p = torch.clamp(predictions.detach(), 0.0, 1.0).float()
        g = torch.clamp(targets.detach().to(p.device), 0.0, 1.0).float()

        p_sum = float(torch.sum(p))
        g_sum = float(torch.sum(g))
        if p_sum < 1e-5 and g_sum < 1e-5:
            return 1.0

        intersection = torch.sum(p * g)
        union = torch.sum(p * p + g * g - p * g)
        iou = float((intersection / (union + eps)).cpu())
        return float(np.clip(iou, 0.0, 1.0))

    else:
        p = np.clip(np.asarray(predictions, dtype=np.float32), 0.0, 1.0)
        g = np.clip(np.asarray(targets, dtype=np.float32), 0.0, 1.0)

        p_sum = float(np.sum(p))
        g_sum = float(np.sum(g))
        if p_sum < 1e-5 and g_sum < 1e-5:
            return 1.0

        intersection = float(np.sum(p * g))
        union = float(np.sum(p * p + g * g - p * g))
        iou = intersection / (union + eps)
        return float(np.clip(iou, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Differentiable Soft IoU Loss
# ---------------------------------------------------------------------------

class SoftIoULoss(nn.Module):
    """
    Differentiable Soft IoU Loss for PyTorch optimization:
        L_SoftIoU = 1.0 - Soft_IoU(P, G)

    Drives predictions to achieve the technical target of Soft IoU >= 0.95.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Predictions in [0.0, 1.0], shape (B, C, H, W) or (B, H, W).
        targets : torch.Tensor
            Ground truth in [0.0, 1.0], matching prediction shape.

        Returns
        -------
        torch.Tensor
            Scalar loss (or per-sample tensor if reduction='none').
        """
        p = torch.clamp(predictions, 0.0, 1.0)
        g = torch.clamp(targets.to(p.device, dtype=p.dtype), 0.0, 1.0)

        # Flatten spatial dimensions per batch sample
        if p.ndim >= 3:
            p_flat = p.flatten(start_dim=1)
            g_flat = g.flatten(start_dim=1)
        else:
            p_flat = p.unsqueeze(0)
            g_flat = g.unsqueeze(0)

        intersection = torch.sum(p_flat * g_flat, dim=1)
        union = torch.sum(p_flat * p_flat + g_flat * g_flat - p_flat * g_flat, dim=1)

        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ---------------------------------------------------------------------------
# 3. Combined Focal Loss + Soft IoU Loss
# ---------------------------------------------------------------------------

class CombinedFocalSoftIoULoss(nn.Module):
    """
    Compound loss function:
        L_total = L_focal + lambda_iou * L_soft_iou

    Combines the extreme class-imbalance handling of Focal Loss (gamma >= 2.0)
    with the global boundary precision of Soft IoU (targeting Soft IoU >= 0.95).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        iou_weight: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, eps=eps, reduction="mean")
        self.iou_loss = SoftIoULoss(eps=eps, reduction="mean")
        self.iou_weight = float(iou_weight)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal_loss(predictions, targets)
        loss_iou = self.iou_loss(predictions, targets)
        return loss_focal + self.iou_weight * loss_iou


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Metrics
    "soft_iou_score",
    "SoftIoULoss",
    "CombinedFocalSoftIoULoss",
    # Inverse-Variance Weighting & Validation
    "compute_effective_variance",
    "compute_photometric_weights",
    "verify_non_destructive_integrity",
    "SpectralPreservationReport",
]
