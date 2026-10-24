"""
src/metrics.py — Top-level alias for spherex_streak.metrics
Phase 4: Evaluation Metrics, Soft IoU & Inverse-Variance Weighting
"""

from spherex_streak.metrics import (
    CombinedFocalSoftIoULoss,
    SoftIoULoss,
    SpectralPreservationReport,
    compute_effective_variance,
    compute_photometric_weights,
    soft_iou_score,
    verify_non_destructive_integrity,
)

__all__ = [
    "soft_iou_score",
    "SoftIoULoss",
    "CombinedFocalSoftIoULoss",
    "compute_effective_variance",
    "compute_photometric_weights",
    "verify_non_destructive_integrity",
    "SpectralPreservationReport",
]
