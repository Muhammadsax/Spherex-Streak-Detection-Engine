"""
spherex_streak.weighting — Phase 4: Inverse-Variance Weighting Module
"""

from spherex_streak.weighting.inv_variance import (
    SpectralPreservationReport,
    compute_effective_variance,
    compute_photometric_weights,
    verify_non_destructive_integrity,
)

__all__ = [
    "compute_effective_variance",
    "compute_photometric_weights",
    "verify_non_destructive_integrity",
    "SpectralPreservationReport",
]
