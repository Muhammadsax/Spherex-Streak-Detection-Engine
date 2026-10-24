"""
detection/pqm.py — Phase 4: Probability Quality Mask (PQM) Engine
==================================================================
Manages continuous Float32 Probability Quality Masks [0.0, 1.0],
computes streak coverage statistics, and interfaces with Inverse-Variance Weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from spherex_streak.weighting.inv_variance import (
    SpectralPreservationReport,
    compute_effective_variance,
    compute_photometric_weights,
    verify_non_destructive_integrity,
)

logger = logging.getLogger(__name__)


@dataclass
class PQMStatistics:
    """Summary metrics of a continuous Probability Quality Mask."""
    shape: Tuple[int, ...]
    min_prob: float
    max_prob: float
    mean_prob: float
    median_prob: float
    contamination_fraction: float       # Fraction of pixels with PQM > display_threshold
    clean_fraction: float               # Fraction of pixels with PQM <= 0.05
    heavy_contamination_count: int      # Count of pixels with PQM >= 0.80


class ProbabilityQualityMaskEngine:
    """
    Engine for creating, validating, and applying Probability Quality Masks (PQM).

    Guarantees:
      - Continuous Float32 precision in [0.0, 1.0].
      - No destructive pixel deletion or zeroing.
      - Seamless coupling with Inverse-Variance Weighting (sigma^2_eff = sigma^2 / w).
    """

    def __init__(self, display_threshold: float = 0.5) -> None:
        self.display_threshold = float(display_threshold)

    def validate_pqm(self, pqm: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Validates and enforces strict PQM technical constraints:
          1. Continuous Float32 data type.
          2. Values strictly bounded within [0.0, 1.0].
          3. Replaces NaNs with 0.0 (clean default).
        """
        if isinstance(pqm, torch.Tensor):
            arr = pqm.detach().cpu().numpy()
        else:
            arr = np.asarray(pqm)

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    def compute_statistics(
        self,
        pqm: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> PQMStatistics:
        """Computes comprehensive quantitative metrics on a PQM."""
        arr = self.validate_pqm(pqm)
        thresh = self.display_threshold if threshold is None else float(threshold)

        total_pixels = max(1, arr.size)
        contaminated = int(np.sum(arr > thresh))
        clean = int(np.sum(arr <= 0.05))
        heavy = int(np.sum(arr >= 0.80))

        return PQMStatistics(
            shape=arr.shape,
            min_prob=float(np.min(arr)),
            max_prob=float(np.max(arr)),
            mean_prob=float(np.mean(arr)),
            median_prob=float(np.median(arr)),
            contamination_fraction=float(contaminated / total_pixels),
            clean_fraction=float(clean / total_pixels),
            heavy_contamination_count=heavy,
        )

    def apply_inverse_variance_weighting(
        self,
        variance: np.ndarray,
        pqm: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes effective variance and downstream photometric weights:
            sigma^2_eff = sigma^2 / (1.0 - PQM)
            W = 1.0 / sigma^2_eff

        Returns
        -------
        Tuple[effective_variance, photometric_weights]
        """
        pqm_clean = self.validate_pqm(pqm)
        eff_var = compute_effective_variance(variance, pqm_clean, eps=eps)
        weights = compute_photometric_weights(eff_var)
        return eff_var, weights

    def generate_display_mask(
        self,
        pqm: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generates a boolean overlay mask for GUI visualization only.
        NOTE: This boolean mask is strictly for display and is NEVER applied
        destructively to the underlying scientific FITS data.
        """
        arr = self.validate_pqm(pqm)
        thresh = self.display_threshold if threshold is None else float(threshold)
        return arr >= thresh
