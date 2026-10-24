"""
spherex_streak — NASA SPHEREx Satellite Streak Detection Engine
================================================================
A lightweight, non-destructive Python toolkit to detect low-SNR satellite streaks
in astronomical FITS images without modifying native spectrophotometric data.
"""

from spherex_streak.io.fits_loader import FITSLoader, SPHEREX_N_BANDS, adaptive_normalise
from spherex_streak.model import (
    DeepStreaksCNN,
    DilatedResidualBlock,
    FocalLoss,
    build_deepstreaks_model,
)
from spherex_streak.preprocessing import (
    AdaptiveBackgroundEstimator,
    AdaptivePreprocessor,
    BackgroundEstimate,
    MultiScaleLineFilter,
    SPHEREX_BANDS_COUNT,
    SpectralDiscriminationEngine,
    SpectralDiscriminationResult,
)

__version__ = "0.1.0"
__author__ = "Spherex Streak Team"

__all__ = [
    # Phase 1: FITS I/O
    "FITSLoader",
    "adaptive_normalise",
    "SPHEREX_N_BANDS",
    # Phase 2: Adaptive Preprocessing & Spectral Verification
    "AdaptiveBackgroundEstimator",
    "BackgroundEstimate",
    "MultiScaleLineFilter",
    "SpectralDiscriminationEngine",
    "SpectralDiscriminationResult",
    "AdaptivePreprocessor",
    "SPHEREX_BANDS_COUNT",
    # Phase 3: DeepStreaks Model & Focal Loss
    "DeepStreaksCNN",
    "DilatedResidualBlock",
    "FocalLoss",
    "build_deepstreaks_model",
]
