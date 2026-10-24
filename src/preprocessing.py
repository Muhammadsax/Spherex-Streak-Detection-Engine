"""
src/preprocessing.py — Top-level alias for spherex_streak.preprocessing
"""

from spherex_streak.preprocessing import (
    AdaptiveBackgroundEstimator,
    AdaptivePreprocessor,
    BackgroundEstimate,
    MultiScaleLineFilter,
    SPHEREX_BANDS_COUNT,
    SpectralDiscriminationEngine,
    SpectralDiscriminationResult,
)

__all__ = [
    "AdaptiveBackgroundEstimator",
    "BackgroundEstimate",
    "MultiScaleLineFilter",
    "SpectralDiscriminationEngine",
    "SpectralDiscriminationResult",
    "AdaptivePreprocessor",
    "SPHEREX_BANDS_COUNT",
]
