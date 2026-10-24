"""
spherex_streak.preprocessing — Phase 2: Adaptive Preprocessing Module
"""

from spherex_streak.preprocessing.adaptive import (
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
