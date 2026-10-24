"""
preprocessing.py — Core entry point for Phase 2: Adaptive Preprocessing & Multi-scale Line Tracking
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
