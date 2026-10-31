"""
src/spherex_streak/timeseries/__init__.py
========================================
Time-series and multi-epoch SPHEREx transient discovery suite.
"""

from spherex_streak.timeseries.irsa_client import (
    CHALLENGE_TARGETS,
    COLLECTION_QR2,
    COLLECTION_QR3,
    DOI_QR2,
    DOI_QR3,
    IRSA_SIA_ENDPOINT,
    EpochFrame,
    SpherexTimeSeriesClient,
)
from spherex_streak.timeseries.processor import (
    SpherexTimeSeriesProcessor,
    TimeSeriesAnalysisResult,
)

__all__ = [
    "CHALLENGE_TARGETS",
    "COLLECTION_QR2",
    "COLLECTION_QR3",
    "DOI_QR2",
    "DOI_QR3",
    "IRSA_SIA_ENDPOINT",
    "EpochFrame",
    "SpherexTimeSeriesClient",
    "SpherexTimeSeriesProcessor",
    "TimeSeriesAnalysisResult",
]
