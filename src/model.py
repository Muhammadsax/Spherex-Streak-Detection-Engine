"""
src/model.py — Top-level alias for spherex_streak.model
Phase 3: DeepStreaks Model Architecture & Focal Loss
"""

from spherex_streak.model import (
    DeepStreaksCNN,
    DilatedResidualBlock,
    FocalLoss,
    build_deepstreaks_model,
)

__all__ = [
    "DeepStreaksCNN",
    "DilatedResidualBlock",
    "FocalLoss",
    "build_deepstreaks_model",
]
