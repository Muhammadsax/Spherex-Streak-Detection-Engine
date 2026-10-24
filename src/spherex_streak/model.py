"""
model.py — Core entry point for Phase 3: DeepStreaks Model Architecture & Focal Loss
=====================================================================================
Adapted from DeepStreaks (arXiv:1904.05920) for SPHEREx astronomical FITS images.
"""

from spherex_streak.models.deepstreaks import DeepStreaksCNN, DilatedResidualBlock
from spherex_streak.models.losses import FocalLoss


def build_deepstreaks_model(
    in_channels: int = 1,
    base_filters: int = 32,
    num_blocks: int = 4,
) -> DeepStreaksCNN:
    """
    Factory function to construct a configured DeepStreaksCNN instance.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1 for FITS science image).
    base_filters : int
        Convolutional filter depth (default: 32).
    num_blocks : int
        Number of dilated residual blocks (default: 4).

    Returns
    -------
    DeepStreaksCNN instance.
    """
    return DeepStreaksCNN(
        in_channels=in_channels,
        base_filters=base_filters,
        num_blocks=num_blocks,
    )


__all__ = [
    "DeepStreaksCNN",
    "DilatedResidualBlock",
    "FocalLoss",
    "build_deepstreaks_model",
]
