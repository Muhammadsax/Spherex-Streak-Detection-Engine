"""
models/deepstreaks.py — Phase 3: DeepStreaks Model Architecture
================================================================
Adapted from the DeepStreaks convolutional framework (arXiv:1904.05920)
optimized for NASA SPHEREx FITS images.

Architecture Highlights:
------------------------
1. Fully Convolutional Architecture:
   - Does NOT use destructive pooling that erases sub-pixel faint streaks.
   - Employs dilated multi-scale convolutions to build a broad receptive field
     capable of following extended satellite tracks across the full field of view.
2. Residual Feature Refinement:
   - Dilated residual blocks with GroupNorm / BatchNorm and LeakyReLU activations.
   - Retains high-frequency edge information to resolve train-track patterns.
3. Continuous Probability Quality Mask (PQM):
   - Strict final Sigmoid activation yielding continuous Float32 values in [0.0, 1.0].
   - Output spatial dimensions exactly match input dimensions (N x M).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DilatedResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions to expand receptive field
    along linear streak directions without reducing spatial resolution.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class DeepStreaksCNN(nn.Module):
    """
    DeepStreaks-inspired Convolutional Neural Network for SPHEREx streak detection.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for single FITS science image, 3 for multi-feature input).
    base_filters : int
        Number of base convolutional filters in the initial layer (default: 32).
    num_blocks : int
        Number of dilated residual blocks (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters

        # 1. Input Stem: Multi-kernel receptive front
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups=min(8, base_filters), num_channels=base_filters),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 2. Dilated Residual Backbone: Progressively wider receptive fields (1, 2, 4, 8)
        dilations = [2 ** (i % 4) for i in range(num_blocks)]
        blocks = []
        for d in dilations:
            blocks.append(DilatedResidualBlock(base_filters, dilation=d))
        self.backbone = nn.Sequential(*blocks)

        # 3. Multi-Scale Context Aggregator
        self.context_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, base_filters // 2), num_channels=base_filters // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 4. Final 1x1 Convolution & Continuous Sigmoid Output
        self.classifier = nn.Conv2d(base_filters // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, H, W).
        return_logits : bool
            If True, returns unactivated logits (useful for numerical stability in BCEWithLogits).
            If False (default), returns continuous PQM values via Sigmoid in [0.0, 1.0].

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 1, H, W) of type float32.
        """
        features = self.stem(x)
        features = self.backbone(features)
        features = self.context_conv(features)
        logits = self.classifier(features)

        if return_logits:
            return logits
        return self.sigmoid(logits)

    @torch.no_grad()
    def predict_pqm(
        self,
        image: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> np.ndarray:
        """
        Non-destructive inference method producing a continuous PQM matrix
        for an input FITS image without modifying original pixel values.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2D array of shape (H, W) or 3D (C, H, W).
        device : str or torch.device, optional
            Computation device ('cpu', 'cuda', etc.). Defaults to model device.

        Returns
        -------
        np.ndarray
            Continuous Float32 matrix of shape (H, W) with values strictly in [0.0, 1.0].
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        if isinstance(image, np.ndarray):
            arr = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif arr.ndim == 3:
                tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
            else:
                raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")
        elif isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = tensor.to(device=device, dtype=torch.float32)
        pqm_tensor = self.forward(tensor, return_logits=False)

        # Squeeze batch and channel to return clean (H, W) float32 numpy array
        pqm_np = pqm_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # Strictly enforce technical guardrail [0.0, 1.0]
        pqm_np = np.clip(pqm_np, 0.0, 1.0)
        return pqm_np
