"""
utils/visualization.py — Astronomical Visualization & Side-by-Side Comparison
==============================================================================
Provides high-fidelity plotting and stretching routines tailored for SPHEREx
FITS science images, continuous PQM confidence maps, and effective variance.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def stretch_astronomical_image(
    image: np.ndarray,
    stretch_type: str = "asinh",
    p_low: float = 0.5,
    p_high: float = 99.5,
    asinh_a: float = 0.1,
) -> np.ndarray:
    """
    Applies standard astronomical contrast stretch (linear, asinh, log10).

    Parameters
    ----------
    image : np.ndarray (H, W) float32
    stretch_type : str
        'linear', 'asinh', or 'log'.
    p_low : float
        Lower clipping percentile.
    p_high : float
        Upper clipping percentile.
    asinh_a : float
        Softening parameter for arcsinh stretch.

    Returns
    -------
    np.ndarray
        Stretched image normalized to [0.0, 1.0].
    """
    arr = np.nan_to_num(image, nan=0.0).astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)

    vmin = float(np.percentile(valid, p_low))
    vmax = float(np.percentile(valid, p_high))
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)

    normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, None)

    if stretch_type.lower() == "asinh":
        stretched = np.arcsinh(normed / asinh_a) / np.arcsinh(1.0 / asinh_a)
    elif stretch_type.lower() == "log":
        stretched = np.log10(1.0 + 9.0 * normed)
    else:  # linear
        stretched = normed

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def render_side_by_side_figure(
    raw_science: np.ndarray,
    pqm_map: np.ndarray,
    effective_variance: np.ndarray,
    photometric_weights: np.ndarray,
    cmap_science: str = "inferno",
    cmap_pqm: str = "viridis",
    display_threshold: float = 0.5,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Renders the 4-panel comprehensive side-by-side comparison figure:
      Panel 1: Original FITS image with streak contour overlay.
      Panel 2: Continuous Probability Quality Mask (PQM [0.0, 1.0]).
      Panel 3: Effective Variance Map (sigma^2_eff = sigma^2 / w).
      Panel 4: Downstream Photometric Weights (W = 1 / sigma^2_eff).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    # 1. Stretched original science
    stretched = stretch_astronomical_image(raw_science, stretch_type="asinh")
    im0 = axes[0].imshow(stretched, cmap=cmap_science, origin="lower")
    axes[0].set_title("Original FITS Science\n(Intact Stars & Spectra)", fontsize=11, fontweight="bold")
    # Overlay subtle contour of streak detection (display-only)
    if np.any(pqm_map >= display_threshold):
        axes[0].contour(pqm_map >= display_threshold, levels=[0.5], colors="cyan", linewidths=1.2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Continuous PQM confidence matrix
    im1 = axes[1].imshow(pqm_map, cmap=cmap_pqm, origin="lower", vmin=0.0, vmax=1.0)
    axes[1].set_title("Probability Quality Mask (PQM)\n(Float32 [0.0, 1.0])", fontsize=11, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Effective Variance Map (log-scaled for dynamic range)
    eff_var_log = np.log10(np.maximum(effective_variance, 1e-4))
    im2 = axes[2].imshow(eff_var_log, cmap="plasma", origin="lower")
    axes[2].set_title(r"Effective Variance $\sigma^2_{\rm eff}$" + "\n" + r"($\sigma^2 / w$, log10)", fontsize=11, fontweight="bold")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Photometric Co-addition Weights
    im3 = axes[3].imshow(photometric_weights, cmap="cividis", origin="lower")
    axes[3].set_title("Photometric Weights W\n(1 / Effective Variance)", fontsize=11, fontweight="bold")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    return fig


def render_spectral_profile_plot(
    wavelengths_um: np.ndarray,
    raw_spectrum: np.ndarray,
    effective_spectrum: np.ndarray,
    pqm_spectrum: np.ndarray,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Renders 1D spectral line profile across 102 SPHEREx bands,
    demonstrating zero flux distortion and non-destructive conservation of
    cosmic absorption signatures (e.g. H2O ice at 3.05 um).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Spectral flux
    ax1.plot(wavelengths_um, raw_spectrum, color="dodgerblue", linewidth=2.0, label="Raw Science Flux I(λ) [Unaltered]")
    ax1.plot(wavelengths_um, effective_spectrum, color="red", linestyle="--", linewidth=1.2, label="Post-PQM Science Flux [Exact Match]")
    ax1.axvspan(2.95, 3.15, color="cyan", alpha=0.15, label=r"H$_2$O Ice 3.05 $\mu$m Absorption")
    ax1.set_ylabel("Relative Flux (e⁻/s)", fontsize=10)
    ax1.set_title("SPHEREx 102-Band Non-Destructive Spectrophotometry", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)

    # Contamination probability along the spectrum
    ax2.plot(wavelengths_um, pqm_spectrum, color="darkorange", linewidth=1.5, label="Contamination PQM(λ)")
    ax2.set_ylabel("PQM", fontsize=9)
    ax2.set_xlabel(r"Wavelength ($\mu$m)", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig
