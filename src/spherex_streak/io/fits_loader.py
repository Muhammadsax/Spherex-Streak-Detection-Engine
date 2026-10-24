"""
fits_loader.py — Phase 1: FITS File I/O & Adaptive Normalization
=================================================================
Safely reads NASA SPHEREx FITS images into float32 NumPy arrays
WITHOUT any destructive cropping, zeroing, or pixel modification.

SPHEREx FITS Layout (per mission spec):
  - Extension 0  : Primary HDU (may be empty or carry a single-band image)
  - Extension 1  : Science image  (SCI)  — the primary data plane
  - Extension 2  : Uncertainty map (UNC) — per-pixel σ values
  - Extension 3  : Coverage mask   (COV) — integer exposure counts

All arrays are cast to float32 and NaN-filled where data is absent
so that downstream modules can handle missing pixels without loss.

Usage
-----
    from spherex_streak.io.fits_loader import FITSLoader

    loader = FITSLoader("path/to/image.fits")
    sci    = loader.science      # float32 ndarray (H, W) or (H, W, B)
    unc    = loader.uncertainty  # float32 ndarray or None
    cov    = loader.coverage     # float32 ndarray or None
    norm   = loader.normalized   # adaptively normalised to [0, 1] float32
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

# Suppress noisy Astropy WCS deprecation warnings
warnings.filterwarnings("ignore", category=FITSFixedWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SPHEREx operates across 102 spectral bands (0.75 – 5.0 µm)
SPHEREX_N_BANDS: int = 102

# Standard FITS extension names used by SPHEREx & SPARKLES
_EXT_SCIENCE     = ("SCI", 1)   # (name, fallback index)
_EXT_UNCERTAINTY = ("UNC", 2)
_EXT_COVERAGE    = ("COV", 3)

# Percentile clipping bounds for adaptive normalisation
_NORM_LOW_PCT  = 0.5   # bottom 0.5 %  — clips extreme dark pixels / cosmic artefacts
_NORM_HIGH_PCT = 99.5  # top    99.5 % — clips saturation spikes


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_cast_float32(data: np.ndarray) -> np.ndarray:
    """Cast any numeric array to float32; replace ±Inf with NaN."""
    arr = np.array(data, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _extract_extension(
    hdul: fits.HDUList,
    ext_name: str,
    ext_index: int,
) -> Optional[np.ndarray]:
    """
    Try to extract a FITS extension by name first, then by index.
    Returns a float32 array or None if the extension is absent.
    """
    # Try by name
    if ext_name in hdul:
        data = hdul[ext_name].data
        if data is not None:
            logger.debug("Loaded extension '%s' by name.", ext_name)
            return _safe_cast_float32(data)

    # Try by index
    if ext_index < len(hdul):
        data = hdul[ext_index].data
        if data is not None:
            logger.debug("Loaded extension %d as fallback for '%s'.", ext_index, ext_name)
            return _safe_cast_float32(data)

    logger.debug("Extension '%s' (index %d) not found — returning None.", ext_name, ext_index)
    return None


def adaptive_normalise(
    image: np.ndarray,
    low_pct: float = _NORM_LOW_PCT,
    high_pct: float = _NORM_HIGH_PCT,
) -> np.ndarray:
    """
    Percentile-based adaptive normalisation to [0.0, 1.0] float32.

    Unlike min-max scaling this is robust to:
      - Hot pixels / cosmic rays (clipped at high_pct)
      - Dark bias artefacts       (clipped at low_pct)
      - Variable sky background across different SPHEREx pointings

    No hardcoded thresholds — percentiles are computed per image.

    Parameters
    ----------
    image    : float32 array of arbitrary shape (H, W) or (H, W, B)
    low_pct  : lower percentile clip  (default 0.5)
    high_pct : upper percentile clip  (default 99.5)

    Returns
    -------
    Normalised float32 array in [0.0, 1.0]; NaN pixels remain NaN.
    """
    finite_pixels = image[np.isfinite(image)]

    if finite_pixels.size == 0:
        logger.warning("Image contains no finite pixels — returning zeros.")
        return np.zeros_like(image, dtype=np.float32)

    p_low  = float(np.percentile(finite_pixels, low_pct))
    p_high = float(np.percentile(finite_pixels, high_pct))

    if np.isclose(p_low, p_high):
        logger.warning(
            "Percentile range is degenerate (p%.1f=%.4f, p%.1f=%.4f). "
            "Image may be uniform — returning zeros.",
            low_pct, p_low, high_pct, p_high,
        )
        return np.zeros_like(image, dtype=np.float32)

    normed = (image - p_low) / (p_high - p_low)
    normed = np.clip(normed, 0.0, 1.0).astype(np.float32)

    # Preserve NaN mask from the original image
    normed[~np.isfinite(image)] = np.nan

    logger.debug(
        "Adaptive normalisation: p%.1f=%.6f, p%.1f=%.6f → clipped to [0, 1].",
        low_pct, p_low, high_pct, p_high,
    )
    return normed


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FITSLoader:
    """
    Non-destructive FITS reader for SPHEREx / SPARKLES images.

    Loads the science, uncertainty, and coverage planes into float32
    NumPy arrays.  The original pixel values are **never** zeroed,
    cropped, or overwritten — preserving all spectral absorption
    fingerprints (H₂O / ice features) for downstream science.

    Parameters
    ----------
    path : str or Path
        Path to a FITS file (`.fits`, `.fits.gz`).

    Attributes
    ----------
    science      : float32 ndarray (H, W) — raw SCI extension
    uncertainty  : float32 ndarray (H, W) or None — per-pixel σ
    coverage     : float32 ndarray (H, W) or None — exposure counts
    header       : astropy FITS header of the science extension
    wcs          : astropy WCS object (or None if WCS is absent)
    normalized   : float32 ndarray (H, W) — adaptively normalised [0,1]
    shape        : (height, width) of the science image
    n_bands      : number of spectral bands (102 for SPHEREx)
    meta         : dict with provenance metadata
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._validate_path()

        self.science:     np.ndarray
        self.uncertainty: Optional[np.ndarray] = None
        self.coverage:    Optional[np.ndarray] = None
        self.header:      fits.Header
        self.wcs:         Optional[WCS]        = None
        self.normalized:  np.ndarray
        self.meta:        Dict                 = {}

        self._load()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """(H, W) shape of the science image."""
        return self.science.shape

    @property
    def n_bands(self) -> int:
        """
        Number of spectral bands.
        Returns SPHEREX_N_BANDS (102) if NAXIS3 is absent or equals 102,
        otherwise returns the actual value from the FITS header.
        """
        return int(self.header.get("NAXIS3", SPHEREX_N_BANDS))

    @property
    def has_uncertainty(self) -> bool:
        return self.uncertainty is not None

    @property
    def has_coverage(self) -> bool:
        return self.coverage is not None

    # ------------------------------------------------------------------
    # Internal loading logic
    # ------------------------------------------------------------------

    def _validate_path(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"FITS file not found: {self._path}")
        if self._path.suffix.lower() not in {".fits", ".fit", ".fts", ".gz"}:
            logger.warning(
                "File '%s' has an unexpected extension. Proceeding anyway.",
                self._path.name,
            )

    def _load(self) -> None:
        logger.info("Opening FITS file: %s", self._path)

        with fits.open(str(self._path), memmap=True, mode="readonly") as hdul:
            hdul.verify("silentfix")  # fix minor header issues in-place (non-destructive)
            self._log_hdu_summary(hdul)

            # --- Science extension -------------------------------------------
            sci = _extract_extension(hdul, *_EXT_SCIENCE)
            if sci is None:
                # Last resort: primary HDU data
                if hdul[0].data is not None:
                    sci = _safe_cast_float32(hdul[0].data)
                    logger.warning(
                        "SCI extension missing — using Primary HDU data instead."
                    )
                else:
                    raise ValueError(
                        f"No science data found in any HDU of '{self._path}'."
                    )

            self.science = sci

            # --- Uncertainty & coverage (optional) ---------------------------
            self.uncertainty = _extract_extension(hdul, *_EXT_UNCERTAINTY)
            self.coverage    = _extract_extension(hdul, *_EXT_COVERAGE)

            # --- Header & WCS ------------------------------------------------
            sci_hdu = (
                hdul["SCI"] if "SCI" in hdul
                else hdul[1] if len(hdul) > 1
                else hdul[0]
            )
            self.header = sci_hdu.header.copy()

            try:
                self.wcs = WCS(self.header, naxis=2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WCS construction failed (%s) — WCS set to None.", exc)
                self.wcs = None

        # --- Adaptive normalisation (after file is closed) -------------------
        self.normalized = adaptive_normalise(self.science)

        # --- Provenance metadata ---------------------------------------------
        self.meta = {
            "path":          str(self._path),
            "shape":         self.shape,
            "n_bands":       self.n_bands,
            "has_unc":       self.has_uncertainty,
            "has_cov":       self.has_coverage,
            "nan_fraction":  float(np.isnan(self.science).mean()),
            "sci_min":       float(np.nanmin(self.science)),
            "sci_max":       float(np.nanmax(self.science)),
            "sci_median":    float(np.nanmedian(self.science)),
        }
        logger.info(
            "Loaded | shape=%s  NaN=%.2f%%  sci ∈ [%.4f, %.4f]",
            self.shape,
            self.meta["nan_fraction"] * 100,
            self.meta["sci_min"],
            self.meta["sci_max"],
        )

    def _log_hdu_summary(self, hdul: fits.HDUList) -> None:
        summary = [(i, h.name, type(h).__name__) for i, h in enumerate(hdul)]
        logger.debug("HDU summary: %s", summary)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_band(self, band_index: int) -> np.ndarray:
        """
        Extract a single spectral band slice from a 3-D data cube.

        Parameters
        ----------
        band_index : int  (0-based, must be < n_bands)

        Returns
        -------
        float32 ndarray of shape (H, W)
        """
        if self.science.ndim != 3:
            raise ValueError(
                f"get_band() requires a 3-D cube, but science array is {self.science.ndim}-D."
            )
        if not (0 <= band_index < self.science.shape[0]):
            raise IndexError(
                f"band_index {band_index} out of range for cube with "
                f"{self.science.shape[0]} bands."
            )
        return self.science[band_index].copy()

    def to_dict(self) -> Dict:
        """Return a dictionary of all data planes + metadata."""
        return {
            "science":     self.science,
            "uncertainty": self.uncertainty,
            "coverage":    self.coverage,
            "normalized":  self.normalized,
            "header":      self.header,
            "wcs":         self.wcs,
            "meta":        self.meta,
        }

    def __repr__(self) -> str:
        return (
            f"FITSLoader("
            f"path={self._path.name!r}, "
            f"shape={self.shape}, "
            f"has_unc={self.has_uncertainty}, "
            f"has_cov={self.has_coverage})"
        )
