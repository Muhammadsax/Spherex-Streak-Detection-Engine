"""
src/fits_loader.py — Top-level alias for spherex_streak.io.fits_loader
"""

from spherex_streak.io.fits_loader import FITSLoader, SPHEREX_N_BANDS, adaptive_normalise

__all__ = ["FITSLoader", "adaptive_normalise", "SPHEREX_N_BANDS"]
