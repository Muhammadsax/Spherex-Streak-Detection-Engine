"""
test_fits_loader.py — Unit tests for Phase 1: FITSLoader
=========================================================
Tests run WITHOUT a real FITS file by constructing synthetic
in-memory FITS HDULists so the suite works in CI with no data.
"""

import io
import numpy as np
import pytest
from astropy.io import fits

from spherex_streak.io.fits_loader import FITSLoader, adaptive_normalise, _safe_cast_float32


# ---------------------------------------------------------------------------
# Helpers — build synthetic FITS files in-memory
# ---------------------------------------------------------------------------

def _make_fits_bytes(
    sci_data: np.ndarray,
    unc_data: np.ndarray | None = None,
    cov_data: np.ndarray | None = None,
) -> bytes:
    """Create a minimal SPHEREx-style FITS file in memory and return its bytes."""
    primary = fits.PrimaryHDU()
    sci_hdu = fits.ImageHDU(data=sci_data.astype(np.float32), name="SCI")
    hdus = [primary, sci_hdu]
    if unc_data is not None:
        hdus.append(fits.ImageHDU(data=unc_data.astype(np.float32), name="UNC"))
    if cov_data is not None:
        hdus.append(fits.ImageHDU(data=cov_data.astype(np.float32), name="COV"))
    hdul = fits.HDUList(hdus)
    buf = io.BytesIO()
    hdul.writeto(buf)
    return buf.getvalue()


@pytest.fixture()
def simple_fits(tmp_path):
    """Write a simple 64×64 synthetic FITS file and return its path."""
    rng  = np.random.default_rng(42)
    sci  = rng.normal(loc=100.0, scale=10.0, size=(64, 64)).astype(np.float32)
    unc  = rng.uniform(0.5, 2.0,             size=(64, 64)).astype(np.float32)
    cov  = rng.integers(1, 10,               size=(64, 64)).astype(np.float32)

    path = tmp_path / "test_image.fits"
    path.write_bytes(_make_fits_bytes(sci, unc, cov))
    return path, sci, unc, cov


# ---------------------------------------------------------------------------
# _safe_cast_float32
# ---------------------------------------------------------------------------

class TestSafeCastFloat32:
    def test_inf_replaced_with_nan(self):
        arr = np.array([1.0, np.inf, -np.inf, 2.0])
        result = _safe_cast_float32(arr)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_output_dtype_is_float32(self):
        arr = np.array([1, 2, 3], dtype=np.float64)
        assert _safe_cast_float32(arr).dtype == np.float32

    def test_finite_values_unchanged(self):
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = _safe_cast_float32(arr)
        np.testing.assert_allclose(result, arr, rtol=1e-6)


# ---------------------------------------------------------------------------
# adaptive_normalise
# ---------------------------------------------------------------------------

class TestAdaptiveNormalise:
    def test_output_in_unit_range(self):
        rng = np.random.default_rng(0)
        img = rng.normal(0, 100, (128, 128)).astype(np.float32)
        normed = adaptive_normalise(img)
        assert float(np.nanmin(normed)) >= 0.0
        assert float(np.nanmax(normed)) <= 1.0

    def test_output_dtype_float32(self):
        img = np.ones((32, 32), dtype=np.float64) * 5.0
        # Uniform image → degenerate range → returns zeros
        normed = adaptive_normalise(img)
        assert normed.dtype == np.float32

    def test_nan_pixels_preserved(self):
        img = np.random.rand(64, 64).astype(np.float32)
        img[10, 10] = np.nan
        normed = adaptive_normalise(img)
        assert np.isnan(normed[10, 10])

    def test_all_nan_returns_zeros(self):
        img = np.full((32, 32), np.nan, dtype=np.float32)
        normed = adaptive_normalise(img)
        assert np.all(normed == 0.0)


# ---------------------------------------------------------------------------
# FITSLoader
# ---------------------------------------------------------------------------

class TestFITSLoader:
    def test_science_loaded_as_float32(self, simple_fits):
        path, sci, *_ = simple_fits
        loader = FITSLoader(path)
        assert loader.science.dtype == np.float32

    def test_science_shape_matches_input(self, simple_fits):
        path, sci, *_ = simple_fits
        loader = FITSLoader(path)
        assert loader.shape == sci.shape

    def test_uncertainty_loaded(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        assert loader.has_uncertainty
        assert loader.uncertainty.dtype == np.float32

    def test_coverage_loaded(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        assert loader.has_coverage
        assert loader.coverage.dtype == np.float32

    def test_normalized_in_unit_range(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        assert float(np.nanmin(loader.normalized)) >= 0.0
        assert float(np.nanmax(loader.normalized)) <= 1.0

    def test_original_science_not_mutated(self, simple_fits):
        """Core non-destructive guarantee: raw SCI pixels must not change."""
        path, original_sci, *_ = simple_fits
        loader = FITSLoader(path)
        np.testing.assert_array_equal(loader.science, original_sci)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FITSLoader(tmp_path / "nonexistent.fits")

    def test_meta_keys_present(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        for key in ("path", "shape", "nan_fraction", "sci_min", "sci_max", "sci_median"):
            assert key in loader.meta

    def test_to_dict_keys(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        d = loader.to_dict()
        assert "science" in d and "normalized" in d and "meta" in d

    def test_repr_contains_shape(self, simple_fits):
        path, *_ = simple_fits
        loader = FITSLoader(path)
        assert "64" in repr(loader)

    def test_get_band_raises_on_2d(self, simple_fits):
        """get_band() must raise for a 2-D (non-cube) image."""
        path, *_ = simple_fits
        loader = FITSLoader(path)
        with pytest.raises(ValueError, match="3-D"):
            loader.get_band(0)
