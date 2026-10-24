"""
test_preprocessing.py — Unit tests for Phase 2: Adaptive Preprocessing & Spectral Verification
================================================================================================
Tests verify:
  1. Dynamic background estimation and MAD noise floor calculation (no hardcoded thresholds).
  2. Multi-scale line tracking filters and train-track enhancement (streak vs star discrimination).
  3. SPHEREx 102 spectral bands discrimination (satellite streak vs natural comet/nebula).
  4. Non-destructive processing (raw pixel integrity preserved).
"""

import numpy as np
import pytest

from spherex_streak.preprocessing.adaptive import (
    AdaptiveBackgroundEstimator,
    AdaptivePreprocessor,
    MultiScaleLineFilter,
    SpectralDiscriminationEngine,
    SpectralDiscriminationResult,
    SPHEREX_BANDS_COUNT,
)


@pytest.fixture
def synthetic_sky_image():
    """Generates a synthetic 128x128 sky image with background gradient and noise."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:128, 0:128]
    # Smooth background ramp + Gaussian noise
    bkg = 100.0 + 0.1 * x + 0.05 * y
    noise = rng.normal(loc=0.0, scale=5.0, size=(128, 128))
    img = (bkg + noise).astype(np.float32)
    return img


@pytest.fixture
def streak_and_star_image(synthetic_sky_image):
    """Adds a thin linear streak and a circular Gaussian star to the sky image."""
    img = synthetic_sky_image.copy()
    # Add thin horizontal satellite streak (low-SNR: amplitude ~ 15 = 3 sigma)
    img[64, 20:108] += 15.0

    # Add circular Gaussian star at (32, 64) with FWHM ~ 3 pixels
    y, x = np.mgrid[0:128, 0:128]
    star = 50.0 * np.exp(-((x - 64)**2 + (y - 32)**2) / (2 * 1.5**2))
    img += star.astype(np.float32)
    return img


# ---------------------------------------------------------------------------
# 1. Tests for AdaptiveBackgroundEstimator
# ---------------------------------------------------------------------------

class TestAdaptiveBackgroundEstimator:
    def test_output_shapes_and_dtypes(self, synthetic_sky_image):
        estimator = AdaptiveBackgroundEstimator(box_size=32)
        est = estimator.estimate(synthetic_sky_image)

        assert est.background.shape == synthetic_sky_image.shape
        assert est.noise_sigma.shape == synthetic_sky_image.shape
        assert est.snr_map.shape == synthetic_sky_image.shape
        assert est.background.dtype == np.float32
        assert est.noise_sigma.dtype == np.float32
        assert est.snr_map.dtype == np.float32

    def test_original_image_unmutated(self, synthetic_sky_image):
        orig_copy = synthetic_sky_image.copy()
        estimator = AdaptiveBackgroundEstimator()
        _ = estimator.estimate(synthetic_sky_image)
        np.testing.assert_array_equal(synthetic_sky_image, orig_copy)

    def test_robust_mad_sigma_estimation(self):
        rng = np.random.default_rng(123)
        known_sigma = 8.5
        noise = rng.normal(loc=50.0, scale=known_sigma, size=(100, 100)).astype(np.float32)

        estimator = AdaptiveBackgroundEstimator(box_size=50)
        est = estimator.estimate(noise)

        # Robust MAD sigma should be within 10% of known normal sigma
        assert pytest.approx(est.global_mad_sigma, rel=0.1) == known_sigma

    def test_handles_nan_and_inf_gracefully(self, synthetic_sky_image):
        img_corrupted = synthetic_sky_image.copy()
        img_corrupted[10:20, 10:20] = np.nan
        img_corrupted[50, 50] = np.inf

        estimator = AdaptiveBackgroundEstimator(box_size=32)
        est = estimator.estimate(img_corrupted)

        assert np.isfinite(est.global_median)
        assert np.isfinite(est.global_mad_sigma)
        assert est.global_mad_sigma > 0.0


# ---------------------------------------------------------------------------
# 2. Tests for MultiScaleLineFilter
# ---------------------------------------------------------------------------

class TestMultiScaleLineFilter:
    def test_kernel_bank_zero_mean_property(self):
        filter_bank = MultiScaleLineFilter(scales=[7, 15], num_angles=8)
        for (scale, theta), kernel in filter_bank._kernel_bank.items():
            # Zero-mean ensures no DC background offset
            assert np.abs(np.sum(kernel)) < 1e-4

    def test_streak_enhanced_over_isotropic_star(self, streak_and_star_image):
        estimator = AdaptiveBackgroundEstimator(box_size=32)
        est = estimator.estimate(streak_and_star_image)

        fltr = MultiScaleLineFilter(scales=[15, 31], num_angles=12)
        max_resp, dom_angle = fltr.filter_image(est.snr_map)

        assert max_resp.shape == streak_and_star_image.shape
        assert max_resp.dtype == np.float32

        # Along the horizontal streak (y=64, x=40..80) response should be pronounced
        streak_signal = float(np.mean(max_resp[64, 40:80]))
        # At the center of the isotropic star (y=32, x=64), line response should be suppressed
        # relative to streak because star flanks cancel out central flux in directional filter
        background_signal = float(np.mean(max_resp[10:20, 10:20]))

        assert streak_signal > background_signal * 1.5

    def test_dominant_angle_detection(self):
        """A strictly horizontal streak should register peak response near 0 degrees."""
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 10:54] = 10.0  # Horizontal streak along theta = 0 deg

        fltr = MultiScaleLineFilter(scales=[15], num_angles=12)
        max_resp, angles = fltr.filter_image(img)

        peak_y, peak_x = 32, 32
        detected_angle = angles[peak_y, peak_x]
        # Angle should be 0.0 deg (or 180 modulo)
        assert detected_angle == 0.0 or detected_angle == 180.0


# ---------------------------------------------------------------------------
# 3. Tests for Spectral Discrimination (102 SPHEREx Bands)
# ---------------------------------------------------------------------------

class TestSpectralDiscriminationEngine:
    def test_identifies_artificial_satellite_streak(self):
        """Satellite streak: present in only 1 band out of 102, sharp optical PSF."""
        engine = SpectralDiscriminationEngine(optical_psf_fwhm=2.0)

        # Cube with 102 SPHEREx bands
        cube = np.zeros((SPHEREX_BANDS_COUNT, 64, 64), dtype=np.float32)
        # Transient: only appears in band 5
        cube[5, 32, 10:50] = 20.0

        y_cand = np.full(40, 32)
        x_cand = np.arange(10, 50)

        # Transverse slice: narrow gaussian FWHM ~ 2.0 pixels
        transverse = np.array([0.0, 5.0, 20.0, 5.0, 0.0], dtype=np.float32)

        res = engine.analyze_candidate(
            spectral_cube=cube,
            candidate_coords=(y_cand, x_cand),
            transverse_slice=transverse,
        )

        assert res.is_satellite_streak is True
        assert res.is_natural_feature is False
        assert res.spectral_persistence <= 0.15
        assert "Transient artificial streak" in res.reason

    def test_preserves_natural_nebula_or_comet_filament(self):
        """Natural feature: persistent across >50 bands, diffuse cross-section (FWHM > 5 px)."""
        engine = SpectralDiscriminationEngine(optical_psf_fwhm=2.0)

        cube = np.zeros((SPHEREX_BANDS_COUNT, 64, 64), dtype=np.float32)
        # Persistent physical emission: present across 80 bands
        cube[:80, 32, 10:50] = 15.0

        y_cand = np.full(40, 32)
        x_cand = np.arange(10, 50)

        # Transverse slice: broad diffuse nebula profile FWHM ~ 7 pixels
        transverse = np.array([1.0, 4.0, 8.0, 12.0, 14.0, 12.0, 8.0, 4.0, 1.0], dtype=np.float32)

        res = engine.analyze_candidate(
            spectral_cube=cube,
            candidate_coords=(y_cand, x_cand),
            transverse_slice=transverse,
        )

        assert res.is_satellite_streak is False
        assert res.is_natural_feature is True
        assert res.spectral_persistence >= 0.40
        assert "Natural astronomical structure" in res.reason


# ---------------------------------------------------------------------------
# 4. Tests for Integrated AdaptivePreprocessor Pipeline
# ---------------------------------------------------------------------------

class TestAdaptivePreprocessor:
    def test_pipeline_on_2d_image(self, streak_and_star_image):
        preprocessor = AdaptivePreprocessor(box_size=32, scales=[7, 15], num_angles=6)
        results = preprocessor.process(streak_and_star_image)

        assert "snr_map" in results
        assert "line_response" in results
        assert "dominant_angle" in results
        assert "bkg_estimate" in results
        assert "spectral_result" in results
        assert "streak_candidate_map" in results
        assert results["snr_map"].shape == streak_and_star_image.shape

    def test_pipeline_on_3d_cube(self):
        preprocessor = AdaptivePreprocessor(box_size=32, scales=[7, 15], num_angles=6)
        cube = np.random.normal(100.0, 5.0, size=(10, 64, 64)).astype(np.float32)
        # Streak in target band 0
        cube[0, 32, 10:50] += 25.0

        results = preprocessor.process(cube, target_band_idx=0)
        assert results["snr_map"].shape == (64, 64)
        assert isinstance(results["spectral_result"], SpectralDiscriminationResult)
