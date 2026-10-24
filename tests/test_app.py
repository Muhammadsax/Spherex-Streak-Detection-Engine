"""
test_app.py — Unit & Integration tests for Phase 5: Streamlit Web Dashboard
===========================================================================
Verifies:
  1. Synthetic SPHEREx sample generator (faint streak, train-track presets).
  2. Model loading and pipeline integration.
  3. Side-by-side visualization and astronomical stretch routines.
  4. Non-destructive end-to-end data processing workflow.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from app.streamlit_app import (
    generate_synthetic_spherex_sample,
    load_deepstreaks_model,
)
from spherex_streak.detection.pqm import ProbabilityQualityMaskEngine
from spherex_streak.io.fits_loader import adaptive_normalise
from spherex_streak.metrics import (
    compute_effective_variance,
    compute_photometric_weights,
    verify_non_destructive_integrity,
)
from spherex_streak.preprocessing import AdaptivePreprocessor
from spherex_streak.utils.visualization import (
    render_side_by_side_figure,
    render_spectral_profile_plot,
    stretch_astronomical_image,
)


class TestAppComponents:
    def test_synthetic_spherex_sample_generator(self):
        """Verifies synthetic sample generation for faint streak and train-track presets."""
        for preset in ["faint_streak", "train_track"]:
            sci, unc, gt_mask = generate_synthetic_spherex_sample(preset_type=preset, size=64)

            assert sci.shape == (64, 64)
            assert unc.shape == (64, 64)
            assert gt_mask.shape == (64, 64)
            assert sci.dtype == np.float32
            assert unc.dtype == np.float32
            assert gt_mask.dtype == np.float32
            assert np.all(np.isfinite(sci))
            assert np.all(unc > 0.0)
            assert np.sum(gt_mask) > 0  # Contains simulated streak pixels

    def test_load_deepstreaks_model(self):
        """Verifies model loading in evaluation mode."""
        model = load_deepstreaks_model()
        assert not model.training
        assert callable(model.predict_pqm)

    def test_stretch_astronomical_image(self):
        """Tests astronomical stretch options: asinh, log, linear."""
        img = np.random.uniform(50.0, 500.0, size=(50, 50)).astype(np.float32)

        for stretch in ["asinh", "log", "linear"]:
            stretched = stretch_astronomical_image(img, stretch_type=stretch)
            assert stretched.shape == (50, 50)
            assert stretched.dtype == np.float32
            assert float(np.min(stretched)) >= 0.0
            assert float(np.max(stretched)) <= 1.0

    def test_render_side_by_side_figure(self):
        """Tests 4-panel figure rendering."""
        h, w = 40, 40
        sci = np.random.uniform(10.0, 100.0, size=(h, w)).astype(np.float32)
        pqm = np.zeros((h, w), dtype=np.float32)
        pqm[20, 10:30] = 0.95
        var = np.full((h, w), 2.0, dtype=np.float32)
        eff_var = compute_effective_variance(var, pqm)
        weights = compute_photometric_weights(eff_var)

        fig = render_side_by_side_figure(
            raw_science=sci,
            pqm_map=pqm,
            effective_variance=eff_var,
            photometric_weights=weights,
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # 4 image panels + colorbars
        plt.close(fig)

    def test_render_spectral_profile_plot(self):
        """Tests 102-band spectral line profile visualization."""
        bands = np.linspace(0.75, 5.0, 102, dtype=np.float32)
        raw_spec = np.full(102, 100.0, dtype=np.float32)
        raw_spec[50] = 60.0  # Dip
        eff_spec = raw_spec.copy()
        pqm_spec = np.zeros(102, dtype=np.float32)

        fig = render_spectral_profile_plot(bands, raw_spec, eff_spec, pqm_spec)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_end_to_end_dashboard_pipeline(self):
        """Full integration test simulating an interactive user session."""
        sci, unc, gt_mask = generate_synthetic_spherex_sample("faint_streak", size=64)
        snapshot = sci.copy()
        variance = np.square(unc)

        # 1. Preprocess
        preprocessor = AdaptivePreprocessor(box_size=16, scales=[7, 15], num_angles=6)
        prep = preprocessor.process(sci)

        # 2. Model Inference
        model = load_deepstreaks_model()
        norm = adaptive_normalise(sci)
        raw_pqm = model.predict_pqm(norm)

        # 3. PQM Engine & Inverse-Variance Weighting
        engine = ProbabilityQualityMaskEngine(display_threshold=0.5)
        pqm = engine.validate_pqm(raw_pqm)
        eff_var = compute_effective_variance(variance, pqm)
        weights = compute_photometric_weights(eff_var)

        # 4. Strict Non-Destructive Integrity Guarantee
        report = verify_non_destructive_integrity(
            raw_science_flux=sci,
            baseline_flux_snapshot=snapshot,
            variance=variance,
            pqm=pqm,
        )

        assert report.raw_flux_unaltered is True
        np.testing.assert_array_equal(sci, snapshot)
        assert report.clean_variance_preserved is True
        assert report.streak_variance_amplified is True
