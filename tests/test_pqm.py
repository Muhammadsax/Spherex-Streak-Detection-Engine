"""
test_pqm.py — Unit tests for ProbabilityQualityMaskEngine
==========================================================
Tests PQM validation, summary statistics computation, display masks,
and coupling with Inverse-Variance Weighting.
"""

import numpy as np
import pytest

from spherex_streak.detection.pqm import ProbabilityQualityMaskEngine, PQMStatistics


class TestProbabilityQualityMaskEngine:
    @pytest.fixture
    def engine(self):
        return ProbabilityQualityMaskEngine(display_threshold=0.5)

    def test_validate_pqm_bounds_and_dtype(self, engine):
        raw = np.array([[-0.5, 0.3], [1.5, np.nan]], dtype=np.float64)
        clean = engine.validate_pqm(raw)

        assert clean.dtype == np.float32
        assert clean[0, 0] == 0.0      # Clamped from -0.5
        assert clean[0, 1] == 0.3
        assert clean[1, 0] == 1.0      # Clamped from 1.5
        assert clean[1, 1] == 0.0      # NaN replaced with clean 0.0

    def test_compute_statistics(self, engine):
        pqm = np.zeros((100, 100), dtype=np.float32)
        pqm[10:20, 10:20] = 0.90  # 100 pixels with heavy contamination

        stats = engine.compute_statistics(pqm)
        assert isinstance(stats, PQMStatistics)
        assert stats.shape == (100, 100)
        assert stats.contamination_fraction == pytest.approx(0.01)
        assert stats.heavy_contamination_count == 100
        assert stats.clean_fraction == pytest.approx(0.99)

    def test_generate_display_mask(self, engine):
        pqm = np.array([[0.1, 0.4], [0.6, 0.95]], dtype=np.float32)
        display_mask = engine.generate_display_mask(pqm, threshold=0.5)

        assert display_mask.shape == (2, 2)
        assert display_mask.dtype == bool
        np.testing.assert_array_equal(display_mask, np.array([[False, False], [True, True]]))

    def test_apply_inverse_variance_weighting(self, engine):
        variance = np.full((4, 4), 5.0, dtype=np.float32)
        pqm = np.zeros((4, 4), dtype=np.float32)
        pqm[0, 0] = 0.5  # 50% contaminated

        eff_var, weights = engine.apply_inverse_variance_weighting(variance, pqm)

        assert eff_var[0, 0] == pytest.approx(10.0, rel=1e-5)
        assert eff_var[1, 1] == pytest.approx(5.0, rel=1e-5)
        assert weights[0, 0] == pytest.approx(0.1, rel=1e-5)
        assert weights[1, 1] == pytest.approx(0.2, rel=1e-5)
