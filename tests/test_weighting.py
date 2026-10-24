"""
test_weighting.py — Unit tests for Inverse-Variance Weighting Module
====================================================================
Tests effective variance mathematical law:
    sigma^2_effective = sigma^2 / w
"""

import numpy as np
import pytest
import torch

from spherex_streak.weighting.inv_variance import (
    compute_effective_variance,
    compute_photometric_weights,
    verify_non_destructive_integrity,
)


class TestWeightingFormulas:
    def test_clean_pixels_unaltered_variance(self):
        variance = np.array([1.5, 3.2, 10.0], dtype=np.float32)
        pqm = np.zeros(3, dtype=np.float32)

        eff_var = compute_effective_variance(variance, pqm)
        np.testing.assert_allclose(eff_var, variance, rtol=1e-6)

    def test_effective_variance_scaling(self):
        variance = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        pqm = np.array([0.0, 0.5, 0.8], dtype=np.float32)

        eff_var = compute_effective_variance(variance, pqm)
        expected = np.array([2.0, 4.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(eff_var, expected, rtol=1e-5)

    def test_photometric_weights_inverse_property(self):
        eff_var = np.array([1.0, 2.0, 4.0, 100.0], dtype=np.float32)
        w = compute_photometric_weights(eff_var)

        expected_w = np.array([1.0, 0.5, 0.25, 0.01], dtype=np.float32)
        np.testing.assert_allclose(w, expected_w, rtol=1e-5)

    def test_torch_tensor_compatibility(self):
        variance = torch.tensor([1.0, 4.0], dtype=torch.float32)
        pqm = torch.tensor([0.0, 0.75], dtype=torch.float32)

        eff_var = compute_effective_variance(variance, pqm)
        expected = torch.tensor([1.0, 16.0], dtype=torch.float32)
        torch.testing.assert_close(eff_var, expected)
