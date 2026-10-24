"""
test_metrics.py — Unit tests for Phase 4: Evaluation Metrics & Soft IoU Validation
===================================================================================
Strict mathematical tests covering:
  1. Soft IoU continuous Float32 metric bounds [0.0, 1.0] and target Soft IoU >= 0.95.
  2. Differentiable SoftIoULoss and CombinedFocalSoftIoULoss gradient backpropagation.
  3. Mathematical law of Inverse-Variance Weighting: sigma^2_eff = sigma^2 / w.
  4. Photometric co-addition weights W = 1 / sigma^2_eff.
  5. Non-destructive conservation of raw spectrophotometric data & H2O ice absorption features.
"""

import numpy as np
import pytest
import torch

from spherex_streak.metrics import (
    CombinedFocalSoftIoULoss,
    SoftIoULoss,
    SpectralPreservationReport,
    compute_effective_variance,
    compute_photometric_weights,
    soft_iou_score,
    verify_non_destructive_integrity,
)


# ---------------------------------------------------------------------------
# 1. Tests for Soft IoU Continuous Metric
# ---------------------------------------------------------------------------

class TestSoftIoUMetric:
    def test_soft_iou_perfect_match(self):
        """Identical predictions and targets must yield Soft IoU == 1.0."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:30, 20:30] = 0.85  # Continuous streak confidence

        score = soft_iou_score(mask, mask)
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_soft_iou_target_benchmark_ge_095(self):
        """
        Technical Requirement: Target Soft IoU >= 0.95.
        High-fidelity predictions closely tracking ground truth must score >= 0.95.
        """
        ground_truth = np.zeros((128, 128), dtype=np.float32)
        ground_truth[60:68, 10:118] = 1.0  # Linear satellite streak

        # Slight Gaussian noise around boundary simulating high-accuracy model (>= 95% fidelity)
        predictions = ground_truth.copy()
        rng = np.random.default_rng(42)
        noise = rng.uniform(-0.02, 0.02, size=ground_truth.shape).astype(np.float32)
        predictions = np.clip(predictions + noise, 0.0, 1.0)

        score = soft_iou_score(predictions, ground_truth)
        assert score >= 0.95, f"Expected Soft IoU >= 0.95, got {score:.4f}"

    def test_soft_iou_disjoint_masks(self):
        """Completely disjoint masks must yield Soft IoU == 0.0."""
        mask_a = np.zeros((50, 50), dtype=np.float32)
        mask_b = np.zeros((50, 50), dtype=np.float32)

        mask_a[5:15, 5:15] = 1.0
        mask_b[30:40, 30:40] = 1.0

        score = soft_iou_score(mask_a, mask_b)
        assert pytest.approx(score, abs=1e-5) == 0.0

    def test_soft_iou_both_empty_masks(self):
        """If an image is completely clean (no streaks in ground truth or prediction), IoU is 1.0."""
        clean_pred = np.zeros((64, 64), dtype=np.float32)
        clean_gt = np.zeros((64, 64), dtype=np.float32)

        score = soft_iou_score(clean_pred, clean_gt)
        assert score == 1.0

    def test_soft_iou_torch_tensor_support(self):
        """Validates torch.Tensor input with identical results to NumPy."""
        pred_np = np.random.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
        gt_np = np.random.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)

        pred_t = torch.from_numpy(pred_np)
        gt_t = torch.from_numpy(gt_np)

        score_np = soft_iou_score(pred_np, gt_np)
        score_t = soft_iou_score(pred_t, gt_t)

        assert pytest.approx(score_np, abs=1e-5) == score_t


# ---------------------------------------------------------------------------
# 2. Tests for Differentiable SoftIoULoss & Combined Loss
# ---------------------------------------------------------------------------

class TestSoftIoULoss:
    def test_soft_iou_loss_bounds_and_zero_on_perfect(self):
        loss_fn = SoftIoULoss()
        target = torch.ones((2, 1, 32, 32), dtype=torch.float32)
        perfect_pred = torch.ones((2, 1, 32, 32), dtype=torch.float32)

        loss = loss_fn(perfect_pred, target)
        assert pytest.approx(float(loss), abs=1e-4) == 0.0

    def test_soft_iou_loss_gradient_backprop(self):
        loss_fn = SoftIoULoss()
        pred = torch.rand((2, 1, 32, 32), dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert float(loss.detach()) >= 0.0

    def test_combined_focal_soft_iou_loss(self):
        loss_fn = CombinedFocalSoftIoULoss(gamma=2.0, alpha=0.25, iou_weight=1.5)
        raw_logits = torch.randn((2, 1, 32, 32), requires_grad=True)
        pred = torch.sigmoid(raw_logits)
        target = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        target[0, 0, 10:20, 10:20] = 1.0

        loss = loss_fn(pred, target)
        loss.backward()

        assert raw_logits.grad is not None
        assert float(loss.detach()) > 0.0


# ---------------------------------------------------------------------------
# 3. Tests for Mathematical Law of Inverse-Variance Weighting
# ---------------------------------------------------------------------------

class TestInverseVarianceWeighting:
    def test_mathematical_law_exact_formula(self):
        """
        Verifies:
            sigma^2_effective = sigma^2 / w
            where w = 1.0 - PQM
        """
        variance = np.full((10, 10), 4.0, dtype=np.float32)  # sigma^2 = 4.0
        pqm = np.zeros((10, 10), dtype=np.float32)

        # 1. Clean region: PQM = 0.0 -> w = 1.0 -> sigma^2_eff = 4.0 / 1.0 = 4.0
        eff_var = compute_effective_variance(variance, pqm)
        np.testing.assert_allclose(eff_var, 4.0, rtol=1e-5)

        # 2. 50% contaminated region: PQM = 0.5 -> w = 0.5 -> sigma^2_eff = 4.0 / 0.5 = 8.0
        pqm_half = np.full((10, 10), 0.5, dtype=np.float32)
        eff_var_half = compute_effective_variance(variance, pqm_half)
        np.testing.assert_allclose(eff_var_half, 8.0, rtol=1e-5)

        # 3. 75% contaminated region: PQM = 0.75 -> w = 0.25 -> sigma^2_eff = 4.0 / 0.25 = 16.0
        pqm_75 = np.full((10, 10), 0.75, dtype=np.float32)
        eff_var_75 = compute_effective_variance(variance, pqm_75)
        np.testing.assert_allclose(eff_var_75, 16.0, rtol=1e-5)

    def test_torch_tensor_inverse_variance_weighting(self):
        var_t = torch.tensor([4.0, 4.0], dtype=torch.float32)
        pqm_t = torch.tensor([0.0, 0.5], dtype=torch.float32)

        eff_var_t = compute_effective_variance(var_t, pqm_t)
        expected = torch.tensor([4.0, 8.0], dtype=torch.float32)
        torch.testing.assert_close(eff_var_t, expected)

    def test_photometric_weights_downweight_streak_pixels(self):
        """
        W = 1.0 / sigma^2_effective
        Contaminated pixels must have near-zero weights in co-addition.
        """
        variance = np.full(4, 2.0, dtype=np.float32)
        pqm = np.array([0.0, 0.1, 0.9, 0.999], dtype=np.float32)

        eff_var = compute_effective_variance(variance, pqm, eps=1e-4)
        weights = compute_photometric_weights(eff_var)

        # Clean pixel (PQM=0.0): weight == 1 / 2.0 = 0.5
        assert pytest.approx(weights[0], abs=1e-4) == 0.5
        # Heavy streak pixel (PQM=0.999): weight must be vanishingly small
        assert weights[3] < 0.001
        assert weights[3] < weights[0] * 0.01


# ---------------------------------------------------------------------------
# 4. Tests for Non-Destructive Spectrophotometric Conservation
# ---------------------------------------------------------------------------

class TestSpectrophotometricIntegrity:
    def test_raw_flux_never_mutated_or_zeroed(self):
        """Core NASA Space Apps guarantee: raw science FITS flux must remain 100% unaltered."""
        rng = np.random.default_rng(99)
        raw_flux = rng.normal(150.0, 15.0, size=(64, 64)).astype(np.float32)
        raw_snapshot = raw_flux.copy()
        variance = np.full((64, 64), 2.5, dtype=np.float32)

        # Apply PQM across horizontal streak
        pqm = np.zeros((64, 64), dtype=np.float32)
        pqm[32, 10:50] = 0.95

        report = verify_non_destructive_integrity(
            raw_science_flux=raw_flux,
            baseline_flux_snapshot=raw_snapshot,
            variance=variance,
            pqm=pqm,
        )

        assert report.raw_flux_unaltered is True
        assert report.clean_variance_preserved is True
        assert report.streak_variance_amplified is True
        # Verify byte-for-byte exact equality
        np.testing.assert_array_equal(raw_flux, raw_snapshot)

    def test_h2o_ice_absorption_feature_preservation(self):
        """
        SPHEREx 102 Spectral Bands Validation:
        Simulate H2O ice absorption signature at 3.05 um (dip in flux across bands 45-55).
        Verify that absorption depth and spectral line profile are preserved intact.
        """
        bands = np.linspace(0.75, 5.0, 102, dtype=np.float32)
        # Baseline continuum emission
        continuum = np.full(102, 100.0, dtype=np.float32)
        # H2O ice 3.05 um absorption feature (Gaussian dip centered at band 50, depth 40%)
        ice_absorption = 40.0 * np.exp(-((bands - 3.05)**2) / (2 * 0.15**2))
        synthetic_spectrum = continuum - ice_absorption

        snapshot = synthetic_spectrum.copy()
        variance = np.full(102, 1.0, dtype=np.float32)
        pqm = np.zeros(102, dtype=np.float32)  # Clean astronomical target

        report = verify_non_destructive_integrity(
            raw_science_flux=synthetic_spectrum,
            baseline_flux_snapshot=snapshot,
            variance=variance,
            pqm=pqm,
            spectral_profile=synthetic_spectrum,
        )

        assert report.absorption_integrity_score == 1.0
        assert report.raw_flux_unaltered is True
        # Confirm absorption depth was preserved exactly identically to baseline snapshot
        measured_depth = float(np.max(synthetic_spectrum) - np.min(synthetic_spectrum))
        expected_depth = float(np.max(snapshot) - np.min(snapshot))
        assert pytest.approx(measured_depth, rel=1e-6) == expected_depth
