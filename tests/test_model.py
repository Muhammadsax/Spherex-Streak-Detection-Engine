"""
test_model.py — Unit tests for Phase 3: DeepStreaks Model & Focal Loss
======================================================================
Strict mathematical tests covering:
  1. Shape consistency and multi-scale resolution preservation.
  2. Continuous Probability Quality Mask (PQM) bounds: Float32 in [0.0, 1.0].
  3. Non-destructive predict_pqm inference guardrails.
  4. Focal Loss mathematical properties (gamma >= 2.0 down-weighting easy background,
     loss sensitivity to low-SNR faint signals, reduction modes, logits vs probabilities).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spherex_streak.model import DeepStreaksCNN, DilatedResidualBlock, FocalLoss, build_deepstreaks_model


# ---------------------------------------------------------------------------
# 1. Tests for DeepStreaksCNN Architecture & PQM Output
# ---------------------------------------------------------------------------

class TestDeepStreaksCNN:
    @pytest.fixture
    def model(self):
        """Construct a lightweight DeepStreaksCNN instance for testing."""
        return build_deepstreaks_model(in_channels=1, base_filters=16, num_blocks=3)

    def test_output_shape_matches_input_dimensions(self, model):
        """Ensure FITS image spatial dimensions (N x M) are preserved exactly without downsampling."""
        for h, w in [(64, 64), (128, 96), (73, 51)]:
            x = torch.randn(2, 1, h, w, dtype=torch.float32)
            out = model(x)
            assert out.shape == (2, 1, h, w), f"Expected shape (2, 1, {h}, {w}), got {out.shape}"
            assert out.dtype == torch.float32

    def test_continuous_pqm_range_strictly_bounded(self, model):
        """PQM values must strictly fall within [0.0, 1.0] via final Sigmoid activation."""
        x = torch.randn(4, 1, 64, 64, dtype=torch.float32) * 50.0  # Wide dynamic range
        pqm = model(x)

        min_val = float(pqm.min().detach())
        max_val = float(pqm.max().detach())
        assert min_val >= 0.0, f"PQM contains negative values: {min_val}"
        assert max_val <= 1.0, f"PQM exceeds upper bound 1.0: {max_val}"

    def test_predict_pqm_numpy_interface(self, model):
        """predict_pqm must accept a 2D float32 numpy array and return an unmutated (H, W) array."""
        h, w = 80, 80
        sample_img = np.random.uniform(0.0, 100.0, size=(h, w)).astype(np.float32)
        orig_copy = sample_img.copy()

        pqm_map = model.predict_pqm(sample_img)

        # Output properties
        assert isinstance(pqm_map, np.ndarray)
        assert pqm_map.shape == (h, w)
        assert pqm_map.dtype == np.float32
        assert np.all(pqm_map >= 0.0)
        assert np.all(pqm_map <= 1.0)

        # Non-destructive guarantee: original image array was not mutated
        np.testing.assert_array_equal(sample_img, orig_copy)

    def test_predict_pqm_handles_nan_and_inf(self, model):
        """Must handle corrupted pixels gracefully without propagating NaNs to the PQM."""
        sample_img = np.ones((64, 64), dtype=np.float32) * 10.0
        sample_img[10, 10] = np.nan
        sample_img[20, 20] = np.inf
        sample_img[30, 30] = -np.inf

        pqm_map = model.predict_pqm(sample_img)
        assert not np.isnan(pqm_map).any(), "PQM contains NaN values"
        assert not np.isinf(pqm_map).any(), "PQM contains Inf values"
        assert np.all(pqm_map >= 0.0) and np.all(pqm_map <= 1.0)

    def test_return_logits_mode(self, model):
        """Verifies that return_logits=True yields logits whose sigmoid matches standard forward."""
        x = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(x, return_logits=True)
            pqm_direct = model(x, return_logits=False)
            pqm_from_logits = torch.sigmoid(logits)

        torch.testing.assert_close(pqm_direct, pqm_from_logits, rtol=1e-5, atol=1e-5)

    def test_dilated_residual_block_gradient_flow(self):
        """Verifies backprop and gradient flow through DilatedResidualBlock."""
        block = DilatedResidualBlock(channels=16, dilation=2)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 2. Tests for Focal Loss Formulation & Technical Requirements
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_focal_loss_initialization_guardrails(self):
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        assert criterion.gamma == 2.0
        assert criterion.alpha == 0.25

        with pytest.raises(ValueError, match="non-negative"):
            FocalLoss(gamma=-1.0)

    def test_easy_background_downweighting_with_gamma_ge_2(self):
        """
        Mathematical proof:
        When gamma >= 2.0, well-classified background pixels (p ~ 0.02, y = 0)
        must produce substantially smaller loss than standard BCE (gamma = 0.0).
        """
        # Easy background: true label is 0, model predicts 0.02 (p_t = 0.98)
        preds = torch.tensor([[0.02]], dtype=torch.float32)
        targets = torch.tensor([[0.0]], dtype=torch.float32)

        loss_bce = FocalLoss(gamma=0.0, alpha=-1.0, reduction="mean")(preds, targets)
        loss_focal_2 = FocalLoss(gamma=2.0, alpha=-1.0, reduction="mean")(preds, targets)
        loss_focal_4 = FocalLoss(gamma=4.0, alpha=-1.0, reduction="mean")(preds, targets)

        # Loss should decay dramatically with higher gamma for easy background
        # (1 - 0.98)^2 = (0.02)^2 = 0.0004 factor!
        ratio_2 = float(loss_focal_2 / loss_bce)
        assert ratio_2 < 0.001, f"Expected >1000x suppression for easy background, got ratio {ratio_2}"
        assert float(loss_focal_4) < float(loss_focal_2)

    def test_hard_faint_streak_gradient_preservation(self):
        """
        Faint streaks near noise floor (p ~ 0.25, y = 1.0):
        The loss should remain significantly active to drive parameter updates.
        """
        preds = torch.tensor([[0.25]], dtype=torch.float32)
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        loss_focal = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")(preds, targets)
        assert float(loss_focal) > 0.05, "Faint streak loss collapsed unexpectedly"

    def test_perfect_predictions_yield_minimal_loss(self):
        """When predictions match targets, focal loss approaches zero."""
        preds = torch.tensor([[0.9999, 0.0001]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        criterion = FocalLoss(gamma=2.0, reduction="mean")
        loss = criterion(preds, targets)
        assert float(loss) < 1e-4

    def test_reduction_modes(self):
        preds = torch.tensor([[0.8, 0.1], [0.3, 0.9]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss_none = FocalLoss(reduction="none")(preds, targets)
        loss_mean = FocalLoss(reduction="mean")(preds, targets)
        loss_sum = FocalLoss(reduction="sum")(preds, targets)

        assert loss_none.shape == (2, 2)
        assert pytest.approx(float(loss_mean)) == float(loss_none.mean())
        assert pytest.approx(float(loss_sum)) == float(loss_none.sum())

    def test_is_logits_mode_consistency(self):
        """is_logits=True with raw logits should match is_logits=False with sigmoid(logits)."""
        logits = torch.randn(4, 1, 16, 16, dtype=torch.float32)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()

        criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
        loss_logits = criterion(logits, targets, is_logits=True)

        probs = torch.sigmoid(logits)
        loss_probs = criterion(probs, targets, is_logits=False)

        # Should be almost identical (within numerical precision)
        assert pytest.approx(float(loss_logits), rel=1e-4) == float(loss_probs)

    def test_end_to_end_training_step(self):
        """Test full training step: model forward -> focal loss -> backward -> optimizer step."""
        model = build_deepstreaks_model(in_channels=1, base_filters=16, num_blocks=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = FocalLoss(gamma=2.0, alpha=0.25)

        # Batch with 1 background image and 1 image with a streak
        x = torch.randn(2, 1, 32, 32, dtype=torch.float32)
        y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
        y[0, 0, 16, 8:24] = 1.0  # Streak in item 0

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss.detach())
        assert float(loss.detach()) > 0.0
