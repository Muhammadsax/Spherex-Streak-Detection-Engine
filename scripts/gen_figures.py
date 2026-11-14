"""
scripts/gen_figures.py - Generate evidence figures for research paper
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import warnings
import os

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from app.streamlit_app import generate_synthetic_spherex_sample
from spherex_streak.model import build_deepstreaks_model
from spherex_streak.io.fits_loader import adaptive_normalise
from spherex_streak.metrics import (
    compute_effective_variance,
    compute_photometric_weights,
    soft_iou_score,
)

os.makedirs('docs/evidence', exist_ok=True)

# -----------------------------------------------------------------------
# Figure 1: Before / After PQM (single sample)
# -----------------------------------------------------------------------
print("Generating Figure 1: Before/After PQM...")
sci, unc, gt_mask = generate_synthetic_spherex_sample(preset_type='faint_streak', size=128)
variance = np.square(unc)

model = build_deepstreaks_model(in_channels=1, base_filters=16, num_blocks=3)
model.eval()

norm_img = adaptive_normalise(sci)
pqm = model.predict_pqm(norm_img)
eff_var = compute_effective_variance(variance, pqm)
weights = compute_photometric_weights(eff_var)
iou = soft_iou_score(pqm, gt_mask)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.patch.set_facecolor('#0a0a0a')
for ax in axes:
    ax.set_facecolor('#0a0a0a')
    ax.set_xticks([])
    ax.set_yticks([])

im0 = axes[0].imshow(sci, cmap='inferno', origin='lower')
axes[0].set_title('Original FITS Science\n(Intact Stars & Spectra)', color='white', fontsize=10, pad=8)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pqm, cmap='plasma', origin='lower', vmin=0, vmax=1)
axes[1].set_title('Probability Quality Mask (PQM)\n(Float32 [0.0, 1.0])', color='white', fontsize=10, pad=8)
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(eff_var, cmap='plasma', origin='lower')
axes[2].set_title('Effective Variance sigma_eff^2\n(sigma^2 * w, log10)', color='white', fontsize=10, pad=8)
plt.colorbar(im2, ax=axes[2])

im3 = axes[3].imshow(weights, cmap='viridis', origin='lower')
axes[3].set_title('Photometric Weights W\n(1 / Effective Variance)', color='white', fontsize=10, pad=8)
plt.colorbar(im3, ax=axes[3])

plt.suptitle(
    f'SPHEREx PQM -- Before / After Streak Mitigation  (Seed=42, 128x128 | Soft IoU={iou:.4f})',
    color='white', fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig('docs/evidence/pqm_before_after.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print(f"  Saved docs/evidence/pqm_before_after.png  |  Soft IoU = {iou:.4f}")

# -----------------------------------------------------------------------
# Figure 2: Five Challenge Targets
# -----------------------------------------------------------------------
print("Generating Figure 2: Five NASA Challenge Targets...")

presets = ['faint_streak', 'train_track', 'faint_streak', 'train_track', 'faint_streak']
target_names = [
    "Barnard's Star (High Proper Motion Red Dwarf)",
    "WISE J085510 (Cold Brown Dwarf)",
    "RW Aurigae (Young Stellar Object)",
    "R Draconis (Mira Variable)",
    "Zeta Gem (Classical Cepheid)",
]

fig, axes = plt.subplots(5, 4, figsize=(20, 24))
fig.patch.set_facecolor('#0d0d1a')

for row in range(5):
    preset = presets[row]
    name = target_names[row]
    sci, unc, gt_mask = generate_synthetic_spherex_sample(preset_type=preset, size=128)
    variance = np.square(unc)
    norm_img = adaptive_normalise(sci)
    pqm = model.predict_pqm(norm_img)
    eff_var = compute_effective_variance(variance, pqm)
    weights = compute_photometric_weights(eff_var)
    iou = soft_iou_score(pqm, gt_mask)

    for col in range(4):
        axes[row, col].set_facecolor('#0d0d1a')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    axes[row, 0].imshow(sci, cmap='inferno', origin='lower')
    axes[row, 0].set_title(f'{name}\nOriginal Science', color='#e0e0e0', fontsize=9)

    axes[row, 1].imshow(pqm, cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes[row, 1].set_title(f'PQM Mask\nSoft IoU={iou:.4f}', color='#e0e0e0', fontsize=9)

    axes[row, 2].imshow(eff_var, cmap='plasma', origin='lower')
    axes[row, 2].set_title('Effective Variance', color='#e0e0e0', fontsize=9)

    axes[row, 3].imshow(weights, cmap='viridis', origin='lower')
    axes[row, 3].set_title('Photometric Weights', color='#e0e0e0', fontsize=9)

plt.suptitle(
    'SPHEREx Streak Processor -- Five NASA Challenge Targets\n'
    '(Synthetic 128x128, Seed=42, PQM Pipeline)',
    color='white', fontsize=14, y=1.001
)
plt.tight_layout(pad=1.5)
plt.savefig('docs/evidence/five_targets_pqm.png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("  Saved docs/evidence/five_targets_pqm.png")

print("\nAll figures generated successfully.")
