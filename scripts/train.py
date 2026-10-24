"""
scripts/train.py — DeepStreaks Model Training Pipeline
======================================================
Trains the DeepStreaksCNN model using Combined Focal Loss (gamma >= 2.0)
and Soft IoU Loss targeting Soft IoU >= 0.95.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spherex_streak.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from spherex_streak.metrics import CombinedFocalSoftIoULoss, soft_iou_score
from spherex_streak.model import build_deepstreaks_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyntheticStreakDataset(Dataset):
    """Generates on-the-fly synthetic SPHEREx streak images for training."""

    def __init__(self, num_samples: int = 100, size: int = 128) -> None:
        self.num_samples = num_samples
        self.size = size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx)
        size = self.size
        # Sky background + noise
        bkg = rng.uniform(50.0, 150.0)
        sigma = rng.uniform(3.0, 7.0)
        img = rng.normal(bkg, sigma, (size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.float32)

        # 80% of samples contain satellite streaks
        if rng.uniform() < 0.8:
            y0 = rng.integers(10, size - 10)
            x_start = rng.integers(5, size // 3)
            x_end = rng.integers(2 * size // 3, size - 5)
            streak_amp = rng.uniform(8.0, 35.0)
            img[y0, x_start:x_end] += streak_amp
            mask[y0, x_start:x_end] = 1.0

        # Normalize to [0, 1]
        norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

        x_tensor = torch.from_numpy(norm).unsqueeze(0).float()
        y_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return x_tensor, y_tensor


def train(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_path: str = "checkpoints/deepstreaks_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = SyntheticStreakDataset(num_samples=64, size=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_deepstreaks_model(in_channels=1, base_filters=16, num_blocks=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = CombinedFocalSoftIoULoss(gamma=2.0, alpha=0.25, iou_weight=1.0)

    best_iou = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        ious = []

        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            with torch.no_grad():
                ious.append(soft_iou_score(preds, y_b))

        avg_loss = total_loss / len(loader)
        avg_iou = float(np.mean(ious))
        logger.info("Epoch [%d/%d] - Loss: %.4f - Mean Soft IoU: %.4f", epoch, epochs, avg_loss, avg_iou)

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best checkpoint (Soft IoU: %.4f) -> %s", best_iou, save_path)

    logger.info("Training complete. Peak Soft IoU: %.4f", best_iou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
