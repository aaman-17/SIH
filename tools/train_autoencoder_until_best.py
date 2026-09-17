"""
Train the CNN autoencoder until its anomaly-separation metric stops improving.

Quality metric
--------------
The old trainer printed only reconstruction MSE, which says nothing about
whether the autoencoder actually SEPARATES anomalies from normal seafloor.
This version:
  - holds out a random 15% of normal patches as validation (never trained on);
  - after each epoch, computes reconstruction errors for val-normal patches
    AND for synthetic "object" patches (crop objects out of generated
    sonar-style images);
  - computes AUROC(normal vs object) - 0.5 is blind guessing, 1.0 is perfect
    separation;
  - keeps the best-AUROC weights, and stops early after `--patience`
    epochs without improvement (instead of always running all epochs and
    keeping whatever the last epoch produced).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder, PATCH_SIZE  # noqa: E402


class PatchDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)


@torch.no_grad()
def _batch_recon_error(model, paths, device, batch_size=32) -> np.ndarray:
    ds = PatchDataset(paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errs = []
    model.eval()
    for batch in dl:
        batch = batch.to(device)
        recon = model(batch)
        err = (recon - batch).pow(2).mean(dim=(1, 2, 3))
        errs.extend(err.cpu().tolist())
    return np.asarray(errs, dtype=np.float32)


def _auroc(normal_errs: np.ndarray, object_errs: np.ndarray) -> float:
    """Rank-based AUROC of the separation between the two error samples."""
    if len(normal_errs) == 0 or len(object_errs) == 0:
        return 0.5
    scores = np.concatenate([normal_errs, object_errs])
    labels = np.concatenate([np.zeros(len(normal_errs)), np.ones(len(object_errs))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels == 1].sum()
    n_pos, n_neg = len(object_errs), len(normal_errs)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _collect_object_patches(n: int, tmp: Path, seed: int) -> list[Path]:
    """Crop object-containing patches from generated images so we can measure
    whether the autoencoder finds man-made shapes 'surprising'. Patches get the
    same denoise+CLAHE treatment the runtime pipeline applies before the AE
    scan, so the eval distribution matches inference conditions."""
    from src.generate_synthetic_data import generate_image
    from src.preprocessing import enhance_contrast_clahe, denoise_speckle

    rng = random.Random(seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        s = seed * 1_000_003 + attempts
        random.seed(s)
        np.random.seed(s % (2**31 - 1))
        img, boxes = generate_image(has_objects=True, max_objects=2)
        if not boxes:
            continue
        cls_id, cx, cy, w, h = boxes[rng.randrange(len(boxes))]
        H, W = img.shape[:2]
        x1 = max(0, int(cx * W - w * W / 2) - 6)
        y1 = max(0, int(cy * H - h * H / 2) - 6)
        x2 = min(W, int(cx * W + w * W / 2) + 6)
        y2 = min(H, int(cy * H + h * H / 2) + 6)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0 or min(patch.shape[:2]) < 12:
            continue
        patch = enhance_contrast_clahe(denoise_speckle(patch))
        p = tmp / f"obj_{len(out):05d}.png"
        cv2.imwrite(str(p), patch)
        out.append(p)
    return out


def train_until_best(patch_dir: Path, out_path: Path, epochs: int = 60,
                     batch_size: int = 16, lr: float = 1e-3, device: str = "cpu",
                     val_frac: float = 0.15, patience: int = 10,
                     n_eval_objects: int = 150, seed: int = 99) -> dict:
    all_paths = sorted(Path(patch_dir).glob("*.png"))
    if len(all_paths) < 40:
        raise SystemExit(f"Too few normal patches in {patch_dir} ({len(all_paths)})")

    rng = random.Random(seed)
    shuffled = all_paths[:]
    rng.shuffle(shuffled)
    n_val = max(8, int(len(shuffled) * val_frac))
    val_paths, train_paths = shuffled[:n_val], shuffled[n_val:]
    print(f"autoencoder: {len(train_paths)} train / {len(val_paths)} val-normal patches")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        obj_paths = _collect_object_patches(n_eval_objects, tmp, seed + 1)
        print(f"autoencoder: {len(obj_paths)} synthetic object patches for AUROC eval")

        dl = DataLoader(PatchDataset(train_paths), batch_size=batch_size, shuffle=True)
        model = ConvAutoencoder().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
        loss_fn = nn.MSELoss()

        best = {"auroc": 0.0, "epoch": -1}
        best_path = tmp / "best_ae.pt"
        stale = 0

        for epoch in range(epochs):
            model.train()
            total = 0.0
            for batch in dl:
                batch = batch.to(device)
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            sched.step()
            avg = total / len(train_paths)

            normal_errs = _batch_recon_error(model, val_paths, device)
            obj_errs = _batch_recon_error(model, obj_paths, device)
            auroc = _auroc(normal_errs, obj_errs)
            marker = ""
            if auroc > best["auroc"] + 0.002:
                best = {"auroc": auroc, "epoch": epoch,
                        "val_normal_mse": float(normal_errs.mean()),
                        "val_object_mse": float(obj_errs.mean())}
                torch.save(model.state_dict(), best_path)
                stale = 0
                marker = "  <- best"
            else:
                stale += 1
            print(f"[autoencoder] epoch {epoch+1}/{epochs} recon={avg:.5f} "
                  f"AUROC={auroc:.4f}{marker}")
            if stale >= patience:
                print(f"[autoencoder] early stop: no AUROC gain for {patience} epochs")
                break

        if best["epoch"] < 0:
            raise SystemExit("Autoencoder never improved on a random-guess AUROC - aborting.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_path, out_path)
        print(f"Saved best autoencoder (epoch {best['epoch']+1}, AUROC={best['auroc']:.4f}) "
              f"-> {out_path}")

        meta = {"best_epoch": best["epoch"] + 1, "auroc": best["auroc"],
                "val_normal_mse": best.get("val_normal_mse"),
                "val_object_mse": best.get("val_object_mse"),
                "train_patches": len(train_paths), "val_patches": len(val_paths)}
        Path(str(out_path) + ".metrics.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                         encoding="utf-8")
        return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch_dir", nargs="?", default="data/normal_patches")
    parser.add_argument("out_path", nargs="?", default="models/autoencoder.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    meta = train_until_best(Path(args.patch_dir), Path(args.out_path),
                            epochs=args.epochs, batch_size=args.batch_size,
                            patience=args.patience, device=args.device)
    print(json.dumps(meta, indent=2))
