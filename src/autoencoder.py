"""
CNN Autoencoder - Unsupervised Anomaly Detection
==================================================
Trained ONLY on normal seafloor patches (no debris). Learns to reconstruct
natural textures (ripples, rock clusters) well. When it is fed a patch
containing an object it has never seen the reconstruction error spikes,
flagging a "potential anomaly" independent of the supervised YOLO detector.
This catches novel debris shapes YOLO wasn't trained on.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

PATCH_SIZE = 64


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(True),   # 32x32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(True),  # 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),  # 8x8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class PatchDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)


def train_autoencoder(patch_dir, out_path, epochs=15, batch_size=16, lr=1e-3, device="cpu"):
    ds = PatchDataset(patch_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dl:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        avg = total_loss / len(ds)
        print(f"[autoencoder] epoch {epoch+1}/{epochs} recon_loss={avg:.5f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved autoencoder weights to {out_path}")
    return model


def load_autoencoder(weights_path, device="cpu"):
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def anomaly_score(model, gray_patch, device="cpu"):
    """Returns reconstruction-error based anomaly score in [0, 1]-ish range
    (not strictly bounded, but empirically small for normal seafloor)."""
    patch = cv2.resize(gray_patch, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32) / 255.0
    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    recon = model(t)
    err = torch.mean((recon - t) ** 2).item()
    return err


if __name__ == "__main__":
    import sys
    patch_dir = sys.argv[1] if len(sys.argv) > 1 else "data/normal_patches"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "models/autoencoder.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    train_autoencoder(patch_dir, out_path)
