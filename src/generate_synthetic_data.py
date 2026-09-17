"""
Synthetic Side-Scan Sonar (SSS) Data Generator
================================================
Real side-scan sonar datasets with debris annotations are not publicly
available / downloadable in this environment. This module procedurally
generates realistic-*looking* side-scan sonar imagery so the full pipeline
(preprocessing -> detection -> anomaly scoring -> geotagging -> dashboard)
can be built, trained, and demonstrated end-to-end.

Swap this out for real .xtf sonar logs (via pyxtf) once you have field data --
the rest of the pipeline does not need to change.

Simulated physics / imaging characteristics:
  - Nadir gap (dark strip directly under the tow-fish)
  - Along-track speckle noise (Rayleigh-ish, typical of acoustic backscatter)
  - Sand ripples / seafloor texture (natural clutter)
  - Debris objects (cylinders/pipes, rectangular ghost-net clumps, shipwreck-like
    blobs) rendered with a bright leading-edge highlight + a dark acoustic
    shadow trailing away from the sonar track - this highlight+shadow pair is
    the single most important visual cue in real SSS imagery.
  - Occasional data dropout stripes (vehicle heave/roll motion artifacts)
"""

import os
import json
import random
import numpy as np
import cv2

random.seed(42)
np.random.seed(42)

IMG_W, IMG_H = 640, 640
CLASSES = ["debris_net", "pipe_cylinder", "shipwreck_debris"]


def _rayleigh_speckle(h, w, scale=18):
    return np.random.rayleigh(scale, size=(h, w)).astype(np.float32)


def _seafloor_texture(h, w):
    """Low-frequency ripples/rock texture as a base backscatter field."""
    base = np.zeros((h, w), dtype=np.float32)
    # sand ripples: several sine bands at slight angles
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    for _ in range(random.randint(2, 4)):
        angle = random.uniform(-0.3, 0.3)
        freq = random.uniform(0.03, 0.09)
        phase = random.uniform(0, np.pi * 2)
        amp = random.uniform(8, 20)
        base += amp * np.sin(freq * (xx * np.cos(angle) + yy * np.sin(angle)) + phase)
    # blotchy rock clusters (natural false-positive sources)
    for _ in range(random.randint(3, 7)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(15, 45)
        yy2, xx2 = np.ogrid[:h, :w]
        mask = (xx2 - cx) ** 2 + (yy2 - cy) ** 2 <= r * r
        base[mask] += random.uniform(15, 35)
    return base


def _add_nadir_gap(img):
    h, w = img.shape
    gap_w = random.randint(18, 30)
    cx = w // 2 + random.randint(-10, 10)
    img[:, max(0, cx - gap_w // 2):cx + gap_w // 2] *= 0.15
    return img


def _add_dropout_stripes(img):
    h, w = img.shape
    for _ in range(random.randint(0, 2)):
        y = random.randint(0, h - 4)
        thickness = random.randint(1, 3)
        img[y:y + thickness, :] *= random.uniform(0.1, 0.4)
    return img


def _draw_object_with_shadow(img, obj_type, cx, cy):
    """Render a bright highlight + dark acoustic shadow, mimicking real SSS
    returns from a raised object insonified from one side."""
    h, w = img.shape
    shadow_dir = 1 if cx < w / 2 else -1  # shadow falls away from sonar track (roughly toward image edge)

    if obj_type == "pipe_cylinder":
        length = random.randint(50, 110)
        thickness = random.randint(6, 12)
        angle = random.uniform(0, np.pi)
        x2 = int(cx + length / 2 * np.cos(angle))
        y2 = int(cy + length / 2 * np.sin(angle))
        x1 = int(cx - length / 2 * np.cos(angle))
        y1 = int(cy - length / 2 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), 220, thickness)
        shadow_len = int(length * 0.9)
        cv2.line(img, (x1 + shadow_dir * 4, y1 + 6), (x1 + shadow_dir * shadow_len, y1 + 6), 5, thickness + 4)
        bw, bh = abs(x2 - x1) + thickness * 2, abs(y2 - y1) + thickness * 2
        bx, by = min(x1, x2) - thickness, min(y1, y2) - thickness

    elif obj_type == "debris_net":
        # irregular tangled clump -> several overlapping blobs
        w_box = random.randint(40, 90)
        h_box = random.randint(40, 90)
        for _ in range(random.randint(4, 8)):
            ox = cx + random.randint(-w_box // 2, w_box // 2)
            oy = cy + random.randint(-h_box // 2, h_box // 2)
            rad = random.randint(6, 16)
            cv2.circle(img, (ox, oy), rad, random.randint(180, 240), -1)
        shadow_w, shadow_h = int(w_box * 0.8), int(h_box * 0.4)
        sx = cx + shadow_dir * w_box // 2
        cv2.ellipse(img, (sx, cy + h_box // 3), (shadow_w // 2, shadow_h // 2), 0, 0, 360, 4, -1)
        bx, by = cx - w_box // 2, cy - h_box // 2
        bw, bh = w_box, h_box

    else:  # shipwreck_debris - larger elongated bright structure w/ long shadow
        length = random.randint(90, 160)
        width = random.randint(20, 40)
        angle = random.uniform(0, np.pi)
        rect = ((cx, cy), (length, width), np.degrees(angle))
        box = cv2.boxPoints(rect).astype(int)
        cv2.fillPoly(img, [box], random.randint(200, 250))
        shadow_len = int(length * 1.1)
        sx2 = int(cx + shadow_dir * shadow_len / 2 * np.cos(angle))
        sy2 = int(cy + shadow_dir * shadow_len / 2 * np.sin(angle))
        cv2.line(img, (cx, cy), (sx2, sy2), 6, width + 6)
        bw, bh = length + 20, width + 20
        bx, by = cx - bw // 2, cy - bh // 2

    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, w - bx), min(bh, h - by)
    return bx, by, bw, bh


def generate_image(has_objects=True, max_objects=3):
    base = 60 + _seafloor_texture(IMG_H, IMG_W)
    speckle = _rayleigh_speckle(IMG_H, IMG_W)
    img = base + speckle
    img = cv2.GaussianBlur(img, (3, 3), 0)

    boxes = []  # (class_id, cx, cy, w, h) normalized
    if has_objects:
        n = random.randint(1, max_objects)
        for _ in range(n):
            cls_id = random.randint(0, len(CLASSES) - 1)
            cx = random.randint(60, IMG_W - 60)
            cy = random.randint(60, IMG_H - 60)
            bx, by, bw, bh = _draw_object_with_shadow(img, CLASSES[cls_id], cx, cy)
            if bw > 4 and bh > 4:
                boxes.append((cls_id, (bx + bw / 2) / IMG_W, (by + bh / 2) / IMG_H, bw / IMG_W, bh / IMG_H))

    img = _add_nadir_gap(img)
    img = _add_dropout_stripes(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, boxes


def build_dataset(root, n_train=240, n_val=60):
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            has_objects = random.random() > 0.15  # ~15% pure-background (hard negatives)
            img, boxes = generate_image(has_objects=has_objects)
            fname = f"sss_{split}_{i:04d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), img)
            with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                for cls_id, cx, cy, w, h in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    # dataset yaml for ultralytics
    yaml_content = f"""path: {root}
train: images/train
val: images/val
names:
  0: debris_net
  1: pipe_cylinder
  2: shipwreck_debris
"""
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"Synthetic dataset written to {root}")


def build_normal_patches(out_dir, n=300, patch=64):
    """Background-only patches (no objects) for autoencoder anomaly training."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    while count < n:
        img, boxes = generate_image(has_objects=False)
        for _ in range(4):
            x = random.randint(0, IMG_W - patch)
            y = random.randint(0, IMG_H - patch)
            crop = img[y:y + patch, x:x + patch]
            cv2.imwrite(os.path.join(out_dir, f"normal_{count:05d}.png"), crop)
            count += 1
            if count >= n:
                break
    print(f"{count} normal patches written to {out_dir}")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_yolo"
    build_dataset(root)
    build_normal_patches("data/normal_patches")
