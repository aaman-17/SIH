"""
Build a FIXED held-out evaluation set for the synthetic sonar detector.

Why this exists
---------------
The bundled YOLO checkpoint's "mAP50 ~= 0.93" was computed by Ultralytics on a
validation split drawn from the same seeded generator run as the training
images, so it measures training-distribution fit, not generalization. To train
repeatedly and keep a model only when it genuinely improves, we need an eval
set that training never sees, generated with a different seed and slightly
harder composition (more objects per image, more clutter, more hard
backgrounds).

The set is written once (deterministic per seed) to data/eval_heldout/ and is
NEVER regenerated or expanded by the training loop. Any future retraining that
wants to claim improvement must evaluate against this same folder.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_synthetic_data import generate_image  # noqa: E402

NAMES_YAML = "names:\n  0: debris_net\n  1: pipe_cylinder\n  2: shipwreck_debris\n"


def generate_split(out_dir: Path, prefix: str, n_images: int, seed: int,
                   max_objects: int, background_frac: float) -> None:
    img_dir = out_dir / "images" / "val"
    lbl_dir = out_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        # Per-image seeding keeps the whole set reproducible regardless of
        # generation order or how many images were requested.
        img_seed = seed * 1_000_003 + i
        random.seed(img_seed)
        np.random.seed(img_seed % (2**31 - 1))

        has_objects = random.random() > background_frac
        img, boxes = generate_image(has_objects=has_objects, max_objects=max_objects)
        fname = f"{prefix}_{i:04d}"
        cv2.imwrite(str(img_dir / f"{fname}.png"), img)
        with open(lbl_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
            for cls_id, cx, cy, w, h in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "eval_heldout")
    parser.add_argument("--n-images", type=int, default=150, help="Eval images (hard, with objects)")
    parser.add_argument("--n-background", type=int, default=30, help="Extra pure-background images")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Must stay different from every training seed used")
    parser.add_argument("--max-objects", type=int, default=5,
                        help="Training generator draws 1-3 objects; eval is harder")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing held-out set (invalidates past comparisons)")
    args = parser.parse_args()

    out_root = args.output
    marker = out_root / "MANIFEST.json"
    if marker.exists() and not args.force:
        print(f"Held-out eval set already exists at {out_root} - refusing to touch it "
              f"(use --force only if you intend to invalidate previous comparisons).")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    # Hard images (multiple objects) + pure-background hard negatives, in one
    # val split. Distinct prefixes avoid any filename collisions.
    generate_split(out_root, "evalhard", args.n_images, args.seed, args.max_objects, 0.0)
    generate_split(out_root, "evalbg", args.n_background, args.seed + 7, 1, 1.0)

    yaml_text = (f"path: {out_root.resolve().as_posix()}\n"
                 f"train: images/val\nval: images/val\n{NAMES_YAML}")
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = {
        "seed": args.seed,
        "hard_images": args.n_images,
        "background_images": args.n_background,
        "max_objects": args.max_objects,
        "note": "never train on this set",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out eval set written: {args.n_images + args.n_background} images -> {out_root}")


if __name__ == "__main__":
    main()
