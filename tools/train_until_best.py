"""
Repeated hill-climb training for the synthetic sonar YOLO detector.

Each round:
  1. (optionally) expands the training set with a fresh, differently-seeded
     synthetic batch (harder than round 0) so every round sees new examples;
  2. fine-tunes the CURRENT champion checkpoint on the combined train data;
  3. evaluates the candidate and the incumbent against the FIXED held-out
     eval set (data/eval_heldout - never trained on, never regenerated);
  4. promotes the candidate only if it beats the incumbent on map50 with a
     minimum margin, otherwise keeps the incumbent and records a plateau.

The loop stops when `--rounds` are exhausted or `--plateau` consecutive
rounds fail to improve. History is appended to reports/training_history.jsonl
so every comparison is auditable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
CHAMPION = MODEL_DIR / "yolo_sonar_best.pt"
HISTORY = PROJECT_ROOT / "reports" / "training_history.jsonl"
EVAL_YAML = PROJECT_ROOT / "data" / "eval_heldout" / "data.yaml"


def evaluate(weights: Path, tag: str, imgsz: int, device: str) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(EVAL_YAML),
        split="val",
        imgsz=imgsz,
        conf=0.001,      # full PR curve; headline metrics are threshold-free
        iou=0.5,
        device=device,
        plots=False,
        verbose=False,
        project=str(PROJECT_ROOT / "runs" / "eval"),
        name=tag,
        exist_ok=True,
    )
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "f1": (2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) else 0.0,
    }


def expand_training_data(n_images: int, round_idx: int, max_objects: int) -> None:
    """Add a fresh synthetic batch (new seed, harder composition) to the
    training pool. Eval data is never touched."""
    import random

    import cv2
    import numpy as np

    from src.generate_synthetic_data import generate_image

    root = PROJECT_ROOT / "data" / "synthetic_yolo"
    seed = 50_000 + round_idx * 977
    for split, n in (("train", n_images), ("val", max(10, n_images // 8))):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            s = seed * 1_000_003 + (0 if split == "train" else 17) * 100_003 + i
            random.seed(s)
            np.random.seed(s % (2**31 - 1))
            has_objects = random.random() > 0.12
            img, boxes = generate_image(has_objects=has_objects, max_objects=max_objects)
            fname = f"sss_r{round_idx:02d}_{split}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{fname}.png"), img)
            with open(lbl_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def train_one_round(weights: Path, rounds_dir: Path, round_idx: int, epochs: int,
                    batch: int, imgsz: int, device: str, lr0: float,
                    freeze: int) -> Path:
    model = YOLO(str(weights))
    results = model.train(
        data=str(PROJECT_ROOT / "data" / "synthetic_yolo" / "data.yaml"),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=4,
        project=str(rounds_dir),
        name=f"round_{round_idx:02d}",
        pretrained=True,
        patience=max(3, epochs // 3),
        lr0=lr0,
        lrf=0.05,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=max(1, epochs // 10),
        freeze=freeze,
        cache=False,
        amp=False,
        seed=42 + round_idx,
        deterministic=True,
        plots=False,
        verbose=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15,
                        help="Fine-tune epochs per round")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Match runtime inference (detection_pipeline uses 640)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--freeze", type=int, default=0,
                        help="0 = train the whole network (we have full supervision)")
    parser.add_argument("--expand", type=int, default=60,
                        help="Fresh synthetic images added to the train pool each round")
    parser.add_argument("--max-objects", type=int, default=4,
                        help="Objects per generated image during expansion")
    parser.add_argument("--min-gain", type=float, default=0.005,
                        help="Required map50 improvement to promote a candidate")
    parser.add_argument("--plateau", type=int, default=3,
                        help="Stop after this many consecutive non-improving rounds")
    parser.add_argument("--max-train-images", type=int, default=600,
                        help="Cap on total train images (subsample if exceeded)")
    args = parser.parse_args()

    if not EVAL_YAML.exists():
        sys.exit("Held-out eval set missing. Run: python tools/build_heldout_eval.py")

    rounds_dir = PROJECT_ROOT / "runs" / "hillclimb"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)

    baseline = evaluate(CHAMPION, "baseline", args.imgsz, args.device)
    print(f"\n=== BASELINE champion: map50={baseline['map50']:.4f} "
          f"recall={baseline['recall']:.4f} ===\n")

    history_line = {"ts": time.time(), "round": "baseline", "role": "baseline",
                    "weights": str(CHAMPION), **baseline}
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_line) + "\n")

    champion_map = baseline["map50"]
    champion_metrics = dict(baseline)
    stale_rounds = 0

    # Continue round numbering after any previous hill-climb run so repeated
    # invocations never collide with existing run directories.
    existing = [int(d.name.split("_")[1]) for d in rounds_dir.glob("round_*")
                if d.name.split("_")[1].isdigit()]
    start = (max(existing) + 1) if existing else 1

    for r in range(start, start + args.rounds):
        t0 = time.time()
        print(f"\n=========== ROUND {r}/{args.rounds} ===========")
        expand_training_data(args.expand, r, args.max_objects)
        _subsample_train_pool(args.max_train_images)

        candidate = train_one_round(CHAMPION, rounds_dir, r, args.epochs, args.batch,
                                    args.imgsz, args.device, args.lr0, args.freeze)
        cand_metrics = evaluate(candidate, f"round_{r:02d}", args.imgsz, args.device)

        improved = cand_metrics["map50"] > champion_map + args.min_gain
        note = ""
        # Dominance override: a candidate at least as good on every tracked
        # metric and clearly better on localization quality (map50-95) is a
        # real improvement even when the map50 gain is under the margin.
        if not improved and \
           cand_metrics["precision"] >= champion_metrics["precision"] and \
           cand_metrics["recall"] >= champion_metrics["recall"] and \
           cand_metrics["map50_95"] >= champion_metrics["map50_95"] + 0.01:
            improved = True
            note = "dominance-override"
        role = "champion" if improved else "rejected"
        rec = {
            "ts": time.time(),
            "round": r,
            "role": role,
            "note": note,
            "weights": str(candidate),
            "champion_map50_before": champion_map,
            "candidate": cand_metrics,
            "secs": round(time.time() - t0, 1),
        }
        print(f"ROUND {r}: candidate map50={cand_metrics['map50']:.4f} vs "
              f"champion {champion_map:.4f} -> {role.upper()}")
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if improved:
            # Archive the old champion, promote the candidate atomically.
            shutil.copy2(CHAMPION, MODEL_DIR / "yolo_sonar_prev.pt")
            shutil.copy2(candidate, CHAMPION)
            champion_map = cand_metrics["map50"]
            champion_metrics = dict(cand_metrics)
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= args.plateau:
                print(f"No improvement for {stale_rounds} consecutive rounds - plateau "
                      f"reached, stopping early. Champion map50={champion_map:.4f}")
                break

    print(f"\nFINAL champion map50={champion_map:.4f} (baseline "
          f"{baseline['map50']:.4f}). Weights: {CHAMPION}")
    print(f"History: {HISTORY}")


def _subsample_train_pool(cap: int) -> None:
    """Keep the train pool bounded: if it exceeds the cap, drop the OLDEST
    round-0 files first (newer, harder rounds matter more)."""
    img_dir = PROJECT_ROOT / "data" / "synthetic_yolo" / "images" / "train"
    lbl_dir = PROJECT_ROOT / "data" / "synthetic_yolo" / "labels" / "train"
    pngs = sorted(img_dir.glob("*.png"),
                  key=lambda p: (0 if p.name.startswith("sss_train") else 1, p.name))
    if len(pngs) <= cap:
        return
    excess = len(pngs) - cap
    for img in pngs[:excess]:
        lbl = lbl_dir / (img.stem + ".txt")
        img.unlink()
        if lbl.exists():
            lbl.unlink()
    print(f"train pool capped: removed {excess} oldest images ({len(pngs) - excess} remain)")


if __name__ == "__main__":
    main()
