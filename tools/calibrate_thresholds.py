"""
Pick the best inference confidence threshold on the held-out eval set.

The dashboard default (0.15) was a guess. This tool runs the champion
detector over the held-out eval images at a range of confidence gates,
scores image-level recall/precision/F2 at each gate, and writes the best
operating point to reports/inference_calibration.json. Recall is weighted
double (F2) because in a debris survey a missed hazard costs much more
than a false alarm that a human reviewer dismisses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "eval_heldout"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "models" / "yolo_sonar_best.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports" / "inference_calibration.json")
    parser.add_argument("--iou-match", type=float, default=0.4,
                        help="IoU above which a prediction counts as a hit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    img_dir = EVAL_DIR / "images" / "val"
    lbl_dir = EVAL_DIR / "labels" / "val"
    images = sorted(img_dir.glob("*.png"))
    if not images:
        sys.exit(f"No eval images found in {img_dir}")

    model = YOLO(str(args.weights))
    gates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    # Collect raw predictions once at the lowest gate, then filter per sweep
    # point (saves one forward pass per gate).
    per_image = []
    for p in images:
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        results = model.predict(cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                                conf=gates[0], imgsz=640, verbose=False,
                                device=args.device)[0]
        preds = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append(([x1, y1, x2, y2], float(box.conf[0]),
                          model.names[int(box.cls[0].item())]))
        gt = []
        with open(lbl_dir / (p.stem + ".txt"), encoding="utf-8") as f:
            W, H = raw.shape[1], raw.shape[0]
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    c, cx, cy, w, h = parts
                    cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                    gt.append((int(c), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        per_image.append((preds, gt))

    sweep = []
    for gate in gates:
        tp = fp = 0
        n_gt = 0
        for preds, gt in per_image:
            kept = [p for p in preds if p[1] >= gate]
            matched = set()
            for pbox, pconf, pcls in kept:
                best_j, best_i = None, args.iou_match
                for j, (gc, gbox) in enumerate(gt):
                    if j in matched or gc != model_names_index(model, pcls):
                        continue
                    v = iou(pbox, gbox)
                    if v > best_i:
                        best_i, best_j = v, j
                if best_j is not None:
                    matched.add(best_j)
                    tp += 1
                else:
                    fp += 1
            n_gt += len(gt)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_gt if n_gt else 0.0
        f2 = (5 * prec * rec / (4 * prec + rec)) if (prec + rec) else 0.0
        sweep.append({"gate": gate, "precision": round(prec, 4),
                      "recall": round(rec, 4), "f2": round(f2, 4),
                      "true_positives": tp, "false_positives": fp,
                      "ground_truths": n_gt})

    best = max(sweep, key=lambda s: (s["f2"], s["recall"]))
    payload = {
        "weights": str(args.weights),
        "eval_set": str(EVAL_DIR),
        "iou_match": args.iou_match,
        "recommended_yolo_conf_gate": best["gate"],
        "recommended_min_report_pct": round(best["gate"] * 100 * 0.8, 1),
        "selection_criterion": "max image-level F2 (recall-weighted)",
        "best": best,
        "sweep": sweep,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sweep, indent=2))
    print(f"\nRecommended YOLO conf gate: {best['gate']} (F2={best['f2']}, "
          f"R={best['recall']}, P={best['precision']}) -> {args.output}")


def model_names_index(model, class_name: str) -> int:
    for k, v in model.names.items():
        if v == class_name:
            return k
    return -1


if __name__ == "__main__":
    main()
