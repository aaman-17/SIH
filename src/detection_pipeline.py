"""
Detection & Confidence Fusion Pipeline
========================================
Combines two independent signals, exactly as in the tech-stack diagram:

  1. YOLO11-Nano (supervised)  -> known object classes + box + base confidence
  2. CNN Autoencoder (unsupervised) -> reconstruction-error anomaly score for
     regions the YOLO detector did NOT already claim - catches novel /
     unlabeled debris shapes.

A noise-filtering stage suppresses likely false positives from natural
acoustic shadows / rock clusters using simple heuristics (aspect ratio,
edge-density, size sanity checks) before assigning a final 0-100% confidence.

IMPORTANT - anomaly thresholding is PER-IMAGE ADAPTIVE, not a fixed constant.
An earlier version used one global reconstruction-error threshold calibrated
against synthetic training data. That worked on synthetic images but broke
badly on real sonar (in testing, one real image produced 2000+ overlapping
"anomaly" boxes) because real acoustic noise statistics differ from the
synthetic approximation. Instead, each image's own grid-cell score
distribution is used to find *outliers relative to that image's own
background*, via a robust median + MAD z-score - self-calibrating regardless
of whether the absolute noise level matches synthetic training data.
"""

import os
import cv2
import numpy as np
import torch
from scipy import ndimage

from src.preprocessing import preprocess
from src.autoencoder import load_autoencoder, anomaly_score, PATCH_SIZE
from src.detection_quality import score_frame

GRID_STRIDE = 48
ANOMALY_Z_THRESHOLD = 4.5   # robust z-score (median/MAD-based) above which a
                             # grid cell counts as an outlier relative to
                             # THIS image's own background - not an absolute
                             # constant, so it self-calibrates per image.
MIN_ABS_ERROR_FLOOR = 0.0015  # guards against flagging trivial noise when an
                               # image's background is nearly perfectly flat
                               # (MAD near zero would otherwise make even tiny
                               # deviations look like huge z-scores)

# Long sonar waterfalls need local windows. A full 1728x5616 strip makes a
# small real target occupy too few model pixels and encourages oversized generic
# boxes. Overlap preserves targets crossing window boundaries.
YOLO_TILE_SIZE = 1024
YOLO_TILE_OVERLAP = 0.25


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter + 1e-6
    return inter / union


def _noise_filter_score(gray, x1, y1, x2, y2):
    """Heuristic natural-clutter suppression. Returns a multiplier in (0,1]
    applied to the raw confidence - lower for boxes that look like smooth
    rock blobs / uniform shadow rather than a hard man-made edge."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.3
    crop = gray[y1:y2, x1:x2]

    edges = cv2.Canny(crop, 50, 150)
    edge_density = edges.mean() / 255.0

    h, w = crop.shape
    aspect = min(h, w) / max(h, w)
    contrast = crop.std()

    score = 0.4 + 0.35 * min(edge_density * 4, 1.0) + 0.15 * aspect + 0.10 * min(contrast / 40, 1.0)
    return float(np.clip(score, 0.25, 1.0))


def _predict_yolo_window(model, gray_window, conf, offset_x=0, offset_y=0):
    bgr = cv2.cvtColor(gray_window, cv2.COLOR_GRAY2BGR)
    results = model.predict(bgr, conf=conf, verbose=False, imgsz=640)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        base_conf = float(box.conf[0].item())
        cls_name = model.names[cls_id]
        dets.append({
            "source": "yolo",
            "class": cls_name,
            "box": (int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y)),
            "base_conf": base_conf,
        })
    return dets


def run_yolo_detection(model, gray_img, conf=0.15, tile_size=YOLO_TILE_SIZE,
                       overlap=YOLO_TILE_OVERLAP):
    """Run YOLO on overlapping local windows and map boxes to source pixels.

    This keeps the public detection schema unchanged while preventing very tall
    waterfall images from being represented as one tiny object inside a global
    letterboxed frame.
    """
    h, w = gray_img.shape[:2]
    xs = _tile_starts(w, tile_size, overlap)
    ys = _tile_starts(h, tile_size, overlap)
    dets = []
    for y in ys:
        for x in xs:
            window = gray_img[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            if window.size == 0 or window.std() < 2:
                continue
            dets.extend(_predict_yolo_window(model, window, conf, x, y))
    return dets


def _shipwreck_candidate(gray_img, yolo_dets):
    """Promote a repeated, spatially coherent large structure for review.

    This is a candidate flag, not a trained shipwreck classifier: the bundled
    weights have no shipwreck class. Repeated detections from overlapping tiles
    can reveal a large object that the model otherwise labels inconsistently.
    """
    candidates = [d for d in yolo_dets if float(d.get("base_conf", 0.0)) >= 0.25]
    if len(candidates) < 3:
        return None
    h, w = gray_img.shape[:2]
    clusters = []
    for det in sorted(candidates, key=lambda d: -float(d.get("base_conf", 0.0))):
        x1, y1, x2, y2 = det["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cx < 0.08 * w or cx > 0.92 * w:
            continue
        best = None
        for cluster in clusters:
            if ((cx - cluster["cx"]) ** 2 + (cy - cluster["cy"]) ** 2) ** 0.5 < 520:
                best = cluster
                break
        if best is None:
            best = {"items": [], "cx": cx, "cy": cy}
            clusters.append(best)
        best["items"].append(det)
        n = len(best["items"])
        best["cx"] = ((n - 1) * best["cx"] + cx) / n
        best["cy"] = ((n - 1) * best["cy"] + cy) / n

    valid = []
    for cluster in clusters:
        items = cluster["items"]
        if len(items) < 3:
            continue
        x1 = max(0, min(d["box"][0] for d in items) - 32)
        y1 = max(0, min(d["box"][1] for d in items) - 32)
        x2 = min(w, max(d["box"][2] for d in items) + 32)
        y2 = min(h, max(d["box"][3] for d in items) + 32)
        bw, bh = x2 - x1, y2 - y1
        if bw < 120 or bh < 160 or bw * bh < 18000:
            continue
        crop = gray_img[y1:y2, x1:x2]
        edge_density = float((cv2.Canny(crop, 60, 160) > 0).mean()) if crop.size else 0.0
        mean_conf = float(np.mean([d["base_conf"] for d in items]))
        if edge_density < 0.025:
            continue
        conf = float(np.clip(0.35 + 0.20 * min(len(items) / 6.0, 1.0) +
                             0.25 * mean_conf + 0.20 * min(edge_density / 0.12, 1.0), 0.35, 0.92))
        valid.append({
            "source": "structure_heuristic",
            "class": "shipwreck_candidate",
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "base_conf": conf,
            "evidence_count": len(items),
        })
    return max(valid, key=lambda d: d["base_conf"]) if valid else None


def run_autoencoder_scan(ae_model, gray_img, existing_boxes, device="cpu"):
    """Slide a coarse grid over regions not already covered by a YOLO box,
    score each cell's reconstruction error, then flag only cells that are
    statistical outliers *relative to this image's own background* (robust
    median/MAD z-score) rather than against a fixed constant. Adjacent
    flagged cells are merged via connected-component labeling on the grid
    (not pairwise box-IoU, which under-merges when adjacent cells only
    partially overlap) so one real anomaly produces one box, not a wall of
    overlapping ones.
    """
    h, w = gray_img.shape
    xs = list(range(0, max(w - PATCH_SIZE, 1), GRID_STRIDE))
    ys = list(range(0, max(h - PATCH_SIZE, 1), GRID_STRIDE))
    n_rows, n_cols = len(ys), len(xs)

    score_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    covered_grid = np.zeros((n_rows, n_cols), dtype=bool)

    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            if any(_iou(box, eb) > 0.15 for eb in existing_boxes):
                covered_grid[gy, gx] = True
                continue
            patch = gray_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if patch.std() < 3:  # flat/dropout/nadir region - not scoreable
                continue
            score_grid[gy, gx] = anomaly_score(ae_model, patch, device=device)

    valid = ~np.isnan(score_grid)
    if valid.sum() < 8:  # not enough data to calibrate against
        return []

    scores = score_grid[valid]
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) * 1.4826 + 1e-9  # -> std-equivalent
    mad = max(mad, MIN_ABS_ERROR_FLOOR)

    z_grid = np.where(valid, (score_grid - median) / mad, 0.0)
    anomaly_mask = valid & (z_grid > ANOMALY_Z_THRESHOLD) & (score_grid > median + MIN_ABS_ERROR_FLOOR)

    labeled, n_clusters = ndimage.label(anomaly_mask, structure=np.ones((3, 3)))
    anomalies = []
    for cluster_id in range(1, n_clusters + 1):
        cell_ys, cell_xs = np.where(labeled == cluster_id)
        x1 = xs[cell_xs.min()]
        y1 = ys[cell_ys.min()]
        x2 = xs[cell_xs.max()] + PATCH_SIZE
        y2 = ys[cell_ys.max()] + PATCH_SIZE
        cluster_z = z_grid[cell_ys, cell_xs].max()
        # map z-score to a 0-1 confidence: z at threshold -> ~0.5, saturating higher
        conf = float(np.clip(0.5 + 0.1 * (cluster_z - ANOMALY_Z_THRESHOLD), 0, 1))
        anomalies.append({
            "source": "autoencoder",
            "class": "unclassified_anomaly",
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "base_conf": conf,
        })
    return anomalies


def detect(yolo_model, ae_model, raw_img, yolo_conf=0.15, device="cpu", shipwreck_model=None, shipwreck_conf=0.20):
    """Full fused detection pipeline. Returns preprocessed image + list of
    final detections with 0-100 confidence scores."""
    clean, valid_mask = preprocess(raw_img)

    yolo_dets = run_yolo_detection(yolo_model, clean, conf=yolo_conf)
    if shipwreck_model is not None:
        shipwreck_dets = run_yolo_detection(shipwreck_model, clean, conf=shipwreck_conf)
        for det in shipwreck_dets:
            det["source"] = "yolo_shipwreck"
            det["class"] = "shipwreck"
        yolo_dets.extend(shipwreck_dets)
    structure_candidate = _shipwreck_candidate(clean, yolo_dets)
    yolo_boxes = [d["box"] for d in yolo_dets]
    ae_dets = run_autoencoder_scan(ae_model, clean, yolo_boxes, device=device)

    # de-duplicate overlapping same-class YOLO boxes (loose conf threshold can
    # produce near-duplicate boxes on the same object)
    yolo_dets.sort(key=lambda d: -d["base_conf"])
    deduped_yolo = []
    for d in yolo_dets:
        if not any(d["class"] == k["class"] and _iou(d["box"], k["box"]) > 0.3 for k in deduped_yolo):
            deduped_yolo.append(d)

    all_dets = deduped_yolo + ae_dets
    if structure_candidate is not None:
        all_dets.append(structure_candidate)
    final = []
    for d in all_dets:
        x1, y1, x2, y2 = d["box"]
        # suppress detections that fall (mostly) in the invalid/nadir mask
        region_mask = valid_mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if region_mask.size > 0 and (region_mask > 0).mean() < 0.4:
            continue
        filt = _noise_filter_score(clean, x1, y1, x2, y2)
        final_conf = float(np.clip(d["base_conf"] * filt, 0, 1)) * 100
        record = {
            "class": d["class"],
            "source": d["source"],
            "box": [x1, y1, x2, y2],
            "confidence_pct": round(final_conf, 1),
        }
        if "evidence_count" in d:
            record["evidence_count"] = int(d["evidence_count"])
        final.append(record)

    final.sort(key=lambda d: -d["confidence_pct"])
    # Add transparent quality evidence without changing the existing detector schema.
    final = score_frame(final)
    return clean, final
