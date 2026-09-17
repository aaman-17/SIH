"""
Optical Marine-Litter Detector
=================================
For ordinary RGB underwater/surface photos - NOT sonar. A genuinely
different computer-vision problem from the acoustic pipeline, and it needs
its own approach.

Honesty about what this is and isn't:
  - There is no bulk-downloadable, real-world annotated underwater-litter
    dataset reachable from this environment (e.g. TACO's images are hosted
    on Flickr, UAVVaste/TrashNet similarly rely on external image hosts -
    all outside this sandbox's network allowlist of GitHub/PyPI/npm/OS
    package mirrors). So this is NOT a custom-trained specialist model.
  - Pretrained COCO-YOLO was tested directly on underwater photos and
    performs poorly: it does not reliably recognize "bottle" underwater at
    all (blue-green color cast + refraction + backlighting are a real
    domain shift it never saw in training) and instead mislabels floating
    plastic as "surfboard", "person", or "bear" at low confidence. Treating
    those literal class names as ground truth would be actively misleading.
  - So COCO-YOLO is used here only as a **generic salient-foreign-object
    proposer**: any detection above a modest confidence, regardless of its
    (unreliable) COCO class name, is surfaced as "possible_debris" with the
    original COCO guess kept only as a secondary hint, not a claim.
  - A classical CV heuristic supplements this for plastic bags/film, which
    COCO has no class for at all: bright, low-saturation, solid (non-
    sliver) blobs, with light gray-world white-balancing first to reduce
    the blue-water color cast.
  - Bottom line: treat this as an assistive first-pass triage tool, not a
    reliable detector. For real deployment, fine-tune on an actual
    underwater-litter dataset (TACO, UAVVaste, DUO, TrashCan) obtained
    outside this sandbox.
"""

import cv2
import numpy as np


def gray_world_white_balance(img_bgr):
    """Cheap color-cast correction - underwater photos skew blue/green,
    which both hurts pretrained-model recognition and the brightness/
    saturation heuristic below."""
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    k = (b.mean() + g.mean() + r.mean()) / 3
    b = np.clip(b * (k / max(b.mean(), 1e-6)), 0, 255)
    g = np.clip(g * (k / max(g.mean(), 1e-6)), 0, 255)
    r = np.clip(r * (k / max(r.mean(), 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def detect_plastic_bags_heuristic(img_bgr, min_area_frac=0.0006, max_area_frac=0.20):
    """Classical CV pass for plastic bags/film (no COCO class exists).
    Bright, low-saturation, solid (non-sliver, non-speckle) blobs. Nearby
    fragments are merged via aggressive morphological closing BEFORE
    contour extraction so one bag doesn't fragment into a dozen boxes."""
    h, w = img_bgr.shape[:2]
    img_area = h * w
    min_area = max(img_area * min_area_frac, 300)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]

    bright_mask = ((v > 190) & (s < 60)).astype(np.uint8) * 255
    surface_band = int(h * 0.12)
    bright_mask[:surface_band, :] = 0

    close_k = max(9, int(min(h, w) * 0.015))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=2)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > img_area * max_area_frac:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = min(bw, bh) / max(bw, bh)
        solidity = area / (bw * bh + 1e-6)
        # require reasonably solid, non-sliver blobs - rejects thin
        # specular streaks / light rays that aren't object-shaped
        if aspect < 0.25 or solidity < 0.45:
            continue
        conf = float(np.clip(0.25 + 0.4 * solidity + 0.25 * aspect, 0.2, 0.9)) * 100
        dets.append({
            "class": "plastic_bag_or_film",
            "source": "cv_heuristic",
            "box": [x, y, x + bw, y + bh],
            "confidence_pct": round(conf, 1),
        })
    return dets


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / union


def detect_optical(yolo_coco_model, img_bgr, yolo_conf=0.05):
    """Runs pretrained-COCO YOLO (as a generic salient-object proposer,
    NOT trusting its class label) + the classical bag/film heuristic, and
    returns a unified detection list in the sonar dashboard's schema."""
    wb = gray_world_white_balance(img_bgr)

    results = yolo_coco_model.predict(wb, conf=yolo_conf, verbose=False)[0]
    yolo_dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        coco_guess = yolo_coco_model.names[cls_id]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        yolo_dets.append({
            "class": "possible_debris",
            "coco_hint": coco_guess,
            "source": "yolo_coco_generic",
            "box": [x1, y1, x2, y2],
            "confidence_pct": round(conf * 100, 1),
        })

    bag_dets = detect_plastic_bags_heuristic(wb)
    for d in bag_dets:
        d["coco_hint"] = None

    all_dets = yolo_dets + bag_dets
    all_dets.sort(key=lambda d: -d["confidence_pct"])
    kept = []
    for d in all_dets:
        if not any(_iou(d["box"], k["box"]) > 0.35 for k in kept):
            kept.append(d)
    return kept
