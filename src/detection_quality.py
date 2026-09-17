"""Detection quality scoring inspired by GhostVision's alpha summarization.

The SIH detector normally processes one image at a time, so this module supports
single-frame scoring and optional multi-frame aggregation. It does not replace
model confidence; it adds transparent evidence fields for reporting.
"""
from __future__ import annotations

from collections import defaultdict
from math import hypot
from statistics import mean, pstdev


def quality_tier(score: float) -> str:
    score = float(score)
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _center(det: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = det["box"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def score_detection(det: dict, persistence_count: int = 1, max_persistence: int = 1,
                    confidence_spread: float = 0.0, center_spread_px: float = 0.0) -> dict:
    """Attach interpretable quality metrics to one detection."""
    confidence = max(0.0, min(100.0, float(det.get("confidence_pct", 0.0)))) / 100.0
    max_persistence = max(1, int(max_persistence))
    persistence = max(0.0, min(1.0, float(persistence_count) / max_persistence))
    stability = max(0.0, 1.0 - min(1.0, float(confidence_spread) / 0.25))
    spatial = max(0.0, 1.0 - min(1.0, float(center_spread_px) / 32.0))
    combined = 0.55 * confidence + 0.25 * persistence + 0.12 * stability + 0.08 * spatial
    score = round(combined * 100.0, 1)
    out = dict(det)
    out.update({
        "quality_score": score,
        "quality_tier": quality_tier(score),
        "persistence_count": int(persistence_count),
        "persistence_score": round(persistence * 100.0, 1),
        "stability_score": round(stability * 100.0, 1),
        "spatial_consistency_score": round(spatial * 100.0, 1),
    })
    return out


def score_frame(detections: list[dict]) -> list[dict]:
    """Score a single frame, preserving the existing detection schema."""
    if not detections:
        return []
    return [score_detection(d) for d in detections]


def aggregate_detections(frames: list[list[dict]], max_center_distance_px: float = 48.0) -> list[dict]:
    """Aggregate detections across sequential frames using class and proximity.

    This is deliberately lightweight and dependency-free. It provides the same
    practical benefit as GhostVision's tracker summary: repeated observations
    increase persistence, while confidence and location variance affect quality.
    """
    tracks: list[dict] = []
    for frame_idx, frame in enumerate(frames):
        for det in frame:
            cx, cy = _center(det)
            candidates = [t for t in tracks if t["class"] == det.get("class") and
                          frame_idx > t["last_frame"] and
                          hypot(cx - t["cx"], cy - t["cy"]) <= max_center_distance_px]
            track = min(candidates, key=lambda t: hypot(cx - t["cx"], cy - t["cy"])) if candidates else None
            if track is None:
                track = {"class": det.get("class"), "detections": [], "frames": [], "cx": cx, "cy": cy, "last_frame": frame_idx}
                tracks.append(track)
            track["detections"].append(det)
            track["frames"].append(frame_idx)
            n = len(track["detections"])
            track["cx"] = ((n - 1) * track["cx"] + cx) / n
            track["cy"] = ((n - 1) * track["cy"] + cy) / n
            track["last_frame"] = frame_idx

    result = []
    max_persistence = max((len(t["detections"]) for t in tracks), default=1)
    for track in tracks:
        ds = track["detections"]
        best = max(ds, key=lambda d: float(d.get("confidence_pct", 0.0)))
        centers = [_center(d) for d in ds]
        spread = [hypot(x - track["cx"], y - track["cy"]) for x, y in centers]
        confidences = [float(d.get("confidence_pct", 0.0)) / 100.0 for d in ds]
        merged = dict(best)
        merged["confidence_pct"] = round(mean(float(d.get("confidence_pct", 0.0)) for d in ds), 1)
        merged["persistence_count"] = len(ds)
        merged["track_frame_span"] = len(set(track["frames"]))
        merged = score_detection(merged, len(ds), max_persistence,
                                 pstdev(confidences) if len(confidences) > 1 else 0.0,
                                 max(spread, default=0.0))
        result.append(merged)
    return sorted(result, key=lambda d: -float(d.get("quality_score", 0.0)))
