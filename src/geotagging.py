"""
Anomalous Reporting & Geotagging Engine
=========================================
Reads sonar navigation metadata (from a sidecar file exported alongside the
sonar log, e.g. a per-ping .csv of lat/lon/heading/altitude - this is what
pyxtf extracts from real .XTF ping headers) and converts each pixel-space
detection into a real-world geotagged hazard record.

If no navigation file is supplied, a synthetic straight-line tow-track is
generated so the dashboard/report still works end-to-end for demo purposes.
Swap `load_nav_track` for a pyxtf-based XTF parser when real logs are used -
everything downstream (interpolation, report schema) stays the same.
"""

import json
import csv
import os
import time
import random
import numpy as np
import pandas as pd

EARTH_R = 6371000.0  # meters


def load_nav_track(nav_csv_path):
    """Expected columns: ping_index,timestamp,lat,lon,heading_deg,altitude_m"""
    df = pd.read_csv(nav_csv_path)
    required = {"ping_index", "lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"nav file must contain columns {required}")
    return df


def synthetic_nav_track(n_pings, start_lat=17.6868, start_lon=83.2185,
                         heading_deg=45.0, ping_spacing_m=0.5, altitude_m=25.0):
    """Generates a straight-line simulated AUV/tow-fish track (used when the
    user has no real navigation sidecar file - e.g. Visakhapatnam coast demo
    coordinates). Real deployments should call load_nav_track() instead."""
    rows = []
    heading_rad = np.radians(heading_deg)
    dlat_per_m = 1 / 111320.0
    dlon_per_m = 1 / (111320.0 * np.cos(np.radians(start_lat)))
    for i in range(n_pings):
        d = i * ping_spacing_m
        lat = start_lat + d * np.cos(heading_rad) * dlat_per_m
        lon = start_lon + d * np.sin(heading_rad) * dlon_per_m
        rows.append({
            "ping_index": i,
            "timestamp": time.time() + i * 0.1,
            "lat": lat,
            "lon": lon,
            "heading_deg": heading_deg,
            "altitude_m": altitude_m,
        })
    return pd.DataFrame(rows)


def pixel_to_geo(nav_df, px_x, px_y, img_w, img_h, swath_width_m=50.0):
    """Maps an (x, y) pixel in the sonar waterfall image to a lat/lon.

    y (row) -> along-track ping index (time axis of the waterfall image)
    x (col) -> across-track range from the nadir (port/starboard swath)
    """
    ping_idx = int(np.clip((px_y / img_h) * (len(nav_df) - 1), 0, len(nav_df) - 1))
    row = nav_df.iloc[ping_idx]
    lat, lon, heading = row["lat"], row["lon"], row.get("heading_deg", 0.0)

    # across-track offset in meters from image center (nadir)
    offset_frac = (px_x / img_w) - 0.5
    offset_m = offset_frac * swath_width_m

    # perpendicular-to-heading bearing for the across-track offset
    perp_heading_rad = np.radians(heading + 90.0)
    dlat = (offset_m * np.cos(perp_heading_rad)) / 111320.0
    dlon = (offset_m * np.sin(perp_heading_rad)) / (111320.0 * np.cos(np.radians(lat)))

    return lat + dlat, lon + dlon


def build_report(detections, nav_df, img_w, img_h, swath_width_m=50.0, source_file="uploaded_sonar_log"):
    records = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        lat, lon = pixel_to_geo(nav_df, cx, cy, img_w, img_h, swath_width_m)
        records.append({
            "detection_id": f"DET-{i+1:04d}",
            "class": det["class"],
            "source_model": det["source"],
            "confidence_pct": det["confidence_pct"],
            "quality_score": det.get("quality_score"),
            "quality_tier": det.get("quality_tier"),
            "persistence_count": det.get("persistence_count", 1),
            "persistence_score": det.get("persistence_score"),
            "stability_score": det.get("stability_score"),
            "spatial_consistency_score": det.get("spatial_consistency_score"),
            "evidence_count": det.get("evidence_count"),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "bbox_px": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "bbox_width_px": x2 - x1,
            "bbox_height_px": y2 - y1,
            "sonar_source_file": source_file,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return records


def save_report(records, out_dir, basename="anomaly_report"):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{basename}.json")
    csv_path = os.path.join(out_dir, f"{basename}.csv")

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    if records:
        flat_rows = []
        for r in records:
            flat = {k: v for k, v in r.items() if k != "bbox_px"}
            flat.update({f"bbox_{k}": v for k, v in r["bbox_px"].items()})
            flat_rows.append(flat)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
    else:
        with open(csv_path, "w") as f:
            f.write("detection_id,class,source_model,confidence_pct,latitude,longitude\n")

    return json_path, csv_path
