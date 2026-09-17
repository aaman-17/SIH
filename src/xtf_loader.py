"""
XTF Sonar Log Ingestion (Data Input stage of the tech stack)
==============================================================
Parses real-world .XTF side-scan sonar files using `pyxtf` and converts them
into:
  1. A grayscale "waterfall" image (rows = pings over time, columns = across-
     track range) that the rest of the pipeline (preprocessing, detection)
     already consumes.
  2. A navigation DataFrame (ping_index, lat, lon, heading_deg, altitude_m)
     compatible with `src.geotagging.build_report`.

This is the real-data counterpart to `generate_synthetic_data.py`. Everything
downstream (preprocessing -> detection -> geotagging -> dashboard) is
identical whether the image came from a real .XTF file or the synthetic
generator - that's the point of separating data ingestion from the model.

Usage:
    from src.xtf_loader import load_xtf
    waterfall_img, nav_df = load_xtf("survey_line_04.xtf")
"""

import numpy as np
import pandas as pd

try:
    import pyxtf
    PYXTF_AVAILABLE = True
except ImportError:
    PYXTF_AVAILABLE = False


def load_xtf(xtf_path, channel=0, normalize=True):
    """Reads an .XTF file and returns (waterfall_gray_uint8, nav_df).

    channel: 0 = port, 1 = starboard (typical dual-channel side-scan sonar).
    Both channels are stitched side-by-side (port | nadir gap | starboard) to
    reproduce the classic side-scan sonar waterfall look.
    """
    if not PYXTF_AVAILABLE:
        raise ImportError("pyxtf is not installed. Run: pip install pyxtf")

    fh, packets = pyxtf.xtf_read(xtf_path)

    ping_packets = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not ping_packets:
        raise ValueError("No sonar ping packets found in this XTF file")

    port_rows, stbd_rows, nav_rows = [], [], []
    for i, ping in enumerate(ping_packets):
        try:
            port = ping.data[0].astype(np.float32)
            stbd = ping.data[1].astype(np.float32) if len(ping.data) > 1 else port.copy()
        except Exception:
            continue
        port_rows.append(port)
        stbd_rows.append(stbd)

        nav_rows.append({
            "ping_index": i,
            "timestamp": getattr(ping, "SonarDateTime", i),
            "lat": getattr(ping, "SensorYcoordinate", getattr(ping, "ShipYcoordinate", 0.0)),
            "lon": getattr(ping, "SensorXcoordinate", getattr(ping, "ShipXcoordinate", 0.0)),
            "heading_deg": getattr(ping, "SensorHeading", getattr(ping, "ShipHeading", 0.0)),
            "altitude_m": getattr(ping, "SensorAltitude", 0.0),
        })

    max_len = max(max(len(r) for r in port_rows), max(len(r) for r in stbd_rows))
    port_arr = np.zeros((len(port_rows), max_len), dtype=np.float32)
    stbd_arr = np.zeros((len(stbd_rows), max_len), dtype=np.float32)
    for i, r in enumerate(port_rows):
        port_arr[i, :len(r)] = r
    for i, r in enumerate(stbd_rows):
        stbd_arr[i, :len(r)] = r

    gap = np.zeros((len(port_rows), max(8, max_len // 40)), dtype=np.float32)
    waterfall = np.concatenate([np.fliplr(port_arr), gap, stbd_arr], axis=1)

    if normalize:
        p1, p99 = np.percentile(waterfall, [1, 99])
        waterfall = np.clip((waterfall - p1) / max(p99 - p1, 1e-6) * 255, 0, 255)

    waterfall_img = waterfall.astype(np.uint8)
    nav_df = pd.DataFrame(nav_rows)
    return waterfall_img, nav_df


if __name__ == "__main__":
    import sys
    import cv2
    if len(sys.argv) < 2:
        print("Usage: python xtf_loader.py <path_to.xtf> [out.png]")
        sys.exit(1)
    img, nav = load_xtf(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "xtf_waterfall.png"
    cv2.imwrite(out, img)
    nav.to_csv("xtf_nav_track.csv", index=False)
    print(f"Wrote {out} ({img.shape}) and xtf_nav_track.csv ({len(nav)} pings)")
