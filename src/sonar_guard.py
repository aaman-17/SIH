"""
Sonar Input Guard
==================
Side-scan sonar exports are acoustic backscatter *intensity* data - even
when saved as a 3-channel PNG/JPG, the R, G, and B channels are identical
(true grayscale). Ordinary underwater photographs have real color content
(blue/green water cast, lighting, object color).

This lets us catch the most common misuse case cheaply and honestly: someone
uploads an optical photo (or any non-sonar image) to the sonar pipeline,
which would otherwise run models that were never trained on that kind of
imagery and produce meaningless, overconfident detections.

Known limitation (documented, not hidden): some sonar-logging software
renders acoustic intensity through a color LUT ("amber", "copper", "bone",
etc.) for display. Such colorized sonar exports would fail this check even
though they are legitimate sonar data. If you work with colorized sonar
software, either export in raw grayscale, or lower `color_std_threshold`.
"""

import cv2
import numpy as np


def channel_decorrelation(img_bgr):
    """Mean pixel-wise std-dev across the B/G/R channels. ~0 for true
    grayscale (sonar); notably >0 for real color photos."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    stacked = np.stack([b, g, r], axis=-1)
    return float(stacked.std(axis=-1).mean())


def saturation_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return float(sat.mean()), float(sat.std())


def looks_like_sonar(img_bgr, color_std_threshold=6.0, saturation_threshold=18.0):
    """Returns (is_sonar: bool, diagnostics: dict).

    Flags as "not sonar" when the image has meaningfully more color content
    than a true single-channel acoustic export would - i.e. real photos,
    not raw/grayscale-exported side-scan sonar waterfalls.
    """
    if img_bgr.ndim == 2:
        return True, {"reason": "already single-channel grayscale", "channel_decorrelation": 0.0}

    decorr = channel_decorrelation(img_bgr)
    sat_mean, sat_std = saturation_stats(img_bgr)

    is_sonar = decorr < color_std_threshold and sat_mean < saturation_threshold

    diagnostics = {
        "channel_decorrelation": round(decorr, 2),
        "mean_saturation": round(sat_mean, 2),
        "saturation_std": round(sat_std, 2),
        "threshold_channel_decorrelation": color_std_threshold,
        "threshold_mean_saturation": saturation_threshold,
    }
    return is_sonar, diagnostics


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    is_sonar, diag = looks_like_sonar(img)
    print("Looks like sonar:", is_sonar)
    print(diag)
