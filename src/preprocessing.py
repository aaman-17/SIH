"""
Preprocessing Module
=====================
Cleans raw side-scan sonar imagery before it hits the detection models.

Handles the core acoustic-imagery challenges called out in the problem
statement:
  - High speckle noise            -> median blur + non-local-means denoise
  - Varying pixel resolutions     -> resize/normalize to a canonical size
  - Acoustic shadows / low dynamic range -> CLAHE contrast enhancement
  - Data dropouts (heave/pitch/roll) -> row-dropout detection + inpainting
"""

import cv2
import numpy as np

CANONICAL_SIZE = None  # do NOT force a fixed square size - real sonar
                        # waterfalls are long/thin (many pings x swath width)
                        # and squashing them to a square destroys the image.
                        # Tiling (see detection_pipeline.py) handles sizing
                        # for the models instead.


def to_grayscale(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def detect_dropout_rows(gray, dark_thresh=8, frac_thresh=0.9):
    """Flag rows that are almost entirely near-black -> vehicle motion dropout."""
    row_means = gray.mean(axis=1)
    dark_frac = (gray < dark_thresh).mean(axis=1)
    return np.where((row_means < dark_thresh * 2) | (dark_frac > frac_thresh))[0]


def inpaint_dropouts(gray, dropout_rows):
    if len(dropout_rows) == 0:
        return gray
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[dropout_rows, :] = 255
    # dilate slightly so inpainting blends edges
    mask = cv2.dilate(mask, np.ones((3, 1), np.uint8), iterations=1)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def denoise_speckle(gray):
    """Median filter removes salt-and-pepper speckle; NLM smooths acoustic
    grain while preserving object edges (important - we don't want to blur
    away the highlight/shadow signature that flags debris)."""
    med = cv2.medianBlur(gray, 3)
    nlm = cv2.fastNlMeansDenoising(med, h=7, templateWindowSize=7, searchWindowSize=21)
    return nlm


def enhance_contrast_clahe(gray, clip_limit=2.5, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def mask_nadir_gap(gray, dark_thresh=12):
    """Return a mask (255 = valid seafloor data) excluding the central nadir
    gap column-band so downstream detectors don't fire on it."""
    col_means = gray.mean(axis=0)
    w = len(col_means)
    center = w // 2
    band = col_means[center - w // 8: center + w // 8]
    if band.mean() < dark_thresh * 3:
        gap_cols = np.where(col_means < dark_thresh * 3)[0]
        gap_cols = gap_cols[(gap_cols > w * 0.3) & (gap_cols < w * 0.7)]
        mask = np.full_like(gray, 255)
        if len(gap_cols) > 0:
            mask[:, gap_cols.min():gap_cols.max() + 1] = 0
        return mask
    return np.full_like(gray, 255)


def preprocess(img, resize_to=CANONICAL_SIZE):
    """Full preprocessing pipeline. Returns (clean_gray_uint8, valid_mask)."""
    gray = to_grayscale(img)
    if resize_to is not None and gray.shape[:2] != resize_to:
        gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

    dropout_rows = detect_dropout_rows(gray)
    gray = inpaint_dropouts(gray, dropout_rows)
    gray = denoise_speckle(gray)
    valid_mask = mask_nadir_gap(gray)
    enhanced = enhance_contrast_clahe(gray)

    return enhanced, valid_mask


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    img = cv2.imread(path)
    clean, mask = preprocess(img)
    cv2.imwrite("preprocessed_debug.png", clean)
    print("Wrote preprocessed_debug.png")
