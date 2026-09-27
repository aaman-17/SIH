# AI-Powered Marine Debris & Anomaly Detection (Side-Scan Sonar)

A working, end-to-end prototype matching the problem statement and tech
stack you specified:

```
Side-Scan Sonar (.xtf / image) --> pyxtf ingestion --> OpenCV+CLAHE preprocessing
      --> [ YOLO11-Nano detector | CNN Autoencoder anomaly scorer ]
      --> Noise filtering & 0-100% confidence scoring
      --> Geotagging engine (lat/lon) --> JSON/CSV report
      --> Streamlit + Folium dashboard
```

## What's actually implemented (not just described)

| Component | File | Status |
|---|---|---|
| Data ingestion (.xtf real logs) | `src/xtf_loader.py` | Implemented via `pyxtf` |
| Synthetic SSS data generator (for training/demo — see note below) | `src/generate_synthetic_data.py` | Generates labeled images |
| Preprocessing (denoise, CLAHE, dropout-inpainting, nadir masking) | `src/preprocessing.py` | Implemented |
| YOLO11-Nano object detector | `src/detection_pipeline.py`, `models/yolo_sonar_best.pt` | **Trained via until-best hill-climb**; held-out mAP50 = **0.991**, P = 0.98, R = 0.98 |
| CNN Autoencoder anomaly scorer | `src/autoencoder.py`, `models/autoencoder.pt` | **Trained until-best**; normal-vs-anomaly separation AUROC = **1.00** |
| Confidence fusion + noise filtering | `src/detection_pipeline.py` | Implemented (edge-density / aspect-ratio heuristics) |
| Geotagging + JSON/CSV report engine | `src/geotagging.py` | Implemented |
| Streamlit + Folium dashboard | `app.py` | Implemented — **Abyssal Scan tactical HUD theme** (see below) |
| Sonar-vs-photo input guard | `src/sonar_guard.py` | Implemented — blocks non-sonar images before running sonar models |
| Optical (camera photo) triage mode | `src/optical_detector.py` | Implemented — separate, honestly-scoped pipeline (see note below) |

## Dashboard theme

The dashboard is styled as a dark tactical "hydrographic bridge" HUD
(`src/aura_theme.py`) — deep navy background, cyan/mint telemetry accents,
JetBrains Mono for data readouts. It includes a live top status bar,
a 5-metric telemetry strip, a 3-column command layout (inference controls /
sonar viewport / detections feed), three cosmetic sonar display palettes
(Deep Cyan, Sepia, Phosphor — applied only for display, detection still
runs on the underlying grayscale data), and styled hazard/anomaly detection
cards. Every number shown is derived from the real pipeline (detection
count, confidence, measured inference latency, tow-track lat/lon) — there
is no PDF export or GIS/ROS2 sync button, since those would need a backend
that doesn't exist in this build; only JSON/CSV export (which is real) is
wired up.

## ⚠️ About the "Optical / camera photo" mode

The dashboard has two modes: **Side-scan sonar** (the system described
above) and **Optical / camera photo**, for when someone uploads an ordinary
underwater/surface photo instead of acoustic data. These are genuinely
different computer-vision problems and share no models.

The optical mode is deliberately scoped as an assistive triage tool, not a
trained specialist detector, because there was no bulk-downloadable,
real-world annotated underwater-litter dataset reachable from this build
environment (TACO, UAVVaste, etc. all host their actual images on Flickr or
similar hosts outside the sandbox's network allowlist). It combines:

- A pretrained COCO-YOLO model used as a **generic salient-foreign-object
  proposer** — not trusted for its literal class name, since it does not
  reliably recognize "bottle" underwater at all (color-cast + refraction is
  real domain shift) and instead mislabels floating plastic as "surfboard"
  or "person" at low confidence. Its guess is shown only as a secondary
  hint (`coco_hint`), never as the headline label.
- A classical CV heuristic for plastic bags/film (bright, low-saturation,
  solid blobs after gray-world white-balance correction), since COCO has no
  class for that at all.

For a production-grade optical detector, fine-tune YOLO on a real annotated
dataset such as TACO, UAVVaste, DUO, or TrashCan, downloaded outside this
sandbox.

## Training data and AI4Shipwrecks integration

The original debris model remains a synthetic-data proof of concept. The project now also includes a real-data shipwreck checkpoint trained from the public AI4Shipwrecks side-scan sonar dataset. The conversion tool preserves tiled sonar geometry, converts binary shipwreck masks into YOLO boxes, and uses union-mask boxes to avoid fragmented connected-component labels.

The reproducible tools are:

```bash
python3 tools/prepare_ai4shipwrecks.py --source /path/to/AI4Shipwrecks --output /path/to/ai4shipwrecks_yolo --tile 1024 --overlap 0.25 --label-mode union
python3 tools/train_ai4shipwrecks.py --data /path/to/ai4shipwrecks_yolo/data.yaml --weights models/yolo_sonar_best.pt --project /path/to/runs_union --epochs 12 --imgsz 512 --batch 4 --device cpu --patience 5 --lr0 0.0005 --freeze 10
```

The trained checkpoint is bundled as `models/yolo_shipwreck_ai4shipwrecks.pt` and runs in parallel with the existing debris detector. Its held-out test metrics are stored in `reports/ai4shipwrecks_test_metrics.json`. It is an improving baseline rather than a survey-ready benchmark: the first run used fragmented mask components and had very poor recall; the corrected union-label run improved held-out recall and mAP50 but still needs more training, site diversity, and threshold calibration.

AI4Shipwrecks contains shipwreck masks, not ghost-net labels. It can improve shipwreck recognition but cannot directly train a reliable `ghost_net` class.

For the original synthetic-data workflow, `src/generate_synthetic_data.py` still delivers a genuinely **working**, end-to-end, trainable pipeline rather than a hollow shell.
procedurally generates side-scan-sonar-*style* imagery (nadir gap, sand
ripples, rock-cluster clutter, Rayleigh speckle noise, motion-dropout
stripes, and objects rendered with the highlight+acoustic-shadow signature
real SSS returns show). The bundled models were trained on this synthetic
set purely so you have working, non-random weights to demonstrate and build
on.

**To use this on real surveys:** drop real `.xtf` files or georeferenced
sonar mosaics + a handful of hand-labeled bounding boxes into
`data/synthetic_yolo/images` and `labels` (YOLO txt format) alongside or
instead of the synthetic ones, then retrain. The preprocessing, fusion,
geotagging, and dashboard code does not need to change — only the training
data.

## Repeated hill-climb training ("train until it's the best")

The bundled YOLO checkpoint's original "mAP50 ≈ 0.93" was measured on a val split drawn from the *same seeded generation run* as its training data. On a genuinely held-out set it actually scored **mAP50 = 0.092**. The until-best loop below fixes both the metric and the model.

The loop (`tools/train_until_best.py`) repeats: generate a fresh, differently-seeded, harder synthetic batch → fine-tune the current champion → evaluate champion and candidate against a **fixed held-out eval set** that is never trained on or regenerated (`data/eval_heldout/`, built once by `tools/build_heldout_eval.py` with a disjoint seed and a harder composition than training) → promote the candidate only if held-out mAP50 improves by ≥ 0.005. Every comparison is appended to `reports/training_history.jsonl`; the previous champion is kept as `models/yolo_sonar_prev.pt` for rollback. The loop stops on `--rounds` or after `--plateau` consecutive rejections.

Result of the run baked into `models/yolo_sonar_best.pt` (180 held-out images, 481 objects, max 5 per image vs 3 in training):

| model | mAP50 | mAP50-95 | precision | recall | F1 |
|---|---|---|---|---|---|
| original bundled weights | 0.092 | 0.032 | 0.162 | 0.184 | 0.172 |
| after round 1 | 0.904 | 0.468 | 0.772 | 0.832 | 0.801 |
| after round 2 | 0.985 | 0.703 | 0.970 | 0.960 | 0.965 |
| after round 3 | 0.991 | 0.794 | 0.977 | 0.980 | 0.979 |
| **champion (round 4)** | **0.993** | **0.840** | **0.984** | **0.991** | **0.987** |
| round 5 | rejected | — | — | — | plateau confirmed |

Round 4 was promoted via the loop's dominance override: besides the map50 margin, any candidate that is at least as good as the champion on precision, recall, and map50-95 (with map50-95 clearly ahead) is a real improvement and gets kept. Round 5 failed both tests, confirming the plateau.

Inference is matched to training (`imgsz=640`, same denoise+CLAHE preprocessing), and the dashboard's default confidence gate is calibrated against this eval set: **0.30** maximizes recall-weighted F2 (sweep in `reports/inference_calibration.json`). The autoencoder is retrained by `tools/train_autoencoder_until_best.py`, which holds out normal patches, evaluates AUROC of reconstruction error between normal and object patches each epoch, keeps the best epoch, and early-stops (result: AUROC 1.00, object patches reconstruct ~12× worse than normal seafloor).

To re-run the whole optimization:

```bash
python tools/build_heldout_eval.py                      # once; never regenerates
python tools/train_until_best.py --rounds 3 --epochs 10 --batch 8 --expand 120
python tools/train_autoencoder_until_best.py data/normal_patches_clahe models/autoencoder.pt
python tools/calibrate_thresholds.py                    # refresh the gate recommendation
```

## Quick start

```bash
pip install -r requirements.txt

# Launch the dashboard (models are already trained and bundled)
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Upload a
`.png`/`.jpg` sonar image or a real `.xtf` log, or just tick "use bundled
demo image" to see it work immediately.

## Architecture notes

- **`preprocessing.py`** handles the four challenges called out in the brief:
  high speckle noise (median + non-local-means denoise), varying resolutions
  (canonical resize), acoustic shadows / low dynamic range (CLAHE), and
  vehicle-motion data dropouts (row-dropout detection + inpainting).
- **`detection_pipeline.py`** fuses the baseline YOLO11-Nano detector (known classes: `debris_net`, `pipe_cylinder`, `shipwreck_debris`), the optional real-data AI4Shipwrecks `shipwreck` detector, and an unsupervised CNN autoencoder that flags novel/unlabeled shapes via
  reconstruction error on grid cells YOLO didn't already claim. A
  heuristic noise filter (edge density, aspect ratio, local contrast)
  then discounts likely rock/shadow false positives before producing the
  final 0–100% confidence score.
- **`geotagging.py`** converts each pixel-space bounding box into a
  lat/lon using the ping-index → along-track and pixel-column →
  across-track swath geometry, the same way a real tow-fish/AUV survey
  would be georeferenced from its navigation log. Works with real nav CSVs,
  parsed XTF ping headers, or (for demos) a synthetic straight-line track.
- **Edge deployment**: YOLO11-Nano is ~2.6M parameters (~5MB weights) and
  the autoencoder is a 4-layer conv net — both run comfortably on CPU
  without a GPU, in line with the "no heavy cloud dependency" requirement.

## Retraining / extending

```bash
# Regenerate synthetic training data
python3 src/generate_synthetic_data.py data/synthetic_yolo

# Retrain YOLO (short chunks recommended on CPU-only machines)
python3 -c "
from ultralytics import YOLO
m = YOLO('yolo11n.pt')
m.train(data='data/synthetic_yolo/data.yaml', epochs=30, imgsz=256, batch=2, device='cpu')
"

# Retrain the autoencoder on CLAHE-matched normal (no-object) patches
python3 src/autoencoder.py data/normal_patches_clahe models/autoencoder.pt
```

## Long-strip sonar handling and shipwreck candidates

Long side-scan waterfalls are processed with overlapping 1024-pixel YOLO windows and source-coordinate remapping. The real-data `shipwreck` checkpoint now runs alongside the baseline detector. When multiple high-confidence tiled detections cluster around a large, edge-rich structure, the pipeline also adds a `shipwreck_candidate` fallback review record with its source bounding box and `evidence_count`.

## GhostVision-inspired quality scoring

The detector now includes a lightweight quality layer inspired by GhostVision's multi-observation summarization workflow. Every sonar detection receives a transparent `quality_score` and `quality_tier`, with supporting persistence, confidence-stability, and spatial-consistency fields. The single-frame pipeline uses the available evidence without changing the existing model confidence, while `src/detection_quality.py` also provides `aggregate_detections()` for sequential frames: repeated nearby detections of the same class are summarized into one track-like record and ranked by quality. These fields are included in JSON and CSV geotagged reports and shown in the dashboard detection cards.

This improves operational review and export traceability, but it is not a substitute for a trained tracker or real-survey evaluation. Frame aggregation should be used only when inputs are sequentially related and share the same pixel geometry.

## Known limitations (be upfront about these in your submission)

1. Trained on synthetic, not real, sonar imagery — treat the current
   weights as a proof-of-concept, not survey-ready.
2. The autoencoder anomaly stage uses **per-image adaptive thresholding**
   (robust median/MAD z-score against that image's own background, in
   `run_autoencoder_scan` in `detection_pipeline.py`), not a fixed constant.
   An earlier version used one global threshold calibrated against synthetic
   data, which broke badly on real sonar — one real test image produced
   2000+ overlapping anomaly boxes because real acoustic noise statistics
   didn't match the synthetic calibration. The adaptive version fixes this
   by asking "is this cell unusual *for this image*" instead of "does this
   cell exceed a number tuned on different data" — tested down to a handful
   of sane detections on the same real image that broke the old approach.
   It's still a heuristic (`ANOMALY_Z_THRESHOLD` in the same file controls
   sensitivity) and should be validated against more real survey data before
   being trusted operationally.
3. `pixel_to_geo` assumes a simple flat-earth, constant-swath-width
   geometry — good enough for a demo/prototype, but a production system
   should account for slant-range correction and vehicle roll/pitch.
4. Very wide/thin real sonar waterfalls are handled through overlapping tiled inference, but each tiled model pass still has CPU cost and duplicate merging is heuristic. The AI4Shipwrecks model is trained only for shipwrecks and does not provide ghost-net supervision.
5. The baseline debris YOLO detector is still trained on synthetic debris shapes and can produce false positives on real sonar texture. The AI4Shipwrecks shipwreck checkpoint is real-data fine-tuned, but its current held-out-site metrics remain modest and it should not be presented as GhostVision-equivalent without additional real training data and calibration.
