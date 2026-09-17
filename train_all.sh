#!/usr/bin/env bash
# Regenerates synthetic training data and retrains both models from scratch.
# On CPU-only machines this can take several minutes.
set -e

echo "== 1/4: Generating synthetic side-scan sonar dataset =="
python3 src/generate_synthetic_data.py data/synthetic_yolo

echo "== 2/4: Building CLAHE-matched normal patches for the autoencoder =="
python3 -c "
import cv2, os
from src.preprocessing import enhance_contrast_clahe, denoise_speckle
src_dir, out_dir = 'data/normal_patches', 'data/normal_patches_clahe'
os.makedirs(out_dir, exist_ok=True)
for f in os.listdir(src_dir):
    img = cv2.imread(os.path.join(src_dir, f), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(out_dir, f), enhance_contrast_clahe(denoise_speckle(img)))
print('done')
"

echo "== 3/4: Training YOLO11-Nano detector (this is the slow step on CPU) =="
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='data/synthetic_yolo/data.yaml',
    epochs=30, imgsz=256, batch=2, device='cpu',
    project='runs', name='sonar_yolo11n',
    workers=0, plots=False, cache=False, amp=False, mosaic=0.0,
)
import shutil
shutil.copy('runs/detect/sonar_yolo11n/weights/best.pt', 'models/yolo_sonar_best.pt')
"

echo "== 4/4: Training CNN autoencoder anomaly scorer =="
python3 src/autoencoder.py data/normal_patches_clahe models/autoencoder.pt

echo "Done. Run: streamlit run app.py"
