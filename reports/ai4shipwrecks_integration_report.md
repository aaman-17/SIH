# AI4Shipwrecks Integration Report

## Training configuration

The public AI4Shipwrecks masks were converted into overlapping 1024-pixel sonar tiles with 25% overlap. Each tile uses a union bounding box around the positive mask pixels, which avoids the fragmented connected-component labels used in the first run. The detector was fine-tuned for 12 epochs at 512-pixel training resolution on CPU, with batch size 4, learning rate 0.0005, and the first 10 model layers frozen.

## Held-out test result

The official held-out test split contains 956 tiles and 251 positive shipwreck tiles. The corrected union-label checkpoint achieved precision 0.2274, recall 0.1255, mAP50 0.0520, and mAP50-95 0.0169 at confidence 0.25. These are baseline results, not production-readiness evidence. The score is substantially more informative than the fragmented-label run, but additional training and site diversity are required.

## SIH integration

The checkpoint is bundled at `models/yolo_shipwreck_ai4shipwrecks.pt`. The SIH pipeline runs it in parallel with the existing synthetic-data debris detector and autoencoder. Long sonar waterfalls use overlapping tiled inference, and shipwreck detections retain their source and quality metadata. The original `shipwreck_candidate` heuristic remains as a fallback review signal.

## Supplied DM Wilson smoke test

The 5616 x 1728 `DM_Wilson_03.png` image produced eight shipwreck detections from the new model and one structure-heuristic shipwreck candidate. The strongest trained-model detection had 62.8% final confidence, and the heuristic candidate had 73.2% final confidence with 40 tiled evidence hits. Because this image is associated with the dataset’s training-site material, it is a debugging smoke test, not an unseen generalization test.

## Limitations

AI4Shipwrecks does not contain ghost-net labels. The new checkpoint therefore improves shipwreck recognition only. The current test mAP remains modest, and the model should not be described as GhostVision-equivalent. A production model requires more real labeled surveys, site-level validation, confidence calibration, temporal association across pings, and a separate ghost-net training set.
