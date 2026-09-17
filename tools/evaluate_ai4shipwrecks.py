from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split='test',
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        device='cpu',
        plots=True,
        project=str(args.output.parent),
        name=args.output.stem,
        exist_ok=True,
    )
    box = metrics.box
    payload = {
        'weights': str(args.weights),
        'data': str(args.data),
        'split': 'test',
        'images': int(getattr(metrics, 'nt_per_class', [0])[0]) if hasattr(metrics, 'nt_per_class') else None,
        'precision': float(box.mp),
        'recall': float(box.mr),
        'map50': float(box.map50),
        'map50_95': float(box.map),
        'per_class_map50': [float(x) for x in getattr(box, 'maps', [])],
    }
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
