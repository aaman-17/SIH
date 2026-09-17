from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--lr0', type=float, default=0.0005)
    parser.add_argument('--freeze', type=int, default=0)
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=2,
        project=str(args.project),
        name='shipwreck_realdata',
        pretrained=True,
        patience=args.patience,
        lr0=args.lr0,
        lrf=0.01,
        optimizer='AdamW',
        freeze=args.freeze,
        cos_lr=True,
        close_mosaic=5,
        cache=False,
        amp=False,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    print('training_complete', results.save_dir)
    print('best_weights', Path(results.save_dir) / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
