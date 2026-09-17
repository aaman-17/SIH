from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

VAL_SITES = {"Montana", "Pewabic", "WP_Rend"}


def starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    out = list(range(0, length - tile + 1, step))
    last = length - tile
    if out[-1] != last:
        out.append(last)
    return out


def site_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def components_to_yolo(mask: np.ndarray, min_area: int, mode: str = "union") -> list[str]:
    binary = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    if int(binary.sum()) < min_area:
        return []
    if mode == "union":
        ys, xs = np.where(binary > 0)
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
        return [f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    labels: list[str] = []
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if int(area) < min_area:
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        labels.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    return labels


def process_split(src_root: Path, out_root: Path, split: str, tile: int, overlap: float, min_area: int, mode: str) -> dict:
    images = src_root / split / "images"
    masks = src_root / split / "labels"
    out_split = "val" if split == "train" else "test"
    count = {"images": 0, "positive": 0, "negative": 0, "objects": 0}
    for image_path in sorted(images.glob("*.png")):
        mask_path = masks / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != mask.shape:
            raise ValueError(f"Invalid image/mask pair: {image_path}")
        h, w = image.shape
        site = site_from_stem(image_path.stem)
        target_split = out_split if split == "test" or site in VAL_SITES else "train"
        for y in starts(h, tile, overlap):
            for x in starts(w, tile, overlap):
                image_tile = image[y:min(y + tile, h), x:min(x + tile, w)]
                mask_tile = mask[y:min(y + tile, h), x:min(x + tile, w)]
                labels = components_to_yolo(mask_tile, min_area, mode)
                tile_name = f"{image_path.stem}_x{x:05d}_y{y:05d}"
                image_out = out_root / "images" / target_split / f"{tile_name}.png"
                label_out = out_root / "labels" / target_split / f"{tile_name}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_out), image_tile)
                label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                count["images"] += 1
                count["objects"] += len(labels)
                count["positive" if labels else "negative"] += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--label-mode", choices=["union", "components"], default="union")
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    counts = {}
    counts["train+val"] = process_split(args.source, args.output, "train", args.tile, args.overlap, args.min_area, args.label_mode)
    counts["test"] = process_split(args.source, args.output, "test", args.tile, args.overlap, args.min_area, args.label_mode)
    (args.output / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "shipwreck"},
    }, sort_keys=False), encoding="utf-8")
    (args.output / "conversion_stats.yaml").write_text(yaml.safe_dump(counts, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(counts, sort_keys=False))


if __name__ == "__main__":
    main()
