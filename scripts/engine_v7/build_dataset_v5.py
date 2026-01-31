import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _imread_korean(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _match_pair(white_path: Path) -> Optional[Path]:
    name = white_path.name
    lower = name.lower()
    if "_white" not in lower:
        return None
    black_name = name[: lower.rfind("_white")] + "_black" + name[lower.rfind("_white") + len("_white") :]
    black_path = white_path.with_name(black_name)
    return black_path if black_path.exists() else None


def _write_label_mask(masks: Dict[str, np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    label = np.zeros((h, w), dtype=np.uint8)
    ring = masks.get("ring")
    dot = masks.get("dot")
    clear = masks.get("clear")
    if ring is not None:
        label[ring] = 1
    if dot is not None:
        label[dot] = 2
    if clear is not None:
        label[clear] = 3
    return label


def build_dataset(data_root: Path, out_root: Path, cfg_path: Path) -> None:
    from src.engine_v7.core.plate.plate_engine import compute_plate_artifacts

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    overlays_dir = out_root / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    white_files = sorted([p for p in data_root.iterdir() if p.is_file() and "_white" in p.name.lower()])
    total = len(white_files)
    success = 0

    for idx, white_path in enumerate(white_files, start=1):
        black_path = _match_pair(white_path)
        if black_path is None:
            continue

        w_img = _imread_korean(white_path)
        b_img = _imread_korean(black_path)
        if w_img is None or b_img is None:
            continue

        artifacts = compute_plate_artifacts(w_img, b_img, cfg)
        masks = artifacts.get("masks")
        white_norm = artifacts.get("white_norm")
        if not masks or white_norm is None:
            continue

        label_mask = _write_label_mask(masks, white_norm.shape[:2])
        base_name = white_path.stem[: white_path.stem.lower().rfind("_white")]

        cv2.imwrite(str(images_dir / f"{base_name}.png"), white_norm)
        cv2.imwrite(str(masks_dir / f"{base_name}_mask.png"), label_mask)

        vis = white_norm.copy()
        vis[label_mask == 1] = (0, 0, 255)
        vis[label_mask == 2] = (0, 255, 0)
        vis[label_mask == 3] = (255, 0, 0)
        overlay = cv2.addWeighted(white_norm, 0.7, vis, 0.3, 0)
        cv2.imwrite(str(overlays_dir / f"{base_name}_vis.png"), overlay)

        success += 1
        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] processed")

    print("Dataset build completed.")
    print(f"- Output: {out_root.resolve()}")
    print(f"- Success: {success} / {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input folder containing *_white.* and *_black.* pairs")
    parser.add_argument("--out", default="dataset_v1", help="Output folder for dataset")
    parser.add_argument("--cfg", default="src/engine_v7/configs/default.json", help="Config path")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    cfg_path = Path(args.cfg)

    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    build_dataset(data_root, out_root, cfg_path)


if __name__ == "__main__":
    main()
