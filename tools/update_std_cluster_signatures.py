import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.core.lens_detector import LensDetector
from src.core.zone_analyzer_2d import InkMaskConfig, bgr_to_lab_float, build_ink_mask, circle_mask


def _read_image(path: Path) -> np.ndarray:
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _compute_clusters(image_bgr: np.ndarray, lens, k: int, r_inner: float, r_outer: float) -> list[dict]:
    h, w = image_bgr.shape[:2]
    cx = float(lens.center_x)
    cy = float(lens.center_y)
    radius = float(lens.radius)

    lens_mask = circle_mask((h, w), cx, cy, radius)
    ink_mask = build_ink_mask(image_bgr, lens_mask, InkMaskConfig())
    ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)

    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    valid = (rr >= radius * r_inner) & (rr <= radius * r_outer) & (lens_mask > 0)
    mask = valid & (ink_mask > 0)

    if np.sum(mask) < max(100, k * 50):
        raise ValueError("Not enough ink pixels for clustering")

    lab = bgr_to_lab_float(image_bgr)
    ab = lab[:, :, 1:3]
    samples = ab[mask].astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, _ = cv2.kmeans(samples, k, None, criteria, 5, flags)

    labels = labels.flatten()
    total_pixels = int(samples.shape[0])

    clusters = []
    for idx in range(k):
        idx_mask = labels == idx
        count = int(np.sum(idx_mask))
        if count == 0:
            continue
        mean_lab = lab[mask][idx_mask].mean(axis=0)
        clusters.append(
            {
                "cluster_id": int(idx),
                "pixels": count,
                "coverage": float(count / total_pixels) if total_pixels > 0 else 0.0,
                "mean_ab": [float(mean_lab[1]), float(mean_lab[2])],
            }
        )

    clusters.sort(key=lambda c: c["coverage"], reverse=True)
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Update STD cluster signatures (mean_ab) from a STD image.")
    parser.add_argument("--sku", required=True, help="SKU config path (e.g., config/sku_db/SKU_INK1.json)")
    parser.add_argument("--image", required=True, help="STD image path")
    args = parser.parse_args()

    sku_path = Path(args.sku)
    img_path = Path(args.image)

    if not sku_path.exists():
        raise SystemExit(f"SKU config not found: {sku_path}")
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    config = json.loads(sku_path.read_text(encoding="utf-8"))
    params = config.get("params", {})
    ink_profile = params.get("ink_profile", {})
    std_clusters = ink_profile.get("std_clusters")
    if not std_clusters:
        raise SystemExit("ink_profile.std_clusters not found in SKU config")

    k = int(ink_profile.get("k", 0))
    if k <= 0:
        raise SystemExit("ink_profile.k must be > 0")

    image = _read_image(img_path)
    lens = LensDetector().detect(image)

    optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
    center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
    r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)
    r_outer = 0.95

    clusters = _compute_clusters(image, lens, k, r_inner, r_outer)
    if len(clusters) != len(std_clusters):
        raise SystemExit(f"Cluster count mismatch: {len(clusters)} vs std_clusters {len(std_clusters)}")

    for idx, std in enumerate(std_clusters):
        std.setdefault("signature", {})
        std["signature"]["mean_ab"] = clusters[idx]["mean_ab"]

    sku_path.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {sku_path} with mean_ab from {img_path}")


if __name__ == "__main__":
    main()
