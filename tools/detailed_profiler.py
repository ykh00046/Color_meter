"""
Detailed radial profiling performance analysis (v7).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.config.v7_paths import V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.signature.radial_signature import to_polar
from src.engine_v7.core.utils import bgr_to_lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sku", default="")
    ap.add_argument("--cfg", default=str(V7_ROOT / "configs" / "default.json"))
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    cfg, _, _ = load_cfg_with_sku(args.cfg, args.sku or None, strict_unknown=False)
    geom = detect_lens_circle(bgr)

    print("=" * 80)
    print("  Detailed Radial Profiling Analysis (v7)")
    print("=" * 80)
    print(f"Image: {image_path.name}")
    print(f"Radius: {geom.r:.1f}px, polar R: {cfg['polar']['R']}, T: {cfg['polar']['T']}")
    print("=" * 80)

    start = time.perf_counter()
    polar = to_polar(bgr, geom, R=int(cfg["polar"]["R"]), T=int(cfg["polar"]["T"]))
    t1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    polar_lab = bgr_to_lab(polar)
    t2 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    L_profile = polar_lab[:, :, 0].mean(axis=0)
    a_profile = polar_lab[:, :, 1].mean(axis=0)
    b_profile = polar_lab[:, :, 2].mean(axis=0)
    t3 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    std_L = polar_lab[:, :, 0].std(axis=0)
    std_a = polar_lab[:, :, 1].std(axis=0)
    std_b = polar_lab[:, :, 2].std(axis=0)
    _ = (std_L, std_a, std_b)
    t4 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    if len(L_profile) >= 11:
        from scipy.signal import savgol_filter

        _ = savgol_filter(L_profile, 11, 3)
        _ = savgol_filter(a_profile, 11, 3)
        _ = savgol_filter(b_profile, 11, 3)
    t5 = (time.perf_counter() - start) * 1000

    total = t1 + t2 + t3 + t4 + t5

    print(f"\n{'Step':<40} {'Time (ms)':<12} {'%':<8}")
    print("-" * 80)
    print(f"{'1. to_polar (warpPolar)':<40} {t1:>10.2f}   {t1/total*100:>6.1f}%")
    print(f"{'2. bgr_to_lab':<40} {t2:>10.2f}   {t2/total*100:>6.1f}%")
    print(f"{'3. Mean per radius':<40} {t3:>10.2f}   {t3/total*100:>6.1f}%")
    print(f"{'4. Std per radius':<40} {t4:>10.2f}   {t4/total*100:>6.1f}%")
    print(f"{'5. Savgol smoothing':<40} {t5:>10.2f}   {t5/total*100:>6.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<40} {total:>10.2f}   {'100.0%':>7}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
