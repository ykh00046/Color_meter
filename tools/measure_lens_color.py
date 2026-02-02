#!/usr/bin/env python3
"""
Measure radial Lab stats using the v7 signature pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config.v7_paths import V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.signature.radial_signature import build_radial_signature, to_polar


def _segment_ranges(segments: list[dict], length: int) -> list[tuple[str, int, int]]:
    ranges = []
    for seg in segments:
        name = str(seg.get("name", "segment"))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 1.0))
        s = int(round(start * length))
        e = int(round(end * length))
        s = max(0, min(length - 1, s))
        e = max(s + 1, min(length, e))
        ranges.append((name, s, e))
    if not ranges:
        ranges.append(("all", 0, length))
    return ranges


def _mean_lab(curve: np.ndarray) -> list[float]:
    if curve.size == 0:
        return [0.0, 0.0, 0.0]
    mean = curve.mean(axis=0)
    return [float(mean[0]), float(mean[1]), float(mean[2])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Lens image path")
    ap.add_argument("--sku", default="", help="SKU id (optional)")
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

    polar = to_polar(
        bgr,
        geom,
        R=int(cfg["polar"]["R"]),
        T=int(cfg["polar"]["T"]),
    )
    mean_curve, p95_curve, _ = build_radial_signature(
        polar,
        r_start=float(cfg["signature"]["r_start"]),
        r_end=float(cfg["signature"]["r_end"]),
    )

    segments = cfg.get("signature", {}).get("segments", [])
    ranges = _segment_ranges(segments, int(mean_curve.shape[0]))

    print(f"\nRadial Lab summary for {image_path}")
    print("=" * 72)
    print(f"Segments: {[name for name, _, _ in ranges]}")

    output = []
    for name, s, e in ranges:
        seg_mean = _mean_lab(mean_curve[s:e])
        seg_p95 = _mean_lab(p95_curve[s:e])
        output.append({"name": name, "mean_lab": seg_mean, "p95_lab": seg_p95})
        print(f"\n{name}")
        print(f"  mean L*a*b*: {seg_mean[0]:.2f}, {seg_mean[1]:.2f}, {seg_mean[2]:.2f}")
        print(f"  p95  L*a*b*: {seg_p95[0]:.2f}, {seg_p95[1]:.2f}, {seg_p95[2]:.2f}")

    print("\n" + "=" * 72)
    print("Suggested JSON snippet:")
    print(json.dumps({"segments": output}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
