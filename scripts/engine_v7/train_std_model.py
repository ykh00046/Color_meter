#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.measure.segmentation.color_masks import build_color_masks
from src.engine_v7.core.signature.fit import fit_std, fit_std_multi, fit_std_per_color
from src.engine_v7.core.signature.model_io import save_model
from src.engine_v7.core.utils import apply_white_balance


def load_cfg(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_curves(curves):
    stack = np.stack(curves, axis=0)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    p05 = np.percentile(stack, 5, axis=0).astype(np.float32)
    p95 = np.percentile(stack, 95, axis=0).astype(np.float32)
    return mean, p95, p05, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stds", nargs="+", help="STD image paths (1 or more)")
    ap.add_argument("--white_stds", nargs="+", help="White STD image paths (plate pair mode)")
    ap.add_argument("--black_stds", nargs="+", help="Black STD image paths (plate pair mode)")
    ap.add_argument("--std_pairs", nargs="+", help="STD pairs as 'white_path:black_path'")
    ap.add_argument("--cfg", default=str(Path("configs") / "default.json"))
    ap.add_argument("--out_prefix", required=True, help="Output prefix (writes .npz + .json)")
    ap.add_argument(
        "--color_mode",
        choices=["aggregate", "per_color"],
        default="aggregate",
        help="Color mode: 'aggregate' (default) for single model, 'per_color' for per-color models",
    )
    ap.add_argument(
        "--expected_ink_count",
        type=int,
        default=None,
        help="Expected number of ink colors (required for per_color mode)",
    )
    args = ap.parse_args()

    # Validate arguments
    if args.color_mode == "per_color":
        if args.expected_ink_count is None:
            raise SystemExit("Error: --expected_ink_count is required when --color_mode=per_color")
        if args.expected_ink_count < 1:
            raise SystemExit("Error: --expected_ink_count must be >= 1")

    cfg = load_cfg(args.cfg)
    wb_cfg = cfg.get("white_balance", {})

    pair_mode = bool(args.std_pairs or args.white_stds or args.black_stds)
    if not pair_mode and not args.stds:
        raise SystemExit("Error: Provide --stds or plate pair inputs (--std_pairs or --white_stds/--black_stds)")

    if pair_mode:
        if args.std_pairs:
            pairs = []
            for item in args.std_pairs:
                if ":" not in item:
                    raise SystemExit("Error: --std_pairs items must be 'white_path:black_path'")
                white_path, black_path = item.split(":", 1)
                pairs.append((white_path, black_path))
        else:
            if not args.white_stds or not args.black_stds:
                raise SystemExit("Error: --white_stds and --black_stds are required together")
            if len(args.white_stds) != len(args.black_stds):
                raise SystemExit("Error: --white_stds and --black_stds count mismatch")
            pairs = list(zip(args.white_stds, args.black_stds))
    else:
        pairs = []

    bgrs = []
    raw_white_bgrs = []
    raw_black_bgrs = []
    if pair_mode:
        for white_path, black_path in pairs:
            white_bgr = cv2.imread(white_path)
            black_bgr = cv2.imread(black_path)
            if white_bgr is None:
                raise SystemExit(f"Failed to read white STD image: {white_path}")
            if black_bgr is None:
                raise SystemExit(f"Failed to read black STD image: {black_path}")
            raw_white_bgrs.append(white_bgr)
            raw_black_bgrs.append(black_bgr)

            white_for_std = white_bgr
            if wb_cfg.get("enabled", False):
                geom = detect_lens_circle(white_bgr)
                white_for_std, _ = apply_white_balance(white_bgr, geom, wb_cfg)
            bgrs.append(white_for_std)
    else:
        for p in args.stds:
            bgr = cv2.imread(p)
            if bgr is None:
                raise SystemExit(f"Failed to read STD image: {p}")
            if wb_cfg.get("enabled", False):
                geom = detect_lens_circle(bgr)
                bgr, _ = apply_white_balance(bgr, geom, wb_cfg)
            bgrs.append(bgr)

    if args.color_mode == "aggregate":
        # Original aggregate mode
        if len(bgrs) == 1:
            model = fit_std(
                bgrs[0],
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )
        else:
            model = fit_std_multi(
                bgrs,
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )

        if pair_mode:
            from src.engine_v7.core.plate.plate_engine import compute_plate_signatures

            plate_results = []
            plate_meta = None
            for white_bgr, black_bgr in zip(raw_white_bgrs, raw_black_bgrs):
                sig, meta = compute_plate_signatures(white_bgr, black_bgr, cfg)
                plate_results.append(sig)
                if plate_meta is None:
                    plate_meta = meta

            ring_lab_mean, ring_lab_p95, ring_lab_p05, ring_lab_std = _aggregate_curves(
                [s["plate_ring_core_radial_lab_mean"] for s in plate_results]
            )
            dot_lab_mean, dot_lab_p95, dot_lab_p05, dot_lab_std = _aggregate_curves(
                [s["plate_dot_core_radial_lab_mean"] for s in plate_results]
            )
            ring_alpha_mean, ring_alpha_p95, ring_alpha_p05, ring_alpha_std = _aggregate_curves(
                [s["plate_ring_core_radial_alpha_mean"] for s in plate_results]
            )
            dot_alpha_mean, dot_alpha_p95, dot_alpha_p05, dot_alpha_std = _aggregate_curves(
                [s["plate_dot_core_radial_alpha_mean"] for s in plate_results]
            )

            model.plate_signatures = {
                "plate_ring_core_radial_lab_mean": ring_lab_mean,
                "plate_ring_core_radial_lab_p95": ring_lab_p95,
                "plate_ring_core_radial_lab_p05": ring_lab_p05,
                "plate_ring_core_radial_lab_std": ring_lab_std,
                "plate_ring_core_radial_alpha_mean": ring_alpha_mean,
                "plate_ring_core_radial_alpha_p95": ring_alpha_p95,
                "plate_ring_core_radial_alpha_p05": ring_alpha_p05,
                "plate_ring_core_radial_alpha_std": ring_alpha_std,
                "plate_dot_core_radial_lab_mean": dot_lab_mean,
                "plate_dot_core_radial_lab_p95": dot_lab_p95,
                "plate_dot_core_radial_lab_p05": dot_lab_p05,
                "plate_dot_core_radial_lab_std": dot_lab_std,
                "plate_dot_core_radial_alpha_mean": dot_alpha_mean,
                "plate_dot_core_radial_alpha_p95": dot_alpha_p95,
                "plate_dot_core_radial_alpha_p05": dot_alpha_p05,
                "plate_dot_core_radial_alpha_std": dot_alpha_std,
            }
            model.plate_meta = plate_meta

        paths = save_model(model, args.out_prefix)
        print(
            json.dumps(
                {
                    "saved": paths,
                    "n_std": model.meta.get("n_std", 1),
                    "color_mode": "aggregate",
                    "plate_pairs": len(raw_white_bgrs) if pair_mode else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.color_mode == "per_color":
        # Per-color mode
        print(f"Per-color mode: segmenting {len(bgrs)} images with expected_ink_count={args.expected_ink_count}")

        # Generate color masks for all images
        color_masks_list = []
        color_metadata_list = []

        for idx, bgr in enumerate(bgrs):
            masks, metadata = build_color_masks(bgr, cfg, expected_k=args.expected_ink_count)
            color_masks_list.append(masks)
            color_metadata_list.append(metadata)

            print(
                f"  Image {idx+1}: Found {len(metadata['colors'])} colors, " f"warnings={metadata.get('warnings', [])}"
            )

        # Train per-color models
        try:
            per_color_models, per_color_metadata = fit_std_per_color(
                bgrs,
                color_masks_list,
                color_metadata_list,
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )
        except ValueError as e:
            raise SystemExit(f"Error training per-color models: {e}")

        # Save per-color models
        out_prefix_path = Path(args.out_prefix)
        colors_dir = out_prefix_path.parent / "colors"
        colors_dir.mkdir(parents=True, exist_ok=True)

        saved_models = {}
        for color_id, model in per_color_models.items():
            color_dir = colors_dir / color_id
            color_dir.mkdir(exist_ok=True)

            color_prefix = str(color_dir / out_prefix_path.name)
            paths = save_model(model, color_prefix)
            saved_models[color_id] = paths

            print(f"  Saved {color_id}: {paths}")

        # Save aggregated color metadata
        color_metadata_path = colors_dir / "color_metadata.json"
        with open(color_metadata_path, "w", encoding="utf-8") as f:
            json.dump(per_color_metadata, f, ensure_ascii=False, indent=2)

        print(
            json.dumps(
                {
                    "color_mode": "per_color",
                    "expected_ink_count": args.expected_ink_count,
                    "saved_models": saved_models,
                    "color_metadata": str(color_metadata_path),
                    "n_std": len(bgrs),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
