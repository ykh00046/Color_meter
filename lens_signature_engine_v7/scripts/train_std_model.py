#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.geometry.lens_geometry import detect_lens_circle
from core.measure.segmentation.color_masks import build_color_masks
from core.signature.fit import fit_std, fit_std_multi, fit_std_per_color
from core.signature.model_io import save_model
from core.utils import apply_white_balance


def load_cfg(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stds", nargs="+", required=True, help="STD image paths (1 or more)")
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

    bgrs = []
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

        paths = save_model(model, args.out_prefix)
        print(
            json.dumps(
                {"saved": paths, "n_std": model.meta.get("n_std", 1), "color_mode": "aggregate"},
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
