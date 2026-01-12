#!/usr/bin/env python3
"""
Quick test script for per-color training functionality.
Creates synthetic 2-color images and trains per-color models.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.measure.segmentation.color_masks import build_color_masks
from core.signature.fit import fit_std_per_color


def create_synthetic_2color_image(output_path: str):
    """Create a synthetic 2-color lens image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cy, cx = 200, 200

    # Outer: Light gap - RGB(180, 180, 180)
    cv2.circle(img, (cx, cy), 180, (180, 180, 180), -1)

    # Inner: Dark ink - RGB(40, 40, 40)
    cv2.circle(img, (cx, cy), 120, (40, 40, 40), -1)

    cv2.imwrite(output_path, img)
    print(f"Created synthetic image: {output_path}")


def main():
    print("=" * 60)
    print("Per-Color Training Test")
    print("=" * 60)

    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp(prefix="per_color_test_"))
    print(f"\nTemporary directory: {temp_dir}")

    try:
        # Load default config
        cfg_path = Path("configs/default.json")
        if not cfg_path.exists():
            print(f"Error: Config not found at {cfg_path}")
            return

        with open(cfg_path) as f:
            cfg = json.load(f)

        # Create 3 synthetic images
        img_paths = []
        for i in range(3):
            img_path = temp_dir / f"std_{i}.png"
            create_synthetic_2color_image(str(img_path))
            img_paths.append(img_path)

        print(f"\nCreated {len(img_paths)} test images")

        # Load images
        bgrs = []
        for p in img_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"Error: Failed to read {p}")
                return
            bgrs.append(bgr)

        print("\nGenerating color masks...")
        # Generate color masks
        color_masks_list = []
        color_metadata_list = []
        expected_k = 2

        for idx, bgr in enumerate(bgrs):
            masks, metadata = build_color_masks(bgr, cfg, expected_k=expected_k)
            color_masks_list.append(masks)
            color_metadata_list.append(metadata)

            print(f"  Image {idx+1}: Found {len(metadata['colors'])} colors")
            for color_info in metadata["colors"]:
                print(
                    f"    {color_info['color_id']}: L*={color_info['lab_centroid'][0]:.1f}, "
                    f"role={color_info['role']}, area={color_info['area_ratio']:.2f}"
                )

        print("\nTraining per-color models...")
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
        except Exception as e:
            print(f"Error training per-color models: {e}")
            import traceback

            traceback.print_exc()
            return

        print(f"\nSuccessfully trained {len(per_color_models)} per-color models:")
        for color_id, model in per_color_models.items():
            print(f"  {color_id}:")
            print(f"    Geometry: cx={model.geom.cx:.1f}, cy={model.geom.cy:.1f}, r={model.geom.r:.1f}")
            print(f"    Signature shape: {model.radial_lab_mean.shape}")
            print(f"    Has std: {model.radial_lab_std is not None}")
            print(f"    n_std: {model.meta.get('n_std')}")

        print("\nPer-color metadata:")
        for color_id, metadata in per_color_metadata.items():
            print(f"  {color_id}: {metadata}")

        print("\n✅ Per-color training test PASSED!")

    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
