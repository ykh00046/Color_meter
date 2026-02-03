"""
V7 Production Colors sub-router.

Routes:
  POST /extract_production_colors - Extract lens colors using Production v1.0.7 algorithm
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from src.config.v7_paths import V7_RESULTS

from .v7_helpers import NumpyEncoder, _require_role, _save_single_upload, logger

router = APIRouter()


def _generate_color_palette_image(colors: dict, output_path: Path) -> bool:
    """
    Generate color palette visualization image.

    Args:
        colors: Dictionary of color categories with RGB values
        output_path: Path to save the palette image

    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if not colors:
            return False

        sorted_colors = sorted(colors.items(), key=lambda x: x[1].get("pct_roi", 0), reverse=True)

        n_colors = len(sorted_colors)
        fig_height = max(2, n_colors * 0.6)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n_colors)
        ax.axis("off")

        for i, (category, info) in enumerate(sorted_colors):
            y_pos = n_colors - 1 - i
            rgb = info.get("color_rgb", [128, 128, 128])
            rgb_normalized = [c / 255.0 for c in rgb[:3]]

            rect = Rectangle((0.5, y_pos + 0.1), 2, 0.8, facecolor=rgb_normalized, edgecolor="black", linewidth=1)
            ax.add_patch(rect)

            name_kr = info.get("color_name_kr", category)
            pct = info.get("pct_roi", 0)
            hex_code = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

            ax.text(3.0, y_pos + 0.5, f"{name_kr}", fontsize=10, va="center", fontweight="bold")
            ax.text(6.0, y_pos + 0.5, f"{hex_code}", fontsize=9, va="center", family="monospace")
            ax.text(8.5, y_pos + 0.5, f"{pct:.1f}%", fontsize=9, va="center")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return True

    except Exception as e:
        logger.warning(f"Failed to generate color palette image: {e}")
        return False


@router.post("/extract_production_colors")
async def extract_production_colors(
    x_user_role: Optional[str] = Header(default=""),
    file: UploadFile = File(...),
    n_clusters: int = Form(8),
    use_dynamic_bg: bool = Form(True),
    use_roi_extraction: bool = Form(True),
    roi_method: str = Form("largest_component"),
    l_weight: float = Form(0.3),
    merge_threshold: float = Form(4.0),
    min_cluster_percentage: float = Form(1.5),
    seed: int = Form(42),
):
    """
    Extract lens colors using Production v1.0.7 algorithm.

    Parameters:
    - file: Image file (JPEG, PNG, etc.)
    - n_clusters: Number of initial clusters (default: 8)
    - use_dynamic_bg: Use dynamic background threshold (default: True)
    - use_roi_extraction: Extract ROI to remove dust/shadow (default: True)
    - roi_method: ROI extraction method ('largest_component' or 'circular')
    - l_weight: L channel weight in Lab clustering (default: 0.3)
    - merge_threshold: Delta-E threshold for cluster merging (default: 4.0)
    - min_cluster_percentage: Minimum cluster percentage to keep (default: 1.5)
    - seed: Random seed for reproducibility (default: 42)

    Returns:
    - colors: Dictionary of detected colors by category
    - main_colors: Top 3 dominant colors
    - color_groups: Colors grouped by tone (Cool/Warm/Natural etc.)
    - summary: Analysis summary
    - coverage_info: ROI and filtering statistics
    - artifacts: Generated image paths
    """
    _require_role("operator", x_user_role)

    # Validate roi_method
    if roi_method not in ("largest_component", "circular"):
        raise HTTPException(status_code=400, detail="roi_method must be 'largest_component' or 'circular'")

    # Create run directory
    run_id = datetime.now().strftime("prod_colors_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    input_path = run_dir / file.filename
    await _save_single_upload(file, input_path)

    # Load image
    bgr = cv2.imread(str(input_path))
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {file.filename}")

    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Import production color extractor
    try:
        from src.engine_v7.core.measure.segmentation.production_color_extractor import (
            extract_colors_production_v107,
            get_color_groups,
        )
    except ImportError as e:
        logger.error(f"Failed to import production_color_extractor: {e}")
        raise HTTPException(status_code=500, detail="Production color extractor module not available")

    # Extract colors
    try:
        colors, coverage_info = extract_colors_production_v107(
            rgb,
            n_clusters=n_clusters,
            seed=seed,
            use_dynamic_bg=use_dynamic_bg,
            use_roi_extraction=use_roi_extraction,
            roi_method=roi_method,
            use_lab_clustering=True,
            l_weight=l_weight,
            use_two_band_sampling=True,
            high_chroma_percentile=40,
            low_chroma_samples_base=2000,
            use_dynamic_low_samples=True,
            dynamic_low_ratio=0.05,
            use_cluster_merging=True,
            merge_threshold=merge_threshold,
            min_cluster_percentage=min_cluster_percentage,
            highlight_l_threshold=92,
            highlight_chroma_threshold=6,
            use_dynamic_highlight_chroma=False,
            enable_auto_retry=True,
            min_clusters_threshold=2,
            use_trimmed_mean=True,
            trim_percent=10,
        )
    except Exception as e:
        logger.exception(f"Color extraction failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Color extraction failed: {str(e)}")

    # Check if extraction succeeded
    if not colors:
        return {
            "success": False,
            "error": coverage_info.get("error", "No colors extracted"),
            "run_id": run_id,
            "coverage_info": coverage_info,
            "artifacts": {"original": f"/v7_results/{run_id}/{file.filename}"},
        }

    # Sort colors by percentage
    sorted_colors = sorted(colors.items(), key=lambda x: x[1].get("pct_roi", 0), reverse=True)
    main_colors = sorted_colors[:3]

    # Get color groups
    color_groups = get_color_groups(colors)

    # Build summary
    dominant_group = None
    if color_groups:
        dominant_group = max(color_groups.items(), key=lambda x: x[1].get("total_pct_roi", 0))[0]

    summary = {
        "image_name": file.filename,
        "total_colors": len(colors),
        "main_color": main_colors[0][1].get("color_name_kr") if main_colors else None,
        "dominant_group": dominant_group,
    }

    # Generate artifacts
    artifacts = {"original": f"/v7_results/{run_id}/{file.filename}"}

    # Generate color palette image
    palette_path = run_dir / "color_palette.png"
    if _generate_color_palette_image(colors, palette_path):
        artifacts["color_palette"] = f"/v7_results/{run_id}/color_palette.png"

    # Convert colors dict for JSON serialization (numpy arrays to lists)
    colors_serializable = {}
    for cat, info in colors.items():
        colors_serializable[cat] = {
            "color_rgb": [int(c) for c in info.get("color_rgb", [0, 0, 0])],
            "pct_roi": float(info.get("pct_roi", 0)),
            "cluster_count": int(info.get("cluster_count", 1)),
            "color_name_kr": info.get("color_name_kr", cat),
            "hex": "#{:02x}{:02x}{:02x}".format(
                int(info["color_rgb"][0]), int(info["color_rgb"][1]), int(info["color_rgb"][2])
            ),
        }

    # Convert main_colors for serialization
    main_colors_serializable = []
    for cat, info in main_colors:
        main_colors_serializable.append(
            {
                "category": cat,
                "color_rgb": [int(c) for c in info.get("color_rgb", [0, 0, 0])],
                "pct_roi": float(info.get("pct_roi", 0)),
                "color_name_kr": info.get("color_name_kr", cat),
                "hex": "#{:02x}{:02x}{:02x}".format(
                    int(info["color_rgb"][0]), int(info["color_rgb"][1]), int(info["color_rgb"][2])
                ),
            }
        )

    # Build response
    response = {
        "success": True,
        "run_id": run_id,
        "colors": colors_serializable,
        "main_colors": main_colors_serializable,
        "color_groups": color_groups,
        "summary": summary,
        "coverage_info": coverage_info,
        "artifacts": artifacts,
    }

    # Save result to JSON
    result_path = run_dir / "production_colors.json"
    result_path.write_text(json.dumps(response, ensure_ascii=False, indent=2, cls=NumpyEncoder), encoding="utf-8")

    # Return JSON-serializable response
    return json.loads(json.dumps(response, cls=NumpyEncoder))
