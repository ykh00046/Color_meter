#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine_v7.core.plate.plate_engine import analyze_plate_pair, compute_plate_artifacts


def _load_cfg(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _blend_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> None:
    if mask is None or not np.any(mask):
        return
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0, dst=base)


def _render_masks_overlay(base_bgr: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    overlay = base_bgr.copy()
    _blend_mask(overlay, masks.get("ring"), (0, 200, 0))
    _blend_mask(overlay, masks.get("dot"), (0, 128, 255))
    _blend_mask(overlay, masks.get("clear"), (0, 255, 255))
    _blend_mask(overlay, masks.get("ink_mask"), (255, 0, 200), alpha=0.25)
    return overlay


def _render_print_masks_overlay(base_bgr: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    overlay = base_bgr.copy()
    _blend_mask(overlay, masks.get("ink_mask"), (255, 128, 0), alpha=0.35)  # print_all
    _blend_mask(overlay, masks.get("ink_mask_core"), (255, 0, 255), alpha=0.35)  # print_core
    _blend_mask(overlay, masks.get("clear"), (0, 255, 255), alpha=0.35)
    _blend_mask(overlay, masks.get("ring"), (0, 200, 0), alpha=0.25)
    _blend_mask(overlay, masks.get("dot"), (0, 128, 255), alpha=0.25)
    return overlay


def _render_ink_labels_overlay(base_bgr: np.ndarray, ink_masks: Dict[str, np.ndarray]) -> np.ndarray:
    overlay = base_bgr.copy()
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for idx, name in enumerate(sorted(ink_masks.keys())):
        color = palette[idx % len(palette)]
        _blend_mask(overlay, ink_masks.get(name), color, alpha=0.5)
    return overlay


def _render_alpha_heatmap(alpha: np.ndarray) -> np.ndarray:
    alpha_u8 = np.clip(alpha, 0.0, 1.0)
    alpha_u8 = (alpha_u8 * 255).astype(np.uint8)
    return cv2.applyColorMap(alpha_u8, cv2.COLORMAP_JET)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--white", required=True, help="Path to white image")
    ap.add_argument("--black", required=True, help="Path to black image")
    ap.add_argument("--cfg", default=str(Path("configs") / "default.json"))
    ap.add_argument("--out_dir", default=str(Path("results") / "v7" / "plate_validate"))
    ap.add_argument("--match_id", default="")
    args = ap.parse_args()

    white_path = Path(args.white)
    black_path = Path(args.black)
    cfg_path = Path(args.cfg)
    out_root = Path(args.out_dir)

    white_bgr = cv2.imread(str(white_path))
    black_bgr = cv2.imread(str(black_path))
    if white_bgr is None:
        raise SystemExit(f"Failed to read white image: {white_path}")
    if black_bgr is None:
        raise SystemExit(f"Failed to read black image: {black_path}")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = _load_cfg(cfg_path)
    plate_cfg = cfg.get("plate", {})

    run_id = datetime.now().strftime("plate_validate_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    match_id = args.match_id.strip() or f"{white_path.stem}_{black_path.stem}"
    plate = analyze_plate_pair(white_bgr, black_bgr, plate_cfg, match_id=match_id)
    plate_path = run_dir / "plate_v1.2.json"
    plate_path.write_text(json.dumps(plate, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = compute_plate_artifacts(white_bgr, black_bgr, cfg)
    alpha = artifacts.get("alpha")
    masks = artifacts.get("masks", {})
    white_norm = artifacts.get("white_norm")
    black_aligned = artifacts.get("black_aligned")

    if white_norm is not None and masks:
        overlay = _render_masks_overlay(white_norm, masks)
        cv2.imwrite(str(run_dir / "plate_masks.png"), overlay)
        print_overlay = _render_print_masks_overlay(white_norm, masks)
        cv2.imwrite(str(run_dir / "debug_print_masks_overlay.png"), print_overlay)

    if alpha is not None:
        alpha_img = _render_alpha_heatmap(alpha)
        cv2.imwrite(str(run_dir / "plate_alpha.png"), alpha_img)

    if white_norm is not None:
        cv2.imwrite(str(run_dir / "white_norm.png"), white_norm)
    if black_aligned is not None:
        cv2.imwrite(str(run_dir / "black_aligned.png"), black_aligned)
    if white_norm is not None and black_aligned is not None:
        blend = cv2.addWeighted(white_norm, 0.5, black_aligned, 0.5, 0.0)
        cv2.imwrite(str(run_dir / "pair_overlay.png"), blend)
    if white_norm is not None and artifacts.get("ink_masks"):
        ink_overlay = _render_ink_labels_overlay(white_norm, artifacts["ink_masks"])
        cv2.imwrite(str(run_dir / "debug_ink_labels_overlay_on_white.png"), ink_overlay)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "plate_json": str(plate_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
