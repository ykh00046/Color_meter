"""
Plate vs Plate-Lite A/B comparison script.

Usage:
  python scripts/ab_compare_plate_lite.py --input PATH_TO_DIR_WITH_WHITE_BLACK
  python scripts/ab_compare_plate_lite.py --pairs_csv pairs.csv

Notes:
  - Plate-Lite requires paired white/black images for each sample.
  - This script compares plate vs plate_lite per sample, not across samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.plate.plate_engine import analyze_plate_lite_pair, analyze_plate_pair
from src.engine_v7.core.utils import cie2000_deltaE

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PairItem:
    white: Path
    black: Path
    sample_id: str


def _collect_pairs_from_dir(input_dir: Path) -> List[PairItem]:
    white_dir = input_dir / "white"
    black_dir = input_dir / "black"
    if not white_dir.exists() or not black_dir.exists():
        raise FileNotFoundError("input dir must contain 'white' and 'black' subdirectories")

    black_map = {p.stem: p for p in black_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
    pairs: List[PairItem] = []
    for white_path in white_dir.iterdir():
        if white_path.suffix.lower() not in VALID_EXTS:
            continue
        black_path = black_map.get(white_path.stem)
        if black_path is None:
            continue
        pairs.append(PairItem(white=white_path, black=black_path, sample_id=white_path.stem))

    if not pairs:
        raise ValueError("no matching white/black pairs found")
    return pairs


def _collect_pairs_from_csv(csv_path: Path) -> List[PairItem]:
    pairs: List[PairItem] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            white = Path(row.get("white", "")).expanduser()
            black = Path(row.get("black", "")).expanduser()
            if not white.exists() or not black.exists():
                continue
            sample_id = row.get("sample_id") or white.stem
            pairs.append(PairItem(white=white, black=black, sample_id=sample_id))
    if not pairs:
        raise ValueError("no valid rows in pairs_csv")
    return pairs


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"failed to load image: {path}")
    return img


def _extract_zone_alpha_plate(plate: Dict[str, Any], zone: str) -> Optional[float]:
    core = plate.get("plates", {}).get(zone, {}).get("core", {})
    alpha = core.get("alpha")
    if not isinstance(alpha, dict):
        return None
    return alpha.get("mean")


def _extract_zone_lab_plate(plate: Dict[str, Any], zone: str) -> Optional[List[float]]:
    core = plate.get("plates", {}).get(zone, {}).get("core", {})
    lab = core.get("lab")
    if not isinstance(lab, dict):
        return None
    return lab.get("mean")


def _extract_zone_alpha_lite(lite: Dict[str, Any], zone: str) -> Optional[float]:
    zone_out = lite.get("zones", {}).get(zone, {})
    if zone_out.get("empty"):
        return None
    return zone_out.get("alpha_mean")


def _extract_zone_lab_lite(lite: Dict[str, Any], zone: str) -> Optional[List[float]]:
    zone_out = lite.get("zones", {}).get(zone, {})
    if zone_out.get("empty"):
        return None
    return zone_out.get("ink_lab")


def _corrcoef_safe(values_a: List[float], values_b: List[float]) -> Optional[float]:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    a = np.array(values_a, dtype=np.float32)
    b = np.array(values_b, dtype=np.float32)
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _collect_zone_alphas(plate: Dict[str, Any], lite: Dict[str, Any]) -> Tuple[List[float], List[float], List[str]]:
    zones = [("ring", "ring_core"), ("dot", "dot_core"), ("clear", "clear")]
    plate_alphas: List[float] = []
    lite_alphas: List[float] = []
    names: List[str] = []
    for plate_zone, lite_zone in zones:
        pa = _extract_zone_alpha_plate(plate, plate_zone)
        la = _extract_zone_alpha_lite(lite, lite_zone)
        if pa is None or la is None:
            continue
        plate_alphas.append(float(pa))
        lite_alphas.append(float(la))
        names.append(plate_zone)
    return plate_alphas, lite_alphas, names


def _delta_e_zone_mean(plate: Dict[str, Any], lite: Dict[str, Any]) -> Optional[float]:
    zones = [("ring", "ring_core"), ("dot", "dot_core")]
    values: List[float] = []
    for plate_zone, lite_zone in zones:
        p_lab = _extract_zone_lab_plate(plate, plate_zone)
        l_lab = _extract_zone_lab_lite(lite, lite_zone)
        if p_lab is None or l_lab is None:
            continue
        de = float(cie2000_deltaE(np.array(p_lab), np.array(l_lab)))
        values.append(de)
    if not values:
        return None
    return float(np.mean(values))


def _stability_score(alpha_runs: List[List[float]]) -> Optional[float]:
    if not alpha_runs:
        return None
    runs = np.array(alpha_runs, dtype=np.float32)
    if runs.ndim != 2 or runs.shape[0] < 2:
        return None
    return float(np.mean(np.std(runs, axis=0)))


def _analyze_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    runs: int,
) -> Dict[str, Any]:
    plate_cfg = cfg.get("plate", {})
    lite_cfg = cfg.get("plate_lite", {})

    plate_times: List[float] = []
    lite_times: List[float] = []
    plate_alpha_runs: List[List[float]] = []
    lite_alpha_runs: List[List[float]] = []

    plate_ref: Optional[Dict[str, Any]] = None
    lite_ref: Optional[Dict[str, Any]] = None

    for i in range(runs):
        t0 = perf_counter()
        plate = analyze_plate_pair(white_bgr, black_bgr, plate_cfg)
        plate_times.append((perf_counter() - t0) * 1000.0)

        t0 = perf_counter()
        lite = analyze_plate_lite_pair(
            white_bgr, black_bgr, lite_cfg, plate_cfg, expected_k=cfg.get("expected_ink_count")
        )
        lite_times.append((perf_counter() - t0) * 1000.0)

        plate_alphas, lite_alphas, _ = _collect_zone_alphas(plate, lite)
        plate_alpha_runs.append(plate_alphas)
        lite_alpha_runs.append(lite_alphas)

        if i == 0:
            plate_ref = plate
            lite_ref = lite

    plate_ref = plate_ref or {}
    lite_ref = lite_ref or {}
    plate_alphas, lite_alphas, zone_names = _collect_zone_alphas(plate_ref, lite_ref)

    alpha_corr = _corrcoef_safe(plate_alphas, lite_alphas)
    alpha_std_diff = None
    if len(plate_alphas) > 1 and len(lite_alphas) > 1:
        alpha_std_diff = float(abs(np.std(plate_alphas) - np.std(lite_alphas)))

    return {
        "plate_time_ms_avg": float(np.mean(plate_times)),
        "plate_time_ms_std": float(np.std(plate_times)),
        "lite_time_ms_avg": float(np.mean(lite_times)),
        "lite_time_ms_std": float(np.std(lite_times)),
        "alpha_correlation": alpha_corr,
        "alpha_std_diff": alpha_std_diff,
        "delta_e_ink_lab": _delta_e_zone_mean(plate_ref, lite_ref),
        "zones_used": zone_names,
        "stability_plate": _stability_score(plate_alpha_runs),
        "stability_lite": _stability_score(lite_alpha_runs),
        "notes": {
            "delta_e_source": "plate.core.lab.mean vs plate_lite.ink_lab (ring/dot zones)",
        },
    }


def _summary_stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    filtered = [v for v in values if v is not None and np.isfinite(v)]
    if not filtered:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.array(filtered, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plate vs Plate-Lite A/B comparison")
    parser.add_argument("--input", type=str, help="Directory containing white/ and black/ subdirs")
    parser.add_argument("--pairs_csv", type=str, help="CSV with columns: white,black,sample_id")
    parser.add_argument("--sku", type=str, default=None, help="Optional SKU code for config override")
    parser.add_argument("--runs", type=int, default=3, help="Runs per sample for stability/time")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()

    if not args.input and not args.pairs_csv:
        print("error: provide --input or --pairs_csv")
        return 2

    if args.input and args.pairs_csv:
        print("error: use only one of --input or --pairs_csv")
        return 2

    if args.input:
        pairs = _collect_pairs_from_dir(Path(args.input))
    else:
        pairs = _collect_pairs_from_csv(Path(args.pairs_csv))

    cfg, sources, warnings = load_cfg_with_sku(
        str(Path("src/engine_v7/configs/default.json")),
        args.sku,
        sku_dir=str(Path("config/sku_db")),
        strict_unknown=False,
    )
    if warnings:
        print(f"warning: unknown cfg keys: {', '.join(warnings)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ab_comparison_{timestamp}.json"
    csv_path = output_dir / f"ab_comparison_summary_{timestamp}.csv"

    results: List[Dict[str, Any]] = []
    for item in pairs:
        white_bgr = _load_image(item.white)
        black_bgr = _load_image(item.black)
        metrics = _analyze_pair(white_bgr, black_bgr, cfg, runs=args.runs)
        results.append(
            {
                "sample_id": item.sample_id,
                "white": str(item.white),
                "black": str(item.black),
                "metrics": metrics,
            }
        )

    summary = {
        "sample_count": len(results),
        "alpha_correlation": _summary_stats([r["metrics"]["alpha_correlation"] for r in results]),
        "alpha_std_diff": _summary_stats([r["metrics"]["alpha_std_diff"] for r in results]),
        "delta_e_ink_lab": _summary_stats([r["metrics"]["delta_e_ink_lab"] for r in results]),
        "plate_time_ms_avg": _summary_stats([r["metrics"]["plate_time_ms_avg"] for r in results]),
        "lite_time_ms_avg": _summary_stats([r["metrics"]["lite_time_ms_avg"] for r in results]),
        "stability_plate": _summary_stats([r["metrics"]["stability_plate"] for r in results]),
        "stability_lite": _summary_stats([r["metrics"]["stability_lite"] for r in results]),
    }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "cfg_sources": sources,
        "sku": args.sku,
        "runs": args.runs,
        "results": results,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "alpha_correlation",
                "alpha_std_diff",
                "delta_e_ink_lab",
                "plate_time_ms_avg",
                "lite_time_ms_avg",
                "stability_plate",
                "stability_lite",
            ]
        )
        for r in results:
            m = r["metrics"]
            writer.writerow(
                [
                    r["sample_id"],
                    m["alpha_correlation"],
                    m["alpha_std_diff"],
                    m["delta_e_ink_lab"],
                    m["plate_time_ms_avg"],
                    m["lite_time_ms_avg"],
                    m["stability_plate"],
                    m["stability_lite"],
                ]
            )

    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
