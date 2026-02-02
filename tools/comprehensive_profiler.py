"""
Comprehensive v7 profiler for inspection steps.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import psutil

from src.config.v7_paths import V7_MODELS, V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.measure.segmentation.color_masks import build_color_masks_with_retry
from src.engine_v7.core.model_registry import load_expected_ink_count, load_pattern_baseline, load_std_models
from src.engine_v7.core.pipeline.analyzer import evaluate_multi
from src.engine_v7.core.signature.radial_signature import build_radial_signature, to_polar


@dataclass
class StepResult:
    name: str
    time_ms: float
    memory_mb: float
    cpu_percent: float


def _step(process: psutil.Process, name: str, func) -> tuple[Any, StepResult]:
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024
    t_start = time.perf_counter()
    result = func()
    t_elapsed = (time.perf_counter() - t_start) * 1000
    mem_end = process.memory_info().rss / 1024 / 1024
    return result, StepResult(
        name=name,
        time_ms=t_elapsed,
        memory_mb=mem_end - mem_start,
        cpu_percent=process.cpu_percent(),
    )


def print_results(results: list[StepResult]) -> None:
    total = sum(r.time_ms for r in results)
    print("\n" + "=" * 90)
    print("  v7 Comprehensive Profiling Results")
    print("=" * 90)
    print(f"\n{'Step':<44} {'Time (ms)':<12} {'%':<8} {'Mem (MB)':<12} {'CPU %':<8}")
    print("-" * 90)
    for r in results:
        pct = (r.time_ms / total * 100) if total > 0 else 0
        print(f"{r.name:<44} {r.time_ms:>10.2f}   {pct:>6.1f}%  {r.memory_mb:>10.2f}   {r.cpu_percent:>6.1f}%")
    print("-" * 90)
    print(f"{'TOTAL':<44} {total:>10.2f}   {'100.0%':>7}  {'-':>10}   {'-':>6}")
    print("=" * 90 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Lens image path")
    ap.add_argument("--sku", required=True)
    ap.add_argument("--ink", default="INK_DEFAULT")
    ap.add_argument("--cfg", default=str(V7_ROOT / "configs" / "default.json"))
    ap.add_argument("--models_root", default=str(V7_MODELS))
    ap.add_argument("--expected_ink_count", type=int, default=None)
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    cfg, _, _ = load_cfg_with_sku(args.cfg, args.sku, strict_unknown=False)
    process = psutil.Process()
    results: list[StepResult] = []

    bgr, step = _step(process, "1. Image Loading", lambda: cv2.imread(str(img_path)))
    results.append(step)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    geom, step = _step(process, "2. Lens Detection", lambda: detect_lens_circle(bgr))
    results.append(step)

    polar, step = _step(
        process,
        "3. Polar Transform",
        lambda: to_polar(bgr, geom, R=int(cfg["polar"]["R"]), T=int(cfg["polar"]["T"])),
    )
    results.append(step)

    _, step = _step(
        process,
        "4. Radial Signature",
        lambda: build_radial_signature(
            polar,
            r_start=float(cfg["signature"]["r_start"]),
            r_end=float(cfg["signature"]["r_end"]),
        ),
    )
    results.append(step)

    expected_ink_count = args.expected_ink_count
    if expected_ink_count is None:
        expected_ink_count = load_expected_ink_count(args.models_root, args.sku, args.ink)

    if expected_ink_count and cfg.get("v2_ink", {}).get("enabled", False):
        _, step = _step(
            process,
            "5. Ink Segmentation",
            lambda: build_color_masks_with_retry(
                bgr, cfg, expected_k=int(expected_ink_count), geom=geom, confidence_threshold=0.7, enable_retry=True
            ),
        )
        results.append(step)

    std_models, reasons = load_std_models(args.models_root, args.sku, args.ink)
    if std_models is None:
        raise SystemExit(f"STD model not found: {reasons}")

    pattern_baseline, _ = load_pattern_baseline(args.models_root, args.sku, args.ink)
    ok_log_context = {
        "sku": args.sku,
        "ink": args.ink,
        "expected_ink_count_input": expected_ink_count,
        "expected_ink_count_registry": expected_ink_count,
    }

    _, step = _step(
        process,
        "6. evaluate_multi (end-to-end)",
        lambda: evaluate_multi(bgr, std_models, cfg, pattern_baseline=pattern_baseline, ok_log_context=ok_log_context),
    )
    results.append(step)

    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
