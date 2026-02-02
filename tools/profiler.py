"""
v7 Performance Profiler for inspection pipeline.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import psutil

from src.config.v7_paths import V7_MODELS, V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.model_registry import load_pattern_baseline, load_std_models
from src.engine_v7.core.pipeline.analyzer import evaluate_multi


@dataclass
class ProfilingResult:
    step_name: str
    elapsed_ms: float
    memory_mb: float
    cpu_percent: float


def _profile_step(process: psutil.Process, name: str, func) -> tuple[object, ProfilingResult]:
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024
    t_start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - t_start) * 1000
    mem_end = process.memory_info().rss / 1024 / 1024
    return result, ProfilingResult(
        step_name=name,
        elapsed_ms=elapsed,
        memory_mb=mem_end - mem_start,
        cpu_percent=process.cpu_percent(),
    )


def print_results(results: List[ProfilingResult], total_time: float) -> None:
    print("\n" + "=" * 80)
    print("  v7 Performance Profiling Results")
    print("=" * 80)
    print(f"\n{'Step':<36} {'Time (ms)':<12} {'%':<8} {'Mem (MB)':<12} {'CPU %':<8}")
    print("-" * 80)
    for r in results:
        pct = (r.elapsed_ms / total_time * 100) if total_time > 0 else 0
        print(
            f"{r.step_name:<36} {r.elapsed_ms:>10.2f}   {pct:>6.1f}%  " f"{r.memory_mb:>10.2f}   {r.cpu_percent:>6.1f}%"
        )
    print("-" * 80)
    print(f"{'TOTAL':<36} {total_time:>10.2f}   {'100.0%':>7}  {'-':>10}   {'-':>6}")
    print("=" * 80 + "\n")


def profile_single(
    image_path: Path, cfg: dict, std_models: dict, pattern_baseline: dict | None
) -> Tuple[List[ProfilingResult], float]:
    process = psutil.Process()
    results: List[ProfilingResult] = []

    bgr, step = _profile_step(process, "1. Image Loading", lambda: cv2.imread(str(image_path)))
    results.append(step)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    _, step = _profile_step(
        process,
        "2. evaluate_multi",
        lambda: evaluate_multi(bgr, std_models, cfg, pattern_baseline=pattern_baseline),
    )
    results.append(step)

    total = sum(r.elapsed_ms for r in results)
    return results, total


def profile_batch(image_paths: List[Path], cfg: dict, std_models: dict, pattern_baseline: dict | None) -> Dict:
    process = psutil.Process()
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024
    peak_mem = mem_start
    t_start = time.perf_counter()

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        evaluate_multi(bgr, std_models, cfg, pattern_baseline=pattern_baseline)
        current_mem = process.memory_info().rss / 1024 / 1024
        peak_mem = max(peak_mem, current_mem)

    t_elapsed = (time.perf_counter() - t_start) * 1000
    return {
        "total_time_ms": t_elapsed,
        "avg_time_ms": t_elapsed / len(image_paths) if image_paths else 0,
        "throughput_per_sec": len(image_paths) / (t_elapsed / 1000) if t_elapsed > 0 else 0,
        "peak_memory_mb": peak_mem - mem_start,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--ink", default="INK_DEFAULT")
    ap.add_argument("--cfg", default=str(V7_ROOT / "configs" / "default.json"))
    ap.add_argument("--models_root", default=str(V7_MODELS))
    ap.add_argument("--images_dir", default=str(Path("data") / "samples"))
    ap.add_argument("--glob", default="*.png")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    images = sorted(images_dir.rglob(args.glob))
    if not images:
        raise SystemExit(f"No images found under {images_dir} with {args.glob}")

    cfg, _, _ = load_cfg_with_sku(args.cfg, args.sku, strict_unknown=False)
    std_models, reasons = load_std_models(args.models_root, args.sku, args.ink)
    if std_models is None:
        raise SystemExit(f"STD model not found: {reasons}")

    pattern_baseline, _ = load_pattern_baseline(args.models_root, args.sku, args.ink)

    results, total = profile_single(images[0], cfg, std_models, pattern_baseline)
    print_results(results, total)

    batch_result = profile_batch(images[:10], cfg, std_models, pattern_baseline)
    print("\nBatch profile (first 10 images):")
    print(f"  Total time:    {batch_result['total_time_ms']:.2f} ms")
    print(f"  Avg per image: {batch_result['avg_time_ms']:.2f} ms")
    print(f"  Throughput:    {batch_result['throughput_per_sec']:.2f} images/sec")
    print(f"  Peak memory:   {batch_result['peak_memory_mb']:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
