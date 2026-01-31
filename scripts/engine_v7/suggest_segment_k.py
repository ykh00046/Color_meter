#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine_v7.core.signature.segment_k_suggest import suggest_segment_k_from_stds


def load_cfg(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stds", nargs="+", required=True, help="STD image paths (>=2 recommended)")
    ap.add_argument("--cfg", default=str(Path("configs") / "default.json"))
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--min_k", type=float, default=2.0)
    ap.add_argument("--max_k", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    segments = cfg["signature"].get("segments", [])
    if not segments:
        raise SystemExit("No signature.segments found in config")

    bgrs = []
    for p in args.stds:
        bgr = cv2.imread(p)
        if bgr is None:
            raise SystemExit(f"Failed to read STD image: {p}")
        bgrs.append(bgr)

    seg_k = suggest_segment_k_from_stds(
        bgrs,
        R=int(cfg["polar"]["R"]),
        T=int(cfg["polar"]["T"]),
        r_start=float(cfg["signature"]["r_start"]),
        r_end=float(cfg["signature"]["r_end"]),
        segments=segments,
        percentile=float(args.percentile),
        min_k=float(args.min_k),
        max_k=float(args.max_k),
    )

    out = {
        "segment_k": seg_k,
        "suggestion": {
            "percentile": args.percentile,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "n_std": len(bgrs),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
