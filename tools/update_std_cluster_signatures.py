"""
Build a v7 ink baseline from STD images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config.v7_paths import V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.measure.baselines.ink_baseline import build_ink_baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--ink", default="INK_DEFAULT")
    ap.add_argument("--expected_k", type=int, required=True)
    ap.add_argument("--low", nargs="+", required=True, help="LOW mode images")
    ap.add_argument("--mid", nargs="+", required=True, help="MID mode images")
    ap.add_argument("--high", nargs="+", required=True, help="HIGH mode images")
    ap.add_argument("--cfg", default=str(V7_ROOT / "configs" / "default.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg, _, _ = load_cfg_with_sku(args.cfg, args.sku, strict_unknown=False)
    images_by_mode = {"LOW": args.low, "MID": args.mid, "HIGH": args.high}

    baseline = build_ink_baseline(images_by_mode, cfg, expected_k=args.expected_k)
    if baseline is None:
        raise SystemExit("Failed to build ink baseline; check inputs and expected_k")

    timestamp = int(time.time())
    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / "v7" / "test" / f"ink_baseline_{args.sku}_{args.ink}_{timestamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved ink baseline: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
