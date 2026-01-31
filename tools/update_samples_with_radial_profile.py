"""
Rebuild v7 inspection results for given images.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config.v7_paths import V7_MODELS, V7_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--ink", default="INK_DEFAULT")
    ap.add_argument("--cfg", default=str(V7_ROOT / "configs" / "default.json"))
    ap.add_argument("--models_root", default=str(V7_MODELS))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--phase", default="INSPECTION", choices=["INSPECTION", "STD_REGISTRATION"])
    ap.add_argument("--expected_ink_count", type=int, default=None)
    args = ap.parse_args()

    script = Path("src") / "engine_v7" / "scripts" / "run_signature_engine.py"
    if not script.exists():
        raise SystemExit(f"run_signature_engine.py not found: {script}")

    out_path = Path(args.out) if args.out else Path("results") / "v7" / "test" / f"v7_rebuild_{args.sku}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--cfg",
        str(args.cfg),
        "--tests",
        *args.images,
        "--sku",
        args.sku,
        "--ink",
        args.ink,
        "--models_root",
        str(args.models_root),
        "--phase",
        args.phase,
        "--out",
        str(out_path),
    ]
    if args.expected_ink_count is not None:
        cmd.extend(["--expected_ink_count", str(args.expected_ink_count)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)

    print(f"Saved v7 result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
