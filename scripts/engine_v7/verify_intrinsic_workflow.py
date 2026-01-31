import argparse
import json
import sys
from pathlib import Path


def _load_k_from_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    analysis = results[0].get("analysis", {})
    ink = analysis.get("ink", {})
    clusters = ink.get("clusters", [])
    if not clusters:
        return None
    return clusters[0].get("intrinsic_k_rgb")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify intrinsic calibration + simulation workflow.")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v7", help="API base URL")
    parser.add_argument("--white", type=Path, required=True, help="Path to empty white plate image")
    parser.add_argument("--black", type=Path, required=True, help="Path to empty black plate image")
    parser.add_argument("--center-crop", type=float, default=0.5, help="Center crop ratio for calibration")
    parser.add_argument("--k-rgb", type=str, default="", help="Override k_rgb JSON array, e.g. [0.2,0.5,0.1]")
    parser.add_argument("--bg", type=str, default="[60,40,20]", help="Background sRGB JSON array")
    parser.add_argument("--thickness", type=float, default=1.0, help="Thickness ratio for simulation")
    parser.add_argument("--sku", type=str, default="", help="Optional SKU for config")
    parser.add_argument("--analysis-json", type=Path, default=None, help="Optional analysis JSON to extract k_rgb")
    args = parser.parse_args()

    try:
        import requests
    except Exception:
        print("requests is required. Install with: pip install requests")
        return 1

    if not args.white.exists() or not args.black.exists():
        print("White/black image paths are invalid.")
        return 1

    calibrate_url = f"{args.base_url}/intrinsic_calibrate"
    simulate_url = f"{args.base_url}/intrinsic_simulate"

    files = {
        "white_file": open(args.white, "rb"),
        "black_file": open(args.black, "rb"),
    }
    data = {
        "mode": "FIXED",
        "center_crop": str(args.center_crop),
    }
    if args.sku:
        data["sku"] = args.sku

    resp = requests.post(calibrate_url, files=files, data=data, timeout=60)
    print("[Calibration] Status:", resp.status_code)
    print("[Calibration] Response:", resp.text)
    if resp.status_code >= 400:
        return 1

    if args.k_rgb:
        k_rgb = json.loads(args.k_rgb)
    elif args.analysis_json and args.analysis_json.exists():
        k_rgb = _load_k_from_file(args.analysis_json)
    else:
        k_rgb = None

    if not k_rgb:
        print("k_rgb not provided. Pass --k-rgb or --analysis-json.")
        return 1

    sim_data = {
        "k_rgb": json.dumps(k_rgb),
        "bg_srgb": args.bg,
        "thickness": str(args.thickness),
        "gamma": "2.2",
    }
    resp = requests.post(simulate_url, data=sim_data, timeout=60)
    print("[Simulate] Status:", resp.status_code)
    print("[Simulate] Response:", resp.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
