#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def _add_missing(missing: List[Dict], kind: str, path: Path, context: Dict) -> None:
    missing.append(
        {
            "kind": kind,
            "path": str(path),
            "context": context,
        }
    )


def _check_pack_schema(pack: Dict, path: Path, invalid: List[Dict]) -> None:
    if pack.get("schema_version") != "approval_pack.v1":
        invalid.append({"kind": "PACK_SCHEMA", "path": str(path)})
        return
    if not pack.get("context", {}).get("sku"):
        invalid.append({"kind": "PACK_MISSING_SKU", "path": str(path)})
    if not pack.get("context", {}).get("ink"):
        invalid.append({"kind": "PACK_MISSING_INK", "path": str(path)})


def _check_result_schema(result: Dict, path: Path, invalid: List[Dict]) -> None:
    if not result.get("phase"):
        invalid.append({"kind": "RESULT_MISSING_PHASE", "path": str(path)})
    if not result.get("results"):
        invalid.append({"kind": "RESULT_MISSING_RESULTS", "path": str(path)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--index", required=True)
    ap.add_argument("--results_dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    index_path = Path(args.index).resolve()
    results_dir = Path(args.results_dir).resolve() if args.results_dir else (base_dir / "results")
    models_dir = base_dir / "models"
    approvals_dir = base_dir / "approvals" / "approval_packs"
    snapshots_dir = models_dir / "active_snapshots"

    index_data = _load_json(index_path) if index_path.exists() else {"items": []}

    referenced_packs: Set[Path] = set()
    referenced_snapshots: Set[Path] = set()
    missing: List[Dict] = []
    invalid: List[Dict] = []

    for item in index_data.get("items", []):
        sku = item.get("sku", "")
        ink = item.get("ink", "")
        for evt in item.get("active_history", []):
            pack_path = evt.get("approval_pack_path", "")
            if pack_path:
                p = _resolve_path(base_dir, pack_path)
                if p.exists():
                    referenced_packs.add(p)
                else:
                    _add_missing(missing, "APPROVAL_PACK_PATH", p, {"sku": sku, "ink": ink})
            for snap in [evt.get("to", {}).get("LOW"), evt.get("to", {}).get("MID"), evt.get("to", {}).get("HIGH")]:
                if snap:
                    p = _resolve_path(models_dir, snap) / "model.json"
                    if p.exists():
                        pass
                    else:
                        _add_missing(missing, "ACTIVE_MODEL_PATH", p, {"sku": sku, "ink": ink})

        snap_path = item.get("active_snapshot", "")
        if snap_path:
            p = _resolve_path(models_dir, snap_path)
            if p.exists():
                referenced_snapshots.add(p)
            else:
                _add_missing(missing, "ACTIVE_SNAPSHOT_PATH", p, {"sku": sku, "ink": ink})

    # Inspect approval packs
    pack_paths = list(approvals_dir.glob("*.json")) if approvals_dir.exists() else []
    for pack_path in pack_paths:
        pack = _load_json(pack_path)
        _check_pack_schema(pack, pack_path, invalid)

        base_snap = pack.get("baseline_active", {}).get("active_snapshot_path", "")
        if base_snap:
            p = _resolve_path(Path("."), base_snap)
            if p.exists():
                referenced_snapshots.add(p)
            else:
                _add_missing(missing, "PACK_BASELINE_SNAPSHOT", p, {"pack": str(pack_path)})

        cand_path = pack.get("candidate_std", {}).get("registration_result_path", "")
        if cand_path:
            p = _resolve_path(Path("."), cand_path)
            if not p.exists():
                _add_missing(missing, "PACK_CANDIDATE_RESULT", p, {"pack": str(pack_path)})

        final_snap = pack.get("final", {}).get("active_snapshot_path", "")
        if final_snap:
            p = _resolve_path(Path("."), final_snap)
            if p.exists():
                referenced_snapshots.add(p)
            else:
                _add_missing(missing, "PACK_FINAL_SNAPSHOT", p, {"pack": str(pack_path)})

    # Inspect results
    result_paths = list(results_dir.rglob("*.json")) if results_dir.exists() else []
    for res_path in result_paths:
        try:
            data = _load_json(res_path)
        except Exception:
            invalid.append({"kind": "RESULT_INVALID_JSON", "path": str(res_path)})
            continue
        if "results" not in data or "phase" not in data:
            continue

        _check_result_schema(data, res_path, invalid)
        if data.get("phase") == "INSPECTION":
            active_snap = data.get("active_snapshot")
            if not active_snap:
                invalid.append({"kind": "RESULT_MISSING_ACTIVE_SNAPSHOT", "path": str(res_path)})
            else:
                p = _resolve_path(base_dir, active_snap)
                if not p.exists():
                    _add_missing(missing, "RESULT_ACTIVE_SNAPSHOT_MISSING", p, {"path": str(res_path)})
                else:
                    referenced_snapshots.add(p)
            for item in data.get("results", []):
                decision = item.get("decision", {})
                if "reason_codes" not in decision:
                    invalid.append({"kind": "RESULT_MISSING_REASON_CODES", "path": str(res_path)})
                    break

    # Orphans
    orphan_packs = [str(p) for p in pack_paths if p not in referenced_packs]
    orphan_snapshots = []
    if snapshots_dir.exists():
        for snap in snapshots_dir.glob("*.json"):
            if snap not in referenced_snapshots:
                orphan_snapshots.append(str(snap))

    report = {
        "generated_at": datetime.now().isoformat(),
        "missing": missing,
        "invalid": invalid,
        "orphans": {
            "approval_packs": orphan_packs,
            "active_snapshots": orphan_snapshots,
        },
        "counts": {
            "missing": len(missing),
            "invalid": len(invalid),
            "orphan_packs": len(orphan_packs),
            "orphan_snapshots": len(orphan_snapshots),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
