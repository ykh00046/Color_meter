#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _age_days(path: Path) -> float:
    ts = path.stat().st_mtime
    return (datetime.now() - datetime.fromtimestamp(ts)).total_seconds() / 86400.0


def _archive_path(base: Path, category: str, src: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m")
    target_dir = base / "archive" / category / stamp
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / src.name


def _collect_referenced_snapshots(base_dir: Path, index_data: Dict) -> Set[Path]:
    refs: Set[Path] = set()
    for item in index_data.get("items", []):
        snap = item.get("active_snapshot", "")
        if snap:
            p = Path(snap)
            if not p.is_absolute():
                p = (base_dir / "models" / snap).resolve()
            refs.add(p)
    return refs


def _is_pack_approved(pack: Dict) -> bool:
    final = pack.get("final", {})
    status = final.get("status", "")
    return status in {"APPROVED", "ROLLED_BACK"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--index", required=True)
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    ap.add_argument("--days_keep_approved", type=int, default=365)
    ap.add_argument("--days_keep_unapplied", type=int, default=90)
    ap.add_argument("--days_keep_snapshots", type=int, default=90)
    ap.add_argument("--days_keep_results", type=int, default=365)
    ap.add_argument("--days_keep_metrics", type=int, default=180)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    index_data = _load_json(Path(args.index).resolve())
    approvals_dir = base_dir / "approvals" / "approval_packs"
    snapshots_dir = base_dir / "models" / "active_snapshots"
    results_dir = base_dir / "results"

    dry_run = not args.apply
    plan = {"move": [], "skip": [], "dry_run": dry_run}

    referenced_snapshots = _collect_referenced_snapshots(base_dir, index_data)

    # Approval packs
    if approvals_dir.exists():
        for pack_path in approvals_dir.glob("*.json"):
            pack = _load_json(pack_path)
            for snap in [
                pack.get("baseline_active", {}).get("active_snapshot_path", ""),
                pack.get("final", {}).get("active_snapshot_path", ""),
            ]:
                if snap:
                    p = Path(snap)
                    if not p.is_absolute():
                        p = (base_dir / snap).resolve()
                    referenced_snapshots.add(p)
            keep_days = args.days_keep_approved if _is_pack_approved(pack) else args.days_keep_unapplied
            if _age_days(pack_path) > keep_days:
                dest = _archive_path(base_dir / "approvals", "approval_packs", pack_path)
                plan["move"].append({"path": str(pack_path), "dest": str(dest), "reason": "approval_pack"})
                if not dry_run:
                    shutil.move(str(pack_path), str(dest))
            else:
                plan["skip"].append({"path": str(pack_path), "reason": "approval_pack"})

    # Active snapshots
    # Include snapshots referenced by results
    if results_dir.exists():
        for item in results_dir.rglob("*.json"):
            if "archive" in item.parts:
                continue
            try:
                data = _load_json(item)
            except Exception:
                continue
            active_snap = data.get("active_snapshot", "")
            if active_snap:
                p = Path(active_snap)
                if not p.is_absolute():
                    p = (base_dir / active_snap).resolve()
                referenced_snapshots.add(p)

    if snapshots_dir.exists():
        for snap_path in snapshots_dir.glob("*.json"):
            keep_days = args.days_keep_approved if snap_path in referenced_snapshots else args.days_keep_snapshots
            if _age_days(snap_path) > keep_days:
                dest = _archive_path(base_dir / "models", "active_snapshots", snap_path)
                plan["move"].append({"path": str(snap_path), "dest": str(dest), "reason": "active_snapshot"})
                if not dry_run:
                    shutil.move(str(snap_path), str(dest))
            else:
                plan["skip"].append({"path": str(snap_path), "reason": "active_snapshot"})

    # Results
    if results_dir.exists():
        for item in results_dir.rglob("*.json"):
            if "archive" in item.parts:
                continue
            if item.name.startswith("metrics"):
                keep_days = args.days_keep_metrics
            else:
                keep_days = args.days_keep_results
            if _age_days(item) > keep_days:
                dest = _archive_path(results_dir, "results", item)
                plan["move"].append({"path": str(item), "dest": str(dest), "reason": "result"})
                if not dry_run:
                    shutil.move(str(item), str(dest))
            else:
                plan["skip"].append({"path": str(item), "reason": "result"})

    out = json.dumps(plan, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
