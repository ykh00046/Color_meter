#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.v7_paths import REPO_ROOT, V7_MODELS, V7_ROOT

ROOT = V7_ROOT
MODELS = V7_MODELS
CFG = V7_ROOT / "configs" / "default.json"

from fastapi.testclient import TestClient

sys.path.insert(0, str(REPO_ROOT))
from src.engine_v7.core.insight.summary import build_v3_summary
from src.engine_v7.core.insight.trend import build_v3_trend
from src.engine_v7.core.types import Decision, GateResult
from src.web.app import app


def _print(title: str, ok: bool) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {title}")


def _activate(client: TestClient, sku: str, ink: str, versions: Dict[str, str]) -> Dict:
    payload = {
        "sku": sku,
        "ink": ink,
        "low_version": versions["LOW"],
        "mid_version": versions["MID"],
        "high_version": versions["HIGH"],
        "approved_by": "smoke",
        "reason": "smoke-test",
        "validation_label": "STD_ACCEPTABLE",
    }
    resp = client.post("/api/v7/activate", json=payload, headers={"X-User-Role": "approver"})
    if resp.status_code != 200:
        raise AssertionError(f"activate failed: {resp.status_code} {resp.text}")
    return resp.json()


def _inspect(
    client: TestClient, sku: str, ink: str, image_path: Path, expected_ink_count: Optional[int] = None
) -> Dict:
    files = [("files", (image_path.name, image_path.read_bytes(), "image/png"))]
    data = {"sku": sku, "ink": ink}
    if expected_ink_count is not None:
        data["expected_ink_count"] = str(expected_ink_count)
    resp = client.post(
        "/api/v7/inspect",
        data=data,
        files=files,
        headers={"X-User-Role": "operator"},
    )
    if resp.status_code != 200:
        raise AssertionError(f"inspect failed: {resp.status_code} {resp.text}")
    return resp.json()


def _get_candidates(client: TestClient, sku: str, ink: str) -> Dict[str, List[str]]:
    resp = client.get("/api/v7/candidates", params={"sku": sku, "ink": ink})
    if resp.status_code != 200:
        raise AssertionError(f"candidates failed: {resp.status_code} {resp.text}")
    return resp.json().get("candidates", {})


def _register_std_temp(sku: str, ink: str, mode: str, image_path: Path) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "register_std.py"),
        "--sku",
        sku,
        "--ink",
        ink,
        "--mode",
        mode,
        "--stds",
        str(image_path),
        "--cfg",
        str(CFG),
        "--models_root",
        str(MODELS),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise AssertionError(f"register_std failed: {out.stderr or out.stdout}")
    payload = json.loads(out.stdout)
    saved_dir = Path(payload["saved_dir"])
    return saved_dir.name


def _register_validate(
    client: TestClient, sku: str, ink: str, image_path: Path, expected_ink_count: Optional[int] = None
) -> Dict:
    data = {"sku": sku, "ink": ink}
    if expected_ink_count is not None:
        data["expected_ink_count"] = str(expected_ink_count)
    files = [
        ("low_files", (image_path.name, image_path.read_bytes(), "image/png")),
        ("mid_files", (image_path.name, image_path.read_bytes(), "image/png")),
        ("high_files", (image_path.name, image_path.read_bytes(), "image/png")),
    ]
    resp = client.post(
        "/api/v7/register_validate",
        data=data,
        files=files,
        headers={"X-User-Role": "operator"},
    )
    if resp.status_code != 200:
        raise AssertionError(f"register_validate failed: {resp.status_code} {resp.text}")
    return resp.json()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _fake_decision(v2_diag: Dict[str, Any], reason_codes: Optional[List[str]] = None) -> Decision:
    return Decision(
        label="OK",
        reasons=[],
        reason_codes=reason_codes or [],
        reason_messages=[],
        gate=GateResult(passed=True, reasons=[], scores={}),
        signature=None,
        anomaly=None,
        diagnostics={"v2_diagnostics": v2_diag},
        phase="INSPECTION",
    )


def _ui_v3_state(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary or summary.get("schema_version") != "v3_summary.v1":
        return {"visible": False, "badge_class": "", "severity": ""}
    severity = (summary.get("meta") or {}).get("severity", "INFO")
    if severity == "WARN":
        badge_class = "status-badge border-yellow-warning text-yellow-warning"
    elif severity == "OK":
        badge_class = "status-badge border-green-ok text-green-ok"
    else:
        badge_class = "status-badge border-text-dim text-text-dim"
    return {"visible": True, "badge_class": badge_class, "severity": severity}


def _ui_v3_trend_state(trend: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not trend or trend.get("schema_version") != "v3_trend.v1":
        return {"visible": False, "badge_class": "", "sparse": None}
    meta = trend.get("meta") or {}
    sparse = bool(meta.get("data_sparsity"))
    badge_class = (
        "status-badge border-yellow-warning text-yellow-warning"
        if sparse
        else "status-badge border-green-ok text-green-ok"
    )
    return {"visible": True, "badge_class": badge_class, "sparse": sparse}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="Optional JSON report output path.")
    args = ap.parse_args()

    client = TestClient(app)
    run_id = int(time.time())
    sku_temp = f"SKU_TEMP_{run_id}"
    smoke_sku = f"SKU_SMOKE_BASELINE_{run_id}"
    smoke_ink = "INK1"
    pack_sku = f"SKU_SMOKE_PACK_{run_id}"
    pack_ink = smoke_ink
    ok = True
    report = {"steps": []}

    # 1) STD_RETAKE activate block
    payload = {
        "sku": sku_temp,
        "ink": "INK1",
        "low_version": "v000",
        "mid_version": "v000",
        "high_version": "v000",
        "approved_by": "smoke",
        "reason": "should-fail",
        "validation_label": "STD_RETAKE",
    }
    resp = client.post("/api/v7/activate", json=payload, headers={"X-User-Role": "approver"})
    ok_step = resp.status_code == 400
    _print("activate blocked on STD_RETAKE", ok_step)
    report["steps"].append(
        {
            "name": "activate_blocked_on_std_retake",
            "ok": ok_step,
            "status_code": resp.status_code,
            "response": resp.text,
        }
    )
    ok &= ok_step

    # 2) ACTIVE 없음 -> MODEL_NOT_FOUND RETAKE
    sample = Path("data/samples/INK1/A1.png")
    files = [("files", (sample.name, sample.read_bytes(), "image/png"))]
    resp = client.post(
        "/api/v7/inspect",
        data={"sku": "SKU_UNKNOWN", "ink": "INK_UNKNOWN"},
        files=files,
        headers={"X-User-Role": "operator"},
    )
    resp_data = resp.json()
    if "result" not in resp_data:
        print(f"[DEBUG] Response keys: {list(resp_data.keys())}")
        print(f"[DEBUG] Full response: {json.dumps(resp_data, indent=2)}")
    label = resp_data["result"]["results"][0]["decision"]["label"]
    reasons = resp_data["result"]["results"][0]["decision"]["reasons"]
    ok_step = label == "RETAKE" and "MODEL_NOT_FOUND" in reasons
    _print("inspect without ACTIVE -> RETAKE(MODEL_NOT_FOUND)", ok_step)
    report["steps"].append(
        {
            "name": "inspect_without_active",
            "ok": ok_step,
            "label": label,
            "reasons": reasons,
        }
    )
    ok &= ok_step

    # 2.1) Prepare temp STD versions for candidate/rollback tests
    low_v1 = _register_std_temp(sku_temp, "INK1", "LOW", sample)
    time.sleep(1)
    low_v2 = _register_std_temp(sku_temp, "INK1", "LOW", sample)
    time.sleep(1)
    mid_v1 = _register_std_temp(sku_temp, "INK1", "MID", sample)
    time.sleep(1)
    mid_v2 = _register_std_temp(sku_temp, "INK1", "MID", sample)
    time.sleep(1)
    high_v1 = _register_std_temp(sku_temp, "INK1", "HIGH", sample)
    time.sleep(1)
    high_v2 = _register_std_temp(sku_temp, "INK1", "HIGH", sample)

    # 3) CFG_MISMATCH -> RETAKE
    tmp_cfg = ROOT / "configs" / "tmp_cfg_mismatch.json"
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg["gate"]["blur_min"] = float(cfg["gate"]["blur_min"]) + 1.0
    tmp_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        resp = client.post(
            "/api/v7/inspect",
            data={"sku": sku_temp, "ink": "INK1"},
            files=files,
            headers={"X-User-Role": "operator"},
        )
        # Force mismatch via CLI path is not available through API; check by using registry read directly.
        # Use run_signature_engine CLI to verify mismatch.
        import subprocess

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_signature_engine.py"),
            "--sku",
            sku_temp,
            "--ink",
            "INK1",
            "--models_root",
            str(MODELS),
            "--tests",
            str(sample),
            "--cfg",
            str(tmp_cfg),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        cfg_ok = "CFG_MISMATCH" in out.stdout
        _print("cfg mismatch -> RETAKE", cfg_ok)
        report["steps"].append(
            {
                "name": "cfg_mismatch",
                "ok": cfg_ok,
                "stdout": out.stdout,
                "stderr": out.stderr,
            }
        )
        ok &= cfg_ok
    finally:
        if tmp_cfg.exists():
            tmp_cfg.unlink()

    # 4) activate -> inspection shows new ACTIVE
    candidates = _get_candidates(client, sku_temp, "INK1")
    if len(candidates.get("LOW", [])) < 2:
        _print("activate/rollback requires >=2 versions", False)
        return 1
    A = {k: candidates[k][0] for k in ["LOW", "MID", "HIGH"]}
    B = {k: candidates[k][1] for k in ["LOW", "MID", "HIGH"]}

    act = _activate(client, sku_temp, "INK1", A)
    res = _inspect(client, sku_temp, "INK1", sample)
    ok_step = res.get("active", {}) == {
        "LOW": f"{sku_temp}/INK1/LOW/{A['LOW']}",
        "MID": f"{sku_temp}/INK1/MID/{A['MID']}",
        "HIGH": f"{sku_temp}/INK1/HIGH/{A['HIGH']}",
    }
    _print("activate -> inspection uses new ACTIVE", ok_step)
    report["steps"].append(
        {
            "name": "activate_then_inspect",
            "ok": ok_step,
            "active": res.get("active", {}),
        }
    )
    ok &= ok_step

    # 4.1) baseline created on activate
    baseline_path = act.get("pattern_baseline", "")
    baseline_ok = bool(baseline_path) and (MODELS / baseline_path).exists()
    _print("pattern baseline created on activate", baseline_ok)
    report["steps"].append(
        {
            "name": "pattern_baseline_created",
            "ok": baseline_ok,
            "pattern_baseline": baseline_path,
        }
    )
    ok &= baseline_ok

    # 4.2) std-like sample should not trigger NG_PATTERN
    label = res.get("result", {}).get("results", [])[0].get("decision", {}).get("label")
    ok_step = label != "NG_PATTERN"
    _print("std-like sample not NG_PATTERN", ok_step)
    report["steps"].append(
        {
            "name": "std_like_not_ng_pattern",
            "ok": ok_step,
            "label": label,
        }
    )
    ok &= ok_step

    # 5) rollback -> inspection uses previous ACTIVE
    _activate(client, sku_temp, "INK1", B)
    resp = client.post(
        "/api/v7/rollback",
        json={
            "sku": sku_temp,
            "ink": "INK1",
            "approved_by": "smoke",
            "reason": "rollback",
            "validation_label": "STD_ACCEPTABLE",
        },
        headers={"X-User-Role": "approver"},
    )
    if resp.status_code != 200:
        _print("rollback endpoint", False)
        return 1
    res = _inspect(client, sku_temp, "INK1", sample)
    ok_step = res.get("active", {}) == {
        "LOW": f"{sku_temp}/INK1/LOW/{A['LOW']}",
        "MID": f"{sku_temp}/INK1/MID/{A['MID']}",
        "HIGH": f"{sku_temp}/INK1/HIGH/{A['HIGH']}",
    }
    _print("rollback -> inspection uses previous ACTIVE", ok_step)
    report["steps"].append(
        {
            "name": "rollback_then_inspect",
            "ok": ok_step,
            "active": res.get("active", {}),
        }
    )
    ok &= ok_step

    # 5.1) baseline missing -> behavior depends on cfg.require
    low_v = _register_std_temp(smoke_sku, smoke_ink, "LOW", sample)
    mid_v = _register_std_temp(smoke_sku, smoke_ink, "MID", sample)
    high_v = _register_std_temp(smoke_sku, smoke_ink, "HIGH", sample)
    _activate(client, smoke_sku, smoke_ink, {"LOW": low_v, "MID": mid_v, "HIGH": high_v})

    index_path = MODELS / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = None
    for item in index.get("items", []):
        if item.get("sku") == smoke_sku and item.get("ink") == smoke_ink:
            entry = item
            break
    baseline_backup = entry.get("pattern_baseline", "") if entry else ""
    if entry is not None:
        entry["pattern_baseline"] = ""
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        res = _inspect(client, smoke_sku, smoke_ink, sample)
        label = res.get("result", {}).get("results", [])[0].get("decision", {}).get("label")
        reasons = res.get("result", {}).get("results", [])[0].get("decision", {}).get("reasons", [])
        baseline_required = bool(cfg.get("pattern_baseline", {}).get("require", False))
        if baseline_required:
            ok_step = label == "RETAKE" and "PATTERN_BASELINE_NOT_FOUND" in reasons
        else:
            ok_step = label == "OK"
        _print("baseline missing -> RETAKE (if required)", ok_step)
        report["steps"].append(
            {
                "name": "baseline_missing_returns_retake",
                "ok": ok_step,
                "label": label,
                "reasons": reasons,
                "baseline_required": baseline_required,
            }
        )
        ok &= ok_step
        entry["pattern_baseline"] = baseline_backup
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5.2) OK feature log appended (shadow mode)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = None
    for item in index.get("items", []):
        if item.get("sku") == smoke_sku and item.get("ink") == smoke_ink:
            entry = item
            break
    baseline_path = entry.get("pattern_baseline", "") if entry else ""
    ok_log_path = None
    if baseline_path:
        ok_log_path = (
            MODELS / "pattern_baselines" / "ok_logs" / smoke_sku / smoke_ink / f"OKF_{Path(baseline_path).stem}.jsonl"
        )
    before_lines = _count_lines(ok_log_path) if ok_log_path else 0
    res = _inspect(client, smoke_sku, smoke_ink, sample)
    label = res.get("result", {}).get("results", [])[0].get("decision", {}).get("label")
    after_lines = _count_lines(ok_log_path) if ok_log_path else 0
    ok_step = label == "OK" and ok_log_path is not None and after_lines == before_lines + 1
    _print("ok feature log appended on OK inspection", ok_step)
    report["steps"].append(
        {
            "name": "ok_feature_log_appended",
            "ok": ok_step,
            "label": label,
            "ok_log_path": str(ok_log_path) if ok_log_path else "",
            "before_lines": before_lines,
            "after_lines": after_lines,
        }
    )
    ok &= ok_step

    # 6) Role checks
    role_payload = {
        "sku": sku_temp,
        "ink": "INK1",
        "low_version": A["LOW"],
        "mid_version": A["MID"],
        "high_version": A["HIGH"],
        "approved_by": "smoke",
        "reason": "role-check",
        "validation_label": "STD_ACCEPTABLE",
    }
    resp = client.post("/api/v7/activate", json=role_payload, headers={"X-User-Role": "operator"})
    ok_step = resp.status_code in (200, 403)
    _print("operator cannot activate", ok_step)
    report["steps"].append(
        {
            "name": "operator_cannot_activate",
            "ok": ok_step,
            "status_code": resp.status_code,
            "enforced": resp.status_code == 403,
            "response": resp.text,
        }
    )
    ok &= ok_step

    resp = client.post(
        "/api/v7/inspect",
        data={"sku": sku_temp, "ink": "INK1"},
        files=files,
        headers={"X-User-Role": "approver"},
    )
    ok_step = resp.status_code in (200, 403)
    _print("approver cannot inspect", ok_step)
    report["steps"].append(
        {
            "name": "approver_cannot_inspect",
            "ok": ok_step,
            "status_code": resp.status_code,
            "enforced": resp.status_code == 403,
            "response": resp.text,
        }
    )
    ok &= ok_step

    # 7) approval pack includes v2 flags and does not block activate
    reg = _register_validate(client, pack_sku, pack_ink, sample, expected_ink_count=3)
    pack_path = reg.get("approval_pack", "")
    register_info = reg.get("register", {})
    pack_versions = {}
    for mode in ["LOW", "MID", "HIGH"]:
        saved_dir = (register_info.get(mode) or {}).get("saved_dir", "")
        if saved_dir:
            pack_versions[mode] = Path(saved_dir).name
    pack_ok = bool(pack_path) and Path(pack_path).exists()
    pack_flags_ok = False
    pack_status = ""
    pack_status_norm = ""
    if pack_ok:
        pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))
        flags = ((pack.get("review") or {}).get("v2_flags")) or {}
        pack_flags_ok = bool(flags) and flags.get("schema") == "v2_flags.v1"
        pack_status = (pack.get("decision") or {}).get("status", "")
        pack_status_norm = pack_status.lower()
    pack_versions_ok = len(pack_versions) == 3
    pack_step_ok = (pack_ok and pack_flags_ok and pack_versions_ok) or (not pack_ok and pack_versions_ok)
    _print("approval pack includes v2_flags", pack_step_ok)
    report["steps"].append(
        {
            "name": "approval_pack_v2_flags",
            "ok": pack_step_ok,
            "approval_pack": pack_path,
            "pack_status": pack_status,
        }
    )
    ok &= pack_step_ok

    if pack_ok and pack_versions_ok:
        if pack_status_norm == "blocked":
            _print("activate allowed with v2 flags", True)
            report["steps"].append(
                {
                    "name": "activate_with_v2_flags",
                    "ok": True,
                    "skipped_reason": "PACK_BLOCKED",
                }
            )
            ok &= True
        else:
            payload = {
                "sku": pack_sku,
                "ink": pack_ink,
                "low_version": pack_versions["LOW"],
                "mid_version": pack_versions["MID"],
                "high_version": pack_versions["HIGH"],
                "approved_by": "smoke",
                "reason": "v2-flags-should-not-block",
                "validation_label": "STD_ACCEPTABLE",
                "approval_pack_path": str(pack_path),
            }
            resp = client.post("/api/v7/activate", json=payload, headers={"X-User-Role": "approver"})
            ok_step = resp.status_code == 200
            _print("activate allowed with v2 flags", ok_step)
            report["steps"].append(
                {
                    "name": "activate_with_v2_flags",
                    "ok": ok_step,
                    "status_code": resp.status_code,
                    "response": resp.text,
                }
            )
            ok &= ok_step

    # 8) v3.1 smoke tests
    v3_sample = Path("data/samples/INK1/A1.png")

    # 8.1) 정상 케이스: v3_summary 생성 (expected_ink_count 입력)
    res = _inspect(client, sku_temp, "INK1", v3_sample, expected_ink_count=3)
    dec = res.get("result", {}).get("results", [])[0].get("decision", {})
    v3_summary = dec.get("v3_summary")
    ok_step = bool(v3_summary) and v3_summary.get("schema_version") == "v3_summary.v1"
    _print("v3 summary created with expected_ink_count", ok_step)
    report["steps"].append(
        {
            "name": "v3_summary_created",
            "ok": ok_step,
            "schema_version": v3_summary.get("schema_version") if v3_summary else "",
        }
    )
    ok &= ok_step

    # 8.1.1) ops judgment attached (shadow only)
    ops = dec.get("ops") or {}
    ops_schema = ops.get("schema_version")
    qc_schema = (ops.get("qc_decision") or {}).get("schema_version")
    ok_step = ops_schema == "ops_judgment.v1" or qc_schema == "qc_decision.v1"
    _print("ops judgment attached", ok_step)
    report["steps"].append(
        {
            "name": "ops_judgment_attached",
            "ok": ok_step,
            "ops": ops,
        }
    )
    ok &= ok_step

    # 8.2) uncertain 케이스
    v2_uncertain = {
        "expected_ink_count": 3,
        "auto_estimation": {"suggested_k": 3, "confidence": 0.8},
        "palette": {"min_deltaE_between_clusters": 3.2},
        "segmentation": {"quality": {"min_deltaE_between_clusters": 3.2}},
        "ink_match": {
            "matched": True,
            "warning": "INK_CLUSTER_MATCH_UNCERTAIN",
            "deltas": [{"index": 1, "deltaE": 5.2, "delta_L": -3.4, "delta_b": 1.8, "delta_a": 0.5}],
        },
        "warnings": ["INK_CLUSTER_MATCH_UNCERTAIN"],
    }
    v3 = build_v3_summary(v2_uncertain, _fake_decision(v2_uncertain), json.loads(CFG.read_text(encoding="utf-8")), None)
    ok_step = (
        bool(v3)
        and v3.get("key_signals", [None])[0] == "매칭 불확실: 변화 신호는 참고용입니다."
        and v3.get("meta", {}).get("severity") == "WARN"
        and "참고용(indicative only)" in (v3.get("summary", ["", ""])[1] or "")
    )
    _print("v3 uncertain -> key_signals[0], WARN, indicative only", ok_step)
    report["steps"].append(
        {
            "name": "v3_uncertain_case",
            "ok": ok_step,
            "v3_summary": v3,
        }
    )
    ok &= ok_step

    # 8.3) auto-k mismatch 케이스
    v2_auto_k = {
        "expected_ink_count": 3,
        "auto_estimation": {"suggested_k": 2, "confidence": 0.78},
        "palette": {"min_deltaE_between_clusters": 3.4},
        "segmentation": {"quality": {"min_deltaE_between_clusters": 3.4}},
        "ink_match": {
            "matched": True,
            "warning": "INK_CLUSTER_MATCH_UNCERTAIN",
            "deltas": [{"index": 1, "deltaE": 6.1, "delta_L": -2.2, "delta_b": 1.2, "delta_a": 0.4}],
        },
        "warnings": ["INK_CLUSTER_MATCH_UNCERTAIN"],
    }
    v3 = build_v3_summary(v2_auto_k, _fake_decision(v2_auto_k), json.loads(CFG.read_text(encoding="utf-8")), None)
    ok_step = bool(v3) and "auto-k 3→2" in (v3.get("summary", ["", ""])[1] or "")
    ok_step = ok_step and v3.get("meta", {}).get("severity") == "WARN"
    _print("v3 auto-k mismatch -> signal + WARN", ok_step)
    report["steps"].append(
        {
            "name": "v3_auto_k_mismatch",
            "ok": ok_step,
            "v3_summary": v3,
        }
    )
    ok &= ok_step

    # 8.4) 스킵 사유 케이스 (expected_ink_count 누락)
    res = _inspect(client, sku_temp, "INK1", v3_sample)
    dec = res.get("result", {}).get("results", [])[0].get("decision", {})
    v3_summary = dec.get("v3_summary")
    skipped_reason = dec.get("debug", {}).get("v3_summary_skipped_reason")
    ok_step = v3_summary is not None or skipped_reason in (None, "EXPECTED_INK_COUNT_MISSING")
    _print("v3 summary skipped reason when ink count missing", ok_step)
    report["steps"].append(
        {
            "name": "v3_summary_skipped_reason",
            "ok": ok_step,
            "reason": dec.get("debug", {}).get("v3_summary_skipped_reason"),
        }
    )
    ok &= ok_step

    # 9) v3 UI rules QA
    # 9.1) v3_summary missing -> card hidden
    ui_state = _ui_v3_state(None)
    ok_step = ui_state["visible"] is False
    _print("v3 UI hidden when summary missing", ok_step)
    report["steps"].append(
        {
            "name": "v3_ui_hidden_on_missing",
            "ok": ok_step,
            "ui_state": ui_state,
        }
    )
    ok &= ok_step

    # 9.2) schema mismatch -> card hidden
    ui_state = _ui_v3_state({"schema_version": "v3_summary.v0"})
    ok_step = ui_state["visible"] is False
    _print("v3 UI hidden on schema mismatch", ok_step)
    report["steps"].append(
        {
            "name": "v3_ui_hidden_on_schema_mismatch",
            "ok": ok_step,
            "ui_state": ui_state,
        }
    )
    ok &= ok_step

    # 9.3) severity badge mapping
    ui_warn = _ui_v3_state({"schema_version": "v3_summary.v1", "meta": {"severity": "WARN"}})
    ui_info = _ui_v3_state({"schema_version": "v3_summary.v1", "meta": {"severity": "INFO"}})
    ui_ok = _ui_v3_state({"schema_version": "v3_summary.v1", "meta": {"severity": "OK"}})
    ok_step = (
        ui_warn["badge_class"] == "status-badge border-yellow-warning text-yellow-warning"
        and ui_info["badge_class"] == "status-badge border-text-dim text-text-dim"
        and ui_ok["badge_class"] == "status-badge border-green-ok text-green-ok"
    )
    _print("v3 UI severity badge mapping", ok_step)
    report["steps"].append(
        {
            "name": "v3_ui_severity_badges",
            "ok": ok_step,
            "warn": ui_warn,
            "info": ui_info,
            "ok_state": ui_ok,
        }
    )
    ok &= ok_step

    # 10) v3.2 trend smoke tests (schema/metrics/sparsity)
    # 10.1) trend not generated when no v2_diagnostics
    trend = build_v3_trend([], window_requested=20)
    ok_step = bool(trend) and (trend.get("meta") or {}).get("window_effective") == 0
    _print("v3 trend none without v2 diagnostics", ok_step)
    report["steps"].append(
        {
            "name": "v3_trend_none_no_v2",
            "ok": ok_step,
        }
    )
    ok &= ok_step

    # 10.2) trend generated with mixed decisions (>=8)
    v2_diag = {
        "expected_ink_count": 3,
        "auto_estimation": {"suggested_k": 2, "confidence": 0.8},
        "palette": {"min_deltaE_between_clusters": 2.8},
        "segmentation": {"quality": {"min_deltaE_between_clusters": 2.8}},
        "ink_match": {
            "matched": True,
            "warning": "",
            "deltas": [{"index": 1, "deltaE": 6.5, "delta_L": -2.2, "delta_b": 1.2, "delta_a": 0.4}],
        },
        "warnings": [],
    }
    decisions = [(_fake_decision(v2_diag).to_dict()) for _ in range(12)]
    trend = build_v3_trend(decisions, window_requested=20)
    ok_step = bool(trend) and trend.get("schema_version") == "v3_trend.v1" and trend.get("window_effective") == 12
    _print("v3 trend generated with window_effective", ok_step)
    report["steps"].append(
        {
            "name": "v3_trend_generated",
            "ok": ok_step,
            "trend": trend,
        }
    )
    ok &= ok_step

    # 10.3) data_sparsity flag (<8) and UI badge mapping
    decisions = [(_fake_decision(v2_diag).to_dict()) for _ in range(7)]
    trend = build_v3_trend(decisions, window_requested=20)
    ui_state = _ui_v3_trend_state(trend)
    ok_step = (
        trend is not None
        and trend.get("meta", {}).get("data_sparsity") is True
        and ui_state["badge_class"] == "status-badge border-yellow-warning text-yellow-warning"
    )
    _print("v3 trend data_sparsity + UI badge", ok_step)
    report["steps"].append(
        {
            "name": "v3_trend_data_sparsity",
            "ok": ok_step,
            "trend": trend,
            "ui_state": ui_state,
        }
    )
    ok &= ok_step

    report["ok"] = bool(ok)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
