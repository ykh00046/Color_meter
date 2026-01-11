from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ink_metrics import deltaE as deltaE_metric
from .ink_metrics import ensure_cie_lab


def _resolve_baseline(baseline: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(baseline, dict):
        return baseline.get("clusters", []), baseline.get("aligned_modes", {})
    if isinstance(baseline, list):
        return baseline, {}
    return [], {}


def align_to_reference(
    clusters_ref: List[Dict[str, Any]],
    clusters_target: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ref_labs = [c["mean_lab"] for c in clusters_ref]
    target_labs = [c["mean_lab"] for c in clusters_target]

    if not ref_labs or not target_labs:
        return clusters_target

    ref_mat = np.array(ref_labs)
    target_mat = np.array(target_labs)

    # Cost matrix: Euclidean distance in Lab space
    cost_matrix = np.linalg.norm(ref_mat[:, None] - target_mat[None, :], axis=2)

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder target clusters to match reference
    aligned_clusters = [clusters_target[i] for i in col_ind]

    # Handle size mismatch if any (though typically same k is used)
    # If target has more clusters, append unmatched ones?
    # For now, assume same k or take matched subset.

    return aligned_clusters


def match_clusters_ab(
    clusters_ref: List[Dict[str, Any]],
    clusters_target: List[Dict[str, Any]],
) -> Tuple[List[int], float]:
    if not clusters_ref or not clusters_target:
        return [], 0.0

    ref_ab = np.array([[c["mean_lab"][1], c["mean_lab"][2]] for c in clusters_ref])
    target_ab = np.array([[c["mean_lab"][1], c["mean_lab"][2]] for c in clusters_target])

    cost_matrix = np.linalg.norm(ref_ab[:, None] - target_ab[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_cost = cost_matrix[row_ind, col_ind].sum()
    return col_ind.tolist(), float(total_cost)


def _normalize_deltae_method(method: str) -> str:
    m = str(method or "76").strip().lower()
    if m in ("2000", "de2000", "ciede2000", "cie2000"):
        return "2000"
    return "76"


def _deltaE76(lab1, lab2) -> float:
    return float(math.sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2))


def _deltaE2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0) -> float:
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0

    Cbar7 = Cbar**7
    G = 0.0
    if Cbar > 0:
        G = 0.5 * (1.0 - math.sqrt(Cbar7 / (Cbar7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def _hp(ap: float, b: float) -> float:
        if ap == 0.0 and b == 0.0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180.0:
            dhp = dh
        elif dh > 180.0:
            dhp = dh - 360.0
        else:
            dhp = dh + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        hbarp = h1p + h2p
    else:
        diff = abs(h1p - h2p)
        if diff <= 180.0:
            hbarp = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360.0:
            hbarp = (h1p + h2p + 360.0) / 2.0
        else:
            hbarp = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hbarp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hbarp))
        + 0.32 * math.cos(math.radians(3.0 * hbarp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hbarp - 63.0))
    )

    dtheta = 30.0 * math.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Rc = 0.0
    if Cbarp > 0:
        Rc = 2.0 * math.sqrt((Cbarp**7) / ((Cbarp**7) + (25.0**7)))

    Sl = 1.0 + (0.015 * ((Lbarp - 50.0) ** 2)) / math.sqrt(20.0 + ((Lbarp - 50.0) ** 2))
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T

    Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def deltaE(lab1, lab2, method="76") -> float:
    m = _normalize_deltae_method(method)
    return _deltaE2000(lab1, lab2) if m == "2000" else _deltaE76(lab1, lab2)


def compute_cluster_deltas(
    baseline: Any, sample_clusters: List[Dict[str, Any]], *, deltaE_method: str = "76"
) -> Dict[str, Any]:
    """
    Match sample clusters to baseline clusters and compute metrics.

    Args:
        baseline: Baseline dict or List of baseline cluster dicts
        sample_clusters: List of sample cluster dicts
        deltaE_method: "76" or "2000"

    Returns:
        {
            "matched": bool,
            "match_cost": float,
            "order": List[int],  # sample_idx for each baseline_idx
            "deltas": List[Dict], # detailed delta info
            "trajectory_summary": Dict
        }
    """
    baseline_clusters, aligned_modes = _resolve_baseline(baseline)
    k = len(baseline_clusters)
    if k == 0 or len(sample_clusters) == 0:
        return {
            "matched": False,
            "reason": "EMPTY_INPUT",
            "ref_k": k,
            "test_k": len(sample_clusters),
            "order": [],
            "deltas": [],
            "cluster_deltas": [],
            "unmatched_ref": list(range(k)),
            "unmatched_test": list(range(len(sample_clusters))),
            "match_cost": 0.0,
            "deltaE_method": "76",
        }

    method = _normalize_deltae_method(deltaE_method)
    # --- Robust matching (allow mismatch) ---
    ref_k = k
    test_k = len(sample_clusters)
    # Build all pair distances (deltaE) and pick best pairs greedily
    pairs = []
    for ri in range(ref_k):
        ref_lab = baseline_clusters[ri].get("mean_lab")
        for ti in range(test_k):
            test_lab = sample_clusters[ti].get("mean_lab")
            if ref_lab is None or test_lab is None:
                continue

            rl = ensure_cie_lab(np.array(ref_lab, dtype=np.float32)).reshape(
                3,
            )
            tl = ensure_cie_lab(np.array(test_lab, dtype=np.float32)).reshape(
                3,
            )
            # method: "76" or "2000"
            de = float(deltaE_metric(tl, rl, method=method))
            pairs.append((de, ri, ti))
    pairs.sort(key=lambda x: x[0])

    used_r = set()
    used_t = set()
    order = [-1] * ref_k  # ref index -> matched test index
    deltas = []
    total_cost = 0.0

    for de, ri, ti in pairs:
        if ri in used_r or ti in used_t:
            continue
        used_r.add(ri)
        used_t.add(ti)
        order[ri] = ti
        total_cost += float(de)
        # build delta entry
        ref = baseline_clusters[ri]
        test = sample_clusters[ti]

        # safer: use the same converted Lab used for deltaE
        rl = ensure_cie_lab(np.array(ref["mean_lab"], dtype=np.float32)).reshape(
            3,
        )
        tl = ensure_cie_lab(np.array(test["mean_lab"], dtype=np.float32)).reshape(
            3,
        )
        delta_L = float(tl[0] - rl[0])

        delta_area_pp = float((test.get("area_ratio", 0.0) - ref.get("area_ratio", 0.0)) * 100.0)
        deltas.append(
            {
                "index": ri,
                "sample_index": ti,
                "deltaE": float(de),
                "delta_L": float(delta_L),
                "delta_area_pp": float(delta_area_pp),
                "sample_hex": test.get("mean_hex") or test.get("mean_hex", None),
                "ref_hex": ref.get("mean_hex") or ref.get("mean_hex", None),
            }
        )
        if len(used_r) >= min(ref_k, test_k):
            break

    unmatched_ref = [i for i in range(ref_k) if i not in used_r]
    unmatched_test = [i for i in range(test_k) if i not in used_t]

    # Trajectory summary: use max deltaE among matched pairs as off-track proxy
    max_off_track = max([d["deltaE"] for d in deltas], default=None)
    traj_summary = {"max_off_track": max_off_track, "on_track_pos_min": 0.0, "on_track_pos_max": 0.0}

    matched = (len(deltas) > 0) and (len(unmatched_ref) == 0) and (len(unmatched_test) == 0)

    # UI expects cluster_deltas sometimes; provide alias
    return {
        "matched": bool(matched),
        "reason": None if matched else ("INK_COUNT_MISMATCH" if (ref_k != test_k) else "PARTIAL_MATCH"),
        "ref_k": ref_k,
        "test_k": test_k,
        "match_cost": float(total_cost),
        "order": order,
        "deltas": deltas,
        "cluster_deltas": deltas,
        "unmatched_ref": unmatched_ref,
        "unmatched_test": unmatched_test,
        "trajectory_summary": traj_summary if traj_summary["max_off_track"] is not None else {},
        "deltaE_method": method,
    }
