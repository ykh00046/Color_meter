"""
Simulation Validation Module

Phase 5: Provides CIEDE2000 color difference calculation and validation utilities
for comparing simulated colors against ground truth measurements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def calculate_delta_e_76(lab1: List[float], lab2: List[float]) -> float:
    """
    Calculate CIE76 color difference (Euclidean distance in Lab space).

    Simple and fast, but less perceptually uniform than CIEDE2000.

    Args:
        lab1: First color [L, a, b]
        lab2: Second color [L, a, b]

    Returns:
        ΔE*ab value
    """
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(dL**2 + da**2 + db**2)


def calculate_delta_e_2000(
    lab1: List[float],
    lab2: List[float],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """
    Calculate CIEDE2000 color difference.

    More perceptually uniform than CIE76, especially for:
    - Low saturation colors
    - Blue region
    - Lightness differences

    Reference: CIE 142-2001

    Args:
        lab1: First color [L, a, b]
        lab2: Second color [L, a, b]
        kL: Lightness weight (default 1.0)
        kC: Chroma weight (default 1.0)
        kH: Hue weight (default 1.0)

    Returns:
        ΔE00 value
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Calculate C'i and h'i
    C1_star = math.sqrt(a1**2 + b1**2)
    C2_star = math.sqrt(a2**2 + b2**2)
    C_bar_star = (C1_star + C2_star) / 2.0

    G = 0.5 * (1 - math.sqrt(C_bar_star**7 / (C_bar_star**7 + 25**7)))

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    # Step 2: Calculate ΔL', ΔC', ΔH'
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = 0.0
    if C1_prime * C2_prime != 0:
        dh = h2_prime - h1_prime
        if abs(dh) <= 180:
            dh_prime = dh
        elif dh > 180:
            dh_prime = dh - 360
        else:
            dh_prime = dh + 360

    dH_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2))

    # Step 3: Calculate CIEDE2000 Color-Difference
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h_bar_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h1_prime - h2_prime) <= 180:
            h_bar_prime = (h1_prime + h2_prime) / 2.0
        else:
            if h1_prime + h2_prime < 360:
                h_bar_prime = (h1_prime + h2_prime + 360) / 2.0
            else:
                h_bar_prime = (h1_prime + h2_prime - 360) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(h_bar_prime - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63))
    )

    SL = 1 + (0.015 * (L_bar_prime - 50) ** 2) / math.sqrt(20 + (L_bar_prime - 50) ** 2)
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T

    dTheta = 30 * math.exp(-(((h_bar_prime - 275) / 25) ** 2))
    RC = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    RT = -RC * math.sin(math.radians(2 * dTheta))

    dE00 = math.sqrt(
        (dL_prime / (kL * SL)) ** 2
        + (dC_prime / (kC * SC)) ** 2
        + (dH_prime / (kH * SH)) ** 2
        + RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
    )

    return dE00


def validate_simulation_accuracy(
    simulated_lab: List[float],
    measured_lab: List[float],
    sample_id: str = "",
    target_delta_e: float = 2.5,
) -> Dict[str, Any]:
    """
    Validate simulation accuracy against ground truth measurement.

    Args:
        simulated_lab: Simulated color [L, a, b]
        measured_lab: Measured color from spectrophotometer [L, a, b]
        sample_id: Optional sample identifier
        target_delta_e: Target ΔE00 threshold (default 2.5)

    Returns:
        Validation result with both ΔE76 and ΔE00
    """
    delta_e_76 = calculate_delta_e_76(simulated_lab, measured_lab)
    delta_e_00 = calculate_delta_e_2000(simulated_lab, measured_lab)

    passed = delta_e_00 < target_delta_e

    return {
        "sample_id": sample_id,
        "simulated_lab": [round(x, 2) for x in simulated_lab],
        "measured_lab": [round(x, 2) for x in measured_lab],
        "delta_e_76": round(delta_e_76, 2),
        "delta_e_00": round(delta_e_00, 2),
        "target_delta_e": target_delta_e,
        "passed": passed,
        "interpretation": _interpret_delta_e(delta_e_00),
    }


def _interpret_delta_e(delta_e: float) -> str:
    """
    Interpret ΔE00 value in human-readable terms.
    """
    if delta_e < 1.0:
        return "Not perceptible by human eye"
    elif delta_e < 2.0:
        return "Perceptible through close observation"
    elif delta_e < 3.5:
        return "Perceptible at a glance"
    elif delta_e < 5.0:
        return "More perceptible than acceptable"
    else:
        return "Clearly different colors"


def batch_validate(
    simulations: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, List[float]]],
    mode: str = "white",
    target_delta_e: float = 2.5,
) -> Dict[str, Any]:
    """
    Batch validate multiple simulations against ground truth.

    Args:
        simulations: List of simulation results from build_simulation_result
        ground_truth: Dict mapping sample_id -> {"on_white_lab": [...], "on_black_lab": [...]}
        mode: "white" or "black"
        target_delta_e: Target ΔE00 threshold

    Returns:
        Batch validation results with summary statistics
    """
    results = []
    delta_e_values = []

    for sim in simulations:
        sample_id = sim.get("ink_id", "unknown")
        simulated_lab = sim["perceived"][f"on_{mode}"]["lab"]

        gt_key = f"on_{mode}_lab"
        if sample_id in ground_truth and gt_key in ground_truth[sample_id]:
            measured_lab = ground_truth[sample_id][gt_key]
            result = validate_simulation_accuracy(simulated_lab, measured_lab, sample_id, target_delta_e)
            results.append(result)
            delta_e_values.append(result["delta_e_00"])

    if not delta_e_values:
        return {"results": [], "summary": None}

    import statistics

    summary = {
        "total_samples": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 3),
        "delta_e_mean": round(statistics.mean(delta_e_values), 2),
        "delta_e_std": round(statistics.stdev(delta_e_values), 2) if len(delta_e_values) > 1 else 0.0,
        "delta_e_max": round(max(delta_e_values), 2),
        "delta_e_min": round(min(delta_e_values), 2),
        "target_delta_e": target_delta_e,
    }

    return {
        "results": results,
        "summary": summary,
    }
