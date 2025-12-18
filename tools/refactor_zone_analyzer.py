"""
Refactor _generate_analysis_summaries() into 3 sub-functions
"""


def refactor_file():
    file_path = "src/core/zone_analyzer_2d.py"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the insertion point (before _generate_analysis_summaries)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "def _generate_analysis_summaries(":
            insert_idx = i
            break

    if insert_idx is None:
        print("ERROR: Could not find _generate_analysis_summaries")
        return

    print(f"Found _generate_analysis_summaries at line {insert_idx + 1}")

    # Create 3 new helper functions
    new_functions = '''

def _build_confidence_breakdown(confidence: float, confidence_factors: Any) -> Dict:
    """
    Confidence breakdown 생성

    Returns:
        confidence_breakdown dict
    """
    return {
        "overall": float(confidence),
        "factors": [
            {
                "name": "pixel_count",
                "weight": 0.30,
                "score": float(confidence_factors.pixel_count_score),
                "contribution": float(0.30 * confidence_factors.pixel_count_score),
                "status": (
                    "good"
                    if confidence_factors.pixel_count_score >= 0.9
                    else ("warning" if confidence_factors.pixel_count_score >= 0.7 else "poor")
                ),
                "description": "Zone별 픽셀 수 충분도",
            },
            {
                "name": "transition",
                "weight": 0.25,
                "score": float(confidence_factors.transition_score),
                "contribution": float(0.25 * confidence_factors.transition_score),
                "status": (
                    "good"
                    if confidence_factors.transition_score >= 0.9
                    else ("warning" if confidence_factors.transition_score >= 0.7 else "poor")
                ),
                "description": "전이 구간 제거 정도 (적을수록 좋음)",
            },
            {
                "name": "uniformity",
                "weight": 0.25,
                "score": float(confidence_factors.std_score),
                "contribution": float(0.25 * confidence_factors.std_score),
                "status": (
                    "good"
                    if confidence_factors.std_score >= 0.9
                    else ("warning" if confidence_factors.std_score >= 0.7 else "poor")
                ),
                "description": "각도 균일성 (std_L 낮을수록 좋음)",
            },
            {
                "name": "sector_uniformity",
                "weight": 0.10,
                "score": float(confidence_factors.sector_uniformity),
                "contribution": float(0.10 * confidence_factors.sector_uniformity),
                "status": (
                    "good"
                    if confidence_factors.sector_uniformity >= 0.9
                    else ("warning" if confidence_factors.sector_uniformity >= 0.7 else "poor")
                ),
                "description": "섹터 간 균일성",
            },
            {
                "name": "lens_detection",
                "weight": 0.10,
                "score": float(confidence_factors.lens_detection),
                "contribution": float(0.10 * confidence_factors.lens_detection),
                "status": (
                    "good"
                    if confidence_factors.lens_detection >= 0.9
                    else ("warning" if confidence_factors.lens_detection >= 0.7 else "poor")
                ),
                "description": "렌즈 검출 신뢰도",
            },
        ],
    }


def _build_analysis_summary(
    max_std_l: float,
    used_fallback_B: bool,
    confidence_factors: Any,
    confidence: float,
    zone_results_raw: List[Dict],
    print_band_area: int,
) -> Dict:
    """
    Analysis summary 생성

    Returns:
        analysis_summary dict
    """
    return {
        "uniformity": {
            "max_std_L": float(max_std_l),
            "threshold_retake": 12.0,
            "threshold_warning": 10.0,
            "status": "good" if max_std_l < 10.0 else ("warning" if max_std_l <= 12.0 else "poor"),
            "impact": (
                "정상" if max_std_l < 10.0 else ("OK_WITH_WARNING 트리거" if max_std_l <= 12.0 else "RETAKE 트리거")
            ),
        },
        "boundary_quality": {
            "B_zone_method": "fallback" if used_fallback_B else "auto_detected",
            "confidence_contribution": float(confidence_factors.transition_score),
            "status": "good" if not used_fallback_B else ("warning" if confidence >= 0.8 else "poor"),
            "impact": "정상" if not used_fallback_B else ("Confidence 페널티" if confidence < 0.85 else "경고"),
        },
        "coverage": {
            "total_pixels": int(sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)),
            "expected_min": int(print_band_area * 0.5),
            "status": (
                "good"
                if sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
                >= print_band_area * 0.5
                else "poor"
            ),
            "impact": (
                "정상"
                if sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
                >= print_band_area * 0.5
                else "RETAKE (R2_CoverageLow)"
            ),
        },
    }


def _build_risk_factors(
    max_std_l: float,
    sector_stats: Dict,
    used_fallback_B: bool,
    confidence: float,
    lens_conf: float,
    zone_results_raw: List[Dict],
    print_band_area: int,
) -> List[Dict[str, Any]]:
    """
    Risk factors 생성

    Returns:
        risk_factors list
    """
    risk_factors: List[Dict[str, Any]] = []

    # Uniformity 위험 요소
    if max_std_l > 12.0:
        risk_factors.append(
            {
                "category": "uniformity",
                "severity": "high",
                "message": f"각도 불균일 높음 (std_L={max_std_l:.1f}, 임계값=12.0)",
                "source": "R4_UniformityLow",
            }
        )
    elif max_std_l > 10.0:
        risk_factors.append(
            {
                "category": "uniformity",
                "severity": "medium",
                "message": f"각도 불균일 경계값 근접 (std_L={max_std_l:.1f}, 경고=10.0)",
                "source": "OK_WITH_WARNING 조건",
            }
        )

    # Sector Uniformity 위험 요소 (국부 불량)
    if sector_stats["enabled"]:
        max_sector_std = sector_stats["max_sector_std_L"]
        worst_zone = sector_stats["worst_zone"]

        if max_sector_std > 8.0:
            risk_factors.append(
                {
                    "category": "sector_uniformity",
                    "severity": "high",
                    "message": f"Zone {worst_zone} 섹터 간 편차 높음 (std_L={max_sector_std:.1f}, 임계값=8.0)",
                    "source": "sector_analysis",
                    "details": {
                        "zone": worst_zone,
                        "max_sector_std_L": float(max_sector_std),
                        "zone_stats": sector_stats["zone_stats"],
                    },
                }
            )
        elif max_sector_std > 5.0:
            risk_factors.append(
                {
                    "category": "sector_uniformity",
                    "severity": "medium",
                    "message": f"Zone {worst_zone} 섹터 간 편차 경계값 근접 (std_L={max_sector_std:.1f}, 경고=5.0)",
                    "source": "sector_analysis",
                    "details": {
                        "zone": worst_zone,
                        "max_sector_std_L": float(max_sector_std),
                        "zone_stats": sector_stats["zone_stats"],
                    },
                }
            )

    # Boundary 위험 요소
    if used_fallback_B:
        severity = "high" if confidence < 0.8 else "medium"
        risk_factors.append(
            {
                "category": "boundary",
                "severity": severity,
                "message": "Zone B 경계 자동 탐지 실패 (fallback 사용)",
                "source": "R3_BoundaryUncertain" if confidence < 0.8 else "경고",
            }
        )

    # Lens detection 위험 요소
    if lens_conf < 0.7:
        risk_factors.append(
            {
                "category": "lens_detection",
                "severity": "high",
                "message": f"렌즈 검출 신뢰도 낮음 (confidence={lens_conf:.2f})",
                "source": "R1_DetectionLow",
            }
        )

    # Coverage 위험 요소
    total_pixels = sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
    min_total_pixels = int(print_band_area * 0.5)
    if total_pixels < min_total_pixels:
        risk_factors.append(
            {
                "category": "coverage",
                "severity": "high",
                "message": f"픽셀 커버리지 부족 ({total_pixels}/{min_total_pixels})",
                "source": "R2_CoverageLow",
            }
        )

    # Zone ΔE 위험 요소
    for zr in zone_results_raw:
        if not zr["is_ok"] and zr["delta_e"] is not None:
            severity = "high" if zr["delta_e"] > zr["threshold"] * 1.5 else "medium"
            risk_factors.append(
                {
                    "category": "zone_quality",
                    "severity": severity,
                    "message": f"Zone {zr['zone_name']} ΔE 초과 (ΔE={zr['delta_e']:.1f} > {zr['threshold']:.1f})",
                    "source": "NG 판정",
                }
            )

    return risk_factors


'''

    # Insert new functions before _generate_analysis_summaries
    lines.insert(insert_idx, new_functions)

    # Now replace the old _generate_analysis_summaries with a simpler version
    # Find the end of _generate_analysis_summaries (before _perform_ink_analysis)
    end_idx = None
    for i in range(insert_idx + 1, len(lines)):
        if lines[i].strip().startswith("def _perform_ink_analysis"):
            end_idx = i
            break

    if end_idx is None:
        print("ERROR: Could not find end of _generate_analysis_summaries")
        return

    print(f"Replacing lines {insert_idx + 1} to {end_idx}")

    # Create new simplified version
    new_implementation = '''def _generate_analysis_summaries(
    confidence_factors: Any,
    zone_results_raw: List[Dict],
    used_fallback_B: bool,
    confidence: float,
    print_band_area: int,
    sector_stats: Dict,
    lens_conf: float,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    confidence_breakdown, analysis_summary, risk_factors 생성

    Returns:
        confidence_breakdown, analysis_summary, risk_factors
    """
    # max_std_l 계산
    max_std_l = max([zr.get("std_lab", [0])[0] for zr in zone_results_raw if zr.get("std_lab")], default=0.0)

    # 3개의 sub-함수 호출 (리팩토링됨)
    confidence_breakdown = _build_confidence_breakdown(confidence, confidence_factors)

    analysis_summary = _build_analysis_summary(
        max_std_l, used_fallback_B, confidence_factors, confidence, zone_results_raw, print_band_area
    )

    risk_factors = _build_risk_factors(
        max_std_l, sector_stats, used_fallback_B, confidence, lens_conf, zone_results_raw, print_band_area
    )

    return confidence_breakdown, analysis_summary, risk_factors


'''

    # Replace old function with new one
    del lines[insert_idx:end_idx]
    lines.insert(insert_idx, new_implementation)

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("OK Refactoring complete!")
    print("  - Extracted 3 helper functions:")
    print("    - _build_confidence_breakdown()")
    print("    - _build_analysis_summary()")
    print("    - _build_risk_factors()")
    print("  - Simplified _generate_analysis_summaries()")


if __name__ == "__main__":
    refactor_file()
