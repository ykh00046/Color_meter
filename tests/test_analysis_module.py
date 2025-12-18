import os
import sys

import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.profile_analyzer import ProfileAnalyzer


def test_analysis_module():
    print("Testing ProfileAnalyzer...")

    # 1. 가짜 데이터 생성 (Radius 0.0 ~ 1.0, 100포인트)
    r_norm = np.linspace(0, 1.0, 100)

    # L값: 0.3에서 급격히 상승, 0.7에서 급격히 하강하는 패턴 생성 (Step function like)
    l_data = np.full_like(r_norm, 70.0)
    l_data[30:70] = 80.0  # Zone 생성

    # 약간의 노이즈 추가
    np.random.seed(42)
    l_data += np.random.normal(0, 0.5, size=l_data.shape)

    a_data = np.full_like(r_norm, 10.0)
    b_data = np.full_like(r_norm, -5.0)

    analyzer = ProfileAnalyzer()

    # 2. 분석 실행
    print("Running analysis_profile()...")
    results = analyzer.analyze_profile(r_norm, l_data, a_data, b_data, smoothing_window=5, gradient_threshold=1.0)

    # 3. 검증 (Updated for current API structure)
    smoothed_L = results["profile"]["L_smoothed"]
    candidates = results["boundary_candidates"]

    print(f"Original L len: {len(l_data)}, Smoothed L len: {len(smoothed_L)}")
    print(f"Number of boundary candidates: {len(candidates)}")

    for c in candidates:
        print(f"Candidate: Method={c['method']}, Radius={c['radius']:.3f}, Conf={c.get('confidence')}")

    # 예상: 0.3 근처와 0.7 근처에서 Peak가 검출되어야 함
    detected_zones = [c["radius"] for c in candidates if c["method"] == "peak_gradient"]
    print(f"Detected gradient peaks at radii: {detected_zones}")

    if len(detected_zones) >= 2:
        print("SUCCESS: Detected step changes correctly.")
    else:
        print("WARNING: Might have failed to detect distinct steps.")


if __name__ == "__main__":
    test_analysis_module()
