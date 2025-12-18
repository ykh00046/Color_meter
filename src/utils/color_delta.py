"""
Color Delta E Calculation Module

CIEDE2000 색차 계산 구현.
NumPy 기반 순수 Python 구현으로 colormath 의존성 제거.

References:
- Sharma, G., Wu, W., & Dalal, E. N. (2005).
  "The CIEDE2000 color-difference formula: Implementation notes,
   supplementary test data, and mathematical observations."
  Color Research & Application, 30(1), 21-30.
"""

from typing import Tuple, Union

import numpy as np


def delta_e_cie2000(
    lab1: Union[Tuple[float, float, float], np.ndarray],
    lab2: Union[Tuple[float, float, float], np.ndarray],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """
    CIEDE2000 색차(ΔE) 계산.

    Args:
        lab1: 첫 번째 색상의 LAB 값 (L*, a*, b*)
        lab2: 두 번째 색상의 LAB 값 (L*, a*, b*)
        kL: 명도(L*) 가중치 (기본값 1.0)
        kC: 채도(C*) 가중치 (기본값 1.0)
        kH: 색상(H*) 가중치 (기본값 1.0)

    Returns:
        ΔE2000 값 (float)

    Examples:
        >>> delta_e_cie2000((50, 2.5, -10), (55, 3.5, -9))
        5.123...
    """
    # LAB 값 추출
    if isinstance(lab1, (tuple, list)):
        L1, a1, b1 = lab1
    else:
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]

    if isinstance(lab2, (tuple, list)):
        L2, a2, b2 = lab2
    else:
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # 1. Calculate C1, C2 (Chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)

    # 2. Calculate C_bar
    C_bar = (C1 + C2) / 2.0

    # 3. Calculate G
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))

    # 4. Calculate a'1, a'2
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    # 5. Calculate C'1, C'2
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    # 6. Calculate h'1, h'2 (hue angle in degrees)
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    if h2_prime < 0:
        h2_prime += 360

    # 7. Calculate ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Calculate Δh'
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360

    # Calculate ΔH'
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

    # 8. Calculate L_bar', C_bar', H_bar'
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    # Calculate H_bar'
    if C1_prime * C2_prime == 0:
        H_bar_prime = h1_prime + h2_prime
    else:
        sum_h = h1_prime + h2_prime
        abs_diff = abs(h1_prime - h2_prime)
        if abs_diff <= 180:
            H_bar_prime = sum_h / 2.0
        elif sum_h < 360:
            H_bar_prime = (sum_h + 360) / 2.0
        else:
            H_bar_prime = (sum_h - 360) / 2.0

    # 9. Calculate T
    T = (
        1.0
        - 0.17 * np.cos(np.radians(H_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * H_bar_prime))
        + 0.32 * np.cos(np.radians(3 * H_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * H_bar_prime - 63))
    )

    # 10. Calculate SL, SC, SH
    SL = 1 + ((0.015 * (L_bar_prime - 50) ** 2) / np.sqrt(20 + (L_bar_prime - 50) ** 2))
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T

    # 11. Calculate RT (rotation term)
    delta_theta = 30 * np.exp(-(((H_bar_prime - 275) / 25) ** 2))
    RC = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    RT = -np.sin(np.radians(2 * delta_theta)) * RC

    # 12. Calculate ΔE2000
    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL)) ** 2
        + (delta_C_prime / (kC * SC)) ** 2
        + (delta_H_prime / (kH * SH)) ** 2
        + RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )

    return float(delta_E)


def delta_e_cie1976(
    lab1: Union[Tuple[float, float, float], np.ndarray], lab2: Union[Tuple[float, float, float], np.ndarray]
) -> float:
    """
    CIEDE1976 색차(ΔE*ab) 계산.

    간단한 유클리드 거리 기반 색차.

    Args:
        lab1: 첫 번째 색상의 LAB 값 (L*, a*, b*)
        lab2: 두 번째 색상의 LAB 값 (L*, a*, b*)

    Returns:
        ΔE*ab 값 (float)

    Examples:
        >>> delta_e_cie1976((50, 2.5, -10), (55, 3.5, -9))
        5.244...
    """
    # LAB 값 추출
    if isinstance(lab1, (tuple, list)):
        L1, a1, b1 = lab1
    else:
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]

    if isinstance(lab2, (tuple, list)):
        L2, a2, b2 = lab2
    else:
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    delta_E = np.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)

    return float(delta_E)


def delta_e_cie1994(
    lab1: Union[Tuple[float, float, float], np.ndarray],
    lab2: Union[Tuple[float, float, float], np.ndarray],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
    K1: float = 0.045,
    K2: float = 0.015,
) -> float:
    """
    CIEDE1994 색차(ΔE*94) 계산.

    Args:
        lab1: 첫 번째 색상의 LAB 값 (L*, a*, b*)
        lab2: 두 번째 색상의 LAB 값 (L*, a*, b*)
        kL: 명도 가중치 (기본값 1.0)
        kC: 채도 가중치 (기본값 1.0)
        kH: 색상 가중치 (기본값 1.0)
        K1: 채도 보정 계수 (기본값 0.045, 그래픽 아트용)
        K2: 색상 보정 계수 (기본값 0.015, 그래픽 아트용)

    Returns:
        ΔE*94 값 (float)

    Note:
        텍스타일용: K1=0.048, K2=0.014
        그래픽 아트용: K1=0.045, K2=0.015 (기본값)
    """
    # LAB 값 추출
    if isinstance(lab1, (tuple, list)):
        L1, a1, b1 = lab1
    else:
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]

    if isinstance(lab2, (tuple, list)):
        L2, a2, b2 = lab2
    else:
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # Calculate C1, C2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)

    # Calculate ΔL, ΔC, Δa, Δb
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2

    # Calculate ΔH
    delta_H_sq = delta_a**2 + delta_b**2 - delta_C**2
    if delta_H_sq < 0:
        delta_H_sq = 0  # Rounding error protection
    delta_H = np.sqrt(delta_H_sq)

    # Calculate SL, SC, SH
    SL = 1.0
    SC = 1.0 + K1 * C1
    SH = 1.0 + K2 * C1

    # Calculate ΔE*94
    delta_E = np.sqrt((delta_L / (kL * SL)) ** 2 + (delta_C / (kC * SC)) ** 2 + (delta_H / (kH * SH)) ** 2)

    return float(delta_E)


# Convenience aliases
delta_e = delta_e_cie2000  # Default to CIEDE2000


if __name__ == "__main__":
    # 테스트 코드
    print("=== Color Delta E Calculation Test ===\n")

    # Test case from Sharma et al. (2005)
    lab1 = (50.0000, 2.6772, -79.7751)
    lab2 = (50.0000, 0.0000, -82.7485)

    de2000 = delta_e_cie2000(lab1, lab2)
    de1976 = delta_e_cie1976(lab1, lab2)
    de1994 = delta_e_cie1994(lab1, lab2)

    print(f"LAB1: {lab1}")
    print(f"LAB2: {lab2}")
    print(f"ΔE2000: {de2000:.4f}")
    print(f"ΔE1976: {de1976:.4f}")
    print(f"ΔE1994: {de1994:.4f}")
    print("\nExpected ΔE2000: ~2.04 (from reference)")

    # Test with identical colors
    print("\n--- Test with identical colors ---")
    lab_same = (50.0, 10.0, -20.0)
    de = delta_e_cie2000(lab_same, lab_same)
    print(f"ΔE2000 (same color): {de:.6f} (should be ~0.0)")

    # Test with very different colors
    print("\n--- Test with very different colors ---")
    white = (100.0, 0.0, 0.0)
    black = (0.0, 0.0, 0.0)
    de = delta_e_cie2000(white, black)
    print(f"ΔE2000 (white vs black): {de:.4f}")
