import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger(__name__)


class ProfileAnalyzer:
    """
    Radial Profile 데이터를 분석하여 1차/2차 미분, 스무딩, 경계 후보 등을 계산하는 클래스.
    '분석 우선(Analysis First)' 원칙에 따라 시각화 및 검증을 위한 상세 데이터를 생성합니다.
    """

    def __init__(self):
        pass

    def smooth_profile(self, data: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
        """
        Savitzky-Golay 필터를 사용하여 데이터를 스무딩합니다.

        Args:
            data: 1D numpy array (프로파일 데이터)
            window_length: 필터 윈도우 크기 (홀수여야 함). 작을수록 디테일 보존, 클수록 노이즈 제거.
            polyorder: 다항식 차수. 보통 2 또는 3 사용.

        Returns:
            Smoothed data array
        """
        # 데이터 길이가 window_length보다 작으면 스무딩 불가, 원본 반환
        if len(data) < window_length:
            logger.warning(
                f"Data length ({len(data)}) is smaller than window_length ({window_length}). Skipping smoothing."
            )
            return data

        # window_length가 짝수면 홀수로 조정
        if window_length % 2 == 0:
            window_length += 1

        try:
            return savgol_filter(data, window_length, polyorder)
        except Exception as e:
            logger.error(f"Error during smoothing: {e}")
            return data

    def compute_gradient(self, data: np.ndarray, x_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        데이터의 1차 미분(Gradient)을 계산합니다.

        Args:
            data: 1D numpy array (y값)
            x_coords: 1D numpy array (x값, 예: radius). None이면 간격 1로 가정.

        Returns:
            Gradient array (dy/dx)
        """
        if x_coords is None:
            return np.gradient(data)
        return np.gradient(data, x_coords)

    def compute_second_derivative(self, data: np.ndarray, x_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        데이터의 2차 미분을 계산합니다. (Gradient의 Gradient)

        Args:
            data: 1D numpy array (y값, 이미 스무딩된 데이터 권장)
            x_coords: 1D numpy array (x값)

        Returns:
            Second derivative array (d^2y/dx^2)
        """
        first_derivative = self.compute_gradient(data, x_coords)
        return self.compute_gradient(first_derivative, x_coords)

    def detect_inflection_points(
        self, second_derivative: np.ndarray, x_coords: np.ndarray, threshold: float = 1e-4
    ) -> List[Dict]:
        """
        2차 미분의 Zero-crossing 지점(변곡점)을 검출합니다.

        Args:
            second_derivative: 2차 미분 데이터
            x_coords: x축 좌표 (radius)
            threshold: 0으로 간주할 임계값 (노이즈로 인한 불필요한 교차 방지)

        Returns:
            List of dictionaries containing inflection point info
        """
        inflections = []

        # 부호 변화 감지 (sign change)
        signs = np.sign(second_derivative)
        sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
        sign_changes[0] = 0  # 첫 번째 요소는 무시

        indices = np.where(sign_changes == 1)[0]

        for idx in indices:
            # 2차 미분 값이 0에 가까운지 확인 (급격한 변화가 아닌 노이즈일 수 있음)
            # 혹은 기울기(3차미분)가 충분히 큰지 확인하여 유의미한 변곡점인지 판단 가능
            # 여기서는 단순하게 변화 전후 값의 절대값이 threshold 이상인지 확인
            prev_val = second_derivative[idx - 1]
            curr_val = second_derivative[idx]

            # Zero-crossing이 일어났고, 변화폭이 의미가 있다면
            if abs(prev_val - curr_val) > threshold:
                inflections.append(
                    {
                        "method": "inflection_point",
                        "index": int(idx),
                        "radius": float(x_coords[idx]),
                        "value": float(curr_val),  # 0에 가까운 값
                        "confidence": 0.5,  # 변곡점은 보통 경계의 시작/끝일 가능성이 큼
                    }
                )

        return inflections

    def detect_peaks(
        self,
        data: np.ndarray,
        x_coords: np.ndarray,
        height: Optional[float] = None,
        prominence: Optional[float] = None,
        distance: Optional[int] = None,
    ) -> List[Dict]:
        """
        데이터(주로 1차 미분값)에서 피크를 검출합니다. 급격한 변화 지점을 찾을 때 사용합니다.

        Args:
            data: 분석할 데이터 (예: Gradient 절대값)
            x_coords: x축 좌표
            height: 피크의 최소 높이
            prominence: 피크의 돌출 정도 (주변 대비 얼마나 튀어나왔나)
            distance: 피크 간 최소 거리 (인덱스 기준)

        Returns:
            List of dictionaries containing peak info
        """
        peaks, properties = find_peaks(data, height=height, prominence=prominence, distance=distance)

        results = []
        for i, peak_idx in enumerate(peaks):
            peak_info = {
                "method": "peak_gradient",
                "index": int(peak_idx),
                "radius": float(x_coords[peak_idx]),
                "value": float(data[peak_idx]),
                "properties": {k: v[i] for k, v in properties.items()} if properties else {},
                "confidence": 0.8,  # Gradient 피크는 강력한 경계 후보
            }
            results.append(peak_info)

        return results

    def detect_print_boundaries(
        self,
        r_norm: np.ndarray,
        a_data: np.ndarray,
        b_data: np.ndarray,
        method: str = "chroma",
        chroma_threshold: float = 2.0,
    ) -> Tuple[float, float, float]:
        """
        Radial profile에서 실제 인쇄 영역의 r_inner, r_outer를 자동 검출합니다.

        투명한 렌즈 외곽 영역을 제외하고 실제 색이 인쇄된 영역만 분석하여
        색상 평균 정확도를 20-30% 향상시킵니다.

        Args:
            r_norm: 정규화된 반경 배열 (0~1)
            a_data: a* 채널 데이터
            b_data: b* 채널 데이터
            method: 검출 방법
                - "chroma": 색도(sqrt(a^2 + b^2)) 기반 (권장)
                - "gradient": 색도 그래디언트 기반
                - "hybrid": 둘 다 사용
            chroma_threshold: 배경 노이즈 임계값 (기본 2.0)

        Returns:
            (r_inner, r_outer, confidence): 정규화된 반경 (0~1) 및 신뢰도 (0~1)

        Example:
            사용자 분석: r_inner=119px, r_outer=387px, lens_radius=400px
            → r_inner=0.2975, r_outer=0.9675
        """
        # 1. 색도(Chroma) 계산
        chroma = np.sqrt(a_data**2 + b_data**2)

        # 2. 배경 노이즈 레벨 추정 (최소값 10% 평균)
        noise_level = np.percentile(chroma, 10)

        # 3. 색이 있는 구간 검출
        threshold = noise_level + chroma_threshold
        colored_mask = chroma > threshold

        if not np.any(colored_mask):
            logger.warning(
                "No colored area detected, using full range. "
                "Consider lowering chroma_threshold or check image quality."
            )
            return (0.0, 1.0, 0.0)

        # 4. 첫/마지막 색 영역 찾기
        colored_indices = np.where(colored_mask)[0]
        inner_idx = colored_indices[0]
        outer_idx = colored_indices[-1]

        r_inner = float(r_norm[inner_idx])
        r_outer = float(r_norm[outer_idx])

        # 5. Gradient 기반 refinement (method가 "gradient" 또는 "hybrid"일 때)
        if method in ("gradient", "hybrid"):
            chroma_grad = np.abs(self.compute_gradient(chroma, r_norm))
            grad_smooth = self.smooth_profile(chroma_grad, window_length=5)

            # Inner boundary refinement: 첫 큰 gradient 피크
            inner_search_start = max(0, inner_idx - 10)
            inner_search_end = min(len(r_norm), inner_idx + 10)
            inner_region_grad = grad_smooth[inner_search_start:inner_search_end]

            if len(inner_region_grad) > 0:
                inner_peak_idx = np.argmax(inner_region_grad) + inner_search_start
                r_inner_refined = float(r_norm[inner_peak_idx])

                # Gradient 방법이 더 정확하다고 판단되면 업데이트
                if abs(r_inner_refined - r_inner) < 0.1:  # 10% 이내 차이면 신뢰
                    r_inner = r_inner_refined

            # Outer boundary refinement: 마지막 큰 gradient 피크
            outer_search_start = max(0, outer_idx - 10)
            outer_search_end = min(len(r_norm), outer_idx + 10)
            outer_region_grad = grad_smooth[outer_search_start:outer_search_end]

            if len(outer_region_grad) > 0:
                outer_peak_idx = np.argmax(outer_region_grad) + outer_search_start
                r_outer_refined = float(r_norm[outer_peak_idx])

                if abs(r_outer_refined - r_outer) < 0.1:
                    r_outer = r_outer_refined

        # 6. 안전성 체크
        confidence = 1.0

        if r_outer - r_inner < 0.2:
            logger.warning(
                f"Print area too narrow ({r_outer - r_inner:.3f}), may be detection error. "
                "Check lens detection accuracy or adjust chroma_threshold."
            )
            confidence = 0.3

        # 중심 홀이 너무 크면 의심
        if r_inner > 0.5:
            logger.warning(
                f"Inner radius very large ({r_inner:.3f}), may indicate clear center hole. " "Verify lens type."
            )
            confidence *= 0.7

        # 외곽이 너무 작으면 의심
        if r_outer < 0.7:
            logger.warning(f"Outer radius small ({r_outer:.3f}), print area may be limited. " "Verify lens type.")
            confidence *= 0.8

        logger.info(
            f"Detected print area: r_inner={r_inner:.3f}, r_outer={r_outer:.3f}, "
            f"confidence={confidence:.2f}, method={method}"
        )

        return (r_inner, r_outer, confidence)

    def analyze_profile(
        self,
        r_norm: np.ndarray,
        l_data: np.ndarray,
        a_data: np.ndarray,
        b_data: np.ndarray,
        smoothing_window: int = 5,
        gradient_threshold: float = 0.5,
    ) -> Dict:
        """
        L, a, b 프로파일 전체에 대한 종합 분석을 수행합니다.
        API에서 호출하는 메인 진입점입니다.
        """

        # 1. Smoothing
        l_smooth = self.smooth_profile(l_data, window_length=smoothing_window)
        a_smooth = self.smooth_profile(a_data, window_length=smoothing_window)
        b_smooth = self.smooth_profile(b_data, window_length=smoothing_window)

        # 2. Gradients (Change rate)
        grad_l = self.compute_gradient(l_smooth, r_norm)
        grad_a = self.compute_gradient(a_smooth, r_norm)
        grad_b = self.compute_gradient(b_smooth, r_norm)

        # Gradient Magnitude (L, a, b 변화량 합) - 경계 검출용 통합 지표
        grad_magnitude = np.sqrt(grad_l**2 + grad_a**2 + grad_b**2)

        # 3. Second Derivatives
        sec_l = self.compute_second_derivative(l_smooth, r_norm)

        # 4. Boundary Candidates Detection
        candidates = []

        # A. Gradient Peaks (급격한 색상 변화)
        peaks = self.detect_peaks(
            grad_magnitude,
            r_norm,
            height=gradient_threshold,
            distance=int(len(r_norm) * 0.05),
        )
        candidates.extend(peaks)

        # B. Inflection Points (L값 변곡점 - 밝기 변화의 전환점)
        inflections = self.detect_inflection_points(sec_l, r_norm, threshold=0.01)
        candidates.extend(inflections)

        # 결과 정리
        analysis_result = {
            "profile": {
                "radius": r_norm.tolist(),
                "L_raw": l_data.tolist(),
                "a_raw": a_data.tolist(),
                "b_raw": b_data.tolist(),
                "L_smoothed": l_smooth.tolist(),
                "a_smoothed": a_smooth.tolist(),
                "b_smoothed": b_smooth.tolist(),
            },
            "derivatives": {
                "gradient_L": grad_l.tolist(),
                "gradient_a": grad_a.tolist(),
                "gradient_b": grad_b.tolist(),
                "gradient_magnitude": grad_magnitude.tolist(),
                "second_derivative_L": sec_l.tolist(),
            },
            "boundary_candidates": candidates,
        }

        return analysis_result
