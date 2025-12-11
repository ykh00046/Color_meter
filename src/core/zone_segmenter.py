"""
Zone Segmenter Module

RadialProfile의 색상 변화를 기반으로 동심원 영역을 분할합니다.
회전 불변성을 유지하는 polar 프로파일을 이용해 변곡점(gradient/ΔE)을 찾고,
필요 시 expected_zones 힌트를 활용해 균등 분할 fallback을 수행합니다.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.core.radial_profiler import RadialProfile

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """색상 영역 데이터"""

    name: str
    r_start: float
    r_end: float
    mean_L: float
    mean_a: float
    mean_b: float
    std_L: float
    std_a: float
    std_b: float
    zone_type: str  # 'pure' or 'mix'


@dataclass
class SegmenterConfig:
    """ZoneSegmenter 설정"""

    detection_method: str = "hybrid"  # gradient, delta_e, hybrid
    min_zone_width: float = 0.03      # 최소 3% 폭
    smoothing_window: int = 11        # Savitzky-Golay 윈도우
    polyorder: int = 3                # Savitzky-Golay 차수
    min_gradient: float = 0.25        # 그래디언트 피크 하한
    min_delta_e: float = 2.0          # ΔE 피크 하한
    expected_zones: Optional[int] = None  # 잉크 수 힌트(1~3)
    transition_buffer_px: int = 0     # 혼합 구간 버퍼(미사용 시 0)


class ZoneSegmentationError(Exception):
    """Zone 분할 실패 예외"""


class ZoneSegmenter:
    """동심원 색상 프로파일을 zone 단위로 분할"""

    def __init__(self, config: SegmenterConfig = SegmenterConfig()):
        self.config = config

    def segment(self, profile: RadialProfile, expected_zones: Optional[int] = None) -> List[Zone]:
        if profile is None:
            raise ValueError("Profile cannot be None")

        hint_zones = expected_zones or self.config.expected_zones

        # 1) 프로파일 평활화
        smooth_profile = self._smooth_profile(profile)

        # 2) 변곡점 검출 (그래디언트 + ΔE)
        grad_pts = self._detect_by_gradient(smooth_profile)
        de_pts = self._detect_by_delta_e(smooth_profile)

        boundaries = sorted(list(set(grad_pts + de_pts)), reverse=True)
        boundaries = self._merge_close_boundaries(boundaries, self.config.min_zone_width)

        # 3) 힌트 기반 개수 보정 또는 fallback
        if hint_zones and hint_zones > 0:
            desired = hint_zones - 1
            if len(boundaries) != desired:
                logger.info(f"Boundary count mismatch (found {len(boundaries)}, expected {desired}); using uniform split.")
                boundaries = self._uniform_boundaries(hint_zones)
        else:
            if not boundaries:
                logger.warning("No boundaries detected and no hint provided. Using default 3-zone split.")
                boundaries = self._uniform_boundaries(3)

        # 4) r 범위 포함 및 최소 폭 병합
        r_max = float(profile.r_normalized[-1])
        r_min = float(profile.r_normalized[0])
        boundaries = [r_max] + boundaries + [r_min]
        boundaries = self._merge_close_boundaries(boundaries, self.config.min_zone_width)

        # 5) Zone 생성
        zones: List[Zone] = []
        labels = self._generate_zone_labels(len(boundaries) - 1)

        for i in range(len(boundaries) - 1):
            r_start = boundaries[i]
            r_end = boundaries[i + 1]
            mask = (profile.r_normalized >= r_end) & (profile.r_normalized < r_start)
            if np.sum(mask) == 0:
                logger.warning(f"No data points in zone {labels[i]} ({r_start:.3f}-{r_end:.3f})")
                continue

            mean_L = float(np.mean(profile.L[mask]))
            mean_a = float(np.mean(profile.a[mask]))
            mean_b = float(np.mean(profile.b[mask]))
            std_L = float(np.std(profile.L[mask]))
            std_a = float(np.std(profile.a[mask]))
            std_b = float(np.std(profile.b[mask]))

            zones.append(
                Zone(
                    name=labels[i],
                    r_start=r_start,
                    r_end=r_end,
                    mean_L=mean_L,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    std_L=std_L,
                    std_a=std_a,
                    std_b=std_b,
                    zone_type="pure" if "-" not in labels[i] else "mix",
                )
            )

        if not zones:
            raise ZoneSegmentationError("No zones created after segmentation")

        logger.info(f"Segmented into {len(zones)} zones: {[z.name for z in zones]}")
        return zones

    def _smooth_profile(self, profile: RadialProfile) -> RadialProfile:
        window = min(self.config.smoothing_window, len(profile.a) - (len(profile.a) + 1) % 2)
        if window >= self.config.polyorder + 2:
            a = savgol_filter(profile.a, window_length=window, polyorder=self.config.polyorder)
            L = savgol_filter(profile.L, window_length=window, polyorder=self.config.polyorder)
            b = savgol_filter(profile.b, window_length=window, polyorder=self.config.polyorder)
        else:
            a, L, b = profile.a, profile.L, profile.b

        return RadialProfile(
            r_normalized=profile.r_normalized,
            L=L,
            a=a,
            b=b,
            std_L=profile.std_L,
            std_a=profile.std_a,
            std_b=profile.std_b,
            pixel_count=profile.pixel_count,
        )

    def _detect_by_gradient(self, profile: RadialProfile) -> List[float]:
        grad = np.gradient(profile.a)
        if grad.size == 0:
            return []
        # 평활화된 그래디언트로 적응형 임계값 계산
        window = min(self.config.smoothing_window, len(grad) - (len(grad) + 1) % 2)
        if window >= self.config.polyorder + 2:
            grad = savgol_filter(grad, window_length=window, polyorder=self.config.polyorder)
        abs_grad = np.abs(grad)
        adaptive = max(self.config.min_gradient, np.percentile(abs_grad, 75))
        distance = max(1, int(self.config.min_zone_width * len(profile.r_normalized)))
        peaks, _ = find_peaks(abs_grad, height=adaptive, distance=distance)
        return profile.r_normalized[peaks].tolist() if len(peaks) > 0 else []

    def _detect_by_delta_e(self, profile: RadialProfile) -> List[float]:
        from src.utils.color_delta import delta_e_cie2000

        r = profile.r_normalized
        if len(r) < 2:
            return []
        delta_e_profile = []
        for i in range(len(r) - 1):
            lab1 = (profile.L[i], profile.a[i], profile.b[i])
            lab2 = (profile.L[i + 1], profile.a[i + 1], profile.b[i + 1])
            delta_e_profile.append(delta_e_cie2000(lab1, lab2))

        delta_e_profile = np.array(delta_e_profile)
        if delta_e_profile.size == 0:
            return []

        adaptive = max(self.config.min_delta_e, np.percentile(delta_e_profile, 75))
        distance = max(1, int(self.config.min_zone_width * len(r)))
        peaks, _ = find_peaks(delta_e_profile, height=adaptive, distance=distance)
        return r[peaks].tolist() if len(peaks) > 0 else []

    def _merge_close_boundaries(self, boundaries: List[float], min_width: float) -> List[float]:
        if len(boundaries) <= 1:
            return boundaries
        merged = sorted(boundaries, reverse=True)
        i = 0
        while i < len(merged) - 1:
            width = merged[i] - merged[i + 1]
            if width < min_width and len(merged) > 2:
                merged.pop(i + 1)
                continue
            i += 1
        return merged

    def _uniform_boundaries(self, zones: int) -> List[float]:
        if zones <= 0:
            return []
        return np.linspace(1.0, 0.0, zones + 1)[1:-1].tolist()  # 내부 경계만 반환

    def _generate_zone_labels(self, n_zones: int) -> List[str]:
        base = ["A", "B", "C", "D", "E"]
        labels = []
        for i in range(n_zones):
            if i < len(base):
                labels.append(base[i])
            else:
                labels.append(f"Z{i+1}")
        return labels

    def evaluate_mix_zone(self, mix_zone: Zone, prev_pure: Zone, next_pure: Zone) -> dict:
        """
        믹스 존이 양 끝 순수 존 사이의 선형 보간 범위 안에 있는지 평가.
        """
        import numpy.linalg as LA

        p = np.array([prev_pure.mean_L, prev_pure.mean_a, prev_pure.mean_b], dtype=float)
        n = np.array([next_pure.mean_L, next_pure.mean_a, next_pure.mean_b], dtype=float)
        m = np.array([mix_zone.mean_L, mix_zone.mean_a, mix_zone.mean_b], dtype=float)

        pn = n - p
        if LA.norm(pn) == 0:
            return {"is_valid": False, "blend_ratio": 0.0, "distance_from_line": float("inf")}

        t = float(np.clip(np.dot(m - p, pn) / (LA.norm(pn) ** 2), 0.0, 1.0))
        projection = p + t * pn
        distance = float(LA.norm(m - projection))

        tol = float(
            np.mean(
                [
                    prev_pure.std_L,
                    prev_pure.std_a,
                    prev_pure.std_b,
                    next_pure.std_L,
                    next_pure.std_a,
                    next_pure.std_b,
                ]
            )
            + 3.0
        )

        return {"is_valid": distance <= tol, "blend_ratio": t, "distance_from_line": distance}
