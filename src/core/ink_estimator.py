"""
Ink Estimator Module (v2.0)

이미지에서 잉크의 개수와 주요 색상을 자동으로 추정하는 독립 모듈입니다.
GMM(Gaussian Mixture Model)과 BIC(Bayesian Information Criterion)를 사용하여
최적의 군집 수를 찾고, 유사한 군집을 병합하거나 혼합색(Mixing)을 보정합니다.

[전제 조건]
입력 이미지는 반드시 White Balance와 노출 보정이 완료된 상태여야 합니다.
그렇지 않으면 L_max(하이라이트)나 Chroma 임계값이 부정확할 수 있습니다.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class InkColor:
    weight: float  # 픽셀 비율 (0.0 ~ 1.0)
    lab: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    hex: str
    is_mix: bool = False  # 혼합색 여부 (보정된 경우)


class InkEstimator:
    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed: int = random_seed

    def lab_to_rgb_hex(self, lab: np.ndarray) -> Tuple[Tuple[int, int, int], str]:
        """Convert CIELAB (L:0-100, a,b:-128-127) to sRGB and HEX."""
        L, a, b = lab.astype(np.float32)

        # DEBUG: Print conversion steps
        print(f"[LAB→RGB] Input LAB: L={L:.2f}, a={a:.2f}, b={b:.2f}")

        # OpenCV Lab uses L in [0,255] scale; a,b shifted by +128.
        # Use uint8 to get 0~255 output range (not normalized 0~1)
        lab_cv = np.array([L * 255.0 / 100.0, a + 128.0, b + 128.0], dtype=np.uint8)
        print(f"[LAB→RGB] OpenCV LAB: L={lab_cv[0]}, a={lab_cv[1]}, b={lab_cv[2]}")

        lab_cv = lab_cv.reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab_cv, cv2.COLOR_Lab2BGR)[0, 0]
        print(f"[LAB→RGB] BGR: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")

        rgb = (int(np.clip(bgr[2], 0, 255)), int(np.clip(bgr[1], 0, 255)), int(np.clip(bgr[0], 0, 255)))
        hexv = "#{:02X}{:02X}{:02X}".format(*rgb)

        print(f"[LAB→RGB] Final RGB: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}, HEX={hexv}")

        return rgb, hexv

    def delta_e76(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        return float(np.linalg.norm(lab1 - lab2))

    def trimmed_mean(self, arr: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
        """Robust mean calculation removing top/bottom outliers per channel."""
        if len(arr) < 10:
            return np.asarray(np.mean(arr, axis=0), dtype=np.float32)
        lo = int(len(arr) * trim_ratio)
        hi = int(len(arr) * (1 - trim_ratio))
        out = []
        for c in range(arr.shape[1]):
            s = np.sort(arr[:, c])
            s = s[lo:hi] if hi > lo else s
            out.append(np.mean(s))
        return np.array(out, dtype=np.float32)

    def sample_ink_pixels(
        self,
        bgr: np.ndarray,
        max_samples: int = 50000,
        chroma_thresh: float = 6.0,
        L_max: float = 98.0,
        L_dark_thresh: float = 45.0,
        downscale_max: int = 1200,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        이미지에서 잉크로 추정되는 픽셀만 샘플링합니다.
        WARNING: 입력 이미지는 WB가 보정된 상태여야 합니다.

        Returns:
            (samples, sampling_info): LAB 샘플 배열과 샘플링 메타 정보
        """
        h, w = bgr.shape[:2]

        # 1. Downscaling (속도 최적화)
        scale = 1.0
        if max(h, w) > downscale_max:
            scale = downscale_max / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 2. Color Conversion
        lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        # Convert OpenCV Lab -> CIELAB (Standard: L 0-100, a,b -128-127)
        L = lab_cv[..., 0] * (100.0 / 255.0)
        a = lab_cv[..., 1] - 128.0
        b = lab_cv[..., 2] - 128.0

        chroma = np.sqrt(a * a + b * b)

        # 3. Filtering
        # - 유채색 잉크: Chroma >= thresh
        # - 무채색(Black) 잉크: L <= dark_thresh (Chroma 낮아도 허용)
        # - 하이라이트 제거: L <= L_max
        is_colored = chroma >= chroma_thresh
        is_dark = L <= L_dark_thresh
        is_not_highlight = L <= L_max

        mask = (is_colored | is_dark) & is_not_highlight

        ys, xs = np.where(mask)
        n_pixels = len(xs)

        # 샘플링 정보 수집
        sampling_info = {
            "chroma_threshold": float(chroma_thresh),
            "L_max": float(L_max),
            "L_dark_threshold": float(L_dark_thresh),
            "downscale_factor": float(scale),
            "original_size": [int(h), int(w)],
            "processed_size": [int(bgr.shape[0]), int(bgr.shape[1])],
            "candidate_pixels": int(n_pixels),
            "highlight_removed": True,
            "chroma_filter_applied": True,
        }

        if n_pixels == 0:
            sampling_info["sampled_pixels"] = 0
            return np.empty((0, 3), dtype=np.float32), sampling_info

        # 4. Sampling Strategy
        # 고해상도 이미지에서도 대표성을 잃지 않도록 최소 샘플 수 보장
        # 전체의 5% 또는 max_samples 중 작은 값 선택 (단, 최소 5000개는 확보 노력)
        target_samples = min(n_pixels, max(5000, min(max_samples, int(n_pixels * 0.05))))

        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(n_pixels, size=target_samples, replace=False)

        samp = np.stack([L[ys[idx], xs[idx]], a[ys[idx], xs[idx]], b[ys[idx], xs[idx]]], axis=1)

        sampling_info["sampled_pixels"] = int(target_samples)
        sampling_info["sampling_ratio"] = float(target_samples / n_pixels) if n_pixels > 0 else 0.0

        return samp.astype(np.float32), sampling_info

    def select_k_clusters(self, samples: np.ndarray, k_min: int = 1, k_max: int = 3) -> Tuple[Any, float]:
        """
        GMM + BIC로 최적 k 선택. 실패 시 KMeans Fallback.

        Returns:
            Tuple[GMM or FakeGMM, BIC score]
        """
        best_gmm = None
        best_bic = np.inf

        for k in range(k_min, k_max + 1):
            try:
                # GMM (Full Covariance for elliptical clusters)
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=self.random_seed,
                    reg_covar=1e-4,  # 안정성 향상
                    n_init=3,  # Local minima 방지
                )
                gmm.fit(samples)
                bic = gmm.bic(samples)

                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception as e:
                logger.warning(f"GMM fit failed for k={k}: {e}. Falling back to KMeans.")
                # Fallback: KMeans does not support BIC directly, so we treat it as high cost
                # or just use it as a candidate if GMM fails entirely.
                pass

        # GMM이 모두 실패했거나 결과가 없으면 KMeans Fallback (k=k_max)
        if best_gmm is None:
            logger.warning("All GMM fits failed. Using KMeans fallback.")
            kmeans = KMeans(n_clusters=k_max, random_state=self.random_seed, n_init=10)
            kmeans.fit(samples)

            # Make a fake GMM-like object for consistency
            class FakeGMM:
                def __init__(self, km: KMeans) -> None:
                    self.means_: np.ndarray = km.cluster_centers_
                    self.weights_: np.ndarray = np.array([np.sum(km.labels_ == i) for i in range(km.n_clusters)]) / len(
                        km.labels_
                    )
                    self.predict = km.predict

            return FakeGMM(kmeans), 0.0

        return best_gmm, best_bic

    def _check_exposure_warnings(self, samples: np.ndarray) -> None:
        """Check for exposure issues and log warnings."""
        if samples.shape[0] <= 0:
            return

        mean_L_all_pixels = np.mean(samples[:, 0])
        if mean_L_all_pixels < 25.0:
            logger.warning(
                f"Low image mean_L ({mean_L_all_pixels:.1f}). Input image might be underexposed or very dark. "
                "Ink estimation reliability may be reduced."
            )
        elif mean_L_all_pixels > 90.0:
            logger.warning(
                f"High image mean_L ({mean_L_all_pixels:.1f}). Input image might be overexposed or too bright. "
                "Ink estimation reliability may be reduced."
            )

    def _robustify_centers(self, centers: np.ndarray, samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Refine cluster centers using trimmed mean to reduce outlier impact.

        Args:
            centers: Cluster center coordinates in LAB space
            samples: All sample points
            labels: Cluster assignments for each sample

        Returns:
            Refined cluster centers
        """
        refined_centers = centers.copy()
        for k in range(len(centers)):
            cluster_samples = samples[labels == k]
            if len(cluster_samples) > 50:
                refined_centers[k] = self.trimmed_mean(cluster_samples, 0.1)
        return refined_centers

    def _merge_close_clusters(
        self, centers: np.ndarray, weights: np.ndarray, merge_de_thresh: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge clusters that are too close in color space.

        Args:
            centers: Cluster center coordinates
            weights: Cluster weights (pixel ratios)
            merge_de_thresh: Delta E threshold for merging

        Returns:
            Tuple of (merged_centers, merged_weights)
        """
        active = list(range(len(centers)))

        while len(active) > 1:
            # Find closest pair
            min_distance = 1e9
            pair = None

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ci, cj = centers[active[i]], centers[active[j]]
                    distance = self.delta_e76(ci, cj)
                    if distance < min_distance:
                        min_distance = distance
                        pair = (active[i], active[j])

            # Merge if close enough
            if pair and min_distance < merge_de_thresh:
                i, j = pair
                wi, wj = weights[i], weights[j]
                w_sum = wi + wj

                if w_sum > 0:
                    centers[i] = (centers[i] * wi + centers[j] * wj) / w_sum
                    weights[i] = w_sum

                active.remove(j)
            else:
                break  # No more merges possible

        merged_centers = centers[active]
        merged_weights = weights[active]

        if merged_weights.sum() > 0:
            merged_weights /= merged_weights.sum()

        return merged_centers, merged_weights

    def _format_ink_results(self, centers: np.ndarray, weights: np.ndarray) -> List[InkColor]:
        """
        Format cluster centers and weights as InkColor objects.

        Args:
            centers: Cluster centers in LAB space
            weights: Cluster weights (pixel ratios)

        Returns:
            List of InkColor objects sorted by L value (darkness)
        """
        # Sort by darkness (L value)
        order = np.argsort(centers[:, 0])
        sorted_centers = centers[order]
        sorted_weights = weights[order]

        inks = []
        for w, c in zip(sorted_weights, sorted_centers):
            rgb, hex_val = self.lab_to_rgb_hex(c)
            inks.append(
                InkColor(
                    weight=float(w),
                    lab=(float(c[0]), float(c[1]), float(c[2])),
                    rgb=rgb,
                    hex=hex_val,
                )
            )
        return inks

    def correct_ink_count_by_mixing(
        self, centers: np.ndarray, weights: np.ndarray, linearity_thresh: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        [Step 4] Human Intuition Logic:
        3개 군집일 때, 중간 톤이 두 극단(Dark, Bright)의 혼합인지 판단하여 2개로 병합.

        개선: 상대 기준 + 다중 조건으로 False merge 방지
        - 절대 거리 → 상대 거리 정규화 (ratio = dist / len_db)
        - 다중 조건 체크 (투영 위치, mid 비중, 상대 거리)
        """
        if len(centers) != 3:
            return centers, weights, False

        # L값 기준으로 정렬 (Dark -> Mid -> Bright)
        order = np.argsort(centers[:, 0])
        sorted_centers = centers[order]
        sorted_weights = weights[order]

        c_dark = sorted_centers[0]
        c_mid = sorted_centers[1]
        c_bright = sorted_centers[2]
        w_dark, w_mid, w_bright = sorted_weights

        # Vector: Dark -> Bright
        vec_db = c_bright - c_dark
        len_db = np.linalg.norm(vec_db)

        if len_db < 1e-6:
            return centers, weights, False

        # Unit vector
        u_db = vec_db / len_db

        # Vector: Dark -> Mid
        vec_dm = c_mid - c_dark

        # Projection of Mid onto DB line
        projection_len = np.dot(vec_dm, u_db)
        projection_ratio = projection_len / len_db  # 0~1 사이면 mid가 양 극단 "사이"

        # Calculate Perpendicular Distance (Linearity Error)
        closest_point = c_dark + u_db * projection_len
        distance = np.linalg.norm(c_mid - closest_point)

        # 🔧 개선: 상대 거리 정규화 (스케일 독립적)
        relative_distance = distance / len_db if len_db > 0 else 999.0

        # 🔧 개선: 다중 조건으로 혼합 판정
        RELATIVE_DIST_THRESH = 0.15  # 상대 거리 임계값 (dist/len_db < 15%)
        MIN_MID_WEIGHT = 0.05  # mid가 최소 5% 비중은 있어야 (너무 작으면 잡음)
        MAX_MID_WEIGHT = 0.7  # mid가 70% 넘으면 주요 잉크일 가능성

        print(f"[MIXING_CHECK] 3 clusters detected - checking if middle is mixed")
        print(f"[MIXING_CHECK] Dark={c_dark}, Mid={c_mid}, Bright={c_bright}")
        print(f"[MIXING_CHECK] Weights: Dark={w_dark:.3f}, Mid={w_mid:.3f}, Bright={w_bright:.3f}")
        print(f"[MIXING_CHECK] Distance: abs={distance:.2f}, relative={relative_distance:.3f} (len_db={len_db:.2f})")
        print(f"[MIXING_CHECK] Projection: ratio={projection_ratio:.3f} (0~1 사이면 중간 위치)")

        # 조건 1: mid가 dark-bright 사이에 위치 (loose margin: -10% ~ 110%)
        cond1_between = -0.1 <= projection_ratio <= 1.1

        # 조건 2: 상대 거리가 임계값 아래
        cond2_close_to_line = relative_distance < RELATIVE_DIST_THRESH

        # 조건 3: mid 비중이 적절 (너무 작으면 잡음, 너무 크면 주요 잉크)
        cond3_mid_weight_ok = MIN_MID_WEIGHT < w_mid < MAX_MID_WEIGHT

        print(f"[MIXING_CHECK] Condition checks:")
        print(f"  - Mid between Dark-Bright: {cond1_between} (ratio={projection_ratio:.3f})")
        print(f"  - Close to line: {cond2_close_to_line} (rel_dist={relative_distance:.3f} < {RELATIVE_DIST_THRESH})")
        print(f"  - Mid weight OK: {cond3_mid_weight_ok} ({MIN_MID_WEIGHT} < {w_mid:.3f} < {MAX_MID_WEIGHT})")

        # 🔧 모든 조건을 만족해야 혼합으로 판정
        if cond1_between and cond2_close_to_line and cond3_mid_weight_ok:
            print(f"[MIXING_CHECK] OK Mid-tone IS mixed (all conditions met). Merging to 2 inks.")

            # Distribute weights based on proximity
            # ratio 0.0 = close to dark, 1.0 = close to bright
            ratio = np.clip(projection_ratio, 0.0, 1.0)

            new_weights = np.zeros(2, dtype=np.float32)
            new_centers = np.array([c_dark, c_bright], dtype=np.float32)

            # Dark weight = Original Dark + Mid * (1 - ratio)
            new_weights[0] = w_dark + w_mid * (1.0 - ratio)
            # Bright weight = Original Bright + Mid * ratio
            new_weights[1] = w_bright + w_mid * ratio

            return new_centers, new_weights, True

        print(f"[MIXING_CHECK] X Mid-tone NOT mixed (conditions failed). Keeping 3 inks.")
        if not cond1_between:
            print(f"[MIXING_CHECK]   → Mid not between Dark-Bright")
        if not cond2_close_to_line:
            print(f"[MIXING_CHECK]   → Mid too far from line (rel_dist={relative_distance:.3f})")
        if not cond3_mid_weight_ok:
            print(f"[MIXING_CHECK]   → Mid weight out of range (weight={w_mid:.3f})")

        return centers, weights, False

    def estimate_from_array(
        self,
        bgr: np.ndarray,
        k_max: int = 3,
        chroma_thresh: float = 6.0,
        L_max: float = 98.0,
        merge_de_thresh: float = 5.0,
        linearity_thresh: float = 3.0,
    ) -> Dict[str, Any]:
        """
        numpy array 이미지에서 잉크를 추정하는 메인 함수.

        Args:
            bgr: OpenCV BGR 이미지 (numpy array)
            k_max: 최대 군집 수
            chroma_thresh: 유채색 잉크 판단 기준
            L_max: 하이라이트 제거 기준
            merge_de_thresh: 유사 색상 병합 기준
            linearity_thresh: 중간 톤 혼합 판단 기준

        Returns:
            잉크 분석 결과 딕셔너리
        """
        # 1. Sample Pixels
        samples, sampling_info = self.sample_ink_pixels(bgr, chroma_thresh=chroma_thresh, L_max=L_max)

        # 2. Pre-check: Exposure warnings
        self._check_exposure_warnings(samples)

        # 3. Check sample sufficiency
        if len(samples) < 500:
            return {
                "ink_count": 0,
                "inks": [],
                "warning": "Insufficient ink pixels. Check image quality or thresholds.",
                "meta": {
                    "sampling_config": sampling_info,
                    "bic": 0.0,
                    "sample_count": len(samples),
                    "correction_applied": False,
                },
            }

        # 4. Select optimal K using GMM + BIC
        gmm, bic = self.select_k_clusters(samples, 1, k_max)
        labels = gmm.predict(samples)
        weights = gmm.weights_.astype(np.float32)
        centers = gmm.means_.astype(np.float32)

        # 5. Robustify Centers using trimmed mean
        centers = self._robustify_centers(centers, samples, labels)

        # 6. Merge close clusters
        centers, weights = self._merge_close_clusters(centers, weights, merge_de_thresh)

        # 7. Correct for mixing if 3 clusters detected
        is_mixed_corrected = False
        if len(centers) == 3:
            centers, weights, is_mixed_corrected = self.correct_ink_count_by_mixing(centers, weights, linearity_thresh)

        # 8. Format results
        inks = self._format_ink_results(centers, weights)

        return {
            "ink_count": len(inks),
            "inks": [asdict(ink) for ink in inks],
            "meta": {
                "bic": float(bic),
                "sample_count": int(len(samples)),
                "correction_applied": is_mixed_corrected,
                "sampling_config": sampling_info,  # 🔧 추가: 샘플링 ROI 정보
            },
        }

    def estimate(
        self,
        image_path: str,
        k_max: int = 3,
        chroma_thresh: float = 6.0,
        L_max: float = 98.0,
        merge_de_thresh: float = 5.0,
        linearity_thresh: float = 3.0,
    ) -> Dict[str, Any]:
        """
        이미지 파일에서 잉크를 추정하는 메인 함수.

        Args:
            image_path: 이미지 파일 경로
            k_max: 최대 군집 수
            chroma_thresh: 유채색 잉크 판단 기준
            L_max: 하이라이트 제거 기준
            merge_de_thresh: 유사 색상 병합 기준
            linearity_thresh: 중간 톤 혼합 판단 기준

        Returns:
            잉크 분석 결과 딕셔너리
        """
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return self.estimate_from_array(
            bgr,
            k_max=k_max,
            chroma_thresh=chroma_thresh,
            L_max=L_max,
            merge_de_thresh=merge_de_thresh,
            linearity_thresh=linearity_thresh,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        estimator = InkEstimator()
        try:
            result = estimator.estimate(sys.argv[1])
            import json

            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python -m src.core.ink_estimator <image_path>")
