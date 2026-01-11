"""
Inspection Pipeline Module

5개 핵심 모듈을 연결하는 엔드투엔드 검사 파이프라인.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.color_evaluator import ColorEvaluationError, ColorEvaluator, InspectionResult
from src.core.image_loader import ImageConfig, ImageLoader
from src.core.lens_detector import DetectorConfig, LensDetectionError, LensDetector
from src.core.quality_metrics import compute_quality_metrics
from src.core.radial_profiler import ProfilerConfig, RadialProfiler
from src.core.zone_segmenter import SegmenterConfig, ZoneSegmentationError, ZoneSegmenter

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """파이프라인 실행 중 발생하는 예외"""

    pass


class InspectionPipeline:
    """
    엔드투엔드 렌즈 색상 검사 파이프라인.

    ImageLoader → LensDetector → RadialProfiler → ZoneSegmenter → ColorEvaluator
    순서로 5개 모듈을 연결하여 최종 판정 결과 생성.
    """

    def __init__(
        self,
        sku_config: Dict[str, Any],
        image_config: Optional[ImageConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        segmenter_config: Optional[SegmenterConfig] = None,
        save_intermediates: bool = False,
    ):
        """
        파이프라인 초기화.

        Args:
            sku_config: SKU별 기준값 설정
            image_config: ImageLoader 설정 (기본값 사용 시 None)
            detector_config: LensDetector 설정 (기본값 사용 시 None)
            profiler_config: RadialProfiler 설정 (기본값 사용 시 None)
            segmenter_config: ZoneSegmenter 설정 (기본값 사용 시 None)
            save_intermediates: 중간 결과 저장 여부
        """
        self.sku_config = sku_config
        self.save_intermediates = save_intermediates

        # 각 모듈 초기화
        self.image_loader = ImageLoader(image_config or ImageConfig())
        self.lens_detector = LensDetector(detector_config or DetectorConfig())

        # profiler 설정: SKU params.optical_clear_ratio -> r_start_ratio에 반영
        profiler_cfg = profiler_config or ProfilerConfig()
        params = sku_config.get("params", {})
        optical_clear = params.get("optical_clear_ratio")
        center_exclude = params.get("center_exclude_ratio")
        if isinstance(optical_clear, (int, float)) and 0 <= optical_clear < 1:
            profiler_cfg.r_start_ratio = float(optical_clear)
        if isinstance(center_exclude, (int, float)) and 0 <= center_exclude < 1:
            profiler_cfg.r_start_ratio = max(profiler_cfg.r_start_ratio, float(center_exclude))
        if profiler_cfg.r_start_ratio > 0:
            logger.info(
                "Applying r_start_ratio=%.3f (optical_clear_ratio=%s, center_exclude_ratio=%s)",
                profiler_cfg.r_start_ratio,
                f"{optical_clear:.3f}" if isinstance(optical_clear, (int, float)) else "None",
                f"{center_exclude:.3f}" if isinstance(center_exclude, (int, float)) else "None",
            )
        self.radial_profiler = RadialProfiler(profiler_cfg)
        seg_cfg = segmenter_config or SegmenterConfig()
        # Apply override params (used by recompute)
        if isinstance(params.get("detection_method"), str):
            seg_cfg.detection_method = params["detection_method"]
        if isinstance(params.get("smoothing_window"), (int, float)):
            seg_cfg.smoothing_window = int(params["smoothing_window"])
        if isinstance(params.get("min_gradient"), (int, float)):
            seg_cfg.min_gradient = float(params["min_gradient"])
        if isinstance(params.get("min_delta_e"), (int, float)):
            seg_cfg.min_delta_e = float(params["min_delta_e"])
        if isinstance(params.get("uniform_split_priority"), bool):
            seg_cfg.uniform_split_priority = params["uniform_split_priority"]
        if isinstance(params.get("expected_zones"), (int, float)):
            seg_cfg.expected_zones = int(params["expected_zones"])
        # Note: expected_zones는 process() 메서드에서 params.expected_zones로 읽어서 segment()에 전달
        self.zone_segmenter = ZoneSegmenter(seg_cfg)
        self.color_evaluator = ColorEvaluator(sku_config)

        logger.info("InspectionPipeline initialized")

    def process(
        self,
        image_path: str,
        sku: str,
        save_dir: Optional[Path] = None,
        run_1d_judgment: bool = True,
        include_dot_stats: bool = True,
    ) -> InspectionResult:
        """
        단일 이미지 처리.

        Args:
            image_path: 입력 이미지 경로
            sku: SKU 코드
            save_dir: 중간 결과 저장 디렉토리 (옵션)

        Returns:
            InspectionResult: 검사 결과

        Raises:
            PipelineError: 파이프라인 실행 중 오류 발생 시
        """
        start_time = datetime.now()
        image_path = Path(image_path)

        logger.info(f"Processing image: {image_path}, SKU: {sku}")

        # PHASE7 Priority 5: 진단 정보 수집
        diagnostics = []
        warnings = []
        suggestions = []

        try:
            # 1. 이미지 로드 및 전처리 (retry 로직 포함)
            logger.debug("Step 1: Loading and preprocessing image")
            max_retries = 3
            image = None

            for attempt in range(max_retries):
                try:
                    image = self.image_loader.load_from_file(image_path)

                    # None 체크
                    if image is None:
                        if attempt < max_retries - 1:
                            logger.warning(f"Image load attempt {attempt+1}/{max_retries} returned None, retrying...")
                            continue
                        else:
                            raise ValueError("Image loader returned None")

                    processed_image = self.image_loader.preprocess(image)

                    # Preprocess 결과도 체크
                    if processed_image is None:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Image preprocess attempt {attempt+1}/{max_retries} returned None, retrying..."
                            )
                            continue
                        else:
                            raise ValueError("Image preprocess returned None")

                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Image load attempt {attempt+1}/{max_retries} failed: {e}, retrying...")
                        continue
                    else:
                        logger.error(f"Failed to load image after {max_retries} attempts: {image_path}")
                        # 파일 존재 여부 확인
                        if not image_path.exists():
                            raise PipelineError(
                                f"Image file not found\n"
                                f"  File: {image_path}\n"
                                f"  Suggestion: Check if the file path is correct"
                            )
                        else:
                            raise PipelineError(
                                f"Image load failed: {e}\n"
                                f"  File: {image_path}\n"
                                f"  Suggestion: Check if file is readable and is a valid image format (JPG, PNG, etc.)"
                            )

            # 2. 렌즈 검출 (상세 에러 메시지)
            logger.debug("Step 2: Detecting lens")
            lens_detection = self.lens_detector.detect(processed_image)

            if lens_detection is None:
                img_h, img_w = processed_image.shape[:2]
                diagnostics.append("✗ Lens detection failed")
                suggestions.append("→ Check if image contains a clear circular lens")
                suggestions.append("→ Try adjusting detector parameters (min_radius, max_radius)")
                raise PipelineError(
                    f"Lens detection failed\n"
                    f"  File: {image_path}\n"
                    f"  Image size: {img_w}x{img_h}\n"
                    f"  Suggestion: Check if image contains a clear circular lens. "
                    f"Try adjusting detector parameters (min_radius, max_radius) in config."
                )

            # 렌즈 검출 성공
            diagnostics.append(
                f"✓ Lens detected: center=({lens_detection.center_x:.1f}, {lens_detection.center_y:.1f}), "
                f"radius={lens_detection.radius:.1f}, confidence={lens_detection.confidence:.2f}"
            )

            if lens_detection.confidence < 0.5:
                logger.warning(f"Low lens detection confidence: {lens_detection.confidence:.2f}")
                warnings.append(f"⚠ Low lens detection confidence: {lens_detection.confidence:.2f}")
                suggestions.append("→ Verify image quality or adjust detector parameters")

            # 3. 극좌표 변환 및 프로파일 추출
            params = self.sku_config.get("params", {})
            num_samples = params.get("num_samples")
            if isinstance(num_samples, (int, float)) and num_samples > 0:
                self.radial_profiler.config.theta_samples = int(num_samples)
            num_points = params.get("num_points")
            if isinstance(num_points, (int, float)) and num_points > 0 and lens_detection.radius > 0:
                r_step = max(1, int(lens_detection.radius / float(num_points)))
                self.radial_profiler.config.r_step_pixels = r_step
            sample_percentile = params.get("sample_percentile")
            if isinstance(sample_percentile, (int, float)) and 0 <= sample_percentile <= 100:
                self.radial_profiler.config.sample_percentile = float(sample_percentile)

            quality_metrics = compute_quality_metrics(
                processed_image,
                lens_detection,
                include_dot_stats=include_dot_stats,
            )
            logger.debug("Step 3: Extracting radial profile")
            radial_profile = self.radial_profiler.extract_profile(processed_image, lens_detection)

            # 4. Zone 분할 (AI 피드백 반영: Ring과 동일한 좌표계 사용)
            logger.debug("Step 4: Segmenting zones")

            # SKU 설정에서 기대 Zone 개수 힌트 추출
            expected_zones = self.sku_config.get("params", {}).get("expected_zones")
            if expected_zones:
                logger.debug(f"Using expected_zones hint: {expected_zones}")

            if not run_1d_judgment:
                suggestions.append("Run 2D analysis for final judgment.")
                inspection_result = InspectionResult(
                    sku=sku,
                    timestamp=datetime.now(),
                    judgment="RETAKE",
                    overall_delta_e=0.0,
                    zone_results=[],
                    ng_reasons=[],
                    confidence=0.0,
                    retake_reasons=[
                        {
                            "code": "1d_judgment_skipped",
                            "reason": "1D judgment was skipped by configuration.",
                            "actions": ["Run 2D analysis for final judgment."],
                            "lever": "use_2d_analysis",
                        }
                    ],
                )

                inspection_result.lens_detection = lens_detection
                inspection_result.zones = []
                inspection_result.image = image
                inspection_result.radial_profile = radial_profile
                inspection_result.metrics = quality_metrics
                inspection_result.diagnostics = diagnostics if diagnostics else None
                inspection_result.warnings = warnings if warnings else None
                inspection_result.suggestions = suggestions if suggestions else None

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(
                    "Processing complete (1D judgment skipped): "
                    f"time={processing_time:.1f}ms, "
                    f"diagnostics={len(diagnostics)}, warnings={len(warnings)}, suggestions={len(suggestions)}"
                )

                if self.save_intermediates and save_dir:
                    self._save_intermediates(
                        save_dir,
                        image_path.stem,
                        {
                            "processed_image": processed_image,
                            "lens_detection": lens_detection,
                            "radial_profile": radial_profile,
                            "zones": [],
                            "inspection_result": inspection_result,
                            "processing_time_ms": processing_time,
                        },
                    )

                return inspection_result

            # 인쇄 영역 범위 추정 (SKU config 기반)
            params = self.sku_config.get("params", {})
            optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
            center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
            r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)  # center exclusion takes priority
            r_outer = 0.95  # 인쇄 영역 끝 (렌즈 외곽 약간 제외)

            # AI 피드백 반영: 반경 기준 명확히 출력
            logger.info(
                f"[ZONE COORD] Zone segmentation using PRINT AREA basis:\n"
                f"  - r_inner={r_inner:.3f} (print start, from optical_clear_ratio={optical_clear_ratio:.3f}, "
                f"center_exclude_ratio={center_exclude_ratio:.3f})\n"
                f"  - r_outer={r_outer:.3f} (print end)\n"
                f"  - lens_radius={lens_detection.radius:.1f}px\n"
                f"  - Normalization: r_norm = (r - {r_inner:.3f}) / ({r_outer:.3f} - {r_inner:.3f})"
            )

            zones = self.zone_segmenter.segment(
                radial_profile, expected_zones=expected_zones, r_inner=r_inner, r_outer=r_outer
            )

            # 진단 정보: Zone 분할 성공
            diagnostics.append(f"✓ Segmented into {len(zones)} zones: {[z.name for z in zones]}")

            # 경고: expected_zones와 불일치
            if expected_zones and len(zones) != expected_zones:
                warnings.append(f"⚠ Expected {expected_zones} zones but got {len(zones)}")
                suggestions.append("→ Adjust min_gradient or min_delta_e parameters")
                suggestions.append(f"→ Or update expected_zones to {len(zones)} if this is correct")

            # AI 피드백 반영: Zone별 실제 픽셀 반경 범위 출력
            logger.info(f"[ZONE RESULT] Created {len(zones)} zones:")
            for z in zones:
                r_start_px = z.r_start * lens_detection.radius
                r_end_px = z.r_end * lens_detection.radius

                # AI 검증 요청: Zone이 Ring과 어느 정도 겹치는지 확인
                ring_overlap = ""
                if z.r_start >= 0.67:
                    ring_overlap = "mainly Ring 2 (outer print)"
                elif z.r_start >= 0.33:
                    ring_overlap = "mainly Ring 1 (middle print)"
                else:
                    ring_overlap = "mainly Ring 0 (inner clear)"

                logger.info(
                    f"  Zone {z.name}: "
                    f"r_norm=[{z.r_end:.3f}, {z.r_start:.3f}), "
                    f"r_pixel=[{r_end_px:.1f}px, {r_start_px:.1f}px), "
                    f"pixels={z.pixel_count}, "
                    f"Lab=({z.mean_L:.1f}, {z.mean_a:.1f}, {z.mean_b:.1f}), "
                    f"{ring_overlap}"
                )

            # 5. 색상 평가 및 판정
            logger.debug("Step 5: Evaluating color quality")
            inspection_result = self.color_evaluator.evaluate(zones, sku, self.sku_config)

            # 시각화를 위한 데이터 추가
            inspection_result.lens_detection = lens_detection
            inspection_result.zones = zones
            inspection_result.image = image  # 원본 이미지 (전처리 전)
            inspection_result.radial_profile = radial_profile
            inspection_result.metrics = quality_metrics

            # PHASE7 Priority 5: 진단 정보, 경고, 제안 추가
            inspection_result.diagnostics = diagnostics if diagnostics else None
            inspection_result.warnings = warnings if warnings else None
            inspection_result.suggestions = suggestions if suggestions else None

            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms

            logger.info(
                f"Processing complete: {inspection_result.judgment}, "
                f"ΔE={inspection_result.overall_delta_e:.2f}, "
                f"time={processing_time:.1f}ms, "
                f"diagnostics={len(diagnostics)}, warnings={len(warnings)}, suggestions={len(suggestions)}"
            )

            # 중간 결과 저장 (옵션)
            if self.save_intermediates and save_dir:
                self._save_intermediates(
                    save_dir,
                    image_path.stem,
                    {
                        "processed_image": processed_image,
                        "lens_detection": lens_detection,
                        "radial_profile": radial_profile,
                        "zones": zones,
                        "inspection_result": inspection_result,
                        "processing_time_ms": processing_time,
                    },
                )

            return inspection_result

        except LensDetectionError as e:
            logger.error(f"Lens detection failed: {e}")
            raise PipelineError(
                f"Pipeline failed at lens detection\n"
                f"  Image: {image_path}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Check image quality, adjust detector config, or verify lens is visible"
            )

        except ZoneSegmentationError as e:
            logger.error(f"Zone segmentation failed: {e}")
            # Zone 분할 실패 시 복구 시도: expected_zones 힌트 사용
            if expected_zones is None:
                logger.info("Attempting recovery with default 3-zone segmentation")
                try:
                    # Recovery에도 r_inner, r_outer 적용
                    params = self.sku_config.get("params", {})
                    optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
                    center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
                    r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)
                    r_outer = 0.95
                    zones = self.zone_segmenter.segment(
                        radial_profile, expected_zones=3, r_inner=r_inner, r_outer=r_outer
                    )
                    logger.info(f"Recovery successful: segmented into {len(zones)} zones")
                    # 처리 계속 진행
                    inspection_result = self.color_evaluator.evaluate(zones, sku, self.sku_config)
                    inspection_result.lens_detection = lens_detection
                    inspection_result.zones = zones
                    inspection_result.image = image
                    inspection_result.radial_profile = radial_profile
                    inspection_result.metrics = quality_metrics
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    logger.info(
                        f"Processing complete (with recovery): {inspection_result.judgment}, "
                        f"time={processing_time:.1f}ms"
                    )
                    return inspection_result
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")

            raise PipelineError(
                f"Pipeline failed at zone segmentation\n"
                f"  Image: {image_path}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Add 'expected_zones' hint to SKU config, or check radial profile quality"
            )

        except ColorEvaluationError as e:
            logger.error(f"Color evaluation failed: {e}")
            raise PipelineError(
                f"Pipeline failed at color evaluation\n"
                f"  Image: {image_path}\n"
                f"  SKU: {sku}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Verify SKU config has baseline values for all detected zones"
            )

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
            raise PipelineError(
                f"Pipeline failed with unexpected error\n"
                f"  Image: {image_path}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Check logs for detailed traceback"
            )

    def process_batch(
        self,
        image_paths: List[str],
        sku: str,
        output_csv: Optional[Path] = None,
        continue_on_error: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[InspectionResult]:
        """
        배치 처리 (옵션으로 병렬 처리 지원).

        Args:
            image_paths: 입력 이미지 경로 리스트
            sku: SKU 코드
            output_csv: 결과 CSV 저장 경로 (옵션)
            continue_on_error: 오류 발생 시 계속 진행 여부
            parallel: 병렬 처리 사용 여부 (기본값: False)
            max_workers: 병렬 처리 시 최대 워커 수 (기본값: 4)

        Returns:
            List[InspectionResult]: 검사 결과 리스트
        """
        logger.info(f"Batch processing {len(image_paths)} images (parallel={parallel})")

        results = []
        errors = []

        if parallel and len(image_paths) > 1:
            # Parallel processing using ThreadPoolExecutor
            import gc
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {executor.submit(self.process, path, sku): path for path in image_paths}

                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_path)):
                    image_path = future_to_path[future]
                    logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

                    try:
                        result = future.result()
                        results.append(result)

                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        errors.append((image_path, str(e)))

                        if not continue_on_error:
                            raise PipelineError(f"Batch processing failed: {e}")

                    # Release memory periodically
                    if i % 10 == 0:
                        gc.collect()

        else:
            # Sequential processing (original behavior)
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

                try:
                    result = self.process(image_path, sku)
                    results.append(result)

                except PipelineError as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    errors.append((image_path, str(e)))

                    if not continue_on_error:
                        raise

        logger.info(f"Batch processing complete: {len(results)} succeeded, {len(errors)} failed")

        if errors:
            logger.error(f"Failed images ({len(errors)}):")
            for path, err in errors:
                logger.error(f"  - {path}: {err}")

        # CSV 저장 (옵션)
        if output_csv and results:
            self._save_results_csv(results, output_csv)

        return results

    def _save_intermediates(self, save_dir: Path, image_name: str, intermediates: Dict[str, Any]):
        """
        중간 결과 저장.

        Args:
            save_dir: 저장 디렉토리
            image_name: 이미지 이름
            intermediates: 중간 결과 딕셔너리
        """
        import cv2

        output_dir = Path(save_dir) / image_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 처리된 이미지 저장
        if "processed_image" in intermediates:
            cv2.imwrite(str(output_dir / "02_preprocessed.jpg"), intermediates["processed_image"])

        # 렌즈 검출 결과 저장
        if "lens_detection" in intermediates and "processed_image" in intermediates:
            img_with_circle = intermediates["processed_image"].copy()
            detection = intermediates["lens_detection"]
            cv2.circle(
                img_with_circle,
                (int(detection.center_x), int(detection.center_y)),
                int(detection.radius),
                (0, 255, 0),
                2,
            )
            cv2.circle(img_with_circle, (int(detection.center_x), int(detection.center_y)), 3, (0, 0, 255), -1)
            cv2.imwrite(str(output_dir / "03_lens_detection.jpg"), img_with_circle)

        # 메타데이터 저장
        metadata = {
            "lens_detection": (
                {
                    "center_x": float(intermediates["lens_detection"].center_x),
                    "center_y": float(intermediates["lens_detection"].center_y),
                    "radius": float(intermediates["lens_detection"].radius),
                    "confidence": float(intermediates["lens_detection"].confidence),
                    "method": intermediates["lens_detection"].method,
                }
                if "lens_detection" in intermediates
                else None
            ),
            "zones": [
                {
                    "name": z.name,
                    "r_start": float(z.r_start),
                    "r_end": float(z.r_end),
                    "mean_lab": [float(z.mean_L), float(z.mean_a), float(z.mean_b)],
                    "zone_type": z.zone_type,
                }
                for z in intermediates.get("zones", [])
            ],
            "processing_time_ms": intermediates.get("processing_time_ms"),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Intermediates saved to {output_dir}")

    def _save_results_csv(self, results: List[InspectionResult], output_path: Path):
        """
        결과를 CSV 파일로 저장.

        Args:
            results: 검사 결과 리스트
            output_path: 출력 CSV 경로
        """
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(["sku", "timestamp", "judgment", "overall_delta_e", "confidence", "ng_reasons"])

            # 데이터
            for result in results:
                writer.writerow(
                    [
                        result.sku,
                        result.timestamp.isoformat(),
                        result.judgment,
                        f"{result.overall_delta_e:.2f}",
                        f"{result.confidence:.2f}",
                        "; ".join(result.ng_reasons) if result.ng_reasons else "",
                    ]
                )

        logger.info(f"Results saved to {output_path}")
