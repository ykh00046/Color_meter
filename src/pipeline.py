"""
Inspection Pipeline Module

5개 핵심 모듈을 연결하는 엔드투엔드 검사 파이프라인.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from src.core.image_loader import ImageLoader, ImageConfig
from src.core.lens_detector import LensDetector, DetectorConfig, LensDetectionError
from src.core.radial_profiler import RadialProfiler, ProfilerConfig
from src.core.zone_segmenter import ZoneSegmenter, SegmenterConfig, ZoneSegmentationError
from src.core.color_evaluator import ColorEvaluator, InspectionResult, ColorEvaluationError

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
        save_intermediates: bool = False
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
        self.radial_profiler = RadialProfiler(profiler_config or ProfilerConfig())
        seg_cfg = segmenter_config or SegmenterConfig()
        # SKU 설정에 expected_zones 힌트가 있으면 주입
        if sku_config and hasattr(seg_cfg, "expected_zones"):
            if isinstance(sku_config, dict) and "expected_zones" in sku_config:
                seg_cfg.expected_zones = sku_config.get("expected_zones")
        self.zone_segmenter = ZoneSegmenter(seg_cfg)
        self.color_evaluator = ColorEvaluator(sku_config)

        logger.info("InspectionPipeline initialized")

    def process(
        self,
        image_path: str,
        sku: str,
        save_dir: Optional[Path] = None
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

        try:
            # 1. 이미지 로드 및 전처리
            logger.debug("Step 1: Loading and preprocessing image")
            image = self.image_loader.load_from_file(image_path)
            processed_image = self.image_loader.preprocess(image)

            # 2. 렌즈 검출
            logger.debug("Step 2: Detecting lens")
            lens_detection = self.lens_detector.detect(processed_image)

            if lens_detection is None:
                raise PipelineError("Lens detection failed")

            if lens_detection.confidence < 0.5:
                logger.warning(f"Low lens detection confidence: {lens_detection.confidence:.2f}")

            # 3. 극좌표 변환 및 프로파일 추출
            logger.debug("Step 3: Extracting radial profile")
            radial_profile = self.radial_profiler.extract_profile(
                processed_image,
                lens_detection
            )

            # 4. Zone 분할
            logger.debug("Step 4: Segmenting zones")
            
            # SKU 설정에서 기대 Zone 개수 힌트 추출
            expected_zones = self.sku_config.get('params', {}).get('expected_zones')
            if expected_zones:
                logger.debug(f"Using expected_zones hint: {expected_zones}")
            
            zones = self.zone_segmenter.segment(radial_profile, expected_zones=expected_zones)

            # 5. 색상 평가 및 판정
            logger.debug("Step 5: Evaluating color quality")
            inspection_result = self.color_evaluator.evaluate(
                zones,
                sku,
                self.sku_config
            )

            # 시각화를 위한 데이터 추가
            inspection_result.lens_detection = lens_detection
            inspection_result.zones = zones
            inspection_result.image = image  # 원본 이미지 (전처리 전)

            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms

            logger.info(
                f"Processing complete: {inspection_result.judgment}, "
                f"ΔE={inspection_result.overall_delta_e:.2f}, "
                f"time={processing_time:.1f}ms"
            )

            # 중간 결과 저장 (옵션)
            if self.save_intermediates and save_dir:
                self._save_intermediates(
                    save_dir,
                    image_path.stem,
                    {
                        'processed_image': processed_image,
                        'lens_detection': lens_detection,
                        'radial_profile': radial_profile,
                        'zones': zones,
                        'inspection_result': inspection_result,
                        'processing_time_ms': processing_time
                    }
                )

            return inspection_result

        except LensDetectionError as e:
            logger.error(f"Lens detection failed: {e}")
            raise PipelineError(f"Pipeline failed at lens detection: {e}")

        except ZoneSegmentationError as e:
            logger.error(f"Zone segmentation failed: {e}")
            raise PipelineError(f"Pipeline failed at zone segmentation: {e}")

        except ColorEvaluationError as e:
            logger.error(f"Color evaluation failed: {e}")
            raise PipelineError(f"Pipeline failed at color evaluation: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
            raise PipelineError(f"Pipeline failed: {e}")

    def process_batch(
        self,
        image_paths: List[str],
        sku: str,
        output_csv: Optional[Path] = None,
        continue_on_error: bool = True,
        parallel: bool = False,
        max_workers: int = 4
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
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import gc

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.process, path, sku): path
                    for path in image_paths
                }

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

        logger.info(
            f"Batch processing complete: {len(results)} succeeded, {len(errors)} failed"
        )

        # CSV 저장 (옵션)
        if output_csv and results:
            self._save_results_csv(results, output_csv)

        return results

    def _save_intermediates(
        self,
        save_dir: Path,
        image_name: str,
        intermediates: Dict[str, Any]
    ):
        """
        중간 결과 저장.

        Args:
            save_dir: 저장 디렉토리
            image_name: 이미지 이름
            intermediates: 중간 결과 딕셔너리
        """
        import cv2
        import numpy as np

        output_dir = Path(save_dir) / image_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 처리된 이미지 저장
        if 'processed_image' in intermediates:
            cv2.imwrite(
                str(output_dir / '02_preprocessed.jpg'),
                intermediates['processed_image']
            )

        # 렌즈 검출 결과 저장
        if 'lens_detection' in intermediates and 'processed_image' in intermediates:
            img_with_circle = intermediates['processed_image'].copy()
            detection = intermediates['lens_detection']
            cv2.circle(
                img_with_circle,
                (int(detection.center_x), int(detection.center_y)),
                int(detection.radius),
                (0, 255, 0),
                2
            )
            cv2.circle(
                img_with_circle,
                (int(detection.center_x), int(detection.center_y)),
                3,
                (0, 0, 255),
                -1
            )
            cv2.imwrite(
                str(output_dir / '03_lens_detection.jpg'),
                img_with_circle
            )

        # 메타데이터 저장
        metadata = {
            'lens_detection': {
                'center_x': float(intermediates['lens_detection'].center_x),
                'center_y': float(intermediates['lens_detection'].center_y),
                'radius': float(intermediates['lens_detection'].radius),
                'confidence': float(intermediates['lens_detection'].confidence),
                'method': intermediates['lens_detection'].method
            } if 'lens_detection' in intermediates else None,
            'zones': [
                {
                    'name': z.name,
                    'r_start': float(z.r_start),
                    'r_end': float(z.r_end),
                    'mean_lab': [float(z.mean_L), float(z.mean_a), float(z.mean_b)],
                    'zone_type': z.zone_type
                }
                for z in intermediates.get('zones', [])
            ],
            'processing_time_ms': intermediates.get('processing_time_ms')
        }

        with open(output_dir / 'metadata.json', 'w') as f:
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

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow([
                'sku',
                'timestamp',
                'judgment',
                'overall_delta_e',
                'confidence',
                'ng_reasons'
            ])

            # 데이터
            for result in results:
                writer.writerow([
                    result.sku,
                    result.timestamp.isoformat(),
                    result.judgment,
                    f"{result.overall_delta_e:.2f}",
                    f"{result.confidence:.2f}",
                    '; '.join(result.ng_reasons) if result.ng_reasons else ''
                ])

        logger.info(f"Results saved to {output_path}")
