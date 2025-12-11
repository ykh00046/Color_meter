"""
Main CLI Entry Point

콘택트렌즈 색상 검사 파이프라인 CLI 프로그램.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List

from src.pipeline import InspectionPipeline, PipelineError
from src.utils.file_io import read_json


# 로깅 설정
def setup_logging(debug: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_sku_config(sku: str, config_dir: Path = Path('config/sku_db')) -> dict:
    """
    SKU 설정 로드.

    Args:
        sku: SKU 코드
        config_dir: SKU 설정 디렉토리

    Returns:
        SKU 설정 딕셔너리

    Raises:
        FileNotFoundError: SKU 설정 파일이 없을 때
    """
    config_path = config_dir / f"{sku}.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"SKU configuration not found: {config_path}\n"
            f"Available SKUs: {[f.stem for f in config_dir.glob('*.json')]}"
        )

    return read_json(config_path)


def process_single_image(args):
    """단일 이미지 처리"""
    logger = logging.getLogger(__name__)

    # SKU 설정 로드
    sku_config = load_sku_config(args.sku)

    # 파이프라인 초기화
    pipeline = InspectionPipeline(
        sku_config,
        save_intermediates=args.save_intermediates
    )

    # 처리
    result = pipeline.process(
        args.image,
        args.sku,
        save_dir=Path(args.output).parent if args.save_intermediates else None
    )

    # 결과 출력
    print("\n" + "="*60)
    print(f"  Inspection Result")
    print("="*60)
    print(f"  Image:      {args.image}")
    print(f"  SKU:        {args.sku}")
    print(f"  Judgment:   {result.judgment}")
    print(f"  Overall ΔE: {result.overall_delta_e:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Timestamp:  {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    if result.ng_reasons:
        print(f"\n  NG Reasons:")
        for reason in result.ng_reasons:
            print(f"    - {reason}")

    print(f"\n  Zone Results:")
    for zr in result.zone_results:
        status = "[OK]" if zr.is_ok else "[NG]"
        print(
            f"    Zone {zr.zone_name}: dE={zr.delta_e:.2f} "
            f"(threshold={zr.threshold:.1f}) {status}"
        )

    print("="*60 + "\n")

    # JSON 저장 (옵션)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'image_path': str(args.image),
            'sku': result.sku,
            'timestamp': result.timestamp.isoformat(),
            'judgment': result.judgment,
            'overall_delta_e': result.overall_delta_e,
            'confidence': result.confidence,
            'zone_results': [
                {
                    'zone_name': zr.zone_name,
                    'measured_lab': list(zr.measured_lab),
                    'target_lab': list(zr.target_lab),
                    'delta_e': zr.delta_e,
                    'threshold': zr.threshold,
                    'is_ok': zr.is_ok
                }
                for zr in result.zone_results
            ],
            'ng_reasons': result.ng_reasons
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Result saved to {output_path}")

    return 0 if result.judgment == 'OK' else 1


def process_batch(args):
    """배치 처리"""
    logger = logging.getLogger(__name__)

    # 이미지 경로 수집
    batch_dir = Path(args.batch)

    if not batch_dir.exists():
        logger.error(f"Batch directory not found: {batch_dir}")
        return 1

    # JPG, PNG 파일 수집
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(batch_dir.glob(ext))

    if not image_paths:
        logger.error(f"No images found in {batch_dir}")
        return 1

    logger.info(f"Found {len(image_paths)} images in {batch_dir}")

    # SKU 설정 로드
    sku_config = load_sku_config(args.sku)

    # 파이프라인 초기화
    pipeline = InspectionPipeline(sku_config)

    # 배치 처리
    results = pipeline.process_batch(
        [str(p) for p in image_paths],
        args.sku,
        output_csv=Path(args.output) if args.output else None,
        continue_on_error=args.continue_on_error
    )

    # 요약 출력
    ok_count = sum(1 for r in results if r.judgment == 'OK')
    ng_count = len(results) - ok_count

    print("\n" + "="*60)
    print(f"  Batch Processing Summary")
    print("="*60)
    print(f"  Total images:  {len(image_paths)}")
    print(f"  Processed:     {len(results)}")
    print(f"  OK:            {ok_count}")
    print(f"  NG:            {ng_count}")
    print(f"  Failed:        {len(image_paths) - len(results)}")

    if args.output:
        print(f"  Results saved: {args.output}")

    print("="*60 + "\n")

    return 0 if ng_count == 0 else 1


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Contact Lens Color Inspection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single image processing
  python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001

  # With JSON output
  python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001 --output results/result.json

  # Batch processing
  python src/main.py --batch data/raw_images/ --sku SKU001 --output results/batch.csv

  # With debug logging
  python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001 --debug
        '''
    )

    # 모드 선택 (단일 or 배치)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--image',
        type=str,
        help='Single image file path'
    )
    mode_group.add_argument(
        '--batch',
        type=str,
        help='Batch directory path'
    )

    # 필수 인자
    parser.add_argument(
        '--sku',
        type=str,
        required=True,
        help='SKU code (e.g., SKU001)'
    )

    # 옵션 인자
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON for single image, CSV for batch)'
    )
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save intermediate results (for debugging)'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help='Continue batch processing on error (default: True)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    try:
        # 모드 분기
        if args.image:
            return process_single_image(args)
        elif args.batch:
            return process_batch(args)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return 2

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
