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
from src.sku_manager import (
    SkuConfigManager,
    SkuNotFoundError,
    SkuAlreadyExistsError,
    InvalidSkuDataError,
    InsufficientSamplesError
)
from src.visualizer import InspectionVisualizer, VisualizerConfig


# 로깅 설정
def setup_logging(debug: bool = False):
    """로깅 설정 (UTF-8 인코딩 지원)"""
    import sys

    level = logging.DEBUG if debug else logging.INFO

    # Windows 콘솔 UTF-8 지원
    if sys.platform == 'win32':
        try:
            # Python 3.7+: UTF-8 모드 활성화
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass  # 실패해도 계속 진행

    # 로깅 핸들러 설정 (UTF-8 인코딩 명시)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # 기본 로거 설정
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True  # 기존 핸들러 제거
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

    # 시각화 (옵션)
    if hasattr(args, 'visualize') and args.visualize:
        visualizer = InspectionVisualizer(VisualizerConfig())

        viz_output = Path(args.viz_output) if hasattr(args, 'viz_output') and args.viz_output else None

        if args.visualize == 'zone_overlay':
            viz_img = visualizer.visualize_zone_overlay(
                result.image,
                result.lens_detection,
                result.zones,
                result,
                show_result=True
            )
            if viz_output:
                visualizer.save_visualization(viz_img, viz_output)
                logger.info(f"Zone overlay saved to {viz_output}")

        elif args.visualize == 'comparison':
            viz_fig = visualizer.visualize_comparison(result.zones, result)
            if viz_output:
                visualizer.save_visualization(viz_fig, viz_output)
                logger.info(f"Comparison chart saved to {viz_output}")
            else:
                import matplotlib.pyplot as plt
                plt.show()

        elif args.visualize == 'all':
            # Zone overlay
            viz_img = visualizer.visualize_zone_overlay(
                result.image,
                result.lens_detection,
                result.zones,
                result,
                show_result=True
            )
            overlay_path = viz_output.parent / f"{viz_output.stem}_overlay{viz_output.suffix}" if viz_output else Path('results') / 'overlay.png'
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            visualizer.save_visualization(viz_img, overlay_path)
            logger.info(f"Zone overlay saved to {overlay_path}")

            # Comparison chart
            viz_fig = visualizer.visualize_comparison(result.zones, result)
            comparison_path = viz_output.parent / f"{viz_output.stem}_comparison{viz_output.suffix}" if viz_output else Path('results') / 'comparison.png'
            visualizer.save_visualization(viz_fig, comparison_path)
            logger.info(f"Comparison chart saved to {comparison_path}")

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

    # 배치 시각화 (옵션)
    if hasattr(args, 'visualize') and args.visualize and results:
        visualizer = InspectionVisualizer(VisualizerConfig())

        viz_output = Path(args.viz_output) if hasattr(args, 'viz_output') and args.viz_output else Path('results') / 'dashboard.png'
        viz_output.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Generating dashboard visualization...")
        viz_fig = visualizer.visualize_dashboard(results)
        visualizer.save_visualization(viz_fig, viz_output)
        logger.info(f"Dashboard saved to {viz_output}")
        print(f"  Dashboard:     {viz_output}")

    return 0 if ng_count == 0 else 1


def cmd_sku_list(args):
    """SKU 목록 조회"""
    manager = SkuConfigManager(db_path=args.db_path)
    skus = manager.list_all_skus()

    if not skus:
        print("No SKUs found.")
        return 0

    # Print table header
    print("\n{:<12} {:<35} {:>6} {:>20}".format(
        "SKU Code", "Description", "Zones", "Created At"
    ))
    print("-" * 75)

    # Print SKU rows
    for sku in skus:
        created_at = sku['created_at'][:19] if sku['created_at'] else ""
        print("{:<12} {:<35} {:>6} {:>20}".format(
            sku['sku_code'],
            sku['description'][:35],
            sku['zones_count'],
            created_at
        ))

    print(f"\nTotal: {len(skus)} SKUs\n")
    return 0


def cmd_sku_show(args):
    """SKU 상세 조회"""
    manager = SkuConfigManager(db_path=args.db_path)

    try:
        sku_data = manager.get_sku(args.sku_code)
    except SkuNotFoundError:
        print(f"Error: SKU {args.sku_code} not found")
        return 1

    # Print SKU details
    print(f"\nSKU Code: {sku_data['sku_code']}")
    print(f"Description: {sku_data['description']}")
    print(f"Default Threshold: {sku_data['default_threshold']}")
    print("\nZones:")

    for zone_name, zone_config in sku_data['zones'].items():
        print(f"  Zone {zone_name}:")
        print(f"    LAB: L={zone_config['L']:.1f}, a={zone_config['a']:.1f}, b={zone_config['b']:.1f}")
        print(f"    Threshold: {zone_config['threshold']:.1f}")
        if 'description' in zone_config:
            print(f"    Description: {zone_config['description']}")

    print("\nMetadata:")
    metadata = sku_data['metadata']
    print(f"  Created: {metadata.get('created_at', 'N/A')}")
    print(f"  Last Updated: {metadata.get('last_updated', 'N/A')}")
    print(f"  Samples: {metadata.get('baseline_samples', 0)}")
    print(f"  Method: {metadata.get('calibration_method', 'N/A')}")
    print()

    return 0


def cmd_sku_create(args):
    """SKU 생성"""
    manager = SkuConfigManager(db_path=args.db_path)

    # Parse zones if provided
    zones = {}
    if args.zone:
        for zone_str in args.zone:
            try:
                # Format: A:70.0:-10.0:-30.0:4.0
                parts = zone_str.split(':')
                if len(parts) != 5:
                    print(f"Error: Invalid zone format: {zone_str}")
                    print("Expected format: NAME:L:a:b:threshold (e.g., A:70.0:-10.0:-30.0:4.0)")
                    return 1

                zone_name = parts[0]
                L, a, b, threshold = map(float, parts[1:])

                zones[zone_name] = {
                    "L": L,
                    "a": a,
                    "b": b,
                    "threshold": threshold
                }
            except ValueError as e:
                print(f"Error: Invalid zone format: {zone_str} ({e})")
                return 1

    try:
        sku_data = manager.create_sku(
            sku_code=args.sku_code,
            description=args.description,
            default_threshold=args.threshold,
            zones=zones,
            author=args.author
        )

        print(f"[OK] SKU {args.sku_code} created successfully")
        print(f"  Description: {sku_data['description']}")
        print(f"  Zones: {len(sku_data['zones'])}")
        print(f"  Saved to: {manager._get_sku_path(args.sku_code)}")
        return 0

    except (SkuAlreadyExistsError, InvalidSkuDataError) as e:
        print(f"Error: {e}")
        return 1


def cmd_sku_generate_baseline(args):
    """베이스라인 자동 생성"""
    manager = SkuConfigManager(db_path=args.db_path)

    # Find images
    image_paths = []
    for pattern in args.images:
        pattern_path = Path(pattern)
        if "*" in pattern:
            image_paths.extend(sorted(pattern_path.parent.glob(pattern_path.name)))
        else:
            image_paths.append(pattern_path)

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"Error: No images found matching patterns: {args.images}")
        return 1

    print(f"Found {len(image_paths)} OK sample images")

    # Check if SKU exists
    if not args.force:
        try:
            existing = manager.get_sku(args.sku_code)
            print(f"Error: SKU {args.sku_code} already exists. Use --force to overwrite")
            return 1
        except SkuNotFoundError:
            pass
    else:
        try:
            manager.delete_sku(args.sku_code)
            print(f"Deleted existing SKU {args.sku_code}")
        except SkuNotFoundError:
            pass

    # Generate baseline
    print(f"Generating baseline for {args.sku_code}...")

    try:
        sku_data = manager.generate_baseline(
            sku_code=args.sku_code,
            ok_images=image_paths,
            description=args.description,
            default_threshold=args.threshold,
            threshold_method=args.method
        )

        print(f"\n[OK] Baseline generated successfully!")
        print(f"  SKU: {sku_data['sku_code']}")
        print(f"  Samples: {sku_data['metadata']['baseline_samples']}")
        print(f"  Zones: {len(sku_data['zones'])}")
        print(f"  Saved to: {manager._get_sku_path(args.sku_code)}")
        return 0

    except (InsufficientSamplesError, InvalidSkuDataError) as e:
        print(f"Error: {e}")
        return 1


def cmd_sku_update(args):
    """SKU 수정"""
    manager = SkuConfigManager(db_path=args.db_path)

    try:
        updates = {}

        if args.description:
            updates['description'] = args.description
        if args.default_threshold is not None:
            updates['default_threshold'] = args.default_threshold

        # Parse zone threshold updates
        if args.zone_threshold:
            for zt_str in args.zone_threshold:
                try:
                    # Format: A:4.5
                    zone_name, threshold = zt_str.split(':')
                    updates[f'zones.{zone_name}.threshold'] = float(threshold)
                except ValueError:
                    print(f"Error: Invalid zone-threshold format: {zt_str}")
                    print("Expected format: ZONE:THRESHOLD (e.g., A:4.5)")
                    return 1

        if not updates:
            print("Error: No updates specified")
            return 1

        sku_data = manager.update_sku(args.sku_code, updates)

        print(f"[OK] SKU {args.sku_code} updated successfully")
        for key, value in updates.items():
            print(f"  {key}: {value}")
        return 0

    except SkuNotFoundError as e:
        print(f"Error: {e}")
        return 1


def cmd_sku_delete(args):
    """SKU 삭제"""
    manager = SkuConfigManager(db_path=args.db_path)

    try:
        # Confirmation
        if not args.yes:
            confirm = input(f"Are you sure you want to delete {args.sku_code}? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled")
                return 0

        manager.delete_sku(args.sku_code)
        print(f"[OK] SKU {args.sku_code} deleted successfully")
        return 0

    except SkuNotFoundError as e:
        print(f"Error: {e}")
        return 1


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Contact Lens Color Inspection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # ========== inspect 명령어 (단일 이미지) ==========
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001
  python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 --output results/result.json
        '''
    )
    inspect_parser.add_argument('--image', required=True, help='Image file path')
    inspect_parser.add_argument('--sku', required=True, help='SKU code')
    inspect_parser.add_argument('--output', help='Output JSON file path')
    inspect_parser.add_argument('--save-intermediates', action='store_true', help='Save intermediate results')
    inspect_parser.add_argument('--visualize', choices=['zone_overlay', 'comparison', 'all'], help='Generate visualization (zone_overlay, comparison, or all)')
    inspect_parser.add_argument('--viz-output', help='Visualization output path (PNG or PDF)')

    # ========== batch 명령어 (배치 처리) ==========
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch process images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m src.main batch --batch data/raw_images/ --sku SKU001
  python -m src.main batch --batch data/raw_images/ --sku SKU001 --output results/batch.csv
        '''
    )
    batch_parser.add_argument('--batch', required=True, help='Batch directory path')
    batch_parser.add_argument('--sku', required=True, help='SKU code')
    batch_parser.add_argument('--output', help='Output CSV file path')
    batch_parser.add_argument('--continue-on-error', action='store_true', default=True, help='Continue on error')
    batch_parser.add_argument('--visualize', action='store_true', help='Generate dashboard visualization')
    batch_parser.add_argument('--viz-output', help='Dashboard output path (PNG or PDF)')

    # ========== sku 명령어 (SKU 관리) ==========
    sku_parser = subparsers.add_parser(
        'sku',
        help='SKU management commands'
    )
    sku_subparsers = sku_parser.add_subparsers(dest='sku_command', help='SKU command')

    # sku list
    sku_list_parser = sku_subparsers.add_parser('list', help='List all SKUs')
    sku_list_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # sku show
    sku_show_parser = sku_subparsers.add_parser('show', help='Show SKU details')
    sku_show_parser.add_argument('sku_code', help='SKU code')
    sku_show_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # sku create
    sku_create_parser = sku_subparsers.add_parser('create', help='Create new SKU')
    sku_create_parser.add_argument('--code', dest='sku_code', required=True, help='SKU code')
    sku_create_parser.add_argument('--description', required=True, help='SKU description')
    sku_create_parser.add_argument('--threshold', type=float, default=3.5, help='Default threshold')
    sku_create_parser.add_argument('--zone', action='append', help='Zone config (NAME:L:a:b:threshold)')
    sku_create_parser.add_argument('--author', default='user', help='Author name')
    sku_create_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # sku generate-baseline
    sku_gen_parser = sku_subparsers.add_parser('generate-baseline', help='Generate baseline from OK samples')
    sku_gen_parser.add_argument('--sku', dest='sku_code', required=True, help='SKU code')
    sku_gen_parser.add_argument('--images', nargs='+', required=True, help='OK sample image paths or patterns')
    sku_gen_parser.add_argument('--description', default='', help='SKU description')
    sku_gen_parser.add_argument('--threshold', type=float, default=3.5, help='Default threshold')
    sku_gen_parser.add_argument('--method', choices=['mean_plus_2std', 'mean_plus_3std', 'fixed'], default='mean_plus_2std', help='Threshold calculation method')
    sku_gen_parser.add_argument('--force', action='store_true', help='Overwrite existing SKU')
    sku_gen_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # sku update
    sku_update_parser = sku_subparsers.add_parser('update', help='Update SKU')
    sku_update_parser.add_argument('sku_code', help='SKU code')
    sku_update_parser.add_argument('--description', help='Update description')
    sku_update_parser.add_argument('--default-threshold', type=float, help='Update default threshold')
    sku_update_parser.add_argument('--zone-threshold', action='append', help='Update zone threshold (ZONE:THRESHOLD)')
    sku_update_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # sku delete
    sku_delete_parser = sku_subparsers.add_parser('delete', help='Delete SKU')
    sku_delete_parser.add_argument('sku_code', help='SKU code')
    sku_delete_parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    sku_delete_parser.add_argument('--db-path', type=Path, default=Path('config/sku_db'), help='SKU database path')

    # Parse arguments
    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0

    try:
        # Command dispatch
        if args.command == 'inspect':
            return process_single_image(args)
        elif args.command == 'batch':
            return process_batch(args)
        elif args.command == 'sku':
            if not args.sku_command:
                sku_parser.print_help()
                return 0

            # SKU command dispatch
            if args.sku_command == 'list':
                return cmd_sku_list(args)
            elif args.sku_command == 'show':
                return cmd_sku_show(args)
            elif args.sku_command == 'create':
                return cmd_sku_create(args)
            elif args.sku_command == 'generate-baseline':
                return cmd_sku_generate_baseline(args)
            elif args.sku_command == 'update':
                return cmd_sku_update(args)
            elif args.sku_command == 'delete':
                return cmd_sku_delete(args)

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
