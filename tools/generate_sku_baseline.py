"""
SKU Baseline Generator

Generate SKU baseline configuration from OK sample images.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sku_manager import (
    SkuConfigManager,
    SkuAlreadyExistsError,
    InsufficientSamplesError
)


def find_images(pattern: str) -> List[Path]:
    """
    Find images matching pattern

    Args:
        pattern: Glob pattern (e.g., "data/raw_images/SKU002_OK_*.jpg")

    Returns:
        List of image paths
    """
    # Parse pattern to get directory and file pattern
    pattern_path = Path(pattern)

    if "*" in pattern:
        # Glob pattern
        if pattern_path.is_absolute():
            directory = pattern_path.parent
            file_pattern = pattern_path.name
        else:
            # Relative to project root
            full_pattern = PROJECT_ROOT / pattern
            directory = full_pattern.parent
            file_pattern = full_pattern.name

        images = sorted(directory.glob(file_pattern))
    else:
        # Single file
        images = [Path(pattern)]

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate SKU baseline from OK sample images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate baseline for SKU002 from glob pattern
  python -m tools.generate_sku_baseline \\
    --sku SKU002 \\
    --images "data/raw_images/SKU002_OK_*.jpg" \\
    --description "Blue colored lens"

  # Specify individual files
  python -m tools.generate_sku_baseline \\
    --sku SKU003 \\
    --images data/raw_images/SKU003_OK_001.jpg data/raw_images/SKU003_OK_002.jpg \\
    --description "Brown colored lens" \\
    --threshold 4.0 \\
    --method mean_plus_3std

  # Overwrite existing SKU
  python -m tools.generate_sku_baseline \\
    --sku SKU001 \\
    --images "data/raw_images/OK_*.jpg" \\
    --description "Updated baseline" \\
    --force
"""
    )

    parser.add_argument(
        "--sku",
        required=True,
        help="SKU code (e.g., SKU002)"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="OK sample image paths or glob patterns"
    )
    parser.add_argument(
        "--description",
        default="",
        help="SKU description"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.5,
        help="Default ΔE threshold (default: 3.5)"
    )
    parser.add_argument(
        "--method",
        choices=["mean_plus_2std", "mean_plus_3std", "fixed"],
        default="mean_plus_2std",
        help="Threshold calculation method (default: mean_plus_2std)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("config/sku_db"),
        help="SKU database directory (default: config/sku_db)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing SKU"
    )

    args = parser.parse_args()

    # Find all images
    all_images = []
    for pattern in args.images:
        images = find_images(pattern)
        all_images.extend(images)

    # Remove duplicates and sort
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"Error: No images found matching patterns: {args.images}")
        return 1

    print(f"Found {len(all_images)} OK sample images:")
    for img in all_images:
        print(f"  - {img.name}")
    print()

    # Check minimum samples
    if len(all_images) < 3:
        print(f"Error: Minimum 3 samples required, found {len(all_images)}")
        return 1

    # Initialize manager
    manager = SkuConfigManager(db_path=args.db_path)

    # Check if SKU exists
    if not args.force:
        try:
            existing = manager.get_sku(args.sku)
            print(f"Error: SKU {args.sku} already exists")
            print(f"Use --force to overwrite")
            print(f"\nExisting SKU:")
            print(f"  Description: {existing['description']}")
            print(f"  Zones: {len(existing['zones'])}")
            print(f"  Created: {existing['metadata']['created_at']}")
            return 1
        except Exception:
            # SKU doesn't exist, proceed
            pass
    else:
        # Delete existing if force mode
        try:
            manager.delete_sku(args.sku)
            print(f"Deleted existing SKU {args.sku}")
        except Exception:
            pass

    # Generate baseline
    print(f"Generating baseline for {args.sku}...")
    print(f"  Method: {args.method}")
    print(f"  Default threshold: {args.threshold}")
    print()

    try:
        sku_data = manager.generate_baseline(
            sku_code=args.sku,
            ok_images=all_images,
            description=args.description or f"Auto-generated {args.sku}",
            default_threshold=args.threshold,
            threshold_method=args.method
        )

        print("✓ Baseline generated successfully!")
        print()
        print(f"SKU Code: {sku_data['sku_code']}")
        print(f"Description: {sku_data['description']}")
        print(f"Samples processed: {sku_data['metadata']['baseline_samples']}")
        print()
        print("Zones:")
        for zone_name, zone_config in sku_data['zones'].items():
            print(f"  Zone {zone_name}:")
            print(f"    LAB: L={zone_config['L']:.1f}, a={zone_config['a']:.1f}, b={zone_config['b']:.1f}")
            print(f"    Threshold: {zone_config['threshold']:.1f}")

            # Show statistics if available
            stats_key = f"zone_{zone_name}"
            if stats_key in sku_data['metadata'].get('statistics', {}):
                stats = sku_data['metadata']['statistics'][stats_key]
                print(f"    Std dev: L±{stats['L_std']:.2f}, a±{stats['a_std']:.2f}, b±{stats['b_std']:.2f}")

        print()
        print(f"Saved to: {manager._get_sku_path(args.sku)}")

        return 0

    except InsufficientSamplesError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: Failed to generate baseline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
