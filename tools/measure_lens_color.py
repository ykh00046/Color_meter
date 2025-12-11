"""
Measure actual LAB colors from lens images to calibrate SKU baseline.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.image_loader import ImageLoader, ImageConfig
from src.core.lens_detector import LensDetector, DetectorConfig
from src.core.radial_profiler import RadialProfiler, ProfilerConfig
from src.core.zone_segmenter import ZoneSegmenter, SegmenterConfig


def measure_colors(image_path: str):
    """Measure LAB colors from lens image"""

    # Initialize modules
    loader = ImageLoader(ImageConfig())
    detector = LensDetector(DetectorConfig())
    profiler = RadialProfiler(ProfilerConfig())
    segmenter = ZoneSegmenter(SegmenterConfig())

    # Process
    image = loader.load_from_file(Path(image_path))
    processed = loader.preprocess(image)
    lens_detection = detector.detect(processed)
    radial_profile = profiler.extract_profile(processed, lens_detection)
    zones = segmenter.segment(radial_profile)

    # Print results
    print(f"\nMeasured LAB colors from {image_path}:")
    print("="*60)

    for zone in zones:
        print(f"\nZone {zone.name} ({zone.zone_type}):")
        print(f"  L*: {zone.mean_L:.1f}")
        print(f"  a*: {zone.mean_a:.1f}")
        print(f"  b*: {zone.mean_b:.1f}")
        print(f"  r_range: {zone.r_end:.3f} - {zone.r_start:.3f}")
        print(f"  std: L={zone.std_L:.1f}, a={zone.std_a:.1f}, b={zone.std_b:.1f}")

    print("\n" + "="*60)
    print("\nSuggested SKU JSON:")
    print("{")
    for zone in zones:
        if zone.zone_type == 'pure':
            print(f'  "{zone.name}": {{')
            print(f'    "L": {zone.mean_L:.1f},')
            print(f'    "a": {zone.mean_a:.1f},')
            print(f'    "b": {zone.mean_b:.1f},')
            # Threshold based on 3*std
            threshold = max(3.0, 3 * max(zone.std_L, zone.std_a, zone.std_b))
            print(f'    "threshold": {threshold:.1f}')
            print(f'  }},')
    print("}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/measure_lens_color.py <image_path>")
        sys.exit(1)

    measure_colors(sys.argv[1])
