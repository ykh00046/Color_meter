"""
Simulation module for Digital Proofing and Color Prediction.

Includes:
- color_simulator: Area-ratio based simulation (legacy)
- mask_compositor: Mask-based pixel synthesis (Direction A)
- validation: Delta E calculation utilities
"""

from .color_simulator import build_simulation_result, simulate_perceived_color
from .mask_compositor import compare_methods, composite_from_masks
