# Color masks module exports
from .color_masks import (  # noqa: E501
    assign_cluster_labels_to_image,
    build_alpha_from_plate_images,
    build_color_masks,
    build_color_masks_v2,
    build_color_masks_with_alpha,
    build_color_masks_with_reference,
    build_color_masks_with_retry,
    calculate_segmentation_confidence,
    compute_cluster_effective_densities,
    filter_masks_by_role,
    lab_cv8_to_hex,
    lab_to_hex,
    stabilize_labels_with_reference,
    visualize_color_masks,
)

__all__ = [
    # Main API
    "build_color_masks",
    "build_color_masks_with_retry",
    "build_color_masks_v2",
    "build_color_masks_with_alpha",
    "build_color_masks_with_reference",
    # Effective density integration
    "compute_cluster_effective_densities",
    "build_alpha_from_plate_images",
    # Label stabilization (Hungarian matching)
    "stabilize_labels_with_reference",
    # Utilities
    "filter_masks_by_role",
    "visualize_color_masks",
    "calculate_segmentation_confidence",
    "assign_cluster_labels_to_image",
    "lab_to_hex",
    "lab_cv8_to_hex",
]
