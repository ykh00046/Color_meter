# Hungarian matcher exports
from .hungarian_matcher import (  # Main API; Types; Utilities
    SCIPY_AVAILABLE,
    MatchResult,
    compute_lab_distance_matrix,
    match_clusters_to_reference,
    reorder_clusters_by_reference,
    stabilize_cluster_labels,
)

__all__ = [
    # Main API
    "match_clusters_to_reference",
    "reorder_clusters_by_reference",
    "stabilize_cluster_labels",
    # Types
    "MatchResult",
    # Utilities
    "compute_lab_distance_matrix",
    "SCIPY_AVAILABLE",
]
