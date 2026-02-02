# Alpha density module exports
from .alpha_density import (  # Main API; Fallback levels; Profile types; Utility functions; Core computation; Config
    DEFAULT_ALPHA_CONFIG,
    AlphaDensityResult,
    AlphaFallbackLevel,
    AlphaRadialProfile,
    AlphaZoneProfile,
    ClusterAlphaResult,
    apply_alpha_fallback,
    build_alpha_map_polar,
    compute_alpha_global,
    compute_alpha_radial_1d,
    compute_alpha_zone,
    compute_effective_density,
    evaluate_alpha_quality_gate,
    extract_alpha_summary,
    extract_effective_densities,
)
from .alpha_polar import PolarAlphaResult, build_polar_alpha_registrationless
from .alpha_verification import AlphaVerificationResult, verify_alpha_agreement
from .transition_detector import TransitionConfig, TransitionResult, compute_transition_weights, create_alpha_weight_map

__all__ = [
    # Main API
    "compute_effective_density",
    "AlphaDensityResult",
    "ClusterAlphaResult",
    # Fallback levels
    "AlphaFallbackLevel",
    # Profile types
    "AlphaRadialProfile",
    "AlphaZoneProfile",
    # Utility functions
    "extract_effective_densities",
    "extract_alpha_summary",
    "build_alpha_map_polar",
    # Core computation
    "compute_alpha_radial_1d",
    "compute_alpha_zone",
    "compute_alpha_global",
    "apply_alpha_fallback",
    # Config
    "DEFAULT_ALPHA_CONFIG",
    "TransitionConfig",
    "TransitionResult",
    "compute_transition_weights",
    "create_alpha_weight_map",
    # Quality gate
    "evaluate_alpha_quality_gate",
    # Re-exported from alpha_polar
    "PolarAlphaResult",
    "build_polar_alpha_registrationless",
    # Re-exported from alpha_verification
    "AlphaVerificationResult",
    "verify_alpha_agreement",
]
