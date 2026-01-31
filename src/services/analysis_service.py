"""
Analysis Service Layer
Centralizes profile analysis logic for reuse across CLI and Web UI.
"""

import logging
from typing import Any, Dict, Optional

from src.engine_v7.core.signature.profile_analysis import analyze_profile

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for analyzing radial profiles.
    Wraps the low-level analyze_profile function and provides a simplified interface for consumers.
    """

    def __init__(self):
        pass

    def analyze_radial_profile(self, profile: Any, lens_radius: float) -> Dict[str, Any]:
        """
        Analyze a radial profile to extract smoothed data, derivatives, and boundary candidates.

        Args:
            profile: The RadialProfile object to analyze.
            lens_radius: The radius of the lens in pixels (for coordinate conversion).
        Returns:
            A dictionary containing the analysis results, serializable to JSON.
        """
        try:
            # Perform comprehensive analysis using v7 engine function
            result = analyze_profile(
                r_norm=profile.r_normalized,
                l_data=profile.L,
                a_data=profile.a,
                b_data=profile.b,
                smoothing_window=5,
                gradient_threshold=0.5,
            )

            # Return as dictionary
            return result

        except Exception as e:
            logger.error(f"Error during profile analysis: {e}", exc_info=True)
            # Return empty structure or re-raise depending on requirements.
            # Here we re-raise to let the caller handle the failure.
            raise
