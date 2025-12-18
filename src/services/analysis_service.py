"""
Analysis Service Layer
Centralizes profile analysis logic for reuse across CLI and Web UI.
"""

import logging
from typing import Any, Dict, Optional

from src.analysis.profile_analyzer import ProfileAnalyzer
from src.core.radial_profiler import RadialProfile

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for analyzing radial profiles.
    Wraps the low-level ProfileAnalyzer and provides a simplified interface for consumers.
    """

    def __init__(self):
        # Default configuration for ProfileAnalyzer
        self.analyzer = ProfileAnalyzer()

    def analyze_radial_profile(
        self, profile: RadialProfile, lens_radius: float, zones_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a radial profile to extract smoothed data, derivatives, and boundary candidates.

        Args:
            profile: The RadialProfile object to analyze.
            lens_radius: The radius of the lens in pixels (for coordinate conversion).
            zones_config: Optional SKU zone configuration. If provided,
                          the baseline Lab values from the first zone (usually 'A')
                          will be used for Delta E calculation.

        Returns:
            A dictionary containing the analysis results, serializable to JSON.
        """
        try:
            # Perform comprehensive analysis using ProfileAnalyzer
            # Using default window=5, threshold=0.5 for now, or could be passed via args
            result = self.analyzer.analyze_profile(
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
