"""
Analysis Service Layer
Centralizes profile analysis logic for reuse across CLI and Web UI.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add root to sys.path to allow importing lens_signature_engine_v7
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lens_signature_engine_v7.core.signature.profile_analysis import analyze_profile
from src.core.radial_profiler import RadialProfile

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for analyzing radial profiles.
    Wraps the low-level analyze_profile function and provides a simplified interface for consumers.
    """

    def __init__(self):
        pass

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
