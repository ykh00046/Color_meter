"""
ColorChecker Reference Data

Standard Lab_CIE values for ColorChecker Classic 24 patches.
Reference: X-Rite ColorChecker, D65 illuminant, 2° observer
"""

from typing import Dict, List, Tuple

# ColorChecker Classic 24 patches (6x4 layout)
# Row 1: Brown, Light Skin, Blue Sky, Foliage, Blue Flower, Bluish Green
# Row 2: Orange, Purplish Blue, Moderate Red, Purple, Yellow Green, Orange Yellow
# Row 3: Blue, Green, Red, Yellow, Magenta, Cyan
# Row 4: White, Neutral 8, Neutral 6.5, Neutral 5, Neutral 3.5, Black

# Explicit patch order (row-major: A1-A6, B1-B6, C1-C6, D1-D6)
# This ensures stable ordering independent of dict implementation
PATCH_ORDER = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",  # Row 1
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",  # Row 2
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",  # Row 3
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",  # Row 4 (Grayscale)
]

# Reference conditions for standard values
REFERENCE_CONDITIONS = {
    "illuminant": "D65",
    "observer": "CIE 1931 2°",
    "source": "X-Rite ColorChecker Classic",
    "note": "Actual capture conditions (white balance, lighting) may differ from reference",
}

COLORCHECKER_24_PATCHES = {
    # Row 1
    "A1": {"name": "dark skin", "lab_cie": [37.99, 13.56, 14.06], "position": (0, 0)},
    "A2": {"name": "light skin", "lab_cie": [65.71, 18.13, 17.81], "position": (0, 1)},
    "A3": {"name": "blue sky", "lab_cie": [49.93, -4.88, -21.93], "position": (0, 2)},
    "A4": {"name": "foliage", "lab_cie": [43.14, -13.10, 21.91], "position": (0, 3)},
    "A5": {"name": "blue flower", "lab_cie": [55.11, 8.84, -25.40], "position": (0, 4)},
    "A6": {"name": "bluish green", "lab_cie": [70.72, -33.40, -0.20], "position": (0, 5)},
    # Row 2
    "B1": {"name": "orange", "lab_cie": [62.66, 36.07, 57.10], "position": (1, 0)},
    "B2": {"name": "purplish blue", "lab_cie": [40.02, 10.41, -45.96], "position": (1, 1)},
    "B3": {"name": "moderate red", "lab_cie": [51.12, 48.24, 16.25], "position": (1, 2)},
    "B4": {"name": "purple", "lab_cie": [30.33, 22.98, -21.59], "position": (1, 3)},
    "B5": {"name": "yellow green", "lab_cie": [72.53, -23.71, 57.26], "position": (1, 4)},
    "B6": {"name": "orange yellow", "lab_cie": [71.94, 19.36, 67.86], "position": (1, 5)},
    # Row 3
    "C1": {"name": "blue", "lab_cie": [28.78, 14.18, -50.30], "position": (2, 0)},
    "C2": {"name": "green", "lab_cie": [55.26, -38.34, 31.37], "position": (2, 1)},
    "C3": {"name": "red", "lab_cie": [42.10, 53.38, 28.19], "position": (2, 2)},
    "C4": {"name": "yellow", "lab_cie": [81.73, 4.04, 79.82], "position": (2, 3)},
    "C5": {"name": "magenta", "lab_cie": [51.94, 49.99, -14.57], "position": (2, 4)},
    "C6": {"name": "cyan", "lab_cie": [51.04, -28.63, -28.64], "position": (2, 5)},
    # Row 4 (Grayscale)
    "D1": {"name": "white", "lab_cie": [96.54, -0.43, 1.19], "position": (3, 0)},
    "D2": {"name": "neutral 8", "lab_cie": [81.26, -0.64, -0.34], "position": (3, 1)},
    "D3": {"name": "neutral 6.5", "lab_cie": [66.77, -0.73, -0.50], "position": (3, 2)},
    "D4": {"name": "neutral 5", "lab_cie": [50.87, -0.15, -0.27], "position": (3, 3)},
    "D5": {"name": "neutral 3.5", "lab_cie": [35.66, -0.42, -1.23], "position": (3, 4)},
    "D6": {"name": "black", "lab_cie": [20.46, -0.08, -0.97], "position": (3, 5)},
}


def get_patch_lab(patch_id: str) -> List[float]:
    """
    Get standard Lab_CIE values for a patch.

    Args:
        patch_id: Patch ID (e.g., "A1", "B3")

    Returns:
        [L*, a*, b*] in CIE scale
    """
    if patch_id not in COLORCHECKER_24_PATCHES:
        raise ValueError(f"Invalid patch ID: {patch_id}")

    return COLORCHECKER_24_PATCHES[patch_id]["lab_cie"]


def get_patch_name(patch_id: str) -> str:
    """Get human-readable name for a patch."""
    if patch_id not in COLORCHECKER_24_PATCHES:
        raise ValueError(f"Invalid patch ID: {patch_id}")

    return COLORCHECKER_24_PATCHES[patch_id]["name"]


def get_patch_position(patch_id: str) -> Tuple[int, int]:
    """
    Get grid position for a patch.

    Returns:
        (row, col) zero-indexed
    """
    if patch_id not in COLORCHECKER_24_PATCHES:
        raise ValueError(f"Invalid patch ID: {patch_id}")

    return COLORCHECKER_24_PATCHES[patch_id]["position"]


def get_all_patch_ids() -> List[str]:
    """Get list of all patch IDs in explicit order (row-major)."""
    return PATCH_ORDER


def get_grayscale_patch_ids() -> List[str]:
    """Get list of grayscale patch IDs (Row 4)."""
    return ["D1", "D2", "D3", "D4", "D5", "D6"]


def get_chromatic_patch_ids() -> List[str]:
    """Get list of chromatic (non-grayscale) patch IDs."""
    all_ids = get_all_patch_ids()
    gray_ids = get_grayscale_patch_ids()
    return [pid for pid in all_ids if pid not in gray_ids]


def get_reference_conditions() -> Dict[str, str]:
    """Get reference conditions for standard Lab values."""
    return REFERENCE_CONDITIONS.copy()
