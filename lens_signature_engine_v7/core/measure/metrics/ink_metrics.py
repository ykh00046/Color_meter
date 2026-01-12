from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def lab_opencv_to_cie(lab: np.ndarray) -> np.ndarray:
    """
    OpenCV Lab (uint8 or float; L:0~255, a/b:0~255 with 128 offset)
    -> CIE Lab (float32; L:0~100, a/b:-128~127)
    Accepts shape (...,3)
    """
    x = np.asarray(lab).astype(np.float32)
    out = x.copy()
    out[..., 0] = out[..., 0] * (100.0 / 255.0)
    out[..., 1] = out[..., 1] - 128.0
    out[..., 2] = out[..., 2] - 128.0
    return out


def ensure_cie_lab(lab: np.ndarray) -> np.ndarray:
    """
    Heuristic: if a/b look like 0~255 scale, treat as OpenCV and convert.
    """
    x = np.asarray(lab).astype(np.float32)
    if x.size == 0:
        return x
    # If a/b mostly in [0,255], convert. (CIE a/b usually centered around 0)
    ab = x[..., 1:3]
    if np.nanmin(ab) >= -5.0 and np.nanmax(ab) <= 260.0 and np.nanmean(ab) > 40.0:
        return lab_opencv_to_cie(x)
    # If L looks like 0~255 scale but a/b already centered, still convert L.
    L = x[..., 0]
    if np.nanmax(L) > 120.0:
        y = x.copy()
        y[..., 0] = y[..., 0] * (100.0 / 255.0)
        return y
    return x


# ----------------------------
# ΔE utilities (76 / CIEDE2000)
# ----------------------------
def _to_cie_vec(lab: Any) -> np.ndarray:
    """
    Return 1x3 CIE Lab float32 vector.
    If ensure_cie_lab() exists, it will be used.
    """
    x = np.asarray(lab, dtype=np.float32).reshape(
        3,
    )
    # ensure_cie_lab is available in this module scope
    try:
        y = ensure_cie_lab(x)
        return np.asarray(y, dtype=np.float32).reshape(
            3,
        )
    except Exception:
        return x


def deltaE76(lab1: Any, lab2: Any) -> float:
    a = _to_cie_vec(lab1)
    b = _to_cie_vec(lab2)
    d = a - b
    return float(np.linalg.norm(d))


def deltaE2000(lab1: Any, lab2: Any, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """
    CIEDE2000 implementation (Sharma et al., 2005).
    Verified against standard test pairs:
      (50, 2.6772, -79.7751) vs (50, 0, -82.7485) -> 2.0425
      (50, 3.1571, -77.2803) vs (50, 0, -82.7485) -> 2.8615
    """
    L1, a1, b1 = map(float, _to_cie_vec(lab1))
    L2, a2, b2 = map(float, _to_cie_vec(lab2))

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0

    C_bar7 = C_bar**7
    G = 0.0
    if C_bar > 0.0:
        G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def _hp(ap: float, bp: float) -> float:
        if ap == 0.0 and bp == 0.0:
            return 0.0
        h = math.degrees(math.atan2(bp, ap))
        return h + 360.0 if h < 0.0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180.0:
            dhp = dh
        elif dh > 180.0:
            dhp = dh - 360.0
        else:
            dhp = dh + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        h_bar_p = h1p + h2p
    else:
        diff = abs(h1p - h2p)
        if diff <= 180.0:
            h_bar_p = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360.0:
            h_bar_p = (h1p + h2p + 360.0) / 2.0
        else:
            h_bar_p = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    Rc = 0.0
    if C_bar_p > 0.0:
        Rc = 2.0 * math.sqrt((C_bar_p**7) / ((C_bar_p**7) + (25.0**7)))

    Sl = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / math.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    Sc = 1.0 + 0.045 * C_bar_p
    Sh = 1.0 + 0.015 * C_bar_p * T
    Rt = -math.sin(math.radians(2.0 * d_theta)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def deltaE(lab1: Any, lab2: Any, method: str = "76") -> float:
    m = str(method or "76").strip().lower()
    if m in ("2000", "de2000", "ciede2000", "cie2000"):
        return deltaE2000(lab1, lab2)
    return deltaE76(lab1, lab2)


def _mean_lab(lab_samples: np.ndarray, labels: np.ndarray, k: int) -> List[np.ndarray]:
    means = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        if idx.size == 0:
            means.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        else:
            means.append(lab_samples[idx].mean(axis=0).astype(np.float32))
    return means


def _area_ratios(labels: np.ndarray, k: int) -> List[float]:
    total = float(labels.size)
    if total <= 0:
        return [0.0 for _ in range(k)]
    ratios = []
    for i in range(k):
        ratios.append(float(np.sum(labels == i) / total))
    return ratios


def _cluster_distribution(lab_samples: np.ndarray, labels: np.ndarray, cluster_id: int) -> Dict[str, Any]:
    """
    Calculate distribution statistics for a single cluster.

    Returns:
        {
            "lab_std": [std_L, std_a, std_b],
            "p10": [p10_L, p10_a, p10_b],
            "p90": [p90_L, p90_a, p90_b],
            "compactness": float (0-1, higher is better),
            "alpha_like": float (0-1, higher = more transparent/gap-like)
        }
    """
    cluster_pixels = lab_samples[labels == cluster_id]

    if cluster_pixels.size == 0 or cluster_pixels.shape[0] == 0:
        return {
            "lab_std": [0.0, 0.0, 0.0],
            "p10": [0.0, 0.0, 0.0],
            "p90": [0.0, 0.0, 0.0],
            "compactness": 0.0,
            "alpha_like": 0.0,
        }

    # Lab standard deviation
    lab_std = np.std(cluster_pixels, axis=0)

    # Percentiles
    p10 = np.percentile(cluster_pixels, 10, axis=0)
    p90 = np.percentile(cluster_pixels, 90, axis=0)

    # Compactness (inverse of mean distance to centroid, normalized)
    centroid = np.mean(cluster_pixels, axis=0)
    distances = np.linalg.norm(cluster_pixels - centroid, axis=1)
    mean_dist = np.mean(distances)
    compactness = 1.0 / (1.0 + mean_dist)  # 0~1, higher is more compact

    # Alpha-like score (transparency indicator)
    # Higher L* = lighter = more transparent
    # Lower chroma (sqrt(a^2 + b^2)) = more neutral = more transparent
    mean_L = float(centroid[0])
    mean_a = float(centroid[1])
    mean_b = float(centroid[2])
    chroma = np.sqrt(mean_a**2 + mean_b**2)

    # Normalize L* to 0-1 (assuming Lab range L:0-100)
    L_norm = np.clip(mean_L / 100.0, 0.0, 1.0)

    # Normalize chroma to 0-1 (typical max chroma ~130 in Lab)
    chroma_norm = np.clip(chroma / 130.0, 0.0, 1.0)

    # Alpha-like: high L + low chroma = high transparency
    alpha_like = 0.6 * L_norm + 0.4 * (1.0 - chroma_norm)

    return {
        "lab_std": [round(float(lab_std[0]), 2), round(float(lab_std[1]), 2), round(float(lab_std[2]), 2)],
        "p10": [round(float(p10[0]), 2), round(float(p10[1]), 2), round(float(p10[2]), 2)],
        "p90": [round(float(p90[0]), 2), round(float(p90[1]), 2), round(float(p90[2]), 2)],
        "compactness": round(float(compactness), 3),
        "alpha_like": round(float(alpha_like), 3),
    }


def separation_ab(mean_labs: List[np.ndarray], area_ratios: List[float]) -> Dict[str, float]:
    if len(mean_labs) < 2:
        return {"min_deltaE": 0.0, "mean_deltaE": 0.0}
    ab = np.array([[m[1], m[2]] for m in mean_labs], dtype=np.float32)
    dists = []
    weights = []
    for i in range(len(ab)):
        for j in range(i + 1, len(ab)):
            d = np.linalg.norm(ab[i] - ab[j])
            dists.append(float(d))
            weights.append(float(min(area_ratios[i], area_ratios[j])))
    if not dists:
        return {"min_deltaE": 0.0, "mean_deltaE": 0.0}
    w = np.array(weights, dtype=np.float32)
    w_sum = float(np.sum(w)) if w.size else 0.0
    if w_sum > 0:
        mean_w = float(np.sum(np.array(dists, dtype=np.float32) * w) / w_sum)
    else:
        mean_w = float(np.mean(dists))
    return {"min_deltaE": float(min(dists)), "mean_deltaE": mean_w}


def build_cluster_stats(
    lab_samples: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    deltaE_method: str = "76",
    separation_d0: float = 3.0,
    separation_k: float = 1.0,
) -> Dict[str, Any]:
    means = _mean_lab(lab_samples, labels, k)
    ratios = _area_ratios(labels, k)
    clusters = []
    for i in range(k):
        mean_lab = means[i]
        rgb = _lab_to_rgb(mean_lab)

        # Calculate distribution metrics (NEW!)
        distribution = _cluster_distribution(lab_samples, labels, i)

        clusters.append(
            {
                "area_ratio": float(ratios[i]),
                "mean_lab": mean_lab.tolist(),
                "mean_ab": [float(mean_lab[1]), float(mean_lab[2])],
                "mean_rgb": rgb,
                "mean_hex": _rgb_to_hex(rgb),
                # NEW: Distribution width information
                "lab_std": distribution["lab_std"],
                "p10": distribution["p10"],
                "p90": distribution["p90"],
                "compactness": distribution["compactness"],
                # NEW: Role discrimination metrics
                "alpha_like": distribution["alpha_like"],
            }
        )
    sep = separation_ab(means, ratios)
    sep_margin = (sep["min_deltaE"] - float(separation_d0)) / max(float(separation_k), 1e-6)

    # NEW: Calculate pairwise ΔE matrix for recipe estimation
    pairwise_deltaE = calculate_pairwise_deltaE(means, method=deltaE_method)

    min_area = float(min(ratios)) if ratios else 0.0
    return {
        "clusters": clusters,
        "quality": {
            "min_area_ratio": min_area,
            "min_deltaE_between_clusters": sep["min_deltaE"],
            "mean_deltaE_between_clusters": sep["mean_deltaE"],
            "deltaE_method": str(deltaE_method),
            "separation_margin": float(sep_margin),
        },
        "pairwise_deltaE": pairwise_deltaE,
    }


def silhouette_ab_proxy(lab_samples: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k < 2 or lab_samples.size == 0 or labels.size == 0:
        return 0.0
    means = _mean_lab(lab_samples, labels, k)
    centers = np.array([[m[1], m[2]] for m in means], dtype=np.float32)
    ab = lab_samples[:, 1:3].astype(np.float32)
    if centers.size == 0 or ab.size == 0:
        return 0.0
    dists = np.linalg.norm(ab[:, None, :] - centers[None, :, :], axis=2)
    own = dists[np.arange(dists.shape[0]), labels]
    other = dists.copy()
    other[np.arange(dists.shape[0]), labels] = np.inf
    nearest = np.min(other, axis=1)
    denom = np.maximum(own, nearest)
    safe = denom > 1e-6
    s = np.zeros_like(own)
    s[safe] = (nearest[safe] - own[safe]) / denom[safe]
    return float(np.mean(s))


def _lab_to_rgb(mean_lab: np.ndarray) -> List[int]:
    """Convert CIE Lab ([L*, a*, b*]) to sRGB [R,G,B] for display."""
    L, a, b = float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])

    # Lab -> XYZ (D65)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t: float) -> float:
        t3 = t * t * t
        return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xr = _f_inv(fx)
    yr = _f_inv(fy)
    zr = _f_inv(fz)

    # Reference white (D65)
    X = 0.95047 * xr
    Y = 1.00000 * yr
    Z = 1.08883 * zr

    # XYZ -> linear sRGB
    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def _gamma(u: float) -> float:
        if u <= 0.0031308:
            return 12.92 * u
        return 1.055 * (u ** (1.0 / 2.4)) - 0.055

    R = _gamma(max(0.0, min(1.0, r_lin)))
    G = _gamma(max(0.0, min(1.0, g_lin)))
    B = _gamma(max(0.0, min(1.0, b_lin)))

    return [int(round(R * 255)), int(round(G * 255)), int(round(B * 255))]


def _rgb_to_hex(rgb: List[int]) -> str:
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """ΔE76 (Euclidean distance in Lab)."""
    d = lab1.astype(np.float32) - lab2.astype(np.float32)
    return float(np.linalg.norm(d))


def _deltaE2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """
    CIEDE2000 ΔE00 implementation.
    Verified against the standard reference example: ~2.0425.
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0

    Cbar7 = Cbar**7
    G = 0.0
    if Cbar > 0:
        G = 0.5 * (1.0 - math.sqrt(Cbar7 / (Cbar7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def _hp(ap: float, b: float) -> float:
        if ap == 0.0 and b == 0.0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180.0:
            dhp = dh
        elif dh > 180.0:
            dhp = dh - 360.0
        else:
            dhp = dh + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        hbarp = h1p + h2p
    else:
        diff = abs(h1p - h2p)
        if diff <= 180.0:
            hbarp = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360.0:
            hbarp = (h1p + h2p + 360.0) / 2.0
        else:
            hbarp = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hbarp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hbarp))
        + 0.32 * math.cos(math.radians(3.0 * hbarp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hbarp - 63.0))
    )

    dtheta = 30.0 * math.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Rc = 0.0
    if Cbarp > 0:
        Rc = 2.0 * math.sqrt((Cbarp**7) / ((Cbarp**7) + (25.0**7)))

    Sl = 1.0 + (0.015 * ((Lbarp - 50.0) ** 2)) / math.sqrt(20.0 + ((Lbarp - 50.0) ** 2))
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T

    Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def calculate_pairwise_deltaE(mean_labs: List[np.ndarray], *, method: str = "76") -> List[List[float]]:
    """
    Calculate pairwise ΔE matrix between all cluster centroids.

    Args:
        mean_labs: List of Lab centroids [L, a, b] for each cluster
        method: "76" or "2000" (default "76")

    Returns:
        k x k matrix where matrix[i][j] = ΔE between cluster i and j
        Diagonal is 0.0, symmetric matrix
    """
    k = len(mean_labs)
    if k == 0:
        return []

    m = str(method).strip().lower()
    if m in ("2000", "de2000", "ciede2000", "cie2000"):
        fn = _deltaE2000
    else:
        fn = _deltaE76

    matrix = []
    for i in range(k):
        row = []
        lab_i = np.array(mean_labs[i], dtype=np.float32)
        for j in range(k):
            if i == j:
                row.append(0.0)
            else:
                lab_j = np.array(mean_labs[j], dtype=np.float32)
                delta_e = fn(lab_i, lab_j)
                row.append(round(delta_e, 2))
        matrix.append(row)

    return matrix


def calculate_radial_presence_curve(
    labels: np.ndarray, cluster_id: int, polar_r: np.ndarray, r_bins: int = 10
) -> List[float]:
    """
    Calculate radial distribution curve for a cluster.

    Args:
        labels: Cluster labels (flattened, same shape as polar_r when flattened)
        cluster_id: Which cluster to analyze
        polar_r: Normalized radial distance [0-1] for each pixel (H x W)
        r_bins: Number of radial bins

    Returns:
        List of area ratios per radial bin [0.0 ~ 1.0]
    """
    if polar_r.size == 0 or labels.size == 0:
        return [0.0] * r_bins

    # Flatten polar_r to match labels
    polar_r_flat = polar_r.flatten()

    if polar_r_flat.shape[0] != labels.shape[0]:
        # Size mismatch, return zeros
        return [0.0] * r_bins

    cluster_mask = labels == cluster_id
    total_cluster_pixels = np.sum(cluster_mask)

    if total_cluster_pixels == 0:
        return [0.0] * r_bins

    curve = []
    for i in range(r_bins):
        r_start = i / r_bins
        r_end = (i + 1) / r_bins

        # Pixels in this radial bin
        bin_mask = (polar_r_flat >= r_start) & (polar_r_flat < r_end)

        # Cluster pixels in this bin
        bin_cluster_pixels = np.sum(cluster_mask & bin_mask)

        # Ratio relative to total cluster pixels
        bin_ratio = float(bin_cluster_pixels / total_cluster_pixels)
        curve.append(round(bin_ratio, 3))

    return curve


def calculate_spatial_prior(radial_presence_curve: List[float]) -> float:
    """
    Calculate spatial prior score from radial presence curve.

    Real ink typically concentrates in mid-to-outer radial zones (bins 3-8).
    Background/gaps tend to be uniform or concentrated at center/far edge.

    Args:
        radial_presence_curve: List of area ratios per radial bin [0.0 ~ 1.0]

    Returns:
        Spatial prior score (0-1, higher = more ink-like spatial pattern)
    """
    if not radial_presence_curve or len(radial_presence_curve) == 0:
        return 0.0

    curve = np.array(radial_presence_curve, dtype=np.float32)
    n_bins = len(curve)

    if n_bins < 3:
        return 0.0

    # Define ink-preferred zone (middle 60% of radius)
    ink_zone_start = int(n_bins * 0.2)  # Skip inner 20%
    ink_zone_end = int(n_bins * 0.8)  # Skip outer 20%

    # Calculate concentration in ink zone
    ink_zone_ratio = float(np.sum(curve[ink_zone_start:ink_zone_end]))

    # Calculate uniformity penalty (high std = scattered = less ink-like)
    uniformity = float(np.std(curve))
    uniformity_penalty = np.clip(uniformity * 2.0, 0.0, 0.3)

    # Spatial prior: high if concentrated in ink zone
    spatial_prior = ink_zone_ratio - uniformity_penalty
    spatial_prior = np.clip(spatial_prior, 0.0, 1.0)

    return round(float(spatial_prior), 3)


def calculate_inkness_score(mean_lab: Any, compactness: float, alpha_like: float, spatial_prior: float) -> float:
    """
    Calculate composite inkness score combining multiple factors.

    Higher score = more ink-like (opaque, colorful, tight cluster, good location)
    Lower score = more gap/background-like (transparent, neutral, scattered)

    Args:
        mean_lab: Cluster centroid [L, a, b]
        compactness: Cluster compactness (0-1)
        alpha_like: Transparency-like score (0-1, high = transparent)
        spatial_prior: Location-based score (0-1, high = ink-like location)

    Returns:
        Inkness score (0-1, higher = more ink-like)
    """
    lab = ensure_cie_lab(np.asarray(mean_lab))
    # Chroma (colorfulness)
    a = float(lab[1])
    b = float(lab[2])
    chroma = np.sqrt(a**2 + b**2)
    chroma_norm = np.clip(chroma / 130.0, 0.0, 1.0)  # Normalize to 0-1

    # Opacity (inverse of alpha_like)
    opacity = 1.0 - alpha_like

    # Weighted combination of factors
    # Weights: chroma 30%, opacity 30%, compactness 20%, spatial 20%
    inkness = 0.30 * chroma_norm + 0.30 * opacity + 0.20 * compactness + 0.20 * spatial_prior

    inkness = np.clip(inkness, 0.0, 1.0)
    return round(float(inkness), 3)
