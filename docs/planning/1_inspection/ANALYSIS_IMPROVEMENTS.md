# Analysis & UI Improvements

Updated: 2025-12-12

This document describes the recommended enhancements to the lens color analysis pipeline and Web UI. It is intended to extend the current “analysis-first, judgment-optional” direction with higher fidelity ring/sector analysis, ink-aware metrics, and reproducible outputs.

## Goals
- Robust against rotation, local non-uniformity, and background variations.
- Ink-aware statistics that exclude transparent areas.
- Minimal manual parameters; presets and sensible defaults.
- Comparable outputs across runs, lots, and SKUs (normalized radii, same grids).
- Clear UI affordances to recompute with parameter tweaks.

## Core Improvements

### 1) Ring (Donut) Segmentation
- **Why**: Rotation-invariant, stable averages even with random dot layouts.
- **How**:
  - Use detected lens center and radius. Keep radii normalized: `r_norm = r_px / lens_radius`.
  - Detect `r_inner` / `r_outer` via gradients/ΔE peaks (with smoothing), fallback to default spans if missing.
  - Split into rings (3–4 for 3-color, 2–3 for 1–2 color). Provide presets:
    - 4-ring preset: 20–35%, 35–55%, 55–75%, 75–95%
    - 3-ring preset: 20–40%, 40–65%, 65–90%
  - Store boundaries as normalized values to compare across resolutions.

### 2) Ring + Sector Grid (Angular Segmentation)
- **Why**: Detect directional non-uniformity/tilt that radial-only misses.
- **How**:
  - Define θ bins (e.g., 12 sectors at 30° or 16 at 22.5°).
  - **Standard**: 0° = 3 o'clock (East), Clockwise direction.
  - Compute per (ring, sector): mean/σ of L*, a*, b*, ΔE to reference, min/max, pixel count.
  - Derived metrics: uniformity index (σ or max−min) per ring and per ink.
- **UI**: Toggle “Enable angular sectors” with ring/sector count controls; render heatmaps for ΔE and variance.

### 3) Ink Mask Application (Dot Fill Analysis)
- **Why**: Remove transparent/background influence; reflect real ink coverage.
- **How**:
  - Build ink mask from the image using Otsu/threshold or clustering (Lab K-means/GMM).
  - During stats, only include pixels where mask=1 within each (ring, sector).
  - Report fill rate (% of area with ink) per ring/sector and per ink cluster.
- **Fallback**: If mask confidence is low, fall back to unmasked averages and flag low confidence in the UI.

### 4) Ink Color Count & Clustering
- **Why**: Avoid manual “ink count” entry; support multi-ink lenses.
- **How**:
  - Cluster in Lab space with K-means or GMM; K ≤ 3 by default. Try K=1..3 and pick the best by inertia/BIC, or respect a user-provided K.
  - Each cluster corresponds to an ink; generate masks to compute per-ink ring/sector stats.
- **Limitations**: If clusters are too similar, expose a warning and allow manual override (e.g., force K=1 or 2).

### 5) ΔE & Reference Handling
- **Reference options**:
  - SKU zone baselines (preferred for judgment).
  - Lot or user-provided reference image(s).
  - Whole-ink average (for within-image contrast only; not for OK/NG).
- **Formula**: **Default CIEDE2000**; optionally show ΔE94/ΔE76 for compatibility.
- **Outputs**: Per ink × ring × sector: L*, a*, b*, ΔE_ref, σ(Lab), fill_rate, pixel_count.

---

## 8. API Schema Contract (New)

The backend (`/inspect`, `/recompute`) MUST return data in this structure to support the Phase 4 UI.

```json
{
  "run_id": "string (uuid)",
  "meta": {
    "image": "filename.jpg",
    "sku": "SKU001",
    "params": {
      "light_source": "D65",
      "magnification": "10x",
      "ring_count": 3,
      "sector_count": 12,
      "delta_e_method": "cie2000",
      "ink_mode": "kmeans",
      "smoothing_window": 5,
      "r_inner_manual": null,
      "r_outer_manual": null
    }
  },
  "lens_info": {
    "center_x": 512.5,
    "center_y": 512.5,
    "radius_px": 450.0,
    "r_inner_px": 132.5,
    "r_outer_px": 383.0,
    "r_norm_factor": 250.5
  },
  "analysis": {
    "profile": {
      "radius": [0.0, 0.01, "..."],
      "L_raw": [...], "a_raw": [...], "b_raw": [...],
      "L_smoothed": [...], "a_smoothed": [...], "b_smoothed": [...]
    },
    "derivatives": {
      "gradient_L": [...],
      "second_derivative_L": [...]
    },
    "angular_profile": {
      "sector_labels": ["S1(0-30)", "S2(30-60)", "..."],
      "rings": [
        {
          "ring_index": 1,
          "r_norm_start": 0.0,
          "r_norm_end": 0.33,
          "stats": { "L_mean": 70.3, "delta_e_mean": 2.3, "coverage_mean": 95.8 },
          "sectors": [
            {
              "sector_idx": 1,
              "angle_start": 0, "angle_end": 30,
              "L": 70.1, "a": 3.2, "b": 15.5,
              "delta_e": 2.5,
              "coverage": 95.5,
              "pixel_count": 1200
            }
          ]
        }
      ]
    },
    "heatmap_grid": [
      [2.5, 2.1, 3.5, "..."],
      [1.2, 1.1, 1.5, "..."]
    ],
    "boundary_candidates": [
      { "method": "gradient", "r_norm": 0.33, "confidence": 0.85 }
    ]
  },
  "overlay_image": "/results/.../overlay.png",
  "judgment": null
}
```

---

## Backend Work Items
- Extend `ProfileAnalyzer` (or new `AngularProfiler`) to:
  - Accept ring/sector binning parameters and return grid stats (mean/σ/min/max/fill_rate/pixel_count).
  - Support ink masks and optional clustering (K-means/GMM) with confidence scores.
  - Normalize radii and emit both px and normalized boundaries.
- Add `/recompute` endpoint: accepts params (rings, sectors, smoothing, thresholds, K, mask mode) and returns updated analysis without re-upload.
- Persist analysis metadata: params used, detected boundaries, cluster centers, mask confidence.
- Safeguards: fallbacks when detection fails (uniform split, single-cluster), warnings in payload.

## Frontend Work Items
- Controls: ring/sector counts, presets, smoothing/peak thresholds, mask toggle + threshold, cluster K override, “Recompute” button.
- Visuals: ring×sector heatmaps (ΔE, variance, fill-rate), canvas overlay with sector lines, clickable table to highlight overlay.
- Data tables: per ink×ring×sector metrics; export CSV/JSON.
- Reference selection: choose SKU baseline vs uploaded reference image(s); show which reference was applied.
- Error messaging: surface fallback usage (e.g., “mask low confidence; using unmasked averages”).

## Recommended Defaults
- Rings: 3 (1–2 inks) or 4 (3 inks); sectors: 12 (30°) or 16 (22.5°).
- Smoothing: Savitzky-Golay window 11, polyorder 3 (auto-clamp to data length).
- Peak detection: height auto from 75th percentile of |grad|; distance = max(1, 3% of samples).
- Mask: Otsu as baseline; cluster mode auto (K=1..3) with inertia/BIC pick; threshold slider exposed when not clustering.
- ΔE: CIE2000 default; show alternative formulas on demand.

## Validation & Testing
- Unit tests: ring/sector binning, mask application, clustering stability (seeded), normalization correctness.
- Golden cases: 1-zone, 2-zone, 3-zone lenses; dot vs. solid patterns; rotated samples to confirm rotation invariance.
- UI cypress/playwright smoke: recompute flow, overlay highlighting, heatmap rendering, preset switching.

## Rollout Plan (Suggested Order)
1) Add ring×sector stats + normalized radii in backend; expose in `/inspect` response.
2) UI heatmaps & overlay with sectors; basic controls for ring/sector counts and smoothing/peak thresholds.
3) Ink mask application (threshold/Otsu) with fill-rate reporting; UI toggle.
4) Clustering for ink count with manual override; per-ink outputs.
5) `/recompute` endpoint + UI recompute flow.
6) Reference selection UI and diff heatmaps for lot comparison.
