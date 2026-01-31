/**
 * Ink Visualization Components
 * Migrated from v7/ink_visuals.js (67 lines)
 * Phase 6: Updated to show 3-column color comparison
 * P2-UI: Added effective density display with phenomenon explanation
 */

import { byId } from '../../utils/helpers.js';
import { labToHex } from '../../utils/helpers.js';

/**
 * Get phenomenon explanation based on alpha value
 * @param {number} alpha - Alpha (transparency) value 0-1
 * @param {number} areaRatio - Area ratio 0-1
 * @returns {Object} { text, color, icon } for UI display
 */
function getAlphaPhenomenon(alpha, areaRatio) {
    if (alpha == null) return { text: "-", color: "text-slate-400", icon: "" };

    const effectiveDensity = alpha * areaRatio;

    if (alpha >= 0.8) {
        return {
            text: "진함",
            color: "text-cyan-400",
            icon: "●",
            detail: "높은 농도 → 색이 선명하게 표현됨"
        };
    } else if (alpha >= 0.6) {
        return {
            text: "보통",
            color: "text-emerald-400",
            icon: "◐",
            detail: "적정 농도 → 자연스러운 색상"
        };
    } else if (alpha >= 0.4) {
        return {
            text: "연함",
            color: "text-amber-400",
            icon: "○",
            detail: "낮은 농도 → 색이 밝게 보임"
        };
    } else {
        return {
            text: "투명",
            color: "text-rose-400",
            icon: "◯",
            detail: "매우 낮은 농도 → 하얗게/투명하게 보임"
        };
    }
}

/**
 * Render ink comparison palette with 3 color columns
 * @param {Object} v2Diag - Diagnostics data with segmentation
 * @param {Object} comparisonData - Optional color comparison data
 */
export function renderInkComparisonPalette(v2Diag, comparisonData = null) {
    const wrap = byId("inkComparisonPalette");
    const empty = byId("inkPaletteEmpty");
    if (!wrap || !empty) return;

    const clusters = v2Diag?.segmentation?.clusters || [];
    const inkMatch = v2Diag?.ink_match || {};
    const clusterDeltas = inkMatch.deltas || inkMatch.cluster_deltas || [];
    const colorComparison = comparisonData?.color_comparison || [];

    wrap.innerHTML = "";
    if (!clusters.length) {
        empty.classList.remove("hidden");
        return;
    }
    empty.classList.add("hidden");

    clusters.forEach((c, idx) => {
        const delta = clusterDeltas[idx] || {};
        const comparison = colorComparison[idx] || {};
        const row = document.createElement("div");
        row.className = "ink-row flex-col items-stretch space-y-3 group relative cursor-help transition-colors hover:bg-white/5";

        const lensHex = comparison.lens_clustering?.hex || c.mean_hex || c.hex_ref || "#333";
        const plateHex = comparison.plate_measurement?.hex || null;

        const dE = delta.deltaE != null ? delta.deltaE.toFixed(1) : "-";
        const dL = delta.delta_L != null ? delta.delta_L.toFixed(1) : "0.0";
        const role = comparison.role?.toUpperCase() || c.role?.toUpperCase() || `INK ${idx + 1}`;

        const inkness = c.inkness_score != null ? (c.inkness_score * 100).toFixed(0) : "-";
        const compactness = c.compactness != null ? c.compactness.toFixed(2) : "-";
        const stdDev = c.lab_std != null ? Math.max(...c.lab_std).toFixed(1) : "-";

        // P2-UI: Effective density and alpha
        const areaRatio = c.area_ratio || 0;
        const alpha = c.alpha_used;
        const effectiveDensity = c.effective_density;
        const phenomenon = getAlphaPhenomenon(alpha, areaRatio);

        // Format display values
        const alphaDisplay = alpha != null ? (alpha * 100).toFixed(0) : "-";
        const effDensityDisplay = effectiveDensity != null ? (effectiveDensity * 100).toFixed(1) : "-";
        const fallbackLevel = c.alpha_fallback_level || "";

        row.title = `[Deep Analysis]\nInkness Score: ${inkness}%\nCompactness: ${compactness}\nColor Uniformity (Std): ${stdDev}\nSpatial Prior: ${(c.spatial_prior || 0).toFixed(2)}\n\n[실효 커버리지]\nAlpha (투명도): ${alphaDisplay}%\nArea: ${(areaRatio * 100).toFixed(1)}%\nEffective: ${effDensityDisplay}%\n${phenomenon.detail || ""}`;

        // Build 2-column swatch HTML
        // Detect plate_lite source for label
        const plateSource = comparison.plate_measurement?.source;
        const isPlate_Lite = plateSource === "plate_lite";
        const plateLabel = isPlate_Lite ? "P-Lite" : "Plate";
        const plateLabelColor = isPlate_Lite ? "text-orange-400" : "text-amber-400";
        const plateSwatchHtml = plateHex
            ? `<div class="flex-1 h-full" style="background:${plateHex}" title="${plateLabel}: ${plateHex}"></div>`
            : `<div class="flex-1 h-full bg-slate-800 flex items-center justify-center" title="No Plate Data"><span class="text-[8px] text-slate-500">N/A</span></div>`;

        // P2-UI: Build effective density badge HTML
        const hasEffectiveDensity = effectiveDensity != null && alpha != null;
        const effDensityBadge = hasEffectiveDensity
            ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono ${phenomenon.color} bg-slate-700/50" title="${phenomenon.detail}">${phenomenon.icon} ${phenomenon.text}</span>`
            : '';

        row.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-bold text-brand-500 tracking-widest">${role}</span>
                    ${c.inkness_score != null ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono ${c.inkness_score > 0.6 ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700 text-text-dim'}">INK ${inkness}%</span>` : ''}
                    ${effDensityBadge}
                </div>
                <span class="text-[10px] font-mono text-text-dim">ΔE ${dE} [CIE]</span>
            </div>
            <div class="flex items-center gap-3">
                <!-- 2-column color swatches with labels -->
                <div class="w-[120px]">
                    <div class="flex mb-0.5">
                        <span class="flex-1 text-[8px] text-center text-cyan-400">Lens</span>
                        <span class="flex-1 text-[8px] text-center ${plateLabelColor}">${plateLabel}</span>
                    </div>
                    <div class="ink-swatch-triple shadow-lg w-[120px] h-12 rounded-md overflow-hidden flex border border-white/10">
                        <div class="flex-1 h-full" style="background:${lensHex}" title="Lens: ${lensHex}"></div>
                        ${plateSwatchHtml}
                    </div>
                </div>
                <div class="flex-1">
                    <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono">
                        <div class="flex justify-between"><span class="text-text-dim">Area</span><span class="text-white">${(areaRatio * 100).toFixed(1)}%</span></div>
                        <div class="flex justify-between"><span class="text-text-dim">L-Shift</span><span class="${parseFloat(dL) > 0 ? 'text-emerald-400' : 'text-rose-400'}">${dL > 0 ? '+' : ''}${dL}</span></div>
                        ${hasEffectiveDensity ? `
                        <div class="flex justify-between"><span class="text-text-dim">Alpha</span><span class="${phenomenon.color}">${alphaDisplay}%</span></div>
                        <div class="flex justify-between"><span class="text-text-dim font-semibold">실효</span><span class="text-white font-semibold">${effDensityDisplay}%</span></div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
        wrap.appendChild(row);
    });
}

export default { renderInkComparisonPalette };
