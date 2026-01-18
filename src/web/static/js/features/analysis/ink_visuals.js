/**
 * Ink Visualization Components
 * Migrated from v7/ink_visuals.js (67 lines)
 */

import { byId } from '../../utils/helpers.js';
import { labToHex } from '../../utils/helpers.js';

/**
 * Render ink comparison palette
 */
export function renderInkComparisonPalette(v2Diag) {
    const wrap = byId("inkComparisonPalette");
    const empty = byId("inkPaletteEmpty");
    if (!wrap || !empty) return;

    const clusters = v2Diag?.segmentation?.clusters || [];
    const inkMatch = v2Diag?.ink_match || {};
    const clusterDeltas = inkMatch.deltas || inkMatch.cluster_deltas || [];

    wrap.innerHTML = "";
    if (!clusters.length) {
        empty.classList.remove("hidden");
        return;
    }
    empty.classList.add("hidden");

    clusters.forEach((c, idx) => {
        const delta = clusterDeltas[idx] || {};
        const row = document.createElement("div");
        row.className = "ink-row flex-col items-stretch space-y-3 group relative cursor-help transition-colors hover:bg-white/5";

        const stdHex = c.mean_hex || "#333";
        const sampleHex = delta.sample_hex || stdHex;
        const dE = delta.deltaE != null ? delta.deltaE.toFixed(1) : "-";
        const dL = delta.delta_L != null ? delta.delta_L.toFixed(1) : "0.0";
        const role = c.role ? c.role.toUpperCase() : `INK ${idx + 1}`;

        const inkness = c.inkness_score != null ? (c.inkness_score * 100).toFixed(0) : "-";
        const compactness = c.compactness != null ? c.compactness.toFixed(2) : "-";
        const stdDev = c.lab_std != null ? Math.max(...c.lab_std).toFixed(1) : "-";

        row.title = `[Deep Analysis]\nInkness Score: ${inkness}%\nCompactness: ${compactness}\nColor Uniformity (Std): ${stdDev}\nSpatial Prior: ${(c.spatial_prior || 0).toFixed(2)}`;

        row.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-bold text-brand-500 tracking-widest">${role}</span>
                    ${c.inkness_score != null ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-mono ${c.inkness_score > 0.6 ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700 text-text-dim'}">INK ${inkness}%</span>` : ''}
                </div>
                <span class="text-[10px] font-mono text-text-dim">ΔE ${dE} [CIE]</span>
            </div>
            <div class="flex items-center gap-4">
                <div class="ink-swatch-split shadow-lg w-10 h-10 rounded-md overflow-hidden flex border border-white/10">
                    <div class="ink-swatch-half flex-1 h-full" style="background:${stdHex}" title="STD Hex: ${stdHex}"></div>
                    <div class="ink-swatch-half flex-1 h-full" style="background:${sampleHex}" title="Sample Hex: ${sampleHex}"></div>
                </div>
                <div class="flex-1 grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono">
                    <div class="flex justify-between"><span>Area</span><span class="text-white">${(c.area_ratio * 100).toFixed(1)}%</span></div>
                    <div class="flex justify-between"><span>L-Shift</span><span class="${parseFloat(dL) > 0 ? 'text-emerald-400' : 'text-rose-400'}">${dL > 0 ? '+' : ''}${dL}</span></div>
                    <div class="col-span-2 text-[9px] text-text-dim mt-1 border-t border-white/5 pt-1 flex justify-between">
                        <span>Uniformity: ${stdDev}</span>
                        <span>Comp: ${compactness}</span>
                    </div>
                </div>
            </div>
        `;
        wrap.appendChild(row);
    });
}

export default { renderInkComparisonPalette };
