/**
 * Diagnostics Visualization Components
 * Migrated from v7/diagnostics_visuals.js (125 lines)
 */

import { byId, safeText } from '../../utils/helpers.js';

/**
 * Render trajectory dashboard
 */
export function renderTrajectoryDashboard(v2Diag, cfg) {
    const summary = v2Diag?.ink_match?.trajectory_summary || {};
    const off = summary.max_off_track;

    safeText("trajOffTrackNumber", off != null ? off.toFixed(2) : "-");
    safeText("trajOffTrackValue", off != null ? `Deviation: ${off.toFixed(3)}` : "-");

    const fill = byId("trajGaugeFill");
    if (fill) fill.style.width = `${Math.min((off || 0) * 10, 100)}%`;

    const badge = byId("trajStatusBadge");
    if (badge) {
        const ok = (off || 0) < 5.0;
        badge.textContent = ok ? "STABLE" : "SHIFTED";
        badge.className = `text-xs font-bold px-2 py-0.5 rounded ${ok ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`;
    }
}

/**
 * Render V3 summary
 */
export function renderV3Summary(v3Summary) {
    if (!v3Summary) return;

    const card = byId("v3SummaryCard");
    const content = byId("v3SummaryContent");
    if (!card || !content) return;

    card.classList.remove("hidden");
    content.innerHTML = `
        <div class="text-sm text-text-secondary">
            <p>V3 Summary: ${v3Summary.status || 'N/A'}</p>
        </div>
    `;
}

/**
 * Render V3 trend
 */
export function renderV3Trend(v3Trend) {
    if (!v3Trend) return;

    const card = byId("v3TrendCard");
    const content = byId("v3TrendContent");
    if (!card || !content) return;

    card.classList.remove("hidden");
    content.innerHTML = `
        <div class="text-sm text-text-secondary">
            <p>Trend data available</p>
        </div>
    `;
}

/**
 * Render ops judgment
 */
export function renderOpsJudgment(ops) {
    if (!ops) return;

    const card = byId("opsJudgmentCard");
    const content = byId("opsJudgmentContent");
    if (!card || !content) return;

    card.classList.remove("hidden");
    const judgment = ops.judgment || "-";
    const grade = ops.quality_grade || "-";

    content.innerHTML = `
        <div class="space-y-2">
            <div class="flex justify-between">
                <span class="text-text-dim">Judgment:</span>
                <span class="font-bold">${judgment}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-text-dim">Quality:</span>
                <span class="font-bold">${grade}</span>
            </div>
        </div>
    `;
}

/**
 * Render worst case
 */
export function renderWorstCase(worstCase) {
    if (!worstCase) return;

    const card = byId("worstCaseCard");
    const content = byId("worstCaseContent");
    if (!card || !content) return;

    card.classList.remove("hidden");
    content.innerHTML = `
        <div class="text-sm text-text-secondary">
            <p>Worst case: ΔE ${worstCase.value?.toFixed(2) || 'N/A'}</p>
        </div>
    `;
}

/**
 * Draw worst case mini (canvas)
 */
export function drawWorstCaseMini(worstCase, cfg) {
    const canvas = byId("worstCaseMini");
    if (!canvas || !worstCase) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Simple visualization placeholder
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(10, 10, 20, 20);
}

/**
 * Draw hotspot overlay (canvas)
 */
export function drawHotspotOverlay(worstCase, cfg) {
    const canvas = byId("hotspotOverlay");
    if (!canvas || !worstCase) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Hotspot visualization placeholder
    if (worstCase.x && worstCase.y) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.beginPath();
        ctx.arc(worstCase.x, worstCase.y, 15, 0, 2 * Math.PI);
        ctx.fill();
    }
}

/**
 * Render engineer KPI
 */
export function renderEngineerKPI(ops) {
    const kpi = ops?.engineer_kpi;
    const panel = byId("engineerKpiPanel");
    const content = byId("kpiContent");

    if (!kpi || !panel || !content) {
        if (panel) panel.classList.add("hidden");
        return;
    }

    panel.classList.remove("hidden");
    content.innerHTML = "";

    // QC
    if (kpi.qc) {
        content.innerHTML += `
            <div class="space-y-2">
                <p class="text-xs font-bold text-text-dim uppercase tracking-wider border-b border-white/5 pb-1">QC METRICS</p>
                <div class="grid grid-cols-2 gap-2 text-xs font-mono">
                    <span class="text-text-secondary">Sharpness</span><span class="text-white text-right">${kpi.qc.sharpness?.toFixed(1) ?? "-"}</span>
                    <span class="text-text-secondary">Offset</span><span class="text-white text-right">${kpi.qc.center_offset_mm?.toFixed(2) ?? "-"} mm</span>
                    <span class="text-text-secondary">Illum Asym</span><span class="text-white text-right">${kpi.qc.illumination_asymmetry?.toFixed(3) ?? "-"}</span>
                </div>
            </div>`;
    }
}

export default {
    renderTrajectoryDashboard,
    renderV3Summary,
    renderV3Trend,
    renderOpsJudgment,
    renderWorstCase,
    drawWorstCaseMini,
    drawHotspotOverlay,
    renderEngineerKPI
};
