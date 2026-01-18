/**
 * Visualization Components - Main Rendering Functions
 * Migrated from v7/visuals.js (396 lines)
 */

import { byId, safeText, getReasonInfo } from '../utils/helpers.js';
import { appState } from '../core/state.js';

// Action Guides
export const ACTION_GUIDES = {
    OK: "정상입니다. 결과가 안정적입니다.",
    RETAKE: "재촬영이 필요합니다. 촬영 조건을 확인해 다시 시도하세요.",
    NG_PATTERN: "패턴 불량이 의심됩니다. 렌즈/도트를 확인해 주세요.",
    NG_COLOR: "색상 불량이 의심됩니다. 잉크 상태를 확인해 주세요.",
    DEFAULT: "판정 정보를 확인해 추가 조치가 필요합니다.",
};

export function getActionGuide(label, codes) {
    let guide = ACTION_GUIDES[label] || ACTION_GUIDES.DEFAULT;
    if (codes && codes.length > 0) guide += `\n(추가 사유: ${codes[0]})`;
    return guide;
}

/**
 * Render judgment bar
 */
export function renderJudgmentBar(decision, data, v2Diag) {
    const label = decision?.label || "UNKNOWN";
    const ops = decision?.ops || {};
    const finalJudgment = ops.judgment || label;
    const grade = ops.quality_grade || "-";

    const policy = decision?.ops?.full_v2_decision?.policy_snapshot || {};
    const inkPolicy = policy.inkness || {};
    const dePolicy = policy.deltaE || {};

    const qualityLevel = inkPolicy.quality_level || "unknown";
    const adjustment = inkPolicy.adjustment || 0.0;
    const adjustReason = inkPolicy.reason || dePolicy.reason || "";

    const card = byId("inspSummaryCard");
    const labelEl = byId("inspLabel");
    const versionEl = byId("inspVersionBadge");
    const uncertainEl = byId("inspUncertainBadge");
    const line1 = byId("inspSummaryLine");

    if (card && labelEl) {
        let qualityBadge = "";
        if (qualityLevel !== "unknown") {
            const qColor = {
                good: "text-emerald-400 border-emerald-500/30",
                medium: "text-amber-400 border-amber-500/30",
                poor: "text-rose-400 border-rose-500/30",
                very_poor: "text-rose-500 border-rose-600"
            }[qualityLevel] || "text-text-dim";
            qualityBadge = `<span class="ml-3 px-2 py-0.5 rounded text-[10px] border ${qColor} font-mono uppercase" title="촬영 품질: ${qualityLevel}">QUAL: ${qualityLevel}</span>`;
        }

        let adjBadge = "";
        if (adjustment > 0) {
            adjBadge = `<span class="ml-2 px-2 py-0.5 rounded text-[10px] border border-amber-500/30 text-amber-400 font-mono" title="임계치 보정: +${adjustment} (${adjustReason})">ADJ +${adjustment.toFixed(2)}</span>`;
        }

        labelEl.innerHTML = `<span class="opacity-50 text-2xl mr-2">${grade}</span>${finalJudgment}${qualityBadge}${adjBadge}`;

        const baseClass = "terminal-panel p-4 flex flex-col md:flex-row items-center justify-between gap-4 border-l-4";
        if (finalJudgment === "PASS" || finalJudgment === "OK") {
            card.className = `${baseClass} border-emerald-500 bg-emerald-500/5`;
            labelEl.className = labelEl.className.replace(/text-\w+-\d+/, "text-emerald-400");
        } else if (finalJudgment === "RETAKE") {
            card.className = `${baseClass} border-amber-500 bg-amber-500/5`;
            labelEl.className = labelEl.className.replace(/text-\w+-\d+/, "text-amber-400");
        } else if (finalJudgment === "FAIL" || finalJudgment.startsWith("NG_")) {
            card.className = `${baseClass} border-rose-500 bg-rose-500/5`;
            labelEl.className = labelEl.className.replace(/text-\w+-\d+/, "text-rose-400");
        } else {
            card.className = `${baseClass} border-cyan-500 bg-cyan-500/5`;
            labelEl.className = labelEl.className.replace(/text-\w+-\d+/, "text-cyan-400");
        }
    }

    if (versionEl) {
        const bestMode = decision?.best_mode ? ` (${decision.best_mode})` : "";
        versionEl.textContent = `CORE: ${label}${bestMode}`;
    }

    const uncertain = Boolean(finalJudgment === "MANUAL_REVIEW" || v2Diag?.ink_match?.warning || (v2Diag?.warnings && v2Diag.warnings.length));
    if (uncertainEl) uncertainEl.classList.toggle("hidden", !uncertain);

    if (line1) {
        if (ops.action_guide) {
            line1.textContent = ops.action_guide;
            line1.className = "text-sm text-amber-200 font-bold mt-1";
        } else {
            const sig = decision?.signature || {};
            const corr = sig.score_corr != null ? sig.score_corr.toFixed(3) : "-";
            line1.textContent = `Correlation ${corr}`;
            line1.className = "text-sm text-text-secondary mt-1";
        }
    }
}

/**
 * Render reason cards
 */
export function renderReasonCards(decision, v2Diag) {
    const container = byId("inspReasonCards");
    if (!container) return;

    const cards = [];

    // GATE
    const gate = decision?.gate;
    const center = gate?.scores?.center_offset_ratio;
    const guidance = gate?.scores?._guidance || {};
    let gateMeta = `Offset ${center != null ? center.toFixed(3) : "-"}`;
    if (Object.keys(guidance).length > 0 && !guidance.status) {
        const firstAction = Object.values(guidance)[0];
        gateMeta = firstAction.split('(')[0].trim() || " 촬영 상태 확인";
    }

    cards.push({
        id: "card-gate",
        mode: "gate",
        title: "GATE CHECK",
        status: gate?.passed ? "PASS" : gate ? "FAIL" : "SKIP",
        reasons: gate?.reasons || [],
        meta: gateMeta
    });

    // PATTERN & COLOR
    const sig = decision?.signature;
    const sigReasons = sig?.reasons || decision?.reason_codes || [];
    const inkReasons = [];
    if (v2Diag?.warnings?.length) inkReasons.push(...v2Diag.warnings);
    if (v2Diag?.ink_match?.warning) inkReasons.push(v2Diag.ink_match.warning);

    let patternStatus = "SKIP";
    if (sig || v2Diag) {
        if (sig?.passed === false) patternStatus = "FAIL";
        else if (inkReasons.length > 0) patternStatus = "WARN";
        else patternStatus = "PASS";
    }

    const corr = sig?.score_corr != null ? sig.score_corr.toFixed(3) : "-";
    const off = v2Diag?.ink_match?.trajectory_summary?.max_off_track;
    cards.push({
        id: "card-pattern",
        mode: "pattern",
        title: "PATTERN & COLOR",
        status: patternStatus,
        reasons: [...sigReasons, ...inkReasons],
        meta: `Corr ${corr} / Off ${off != null ? off.toFixed(2) : "-"}`
    });

    container.innerHTML = "";
    cards.forEach((item) => {
        const card = document.createElement("div");
        card.id = item.id;
        card.className = "reason-card space-y-3 cursor-pointer transition-all hover:bg-slate-800 border-2 border-transparent";
        card.onclick = () => {
            // Import dynamically if needed
            if (window.v7?.inspection?.applyInspectionMode) {
                window.v7.inspection.applyInspectionMode(item.mode);
            }
        };

        const reasonCode = item.reasons?.[0];
        const info = getReasonInfo(reasonCode);
        let statusColor = "text-text-secondary";
        if (item.status === "PASS") statusColor = "text-emerald-400";
        if (item.status === "FAIL") statusColor = "text-rose-400";
        if (item.status === "WARN") statusColor = "text-amber-400";

        card.innerHTML = `
            <div class="flex items-center justify-between pointer-events-none">
                <span class="reason-title">${item.title}</span>
                <span class="reason-status ${statusColor}">${item.status}</span>
            </div>
            <div class="space-y-1 pointer-events-none">
                <div class="text-sm font-bold text-white mb-1">${item.meta}</div>
                <p class="text-xs text-text-dim truncate">${reasonCode ? info.title : "No Issues"}</p>
            </div>`;
        container.appendChild(card);
    });
}

/**
 * Render key metrics
 */
export function renderKeyMetrics(decision, resultItem, cfg) {
    const sig = decision?.signature || {};
    const gate = decision?.gate || {};
    const anomaly = decision?.anomaly || {};
    const diag = decision?.diagnostics || {};
    const patternColor = decision?.pattern_color || {};
    const color = patternColor.color_histogram?.direction || diag.color?.overall?.direction || {};
    const dot = diag.pattern?.dot || {};

    const updateMetric = (id, val) => safeText(id, val ?? "-");

    updateMetric("metric-best-mode", resultItem?.decision?.best_mode ? `MODE ${resultItem.decision.best_mode}` : "-");
    updateMetric("metric-corr", sig.score_corr?.toFixed(3));
    updateMetric("metric-de-mean", sig.delta_e_mean?.toFixed(2));
    updateMetric("metric-de-p95", sig.delta_e_p95?.toFixed(2));
    updateMetric("metric-fail", sig.fail_ratio?.toFixed(2));
    updateMetric("metric-center", gate?.scores?.center_offset_ratio?.toFixed(3));

    const blurScore = gate?.scores?.sharpness_score ?? gate?.scores?.blur_var;
    updateMetric("metric-blur", blurScore?.toFixed(1));
    const guidance = gate?.scores?._guidance || {};
    let blurMsg = dot.edge_sharpness_delta != null ? `Δ${dot.edge_sharpness_delta.toFixed(3)}` : "-";
    if (guidance.focus) blurMsg = "⚠️ 초점 확인";
    updateMetric("metric-blur-delta", blurMsg);

    updateMetric("metric-illum", gate?.scores?.illum_sym?.toFixed(3));
    updateMetric("metric-angular", anomaly?.scores?.angular_uniformity?.toFixed(3));
    updateMetric("metric-dot-count", anomaly?.debug?.blob_debug?.blob_count ?? anomaly?.scores?.center_blob_count ?? "-");
    updateMetric("metric-coverage", dot.coverage_sample?.toFixed(2));
    updateMetric("metric-coverage-delta", dot.coverage_delta_pp != null ? `Δ${dot.coverage_delta_pp.toFixed(2)}pp` : "-");
    updateMetric("metric-sharpness", dot.edge_sharpness_sample?.toFixed(3));
    updateMetric("metric-sharpness-delta", dot.edge_sharpness_delta != null ? `Δ${dot.edge_sharpness_delta.toFixed(3)}` : "-");
    updateMetric("metric-delta-l", color.delta_L?.toFixed(2));
    updateMetric("metric-delta-a", color.delta_a?.toFixed(2));
    updateMetric("metric-delta-b", color.delta_b?.toFixed(2));

    const colorCard = byId("detailColor");
    if (colorCard) colorCard.classList.toggle("hidden", !(color.delta_L != null || color.delta_a != null || color.delta_b != null));
}

/**
 * Reset inspection UI
 */
export function resetInspectionUI() {
    const resetText = (id, text = "-") => safeText(id, text);
    resetText("inspLabel", "RUNNING...");

    const card = byId("inspSummaryCard");
    if (card) {
        card.setAttribute("class", "terminal-panel p-4 flex flex-col md:flex-row items-center justify-between gap-4 border-l-4 border-slate-600");
    }

    const metrics = ["metric-best-mode", "metric-corr", "metric-de-mean", "metric-de-p95", "metric-fail", "metric-center", "metric-blur", "metric-blur-delta", "metric-illum", "metric-angular", "metric-dot-count", "metric-coverage", "metric-coverage-delta", "metric-sharpness", "metric-sharpness-delta", "metric-delta-l", "metric-delta-a", "metric-delta-b"];
    metrics.forEach(id => resetText(id));

    const img = byId("inspMainImg");
    if (img) img.src = "";
    byId("inspImageEmpty")?.classList.remove("hidden");

    const canvas = byId("hotspotOverlay");
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    const reasonCards = byId("inspReasonCards");
    if (reasonCards) reasonCards.innerHTML = "";

    const palette = byId("inkComparisonPalette");
    if (palette) palette.innerHTML = "";
}

/**
 * Render color chart (Chart.js)
 */
export function renderColorChart(diagnostics) {
    const canvas = byId("colorChart");
    if (!canvas) return;

    // Destroy existing chart
    const state = appState.getState('inspection');
    if (state?.colorChartInstance) {
        state.colorChartInstance.destroy();
    }

    const overall = diagnostics?.color?.overall?.direction || { delta_L: 0, delta_a: 0, delta_b: 0 };

    try {
        const chartInstance = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: ["L", "a", "b"],
                datasets: [{
                    data: [overall.delta_L, overall.delta_a, overall.delta_b],
                    backgroundColor: [
                        overall.delta_L > 0 ? "#f8fafc" : "#475569",
                        overall.delta_a > 0 ? "#ef4444" : "#22c55e",
                        overall.delta_b > 0 ? "#eab308" : "#3b82f6"
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                    y: { grid: { display: false }, ticks: { color: "#94a3b8" } }
                }
            }
        });

        appState.mergeState('inspection', { colorChartInstance: chartInstance });
    } catch (e) {
        console.error('Failed to render color chart:', e);
    }
}

// Additional rendering functions (simplified)
export function renderOperatorPanels() { /* Legacy support */ }
export function renderTrajectoryDashboard() { /* TODO */ }
export function renderV3Summary() { /* TODO */ }
export function renderV3Trend() { /* TODO */ }
export function renderOpsJudgment() { /* TODO */ }
export function renderWorstCase() { /* TODO */ }
export function drawWorstCaseMini() { /* TODO */ }
export function drawHotspotOverlay() { /* TODO */ }

export default {
    ACTION_GUIDES,
    getActionGuide,
    renderJudgmentBar,
    renderReasonCards,
    renderKeyMetrics,
    resetInspectionUI,
    renderColorChart,
    renderOperatorPanels,
    renderTrajectoryDashboard,
    renderV3Summary,
    renderV3Trend,
    renderOpsJudgment,
    renderWorstCase,
    drawWorstCaseMini,
    drawHotspotOverlay
};
