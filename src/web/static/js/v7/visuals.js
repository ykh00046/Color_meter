(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

v7.render.ACTION_GUIDES = {
    OK: "정상입니다. 결과가 안정적입니다.",
    RETAKE: "재촬영이 필요합니다. 촬영 조건을 확인해 다시 시도하세요.",
    NG_PATTERN: "패턴 불량이 의심됩니다. 렌즈/도트를 확인해 주세요.",
    NG_COLOR: "색상 불량이 의심됩니다. 잉크 상태를 확인해 주세요.",
    DEFAULT: "판정 정보를 확인해 추가 조치가 필요합니다.",
};

v7.render.getActionGuide = (label, codes) => {
    let guide = v7.render.ACTION_GUIDES[label] || v7.render.ACTION_GUIDES.DEFAULT;
    if (codes && codes.length > 0) guide += `\n(추가 사유: ${codes[0]})`;
    return guide;
};

v7.render.renderJudgmentBar = (decision, data, v2Diag) => {
    const label = decision?.label || "UNKNOWN";
    const ops = decision?.ops || {};
    const finalJudgment = ops.judgment || label;
    const grade = ops.quality_grade || "-";

    // Policy Snapshot Data
    const policy = decision?.ops?.full_v2_decision?.policy_snapshot || {};
    const inkPolicy = policy.inkness || {};
    const dePolicy = policy.deltaE || {};

    const qualityLevel = inkPolicy.quality_level || "unknown";
    const adjustment = inkPolicy.adjustment || 0.0;
    const adjustReason = inkPolicy.reason || dePolicy.reason || "";

    const card = v7.utils.byId("inspSummaryCard");
    const labelEl = v7.utils.byId("inspLabel");
    const versionEl = v7.utils.byId("inspVersionBadge");
    const uncertainEl = v7.utils.byId("inspUncertainBadge");
    const line1 = v7.utils.byId("inspSummaryLine");

    if (card && labelEl) {
        // Display Grade + Judgment + Quality Badges
        let qualityBadge = "";
        if (qualityLevel !== "unknown") {
            const qColor = {good: "text-emerald-400 border-emerald-500/30", medium: "text-amber-400 border-amber-500/30", poor: "text-rose-400 border-rose-500/30", very_poor: "text-rose-500 border-rose-600"}[qualityLevel] || "text-text-dim";
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
        // Use Action Guide if available
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
};

v7.render.renderEngineerKPI = (ops) => {
    const kpi = ops?.engineer_kpi;
    const panel = v7.utils.byId("engineerKpiPanel");
    const content = v7.utils.byId("kpiContent");

    if (!kpi || !panel || !content) {
        if(panel) panel.classList.add("hidden");
        return;
    }

    panel.classList.remove("hidden");
    content.innerHTML = "";

    // 1. QC
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

    // 2. Pattern
    if (kpi.pattern) {
        content.innerHTML += `
            <div class="space-y-2">
                <p class="text-xs font-bold text-text-dim uppercase tracking-wider border-b border-white/5 pb-1">PATTERN</p>
                <div class="grid grid-cols-2 gap-2 text-xs font-mono">
                    <span class="text-text-secondary">Correlation</span><span class="text-white text-right">${kpi.pattern.correlation?.toFixed(3) ?? "-"}</span>
                    <span class="text-text-secondary">Ring Contrast</span><span class="text-white text-right">${kpi.pattern.ring_contrast?.toFixed(3) ?? "-"}</span>
                    <span class="text-text-secondary">Ang. Unif</span><span class="text-white text-right">${kpi.pattern.angular_uniformity?.toFixed(3) ?? "-"}</span>
                </div>
            </div>`;
    }

    // 3. Ink
    if (kpi.ink) {
        const ratios = (kpi.ink.ink_area_ratios || []).map(r => (r*100).toFixed(1) + "% ").join(", ");
        content.innerHTML += `
            <div class="space-y-2">
                <p class="text-xs font-bold text-text-dim uppercase tracking-wider border-b border-white/5 pb-1">INK STATS</p>
                <div class="grid grid-cols-2 gap-2 text-xs font-mono">
                    <span class="text-text-secondary">Count</span><span class="text-white text-right">${kpi.ink.detected_count ?? "-"}</span>
                    <span class="text-text-secondary">Cluster Conf</span><span class="text-white text-right">${kpi.ink.clustering_confidence?.toFixed(2) ?? "-"}</span>
                    <span class="text-text-secondary col-span-2 truncate" title="${ratios}">Areas: ${ratios}</span>
                </div>
            </div>`;
    }

    // 4. Defect
    if (kpi.defect) {
        content.innerHTML += `
            <div class="space-y-2">
                <p class="text-xs font-bold text-text-dim uppercase tracking-wider border-b border-white/5 pb-1">DEFECTS</p>
                <div class="grid grid-cols-2 gap-2 text-xs font-mono">
                    <span class="text-text-secondary">Blob Count</span><span class="text-white text-right">${kpi.defect.blob_count ?? "-"}</span>
                    <span class="text-text-secondary">Total Area</span><span class="text-white text-right">${kpi.defect.blob_total_area ?? "-"} px</span>
                </div>
            </div>`;
    }
};

v7.render.renderReasonCards = (decision, v2Diag) => {
    const container = v7.utils.byId("inspReasonCards");
    if (!container) return;
    const cards = [];

    // 1. GATE
    const gate = decision?.gate;
    const center = gate?.scores?.center_offset_ratio;

    // Extract Guidance
    const guidance = gate?.scores?._guidance || {};
    let gateMeta = `Offset ${center != null ? center.toFixed(3) : "-"}`;
    if (Object.keys(guidance).length > 0 && !guidance.status) {
        const firstAction = Object.values(guidance)[0];
        gateMeta = firstAction.split('(')[0].trim() || " 촬영 상태 확인";
    }

    cards.push({ id: "card-gate", mode: "gate", title: "GATE CHECK", status: gate?.passed ? "PASS" : gate ? "FAIL" : "SKIP", reasons: gate?.reasons || [], meta: gateMeta });

    // 2. PATTERN & COLOR (Integrated)
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
    cards.push({ id: "card-pattern", mode: "pattern", title: "PATTERN & COLOR", status: patternStatus, reasons: [...sigReasons, ...inkReasons], meta: `Corr ${corr} / Off ${off != null ? off.toFixed(2) : "-"}` });

    container.innerHTML = "";
    cards.forEach((item) => {
        const card = document.createElement("div");
        card.id = item.id;
        card.className = "reason-card space-y-3 cursor-pointer transition-all hover:bg-slate-800 border-2 border-transparent";
        card.onclick = () => v7.inspection.applyInspectionMode(item.mode);
        const reasonCode = item.reasons?.[0];
        const info = v7.utils.getReasonInfo(reasonCode);
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
};

v7.render.renderInkComparisonPalette = (v2Diag) => {
    const wrap = v7.utils.byId("inkComparisonPalette");
    const empty = v7.utils.byId("inkPaletteEmpty");
    if (!wrap || !empty) return;

    const clusters = v2Diag?.segmentation?.clusters || [];
    const inkMatch = v2Diag?.ink_match || {};
    // Patch: Backend returns 'deltas', but older code might look for 'cluster_deltas'
    const clusterDeltas = inkMatch.deltas || inkMatch.cluster_deltas || [];

    wrap.innerHTML = "";
    if (!clusters.length) { empty.classList.remove("hidden"); return; }
    empty.classList.add("hidden");

    clusters.forEach((c, idx) => {
        const delta = clusterDeltas[idx] || {};
        const row = document.createElement("div");
        row.className = "ink-row flex-col items-stretch space-y-3 group relative cursor-help transition-colors hover:bg-white/5";

        const stdHex = c.mean_hex || "#333";
        const sampleHex = delta.sample_hex || stdHex;
        const dE = delta.deltaE != null ? delta.deltaE.toFixed(1) : "-";
        const dL = delta.delta_L != null ? delta.delta_L.toFixed(1) : "0.0";
        const role = c.role ? c.role.toUpperCase() : `INK ${idx+1}`;

        // Deep Metrics
        const inkness = c.inkness_score != null ? (c.inkness_score * 100).toFixed(0) : "-";
        const compactness = c.compactness != null ? c.compactness.toFixed(2) : "-";
        const stdDev = c.lab_std != null ? Math.max(...c.lab_std).toFixed(1) : "-"; // Max std across L,a,b

        // Tooltip Content
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
                    <div class="flex justify-between"><span>Area</span><span class="text-white">${(c.area_ratio*100).toFixed(1)}%</span></div>
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
};

v7.render.renderKeyMetrics = (decision, resultItem, cfg) => {
    const sig = decision?.signature || {};
    const gate = decision?.gate || {};
    const anomaly = decision?.anomaly || {};
    const diag = decision?.diagnostics || {};
    const patternColor = decision?.pattern_color || {};
    const color = patternColor.color_histogram?.direction || diag.color?.overall?.direction || {};
    const dot = diag.pattern?.dot || {};

    const updateMetric = (id, val) => { const el = v7.utils.byId(id); if(el) el.textContent = val ?? "-"; };
    updateMetric("metric-best-mode", resultItem?.decision?.best_mode ? `MODE ${resultItem.decision.best_mode}` : "-");
    updateMetric("metric-corr", sig.score_corr?.toFixed(3));
    updateMetric("metric-de-mean", sig.delta_e_mean?.toFixed(2));
    updateMetric("metric-de-p95", sig.delta_e_p95?.toFixed(2));
    updateMetric("metric-fail", sig.fail_ratio?.toFixed(2));
    updateMetric("metric-center", gate?.scores?.center_offset_ratio?.toFixed(3));

    // Show Sharpness + Guidance
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

    const colorCard = v7.utils.byId("detailColor");
    if(colorCard) colorCard.classList.toggle("hidden", !(color.delta_L != null || color.delta_a != null || color.delta_b != null));
};

v7.render.resetInspectionUI = () => {
    const resetText = (id, text = "-") => { const el = v7.utils.byId(id); if(el) el.textContent = text; };
    resetText("inspLabel", "RUNNING...");
    v7.utils.byId("inspSummaryCard")?.setAttribute("class", "terminal-panel p-4 flex flex-col md:flex-row items-center justify-between gap-4 border-l-4 border-slate-600");
    const metrics = ["metric-best-mode", "metric-corr", "metric-de-mean", "metric-de-p95", "metric-fail", "metric-center", "metric-blur", "metric-blur-delta", "metric-illum", "metric-angular", "metric-dot-count", "metric-coverage", "metric-coverage-delta", "metric-sharpness", "metric-sharpness-delta", "metric-delta-l", "metric-delta-a", "metric-delta-b", "v2-ink-count", "v2-suggested-k", "v2-confidence", "v2-offtrack", "v2-ontrack", "v2-warnings", "trajOffTrackNumber", "trajOffTrackValue"];
    metrics.forEach(id => resetText(id));
    const img = v7.utils.byId("inspMainImg"); if(img) img.src = "";
    v7.utils.byId("inspImageEmpty")?.classList.remove("hidden");
    const canvas = v7.utils.byId("hotspotOverlay"); if(canvas) canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    v7.utils.byId("inspReasonCards").innerHTML = "";
    v7.utils.byId("inkComparisonPalette").innerHTML = "";
    const kpiPanel = v7.utils.byId("engineerKpiPanel");
    if (kpiPanel) kpiPanel.classList.add("hidden");
};

v7.render.renderColorChart = (diagnostics) => {
    const canvas = v7.utils.byId("colorChart");
    if (!canvas) return;
    if (v7.state.colorChartInstance) v7.state.colorChartInstance.destroy();
    const overall = diagnostics?.color?.overall?.direction || { delta_L: 0, delta_a: 0, delta_b: 0 };
    try {
        v7.state.colorChartInstance = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: ["L", "a", "b"],
                datasets: [{
                    data: [overall.delta_L, overall.delta_a, overall.delta_b],
                    backgroundColor: [overall.delta_L > 0 ? "#f8fafc" : "#475569", overall.delta_a > 0 ? "#ef4444" : "#22c55e", overall.delta_b > 0 ? "#eab308" : "#3b82f6"],
                    borderRadius: 4
                }]
            },
            options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", size: 10 } }, y: { grid: { display: false }, ticks: { color: "#94a3b8" } } } }
        });
    } catch (e) { console.error(e); }
};

v7.render.renderOperatorPanels = (dec, v2Diag, v2Meta, cfg, result, resItem) => {
    // Legacy support or specific operator panel rendering if needed
};

v7.render.renderTrajectoryDashboard = (v2Diag, cfg) => {
    const summary = v2Diag?.ink_match?.trajectory_summary || {};
    const off = summary.max_off_track;
    v7.utils.safeText("trajOffTrackNumber", off != null ? off.toFixed(2) : "-");
    v7.utils.safeText("trajOffTrackValue", off != null ? `Deviation: ${off.toFixed(3)}` : "-");
    const fill = v7.utils.byId("trajGaugeFill");
    if (fill) fill.style.width = `${Math.min((off || 0) * 10, 100)}%`;
    const badge = v7.utils.byId("trajStatusBadge");
    if (badge) {
        const ok = (off || 0) < 5.0;
        badge.textContent = ok ? "STABLE" : "SHIFTED";
        badge.className = `text-xs font-bold px-2 py-0.5 rounded ${ok ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`;
    }
};

v7.render.buildInspCommentary = (decision, v2Diag, data) => {
    const label = decision?.label || "UNKNOWN";
    return `Final Judgment: ${label} · Analysis Complete.`;
};

v7.render.renderRegFailures = (validation) => {
    const panel = v7.utils.byId("regFailurePanel");
    const list = v7.utils.byId("regFailureList");
    if (!panel || !list) return;
    const items = Array.isArray(validation?.results) ? validation.results : [];
    const failures = items.filter((item) => item?.decision?.label && item.decision.label !== "STD_ACCEPTABLE");
    if (!failures.length) { panel.classList.add("hidden"); list.innerHTML = ""; return; }
    list.innerHTML = failures.map(item => {
        const reasons = (item.decision?.reason_codes || item.decision?.reason_messages || []).slice(0, 4);
        return `<div class="p-4 rounded-xl border border-red-error/40 bg-red-error/10 space-y-2"><div class="flex items-center justify-between"><span class="text-xs font-mono text-red-error">${item.decision?.label || "FAILED"}</span><span class="text-[10px] text-text-dim font-mono">${item.path ? item.path.split(/[\\/]/).pop() : "-"}</span></div><ul class="list-disc list-inside text-xs text-text-secondary space-y-1">${reasons.map((r) => `<li>${r}</li>`).join("") || "<li>Unknown failure</li>"}</ul></div>`;
    }).join("");
    panel.classList.remove("hidden");
};

})();
