(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
// Diagnostics and overlays for v7 UI (globals).
v7.render.renderV3Summary = (summary) => {
    const card = v7.utils.byId("v3SummaryCard");
    const lines = v7.utils.byId("v3SummaryLines");
    const signals = v7.utils.byId("v3KeySignals");
    const meta = v7.utils.byId("v3MetaLine");
    const badge = v7.utils.byId("v3SeverityBadge");
    if (!card || !lines || !signals || !meta || !badge) return;

    if (!summary) {
        card.classList.add("hidden");
        return;
    }

    lines.innerHTML = (summary.summary || []).map((line) => `<li>${line}</li>`).join("");
    signals.innerHTML = (summary.key_signals || []).map((line) => `<li>${line}</li>`).join("");
    meta.textContent = summary.meta?.trend_line || summary.meta?.generated_at || "";

    const severity = summary.meta?.severity || "INFO";
    badge.textContent = severity;
    badge.className = `status-badge ${severity === "WARN" ? "border-yellow-warning text-yellow-warning" : "border-text-dim text-text-dim"}`;
    card.classList.remove("hidden");
};

v7.render.renderV3Trend = (trend) => {
    const card = v7.utils.byId("v3TrendCard");
    const signals = v7.utils.byId("v3TrendSignals");
    const meta = v7.utils.byId("v3TrendMeta");
    const badge = v7.utils.byId("v3TrendBadge");
    if (!card || !signals || !meta || !badge) return;

    if (!trend) {
        card.classList.add("hidden");
        return;
    }

    signals.innerHTML = (trend.signals || []).map((line) => `<li>${line}</li>`).join("");
    meta.textContent = trend.trend_line || "";
    badge.textContent = trend.meta?.confidence || "LOW";
    badge.className = "status-badge border-text-dim text-text-dim";
    card.classList.remove("hidden");
};

v7.render.drawHotspotOverlay = (worstCase, cfg) => {
    const canvas = v7.utils.byId("hotspotOverlay");
    const viewer = v7.utils.byId("analysisViewer");
    if (!canvas || !viewer) return;

    const ctx = canvas.getContext("2d");
    const rect = viewer.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!worstCase?.hotspot) {
        v7.state.worstHotspot = null;
        return;
    }

    const hotspot = worstCase.hotspot;
    const polar = cfg?.polar || {};
    const tBins = polar.T || 360;
    const rBins = polar.R || 260;
    const x = (hotspot.theta_bin / tBins) * canvas.width;
    const y = (hotspot.r_bin / rBins) * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(239, 68, 68, 0.35)";
    ctx.fill();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();

    v7.state.worstHotspot = { x, y, value: hotspot.value };
};

v7.render.drawWorstCaseMini = (worstCase, cfg) => {
    const canvas = v7.utils.byId("worstCaseMini");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(12, 18, 28, 0.9)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!worstCase?.hotspot) return;
    const polar = cfg?.polar || {};
    const x = (worstCase.hotspot.theta_bin / (polar.T || 360)) * canvas.width;
    const y = (worstCase.hotspot.r_bin / (polar.R || 260)) * canvas.height;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(239, 68, 68, 0.75)";
    ctx.fill();
};

v7.render.renderOpsJudgment = (ops) => {
    const card = v7.utils.byId("opsJudgmentCard");
    const badge = v7.utils.byId("opsJudgmentBadge");
    const reasons = v7.utils.byId("opsJudgmentReasons");
    const signals = v7.utils.byId("opsJudgmentSignals");
    if (!card || !badge || !reasons || !signals) return;

    if (!ops) {
        card.classList.add("hidden");
        return;
    }

    badge.textContent = ops.judgment || "-";
    badge.className = `status-badge ${ops.judgment === "OK" ? "border-green-ok text-green-ok" : "border-yellow-warning text-yellow-warning"}`;
    reasons.innerHTML = (ops.reasons || []).map((r) => `<li>${r}</li>`).join("");
    signals.innerHTML = (ops.top_signals || []).map((s) => `<li>${s.code}: ${typeof s.value === "string" ? s.value : JSON.stringify(s.value)}</li>`).join("");
    card.classList.remove("hidden");
};

v7.render.renderWorstCase = (worstCase) => {
    const card = v7.utils.byId("worstCaseCard");
    if (!card) return;
    if (!worstCase) {
        card.classList.add("hidden");
        v7.utils.safeText("worst-max", "-");
        v7.utils.safeText("worst-p95", "-");
        v7.utils.safeText("worst-p99", "-");
        v7.utils.safeText("worst-std", "-");
        v7.utils.safeText("worst-coverage", "-");
        v7.utils.safeText("worst-hotspot", "-");
        return;
    }
    v7.utils.safeText("worst-max", worstCase.max_deltaE != null ? worstCase.max_deltaE.toFixed(2) : "-");
    v7.utils.safeText("worst-p95", worstCase.p95_deltaE != null ? worstCase.p95_deltaE.toFixed(2) : "-");
    v7.utils.safeText("worst-p99", worstCase.p99_deltaE != null ? worstCase.p99_deltaE.toFixed(2) : "-");
    v7.utils.safeText("worst-std", worstCase.std_deltaE != null ? worstCase.std_deltaE.toFixed(2) : "-");
    v7.utils.safeText("worst-coverage", worstCase.coverage_ratio != null ? worstCase.coverage_ratio.toFixed(2) : "-");
    if (worstCase.hotspot) {
        const value = worstCase.hotspot.value != null ? worstCase.hotspot.value.toFixed(1) : "-";
        v7.utils.safeText("worst-hotspot", `theta ${worstCase.hotspot.theta_bin}, r ${worstCase.hotspot.r_bin} (ΔE ${value})`);
    } else {
        v7.utils.safeText("worst-hotspot", "-");
    }
    card.classList.remove("hidden");
};

})();
