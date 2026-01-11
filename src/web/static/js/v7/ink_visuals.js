(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
// Ink-related visuals for v7 UI (globals).
v7.render.renderInkJudgment = (v2Diag, v2Meta) => {
    const card = v7.utils.byId("inkJudgeCard");
    const badge = v7.utils.byId("inkJudgeBadge");
    const line = v7.utils.byId("inkJudgeLine");
    const reasons = v7.utils.byId("inkJudgeReasons");
    if (!card || !badge || !line || !reasons) return;

    if (!v2Diag) {
        badge.textContent = "N/A";
        line.textContent = "V2 diagnostics unavailable.";
        reasons.innerHTML = "";
        return;
    }

    const warnings = v2Diag.warnings || [];
    const matchWarning = v2Diag.ink_match?.warning;
    const summary = v2Diag.ink_match?.trajectory_summary;
    const maxOff = summary?.max_off_track;

    const hasWarning = warnings.length > 0 || matchWarning;
    badge.textContent = hasWarning ? "CHECK" : "OK";
    badge.className = `status-badge ${hasWarning ? "border-yellow-warning text-yellow-warning" : "border-green-ok text-green-ok"}`;
    line.textContent = `Ink STD=${v2Diag.expected_ink_count || "-"} | Suggested=${v2Diag.auto_estimation?.suggested_k ?? "-"}`;

    const lines = [];
    if (matchWarning) lines.push(matchWarning);
    if (typeof maxOff === "number") lines.push(`max off-track ${maxOff.toFixed(2)}`);
    warnings.forEach((w) => lines.push(w));
    reasons.innerHTML = lines.map((item) => `<li>${item}</li>`).join("");
};

v7.render.renderTrajectoryDashboard = (v2Diag, cfg) => {
    const badge = v7.utils.byId("trajStatusBadge");
    const dot = v7.utils.byId("trajAlertDot");
    const posText = v7.utils.byId("trajPositionText");
    const rangeEl = v7.utils.byId("trajRange");
    const pinEl = v7.utils.byId("trajPin");
    const perInkList = v7.utils.byId("trajPerInkList");
    const offText = v7.utils.byId("trajOffTrackValue");
    const offNum = v7.utils.byId("trajOffTrackNumber");
    const fillEl = v7.utils.byId("trajGaugeFill");
    const limitEl = v7.utils.byId("trajGaugeLimit");
    const captionEl = v7.utils.byId("trajGaugeCaption");

    if (!badge || !dot || !posText || !rangeEl || !pinEl || !perInkList || !offText || !offNum || !fillEl || !limitEl || !captionEl) {
        return;
    }

    const summary = v2Diag?.ink_match?.trajectory_summary || {};
    const maxOff = summary.max_off_track;
    const minPos = summary.on_track_pos_min;
    const maxPos = summary.on_track_pos_max;
    const limit = cfg?.v2_ink?.trajectory_max_off_track || 10;
    const displayMax = Math.max(limit, maxOff || 0.0001);

    if (maxOff == null) {
        badge.textContent = "N/A";
        dot.className = "traj-alert-dot";
        posText.textContent = "-";
        offText.textContent = "-";
        offNum.textContent = "-";
        rangeEl.style.width = "0%";
        pinEl.style.left = "0%";
        perInkList.innerHTML = "";
        fillEl.style.width = "0%";
        limitEl.style.left = "0%";
        captionEl.textContent = "-";
        return;
    }

    const statusOk = maxOff <= limit;
    badge.textContent = statusOk ? "STABLE" : "WARNING";
    badge.className = `traj-alert-text ${statusOk ? "ok" : "alert"}`;
    dot.className = `traj-alert-dot ${statusOk ? "ok" : "alert"}`;
    offText.textContent = `${maxOff.toFixed(2)} (Limit ${limit.toFixed(1)})`;
    offNum.textContent = maxOff.toFixed(2);

    const minVal = typeof minPos === "number" ? v7.utils.clamp(minPos, 0, 100) : 0;
    const maxVal = typeof maxPos === "number" ? v7.utils.clamp(maxPos, 0, 100) : 0;
    const pin = v7.utils.clamp((minVal + maxVal) / 2, 0, 100);

    rangeEl.style.left = `${Math.min(minVal, maxVal)}%`;
    rangeEl.style.width = `${Math.abs(maxVal - minVal)}%`;
    pinEl.style.left = `${pin}%`;
    posText.textContent = `${minVal.toFixed(1)}~${maxVal.toFixed(1)}%`;

    const fillPct = v7.utils.clamp((maxOff / displayMax) * 100, 0, 100);
    const limitPct = v7.utils.clamp((limit / displayMax) * 100, 0, 100);
    fillEl.style.width = `${fillPct}%`;
    limitEl.style.left = `${limitPct}%`;
    fillEl.classList.toggle("warn", !statusOk);
    captionEl.textContent = statusOk ? "Within limit" : "Over limit";

    const deltas = v2Diag?.ink_match?.deltas || [];
    perInkList.innerHTML = deltas.map((item, idx) => {
        const t = item.trajectory;
        const name = `INK ${idx + 1}`;
        const onTrack = t?.on_track_pos != null ? `${t.on_track_pos.toFixed(1)}%` : "-";
        const off = t?.off_track_error != null ? t.off_track_error.toFixed(2) : "-";
        return `<div class="traj-ink-item"><strong>${name}</strong><span>On-track ${onTrack} | Off ${off}</span></div>`;
    }).join("");
};

v7.render.renderInkComponentStatus = (inkMatch, cfg) => {
    const grid = v7.utils.byId("inkComponentGrid");
    if (!grid) return;

    const deltas = inkMatch?.deltas || [];
    if (!deltas.length) {
        grid.innerHTML = `<p class="text-xs text-text-dim">No ink match data.</p>`;
        return;
    }

    grid.innerHTML = deltas.map((item, idx) => {
        const refHex = v7.utils.labToHex(item.baseline_mean_lab) || "#334155";
        const sampHex = v7.utils.labToHex(item.sample_mean_lab) || "#475569";
        const deltaE = item.deltaE != null ? item.deltaE.toFixed(2) : "-";
        const warn = item.deltaE != null && cfg?.v2_ink?.match_max_deltaE != null
            ? item.deltaE > cfg.v2_ink.match_max_deltaE
            : false;

        return `
            <div class="ink-item ${warn ? "warn" : ""}">
                <div class="ink-chip-pair">
                    <div class="ink-chip-dot" style="background:${refHex}"></div>
                    <span class="ink-chip-sep">vs</span>
                    <div class="ink-chip-dot" style="background:${sampHex}"></div>
                </div>
                <div class="ink-item-meta">
                    <strong>INK ${idx + 1}</strong>
                    <span>DeltaE ${deltaE}</span>
                </div>
                <span class="ink-status ${warn ? "warn" : "ok"}">${warn ? "!" : "OK"}</span>
            </div>
        `;
    }).join("");
};

v7.render.updateInk3Focus = (palette) => {
    const swatch = v7.utils.byId("ink3Swatch");
    const hexEl = v7.utils.byId("ink3Hex");
    const areaEl = v7.utils.byId("ink3Area");
    const labEl = v7.utils.byId("ink3Lab");
    const rgbEl = v7.utils.byId("ink3Rgb");
    const badge = v7.utils.byId("ink3OverlayBadge");
    const viewer = v7.utils.byId("analysisViewer");
    const focusBadge = v7.utils.byId("ink3FocusBadge");

    if (!swatch || !hexEl || !areaEl || !labEl || !rgbEl) return;

    const colors = palette?.colors || [];
    if (colors.length < 3) {
        swatch.style.background = "var(--surface-dark)";
        hexEl.textContent = "-";
        areaEl.textContent = "-";
        labEl.textContent = "-";
        rgbEl.textContent = "-";
        if (badge) badge.classList.add("hidden");
        if (viewer) viewer.style.borderColor = "var(--border-dim)";
        if (focusBadge) focusBadge.textContent = "INK3";
        return;
    }

    const ink3 = colors[2];
    const hex = ink3.mean_hex || "#222222";
    const area = ink3.area_ratio != null ? `${(ink3.area_ratio * 100).toFixed(1)}%` : "-";
    const lab = Array.isArray(ink3.mean_lab) ? ink3.mean_lab.map((v) => v.toFixed(1)).join(", ") : "-";
    const rgb = Array.isArray(ink3.mean_rgb) ? ink3.mean_rgb.join(", ") : "-";

    swatch.style.background = hex;
    hexEl.textContent = hex;
    areaEl.textContent = area;
    labEl.textContent = lab;
    rgbEl.textContent = rgb;
    if (badge) {
        badge.classList.remove("hidden");
        badge.style.borderColor = hex;
        badge.style.color = hex;
    }
    if (viewer) viewer.style.borderColor = hex;
    if (focusBadge) focusBadge.textContent = "INK3";
};

})();
