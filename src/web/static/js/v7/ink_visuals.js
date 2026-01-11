(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
// Ink-related visuals for v7 UI (globals).
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

})();
