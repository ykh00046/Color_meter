(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
v7.inspection = v7.inspection || {};
// Inspection flow for v7 UI.
v7.inspection.applyInspectionMode = (mode) => {
    v7.state.inspectMode = mode || "signature";
    const view = v7.utils.byId("view-inspection");
    if (view) view.dataset.mode = v7.state.inspectMode;

    const showAll = mode === "all";
    const showSignature = showAll || mode === "signature";
    const showGate = showAll || mode === "gate";
    const showInk = showAll || mode === "ink";

    const toggle = (id, show) => {
        const el = v7.utils.byId(id);
        if (!el) return;
        el.classList.toggle("hidden", !show);
    };

    toggle("detailSignature", showSignature);
    toggle("detailGate", showGate);
    toggle("detailAnomaly", showGate);
    toggle("detailColor", showSignature);
    toggle("worstCaseCard", showSignature);
    toggle("v3SummaryCard", showSignature);
    toggle("opsJudgmentCard", showSignature);
    toggle("v3TrendCard", showSignature);
    toggle("detailInkOverview", showInk);
    toggle("inkPanelStack", showInk);

    // Update Active Card State
    document.querySelectorAll(".reason-card").forEach(c => {
        c.classList.remove("ring-2", "ring-brand-500", "bg-slate-800");
    });
    const activeCard = v7.utils.byId(`card-${v7.state.inspectMode}`);
    if (activeCard) {
        activeCard.classList.add("ring-2", "ring-brand-500", "bg-slate-800");
    }
};

v7.inspection.initHotspotOverlay = (worstCase, cfg) => {
    v7.state.worstCfg = cfg;
    v7.render.drawWorstCaseMini(worstCase, cfg);
    v7.render.drawHotspotOverlay(worstCase, cfg);

    const tooltip = v7.utils.byId("hotspotTooltip");
    const viewer = v7.utils.byId("analysisViewer");
    if (!tooltip || !viewer) return;

    if (v7.state.hotspotBound) return;
    v7.state.hotspotBound = true;

    viewer.addEventListener("mousemove", (event) => {
        const hotspot = v7.state.worstHotspot;
        if (!hotspot) {
            tooltip.classList.add("hidden");
            return;
        }
        const rect = viewer.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const dx = x - hotspot.x;
        const dy = y - hotspot.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 18) {
            tooltip.classList.remove("hidden");
            tooltip.style.left = `${hotspot.x}px`;
            tooltip.style.top = `${hotspot.y}px`;
            tooltip.textContent = `ΔE ${hotspot.value?.toFixed ? hotspot.value.toFixed(1) : hotspot.value}`;
        } else {
            tooltip.classList.add("hidden");
        }
    });

    viewer.addEventListener("mouseleave", () => {
        tooltip.classList.add("hidden");
    });
};

v7.actions.initInspection = function initInspection() {
    const view = v7.utils.byId("view-inspection");
    const select = v7.utils.byId("inspProductSelect");
    const btnInspect = v7.utils.byId("btnInspect");
    const fileInput = v7.utils.byId("inspFiles");
    const fileName = v7.utils.byId("inspFileName");
    const detailToggle = v7.utils.byId("opDetailToggle");

    if (detailToggle && view) {
        detailToggle.addEventListener("click", () => {
            const opened = view.classList.toggle("detail-open");
            const openLabel = v7.t("labels.detailOpen", "Summary");
            const closeLabel = v7.t("labels.detailClose", "Details");
            detailToggle.textContent = opened ? openLabel : closeLabel;
        });
    }

    if (fileInput && fileName) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files?.[0];
            fileName.textContent = file ? file.name : v7.t("labels.filePlaceholder", "Drop or click to select a file.");
        });
    }

    const modeButtons = document.querySelectorAll("[data-insp-mode]");
    modeButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const mode = btn.dataset.inspMode || "signature";
            v7.inspection.applyInspectionMode(mode);
        });
    });
    v7.inspection.applyInspectionMode(v7.state.inspectMode);

    if (select) {
        select.addEventListener("change", async () => {
            v7.products.setInspInkCountDisplay(null);
            if (!select.value) return;
            const res = await v7.products.ensureProductActive(select.value);
            v7.products.setInspInkCountDisplay(res.expectedInkCount);
        });
    }

    const primary = v7.utils.byId("inspActionPrimary");
    const secondary = v7.utils.byId("inspActionSecondary");
    const actionHandler = (btn) => {
        if (!btn) return;
        btn.addEventListener("click", () => {
            const action = btn.dataset.action;
            if (action === "reinspect" && btnInspect) return btnInspect.click();
            if (action === "retake") return alert(v7.t("alerts.retakeChecklist", "Check retake checklist."));
            if (action === "share") return alert(v7.t("alerts.shareLinkCopied", "Share link copied."));
            if (action === "confirm") return alert(v7.t("alerts.confirmed", "Confirmed."));
            if (action === "approve") return alert(v7.t("alerts.requestedApproval", "Approval requested."));
            if (action === "pin") return alert(v7.t("alerts.pinnedEvidence", "Evidence panel pinned."));
            if (action === "note") return alert(v7.t("alerts.notePrompt", "Leave a note."));
            if (action === "detail") {
                const dec = v7.state.lastInspection?.result?.results?.[0]?.decision;
                const reasons = dec?.reason_codes || dec?.reasons || [];
                if (reasons.length > 0) {
                    alert("상세 사유:\n" + reasons.join("\n"));
                } else {
                    alert("특이 사유가 없습니다.");
                }
                view?.classList.add("detail-open");
                return;
            }
        });
    };
    actionHandler(primary);
    actionHandler(secondary);

    const viewer = v7.utils.byId("analysisViewer");
    const wrap = v7.utils.byId("analysisCanvasWrap");
    if (viewer && wrap) v7.viewer.initPanzoom(viewer, wrap);

    if (!btnInspect) return;
    btnInspect.addEventListener("click", async () => {
        try {
            const sku = select?.value;
            if (!sku) return alert(v7.t("alerts.needProductSelect", "Select a product."));

            const res = await v7.products.ensureProductActive(sku);
            if (!res.ok) return;

            const file = fileInput?.files?.[0];
            if (!file) return alert(v7.t("alerts.needInspectFile", "Select an inspection image."));

            btnInspect.disabled = true;
            btnInspect.textContent = v7.t("buttons.inspectRunning", "Running...");

            // Step 1: Reset UI
            v7.render.resetInspectionUI();
            v7.utils.byId("inspResultArea")?.classList.remove("hidden"); // Show area to display "Running..." state

            const expectedInk = v7.utils.byId("inspInkCountDisplay")?.dataset?.value || res.expectedInkCount || "";
            const inspMode = v7.utils.byId("inspMode")?.value || "all";
            const fd = new FormData();
            fd.append("sku", sku);
            fd.append("ink", "INK_DEFAULT");
            fd.append("mode", inspMode);
            if (expectedInk) fd.append("expected_ink_count", expectedInk);
            fd.append("files", file);

            const data = await v7.api.apiCall("/api/v7/inspect", "POST", fd, "operator", false);
            const result = data.result || {};
            const resultItem = result.results?.[0];
            if (!resultItem) throw new Error("Inspection result missing.");

            const decision = resultItem.decision || {};
            const v2Diag = decision?.diagnostics?.v2_diagnostics || decision?.diagnostics?.v2 || {};
            const v2Meta = decision?.diagnostics?.v2_meta || {};
            const cfg = result.cfg || {};

            v7.state.lastInspection = data;

            const artifact = result.artifacts?.images?.[0] || {};
            v7.state.currentArtifacts = {
                overlay: artifact.overlay || "",
                heatmap: artifact.heatmap || ""
            };
            if (typeof setViz === "function") setViz("overlay");

            // ✨ UI Improvements: Render enhanced UI components
            const ops = decision?.ops || {};
            if (v7.uiImprovements) {
                // 1. Unified Summary Card - Top-level overview
                v7.uiImprovements.renderUnifiedSummary(decision, v2Diag, ops);

                // 2. Same-Scale Panels - Core and Ink with consistent badges
                v7.uiImprovements.renderSameScalePanels(decision, v2Diag);

                // 3. Direction Clarity - ROI vs Global
                v7.uiImprovements.renderDirectionClarified(v2Diag);

                // 4. Pattern Color Score - With policy label
                v7.uiImprovements.renderPatternColorScore(ops);

                // 5. Forced K Badge - Show when k is forced
                v7.uiImprovements.renderForcedKBadge(v2Diag);

                // 6. Radial Profile Summary - Collapsible detailed view
                if (decision?.diagnostics?.radial) {
                    v7.uiImprovements.renderRadialSummary(decision.diagnostics.radial);
                }
            }

            // Original rendering functions
            v7.render.renderJudgmentBar(decision, data, v2Diag);
            v7.render.renderReasonCards(decision, v2Diag);
            v7.render.renderOperatorPanels(decision, v2Diag, v2Meta, cfg, result, resultItem);
            v7.render.renderKeyMetrics(decision, resultItem, cfg);
            // v7.render.renderModuleTabs removed (merged into cards)
            v7.render.renderTrajectoryDashboard(v2Diag, cfg);
            v7.render.renderInkComparisonPalette(v2Diag); // New Professional Palette
            v7.render.renderV3Summary(decision?.v3_summary);
            v7.render.renderV3Trend(decision?.v3_trend);
            v7.render.renderOpsJudgment(decision?.ops);
            v7.render.renderWorstCase(decision?.diagnostics?.worst_case);
            v7.render.renderColorChart(decision?.diagnostics);

            // Default to 'pattern' view for comprehensive analysis
            const targetMode = (inspMode === "all") ? "pattern" : inspMode;
            v7.inspection.applyInspectionMode(targetMode);

            v7.utils.byId("inspResultArea")?.classList.remove("hidden");
            if (v7.tabs?.switchResultTab) v7.tabs.switchResultTab("v7");
            if (typeof v7.viewer.resetPanzoom === "function") v7.viewer.resetPanzoom();
        } catch (err) {
            alert(`검사 실패: ${err.message}`);
        } finally {
            btnInspect.disabled = false;
            btnInspect.textContent = v7.t("buttons.inspectStart", "Inspect");
        }
    });
};

})();
