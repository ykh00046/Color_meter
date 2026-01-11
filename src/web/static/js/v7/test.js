(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
﻿v7.actions.initTestTab = function initTestTab() {
    const btnTestRun = byId("btnTestRun");
    if (!btnTestRun) return;
    btnTestRun.addEventListener("click", async () => {
        try {
            const mode = byId("testMode")?.value || "all";
            const inkCount = byId("testInkCount")?.value;

            const lowFile = byId("testLow")?.files?.[0];
            const midFile = byId("testMid")?.files?.[0];
            const highFile = byId("testHigh")?.files?.[0];
            if (!lowFile || !midFile || !highFile) {
                return alert("LOW/MID/HIGH 이미지를 선택하세요.");
            }
            if ((mode === "ink" || mode === "all") && !inkCount) {
                return alert("잉크 수를 선택하세요.");
            }

            btnTestRun.disabled = true;
            btnTestRun.textContent = "테스트 중...";
            byId("testResultArea")?.classList.add("hidden");

            const fd = new FormData();
            fd.append("ink", "INK_DEFAULT");
            fd.append("mode", mode);
            if (inkCount) fd.append("expected_ink_count", inkCount);
            fd.append("low_file", lowFile);
            fd.append("mid_file", midFile);
            fd.append("high_file", highFile);

            const data = await v7.api.apiCall("/api/v7/test_run", "POST", fd, "operator", false);
            const report = data.report || {};
            const results = report.results || {};

            const gate = results.gate || {};
            const gateLine = ["LOW", "MID", "HIGH"].map((m) => {
                const passed = gate?.[m]?.gate?.passed;
                const label = passed === true ? "OK" : (passed === false ? "FAIL" : "-");
                return `${m}:${label}`;
            }).join(" ");

            const regSummary = results.registration_summary || {};
            const regLine = results.registration_summary
                ? `REG sep=${regSummary.separation_ok ?? "-"} order=${regSummary.order_ok ?? "-"}`
                : "";

            const inkBaseline = results.ink_baseline || {};
            const inkLine = results.ink_baseline
                ? `INK baseline=${inkBaseline.generated ? "OK" : "FAIL"}`
                : "";

            const summaryParts = [gateLine, regLine, inkLine].filter(Boolean);
            v7.utils.safeText("testSummaryLine", summaryParts.join(" | ") || "-");
            v7.utils.safeText("testReportPath", data.report_path || "-");
            v7.utils.safeText("testReportJson", JSON.stringify(report, null, 2));

            if (v7.render.renderTestInkPalette) {
                v7.render.renderTestInkPalette(inkBaseline?.baseline, results.v2_diagnostics);
            }

            const badge = byId("testStatusBadge");
            if (badge) {
                const gateFailed = ["LOW", "MID", "HIGH"].some((m) => gate?.[m]?.gate?.passed === false);
                const regFailed = Boolean(
                    results.registration_summary
                    && (regSummary.separation_ok === false || regSummary.order_ok === false || (regSummary.unstable_modes || []).length)
                );
                const inkFailed = results.ink_baseline && inkBaseline.generated === false;
                const ok = !(gateFailed || regFailed || inkFailed);
                badge.textContent = ok ? "OK" : "CHECK";
                badge.className = ok
                    ? "status-badge border-green-ok text-green-ok"
                    : "status-badge border-yellow-warning text-yellow-warning";
            }

            byId("testResultArea")?.classList.remove("hidden");
        } catch (err) {
            alert("테스트 실패: " + err.message);
        } finally {
            btnTestRun.disabled = false;
            btnTestRun.textContent = "테스트 실행";
        }
    });
};

})();
