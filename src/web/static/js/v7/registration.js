(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
﻿v7.actions.initRegistration = function initRegistration() {
    const btnReg = byId("btnRegister");
    if (btnReg) {
        btnReg.addEventListener("click", async () => {
            try {
                const sku = byId("regSku")?.value.trim();
                if (!sku) return alert("제품명을 입력하세요.");
                const regInkCount = byId("regInkCount")?.value;
                if (!regInkCount) return alert("잉크 수를 선택하세요.");
                btnReg.disabled = true;
                btnReg.textContent = "등록 중...";
                byId("regResultArea")?.classList.add("hidden");

                const fd = new FormData();
                fd.append("sku", sku);
                fd.append("ink", "INK_DEFAULT");
                fd.append("created_by", "SYSTEM");
                if (regInkCount) fd.append("expected_ink_count", regInkCount);
                ["regLow", "regMid", "regHigh"].forEach(id => {
                    Array.from(byId(id)?.files || []).forEach(f => fd.append(id.replace("reg", "").toLowerCase() + "_files", f));
                });

                const data = await v7.api.apiCall("/api/v7/register_validate", "POST", fd, "operator", false);
                const allowed = data.summary?.activation_allowed;
                const status = data.status || (allowed ? "ACTIVATED" : "VALIDATION_FAILED");

                v7.utils.safeText("regSummaryText", `결과: ${allowed ? "SUCCESS" : "FAILED"}\n${JSON.stringify(data.summary?.label_counts || {})}`);
                if (typeof v7.render.renderRegFailures === "function") {
                    v7.render.renderRegFailures(data.validation);
                } else {
                    console.warn("[registration] renderRegFailures not available");
                }

                byId("regResultArea")?.classList.remove("hidden");
                const goActivate = byId("regBtnGoActivate");
                if (goActivate) goActivate.classList.add("hidden");

                if (status === "ACTIVATED") {
                    // Success Case: Fully Registered and Activated
                    if (byId("regStatusLabel")) {
                        byId("regStatusLabel").textContent = "등록 및 활성화 완료";
                        byId("regStatusLabel").className = "text-sm font-black mb-4 text-emerald-400";
                    }
                    if (byId("regStatus")) {
                        byId("regStatus").textContent = "ACTIVE";
                        byId("regStatus").className = "status-badge border-green-ok text-green-ok";
                    }

                    // Update Product List
                    const saved = JSON.parse(localStorage.getItem("v7_products") || "[]");
                    if (!saved.includes(sku)) {
                        saved.push(sku);
                        localStorage.setItem("v7_products", JSON.stringify(saved));
                        await v7.products.loadProducts();
                    }

                    alert(`[${sku}] 제품 등록 및 활성화가 성공적으로 완료되었습니다.`);
                } else if (status === "VALIDATION_FAILED") {
                    // Failure Case: Some samples failed validation
                    if (byId("regStatusLabel")) {
                        byId("regStatusLabel").textContent = "검증 실패 (조건 미흡)";
                        byId("regStatusLabel").className = "text-sm font-black mb-4 text-rose-400";
                    }
                    if (byId("regStatus")) {
                        byId("regStatus").textContent = "FAILED";
                        byId("regStatus").className = "status-badge border-red-error text-red-error";
                    }

                    const runId = data.staging?.run_id;
                    if (runId && confirm("검증에 실패한 샘플이 있습니다. 등록을 취소하고 임시 파일을 삭제할까요?")) {
                        await v7.api.apiCall("/api/v7/register_cleanup", "POST", { run_id: runId }, "operator", true);
                        byId("regResultArea")?.classList.add("hidden");
                    } else if (runId) {
                        // User chose to keep it (maybe for manual activation later)
                        if (goActivate) goActivate.classList.remove("hidden");
                    }
                } else if (status === "ACTIVATION_ERROR") {
                    alert("검증은 통과했으나 자동 활성화 중 오류가 발생했습니다: " + data.error);
                    if (goActivate) goActivate.classList.remove("hidden");
                }
            } catch (err) {
                alert("등록 프로세스 오류: " + err.message);
            } finally {
                btnReg.disabled = false;
                btnReg.textContent = "등록 및 자동 활성화 시작";
            }
        });
    }

    const btnLoad = byId("btnLoadCandidates");
    if (btnLoad) {
        btnLoad.addEventListener("click", async () => {
            try {
                const sku = byId("actProductSelect")?.value;
                if (!sku) return alert("제품을 선택하세요.");
                btnLoad.disabled = true;
                const [status, cand, v2] = await Promise.all([
                    v7.api.apiCall(`/api/v7/status?sku=${sku}&ink=INK_DEFAULT`, "GET"),
                    v7.api.apiCall(`/api/v7/candidates?sku=${sku}&ink=INK_DEFAULT`, "GET"),
                    v7.api.apiCall(`/api/v7/v2_metrics?sku=${sku}&ink=INK_DEFAULT`, "GET")
                ]);

                const box = byId("activeInfoBox");
                if (box) {
                    if (status.active?.HIGH) {
                        box.innerHTML = Object.entries(status.active)
                            .map(([m, p]) => `<div class="flex justify-between"><span>${m}</span><span>${p.split("/").pop()}</span></div>`)
                            .join("");
                        box.classList.remove("italic");
                    } else {
                        box.textContent = "활성 정보 없음";
                        box.classList.add("italic");
                    }
                }

                ["actLow", "actMid", "actHigh"].forEach(id => {
                    const el = byId(id);
                    if (!el) return;
                    const list = cand.candidates?.[id.replace("act", "").toUpperCase()] || [];
                    el.innerHTML = list.map(v => `<option value="${v}">${v}</option>`).join("") || "<option disabled>목록 없음</option>";
                });

                const reviewData = v2.v2_review || v2.metrics?.v2_review || v2.metrics;
                renderV2Review(reviewData, "v2AlertList", "v2ApproverAlert");

                byId("actWorkArea")?.classList.remove("hidden");
            } catch (e) {
                alert("상태 조회 실패");
            } finally {
                btnLoad.disabled = false;
            }
        });
    }

    const btnAct = byId("btnActivate");
    if (btnAct) {
        btnAct.addEventListener("click", async () => {
            const sku = byId("actProductSelect")?.value;
            btnAct.disabled = true;
            try {
                const payload = {
                    sku: sku,
                    ink: "INK_DEFAULT",
                    low_version: byId("actLow").value,
                    mid_version: byId("actMid").value,
                    high_version: byId("actHigh").value,
                    approved_by: "ADMIN_MANUAL",
                    validation_label: "STD_ACCEPTABLE"
                };
                await v7.api.apiCall("/api/v7/activate", "POST", payload, "approver");
                alert(`[${sku}] 활성화되었습니다.`);
                switchTab("inspection");
            } catch (err) {
                alert("활성화 실패: " + err.message);
            } finally {
                btnAct.disabled = false;
            }
        });
    }
};

})();
