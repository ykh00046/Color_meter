(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
﻿// Product/STD helpers
v7.products.removeSkuFromLocalStorage = function removeSkuFromLocalStorage(sku) {
    try {
        const saved = JSON.parse(localStorage.getItem("v7_products") || "[]");
        const filtered = saved.filter(item => item !== sku);
        localStorage.setItem("v7_products", JSON.stringify(filtered));
    } catch (e) { console.error(e); }
};

v7.products.setInspInkCountDisplay = function setInspInkCountDisplay(value) {
    const el = byId("inspInkCountDisplay");
    if (!el) return;
    if (value === null || value === undefined || value === "") {
        el.value = "-";
        el.dataset.value = "";
        return;
    }
    el.value = String(value);
    el.dataset.value = String(value);
};

v7.products.ensureProductActive = async function ensureProductActive(sku) {
    try {
        const status = await v7.api.apiCall(`/api/v7/status?sku=${sku}&ink=INK_DEFAULT`, "GET");
        const expectedInkCount = status.expected_ink_count ?? "";
        if (status.status === "DELETED" || status.status === "NOT_FOUND") {
            alert("선택한 STD가 목록에 없습니다.");
            v7.products.removeSkuFromLocalStorage(sku);
            await v7.products.loadProducts();
            return { ok: false, expectedInkCount: "" };
        }
        if (status.status === "ACTIVE") return { ok: true, expectedInkCount };
        alert("활성화되지 않은 STD입니다. 활성화를 먼저 진행하세요.");
        return { ok: false, expectedInkCount };
    } catch (e) {
        alert("상태 조회 실패");
        return { ok: false, expectedInkCount: "" };
    }
};

v7.products.loadProducts = async function loadProducts() {
    try {
        const saved = JSON.parse(localStorage.getItem("v7_products") || '["PRODUCT_A_V1"]');
        const statuses = await Promise.all(saved.map(async (sku) => {
            try {
                const status = await v7.api.apiCall(`/api/v7/status?sku=${sku}&ink=INK_DEFAULT`, "GET");
                return { sku, status };
            } catch (e) {
                return { sku, status: { status: "ERROR" } };
            }
        }));

        const filtered = [];
        const activeOnly = [];
        v7.state.productStatus = {};
        statuses.forEach(({ sku, status }) => {
            if (status.status === "DELETED" || status.status === "NOT_FOUND") {
                return;
            }
            filtered.push(sku);
            if (status.status === "ACTIVE") {
                activeOnly.push(sku);
            }
            v7.state.productStatus[sku] = status;
        });

        if (filtered.length !== saved.length) {
            localStorage.setItem("v7_products", JSON.stringify(filtered));
        }

        ["inspProductSelect", "actProductSelect", "delProductSelect"].forEach(id => {
            const select = byId(id);
            if (!select) return;
            const current = select.value;
            select.innerHTML = `<option value="" disabled ${!current ? "selected" : ""}>제품을 선택하세요.</option>`;
            const source = id === "inspProductSelect" ? activeOnly : filtered;
            source.forEach(sku => {
                const opt = document.createElement("option");
                opt.value = sku;
                opt.textContent = sku;
                select.appendChild(opt);
            });
            if (current && source.includes(current)) select.value = current;
        });

        const inspSelect = byId("inspProductSelect");
        if (inspSelect && inspSelect.value) {
            const res = await v7.products.ensureProductActive(inspSelect.value);
            v7.products.setInspInkCountDisplay(res.expectedInkCount);
        }
    } catch (e) { console.error(e); }
};

})();
