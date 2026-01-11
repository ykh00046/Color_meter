(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
﻿async function loadStdEntries() {
    const sku = byId("stdFilterSku")?.value.trim() || "";
    const ink = byId("stdFilterInk")?.value.trim() || "";
    const status = byId("stdFilterStatus")?.value || "";
    const params = new URLSearchParams();
    if (sku) params.set("sku", sku);
    if (ink) params.set("ink", ink);
    if (status) params.set("status", status);

    const url = `/api/v7/entries${params.toString() ? `?${params.toString()}` : ""}`;
    const res = await v7.api.apiCall(url, "GET", null, "operator", false);
    const items = res.items || [];

    const body = byId("stdTableBody");
    const empty = byId("stdEmpty");
    if (!body || !empty) return;
    body.innerHTML = "";
    empty.classList.toggle("hidden", items.length > 0);

    items.forEach(item => {
        const active = item.active_versions || {};
        const versions = ["LOW", "MID", "HIGH"]
            .map(mode => active[mode] ? `${mode}: ${active[mode]}` : `${mode}: -`)
            .join("<br>");

        const created = item.created_at || "-";
        const activated = item.activated_at || "-";
        const statusText = item.status || "-";

        const row = document.createElement("tr");
        row.innerHTML = `
            <td class="text-white font-bold">${item.sku || "-"}</td>
            <td>${item.ink || "-"}</td>
            <td class="text-amber-bright">${statusText}</td>
            <td class="text-text-dim">${versions}</td>
            <td>${created}</td>
            <td>${activated}</td>
            <td>
                <div class="flex flex-col gap-2">
                    <button class="btn-secondary text-xs px-3 py-2" data-action="soft">SOFT DELETE</button>
                    <button class="btn-secondary text-xs px-3 py-2" data-action="hard">HARD DELETE</button>
                </div>
            </td>
        `;

        row.querySelectorAll("button").forEach(btn => {
            btn.addEventListener("click", async () => {
                const hardDelete = btn.dataset.action === "hard";
                const label = hardDelete ? "완전 삭제" : "삭제";
                if (!confirm(`[${item.sku}/${item.ink}] ${label}를 진행할까요?`)) return;
                const payload = {
                    sku: item.sku,
                    ink: item.ink,
                    deleted_by: "UI_ADMIN",
                    reason: hardDelete ? "hard delete from UI" : "soft delete from UI",
                    hard_delete: hardDelete
                };
                await v7.api.apiCall("/api/v7/delete_entry", "POST", payload, "admin", true);
                v7.products.removeSkuFromLocalStorage(item.sku);
                await v7.products.loadProducts();
                await loadStdEntries();
            });
        });

        body.appendChild(row);
    });
}

v7.actions.initStdAdmin = function initStdAdmin() {
    const btnStdReload = byId("btnStdReload");
    if (btnStdReload) btnStdReload.addEventListener("click", loadStdEntries);
    const btnStdSearch = byId("btnStdSearch");
    if (btnStdSearch) btnStdSearch.addEventListener("click", loadStdEntries);
    const stdTab = byId("tab-std-admin");
    if (stdTab) stdTab.addEventListener("click", () => setTimeout(loadStdEntries, 0));
};

})();
