/**
 * Product/SKU management utilities
 * Migrated from v7/products.js
 */

import { appState } from '../../core/state.js';
import { apiClient } from '../../core/api.js';
import { byId } from '../../utils/helpers.js';

const PRODUCTS_STORAGE_KEY = 'v7_products';
const DEFAULT_PRODUCTS = ['PRODUCT_A_V1'];

// Local Storage Helpers
export function removeSkuFromLocalStorage(sku) {
    try {
        const saved = JSON.parse(localStorage.getItem(PRODUCTS_STORAGE_KEY) || '[]');
        const filtered = saved.filter(item => item !== sku);
        localStorage.setItem(PRODUCTS_STORAGE_KEY, JSON.stringify(filtered));
    } catch (e) {
        console.error('Failed to remove SKU from localStorage:', e);
    }
}

export function setInspInkCountDisplay(value) {
    const el = byId("inspInkCountDisplay");
    if (!el) return;

    if (value === null || value === undefined || value === "") {
        el.value = "-";
        el.dataset.value = "";
        return;
    }
    el.value = String(value);
    el.dataset.value = String(value);
}

// Product Status Management
export async function ensureProductActive(sku) {
    try {
        const url = `/api/v7/status?sku=${sku}&ink=INK_DEFAULT`;
        const status = await apiClient.get(url);
        const expectedInkCount = status.expected_ink_count ?? "";

        if (status.status === "DELETED" || status.status === "NOT_FOUND") {
            alert("선택한 STD가 목록에 없습니다.");
            removeSkuFromLocalStorage(sku);
            await loadProducts();
            return { ok: false, expectedInkCount: "" };
        }

        if (status.status === "ACTIVE") {
            return { ok: true, expectedInkCount };
        }

        alert("활성화되지 않은 STD입니다. 활성화를 먼저 진행하세요.");
        return { ok: false, expectedInkCount };
    } catch (e) {
        console.error('Failed to check product status:', e);
        alert("상태 조회 실패");
        return { ok: false, expectedInkCount: "" };
    }
}

// Load Products from API
export async function loadProducts() {
    try {
        const saved = JSON.parse(localStorage.getItem(PRODUCTS_STORAGE_KEY) || JSON.stringify(DEFAULT_PRODUCTS));

        // Fetch all product statuses
        const statuses = await Promise.all(saved.map(async (sku) => {
            try {
                const url = `/api/v7/status?sku=${sku}&ink=INK_DEFAULT`;
                const status = await apiClient.get(url);
                return { sku, status };
            } catch (e) {
                return { sku, status: { status: "ERROR" } };
            }
        }));

        // Filter products
        const filtered = [];
        const activeOnly = [];
        const productStatus = {};

        statuses.forEach(({ sku, status }) => {
            if (status.status === "DELETED" || status.status === "NOT_FOUND") {
                return;
            }
            filtered.push(sku);
            if (status.status === "ACTIVE") {
                activeOnly.push(sku);
            }
            productStatus[sku] = status;
        });

        // Update localStorage if needed
        if (filtered.length !== saved.length) {
            localStorage.setItem(PRODUCTS_STORAGE_KEY, JSON.stringify(filtered));
        }

        // Update state
        appState.setState('productStatus', productStatus);

        // Update all product select elements
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

            if (current && source.includes(current)) {
                select.value = current;
            }
        });

        // Update ink count display for inspection
        const inspSelect = byId("inspProductSelect");
        if (inspSelect && inspSelect.value) {
            const res = await ensureProductActive(inspSelect.value);
            setInspInkCountDisplay(res.expectedInkCount);
        }
    } catch (e) {
        console.error('Failed to load products:', e);
    }
}

// Auto-refresh products on select change
export function setupProductListeners() {
    const refreshBtn = byId("btnRefreshProducts");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", loadProducts);
    }

    const inspSelect = byId("inspProductSelect");
    if (inspSelect) {
        inspSelect.addEventListener("change", async (e) => {
            const sku = e.target.value;
            if (sku) {
                const res = await ensureProductActive(sku);
                setInspInkCountDisplay(res.expectedInkCount);
            }
        });
    }
}
