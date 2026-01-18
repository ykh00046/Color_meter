/**
 * Inspection Feature Module
 * Migrated from v7/inspection.js
 */

import { appState } from '../../core/state.js';
import { apiClient } from '../../core/api.js';
import { byId } from '../../utils/helpers.js';
import { t } from '../../utils/i18n.js';
import { ensureProductActive, setInspInkCountDisplay } from './products.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Apply inspection mode (all, signature, gate, ink)
 * @param {string} mode - Inspection mode
 */
export function applyInspectionMode(mode = 'signature') {
    appState.setState('inspection.inspectMode', mode);

    const view = byId('view-inspection');
    if (view) view.dataset.mode = mode;

    const showAll = mode === 'all';
    const showSignature = showAll || mode === 'signature';
    const showGate = showAll || mode === 'gate';
    const showInk = showAll || mode === 'ink';

    const toggle = (id, show) => {
        const el = byId(id);
        if (el) el.classList.toggle('hidden', !show);
    };

    toggle('detailSignature', showSignature);
    toggle('detailGate', showGate);
    toggle('detailAnomaly', showGate);
    toggle('detailColor', showSignature);
    toggle('worstCaseCard', showSignature);
    toggle('v3SummaryCard', showSignature);
    toggle('opsJudgmentCard', showSignature);
    toggle('v3TrendCard', showSignature);
    toggle('detailInkOverview', showInk);
    toggle('inkPanelStack', showInk);

    // Update active card
    document.querySelectorAll('.reason-card').forEach(c => {
        c.classList.remove('ring-2', 'ring-brand-500', 'bg-slate-800');
    });
    const activeCard = byId(`card-${mode}`);
    if (activeCard) {
        activeCard.classList.add('ring-2', 'ring-brand-500', 'bg-slate-800');
    }
}

/**
 * Run inspection
 * @returns {Promise<void>}
 */
export async function runInspection() {
    const select = byId('inspProductSelect');
    const fileInput = byId('inspFiles');
    const btnInspect = byId('btnInspect');

    try {
        const sku = select?.value;
        if (!sku) {
            showNotification('오류', t('alerts.needProductSelect', '제품을 선택하세요'), 'error');
            return;
        }

        const res = await ensureProductActive(sku);
        if (!res.ok) return;

        const file = fileInput?.files?.[0];
        if (!file) {
            showNotification('오류', t('alerts.needInspectFile', '검사 이미지를 선택하세요'), 'error');
            return;
        }

        // Set loading state
        if (btnInspect) {
            btnInspect.disabled = true;
            btnInspect.textContent = t('buttons.inspectRunning', '검사 중...');
        }

        appState.setState('inspection.isProcessing', true);
        byId('inspResultArea')?.classList.remove('hidden');

        // Prepare form data
        const expectedInk = byId('inspInkCountDisplay')?.dataset?.value || res.expectedInkCount || '';
        const inspMode = byId('inspMode')?.value || 'all';
        const fd = new FormData();
        fd.append('sku', sku);
        fd.append('ink', 'INK_DEFAULT');
        fd.append('mode', inspMode);
        if (expectedInk) fd.append('expected_ink_count', expectedInk);
        fd.append('files', file);

        // Call API
        const data = await apiClient.post('/v7/inspect', fd, 'multipart');
        const result = data.result || {};
        const resultItem = result.results?.[0];

        if (!resultItem) {
            throw new Error('Inspection result missing');
        }

        // Store result in state
        appState.setState('inspection.lastResult', data);

        const artifact = result.artifacts?.images?.[0] || {};
        appState.setState('inspection.currentArtifacts', {
            overlay: artifact.overlay || '',
            heatmap: artifact.heatmap || ''
        });

        // Render results (placeholder - would import render functions)
        renderInspectionResults(data, result, resultItem);

        showNotification('완료', '검사가 완료되었습니다', 'success');

    } catch (err) {
        console.error('Inspection failed:', err);
        showNotification('오류', `검사 실패: ${err.message}`, 'error');
    } finally {
        if (btnInspect) {
            btnInspect.disabled = false;
            btnInspect.textContent = t('buttons.inspectStart', '검사 실행');
        }
        appState.setState('inspection.isProcessing', false);
    }
}

/**
 * Render inspection results (placeholder)
 * @param {Object} data - Full response data
 * @param {Object} result - Result object
 * @param {Object} resultItem - Result item
 * @private
 */
function renderInspectionResults(data, result, resultItem) {
    // TODO: Import and call rendering functions from visuals module
    // This would include:
    // - renderJudgmentBar(decision, data, v2Diag);
    // - renderReasonCards(decision, v2Diag);
    // - renderKeyMetrics(decision, resultItem, cfg);
    // etc.

    console.log('Rendering results:', { data, result, resultItem });
}

/**
 * Initialize inspection UI
 */
export function initInspection() {
    const view = byId('view-inspection');
    const select = byId('inspProductSelect');
    const btnInspect = byId('btnInspect');
    const fileInput = byId('inspFiles');
    const fileName = byId('inspFileName');
    const detailToggle = byId('opDetailToggle');

    // Detail toggle
    if (detailToggle && view) {
        detailToggle.addEventListener('click', () => {
            const opened = view.classList.toggle('detail-open');
            detailToggle.textContent = opened
                ? t('labels.detailOpen', 'Summary')
                : t('labels.detailClose', 'Details');
        });
    }

    // File input
    if (fileInput && fileName) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files?.[0];
            fileName.textContent = file
                ? file.name
                : t('labels.filePlaceholder', 'Drop or click to select a file.');
        });
    }

    // Mode buttons
    document.querySelectorAll('[data-insp-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.inspMode || 'signature';
            applyInspectionMode(mode);
        });
    });

    // Product select
    if (select) {
        select.addEventListener('change', async () => {
            setInspInkCountDisplay(null);
            if (select.value) {
                const res = await ensureProductActive(select.value);
                setInspInkCountDisplay(res.expectedInkCount);
            }
        });
    }

    // Inspect button
    if (btnInspect) {
        btnInspect.addEventListener('click', runInspection);
    }

    // Initialize mode
    const currentMode = appState.getState('inspection.inspectMode') || 'signature';
    applyInspectionMode(currentMode);
}

export default { initInspection, runInspection, applyInspectionMode };
