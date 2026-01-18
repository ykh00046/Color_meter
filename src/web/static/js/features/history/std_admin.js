/**
 * STD History/Admin Feature Module
 * Simplified from v7/std_admin.js
 */

import { apiClient } from '../../core/api.js';
import { byId } from '../../utils/helpers.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Load STD history
 * @returns {Promise<void>}
 */
export async function loadSTDHistory() {
    try {
        const data = await apiClient.get('/v7/std_history');
        renderSTDHistory(data);
    } catch (err) {
        console.error('Failed to load STD history:', err);
        showNotification('오류', 'STD 이력 조회 실패', 'error');
    }
}

/**
 * Render STD history (placeholder)
 * @param {Array} data - History data
 * @private
 */
function renderSTDHistory(data) {
    console.log('Rendering STD history:', data);
    // TODO: Implement table rendering
}

/**
 * Initialize STD admin UI
 */
export function initStdAdmin() {
    loadSTDHistory();

    const refreshBtn = byId('btnRefreshHistory');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadSTDHistory);
    }
}

export default { initStdAdmin, loadSTDHistory };
