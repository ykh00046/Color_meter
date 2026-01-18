/**
 * Test Lab Feature Module
 * Simplified from v7/test.js
 */

import { apiClient } from '../../core/api.js';
import { byId } from '../../utils/helpers.js';
import { showNotification } from '../../utils/notifications.js';
import { t } from '../../utils/i18n.js';

/**
 * Run test
 * @returns {Promise<void>}
 */
export async function runTest() {
    const btnTest = byId('btnTest');

    try {
        if (btnTest) {
            btnTest.disabled = true;
            btnTest.textContent = t('buttons.testRunning', '테스트 중...');
        }

        const data = await apiClient.get('/v7/test');

        showNotification('완료', '테스트 완료', 'success');
        console.log('Test result:', data);

    } catch (err) {
        console.error('Test failed:', err);
        showNotification('오류', `테스트 실패: ${err.message}`, 'error');
    } finally {
        if (btnTest) {
            btnTest.disabled = false;
            btnTest.textContent = t('buttons.testStart', '테스트 실행');
        }
    }
}

/**
 * Initialize test UI
 */
export function initTestTab() {
    const btnTest = byId('btnTest');
    if (btnTest) {
        btnTest.addEventListener('click', runTest);
    }
}

export default { initTestTab, runTest };
