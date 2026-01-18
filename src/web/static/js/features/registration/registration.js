/**
 * STD Registration Feature Module
 * Simplified from v7/registration.js
 */

import { apiClient } from '../../core/api.js';
import { appState } from '../../core/state.js';
import { byId } from '../../utils/helpers.js';
import { t } from '../../utils/i18n.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Register STD
 * @returns {Promise<void>}
 */
export async function registerSTD() {
    const productInput = byId('regProductName');
    const inkCountInput = byId('regInkCount');
    const fileInput = byId('regFiles');
    const btnRegister = byId('btnRegister');

    try {
        const productName = productInput?.value?.trim();
        if (!productName) {
            showNotification('오류', t('alerts.needProductName', '제품명을 입력하세요'), 'error');
            return;
        }

        const inkCount = inkCountInput?.value;
        if (!inkCount) {
            showNotification('오류', t('alerts.needInkCount', '잉크 개수를 입력하세요'), 'error');
            return;
        }

        const file = fileInput?.files?.[0];
        if (!file) {
            showNotification('오류', '이미지를 선택하세요', 'error');
            return;
        }

        if (btnRegister) {
            btnRegister.disabled = true;
            btnRegister.textContent = t('buttons.registerRunning', '등록 중...');
        }

        const fd = new FormData();
        fd.append('sku', productName);
        fd.append('ink', 'INK_DEFAULT');
        fd.append('expected_ink_count', inkCount);
        fd.append('files', file);

        const data = await apiClient.post('/v7/register', fd, 'multipart');

        showNotification('완료', 'STD 등록이 완료되었습니다', 'success');

        // Clear form
        if (productInput) productInput.value = '';
        if (inkCountInput) inkCountInput.value = '';
        if (fileInput) fileInput.value = '';

    } catch (err) {
        console.error('Registration failed:', err);
        showNotification('오류', `등록 실패: ${err.message}`, 'error');
    } finally {
        if (btnRegister) {
            btnRegister.disabled = false;
            btnRegister.textContent = t('buttons.registerStart', '등록 시작');
        }
    }
}

/**
 * Initialize registration UI
 */
export function initRegistration() {
    const btnRegister = byId('btnRegister');
    if (btnRegister) {
        btnRegister.addEventListener('click', registerSTD);
    }
}

export default { initRegistration, registerSTD };
