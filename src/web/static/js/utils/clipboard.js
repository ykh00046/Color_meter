/**
 * Clipboard utilities for copying data
 * Migrated from v7/copy.js
 */

import { showNotification } from './notifications.js';

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} Success status
 */
export async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('복사 완료', '클립보드에 복사되었습니다', 'success');
        return true;
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('복사 실패', '클립보드 복사에 실패했습니다', 'error');
        return false;
    }
}

/**
 * Copy JSON data to clipboard
 * @param {Object} data - Data to copy as JSON
 * @param {boolean} pretty - Pretty print JSON
 * @returns {Promise<boolean>} Success status
 */
export async function copyJsonToClipboard(data, pretty = true) {
    try {
        const text = pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
        return await copyToClipboard(text);
    } catch (error) {
        console.error('Failed to copy JSON:', error);
        showNotification('복사 실패', 'JSON 변환에 실패했습니다', 'error');
        return false;
    }
}

/**
 * Copy element's text content to clipboard
 * @param {string} elementId - Element ID
 * @returns {Promise<boolean>} Success status
 */
export async function copyElementText(elementId) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element not found: ${elementId}`);
        return false;
    }

    return await copyToClipboard(element.textContent || element.innerText);
}
