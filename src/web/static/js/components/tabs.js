/**
 * Result Tabs Component
 * Simplified from v7/result_tabs.js
 */

import { byId } from '../utils/helpers.js';

/**
 * Switch result tab
 * @param {string} tabName - Tab name (v7, legacy, etc.)
 */
export function switchResultTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('[data-result-tab]').forEach(tab => {
        tab.classList.add('hidden');
    });

    // Show selected tab
    const selectedTab = byId(`tab-${tabName}`);
    if (selectedTab) {
        selectedTab.classList.remove('hidden');
    }

    // Update tab buttons
    document.querySelectorAll('[data-tab-button]').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tabButton === tabName) {
            btn.classList.add('active');
        }
    });
}

/**
 * Initialize result tabs
 */
export function initResultTabs() {
    document.querySelectorAll('[data-tab-button]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tabButton;
            if (tabName) {
                switchResultTab(tabName);
            }
        });
    });
}

export default { initResultTabs, switchResultTab };
