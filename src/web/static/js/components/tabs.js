/**
 * Result Tabs Component
 * Simplified from v7/result_tabs.js
 * Enhanced with ARIA attributes and keyboard navigation.
 */

import { byId } from '../utils/helpers.js';

/**
 * Switch result tab with ARIA support.
 * @param {string} tabName - Tab name (v7, legacy, etc.)
 */
export function switchResultTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('[data-tab-button]').forEach(btn => {
        const isActive = btn.dataset.tabButton === tabName;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', String(isActive));
        btn.setAttribute('tabindex', isActive ? '0' : '-1');
    });

    // Update tab panels
    document.querySelectorAll('[data-result-tab]').forEach(tab => {
        const isVisible = tab.dataset.resultTab === tabName;
        tab.classList.toggle('hidden', !isVisible);
    });
}

/**
 * Initialize result tabs with click handlers and keyboard navigation.
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

    // Keyboard navigation: Arrow Left/Right, Home, End
    const tabLists = document.querySelectorAll('[role="tablist"]');
    tabLists.forEach(tabList => {
        tabList.addEventListener('keydown', (e) => {
            const tabs = [...tabList.querySelectorAll('[role="tab"]:not([style*="display:none"])')];
            const idx = tabs.indexOf(e.target);
            if (idx < 0) return;

            let next = -1;
            if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length;
            else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length;
            else if (e.key === 'Home') next = 0;
            else if (e.key === 'End') next = tabs.length - 1;

            if (next >= 0) {
                e.preventDefault();
                tabs[next].focus();
                tabs[next].click();
            }
        });
    });
}

export default { initResultTabs, switchResultTab };
