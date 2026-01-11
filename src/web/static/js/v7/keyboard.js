/**
 * Keyboard Shortcuts Module for Lens Signature Engine v7
 *
 * Provides intuitive keyboard navigation and shortcuts:
 * - g: Gate mode
 * - s: Signature mode
 * - i: Ink mode
 * - a: All mode
 * - Space: Toggle overlay
 * - /: Show help
 * - Esc: Close panels/modals
 * - Arrow keys: Navigate between results
 * - Ctrl+C: Copy result to clipboard
 * - Ctrl+S: Save/export current result
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.keyboard = v7.keyboard || {};

    // State
    let state = {
        enabled: true,
        shortcuts: {},
        helpVisible: false,
        lastKeyTime: 0,
        keySequence: []
    };

    // Default shortcuts configuration
    const DEFAULT_SHORTCUTS = {
        // Mode switching
        'g': { action: 'switchToGate', description: 'Switch to Gate mode', category: 'Mode' },
        's': { action: 'switchToSignature', description: 'Switch to Signature mode', category: 'Mode' },
        'i': { action: 'switchToInk', description: 'Switch to Ink mode', category: 'Mode' },
        'a': { action: 'switchToAll', description: 'Switch to All mode', category: 'Mode' },

        // Visualization
        ' ': { action: 'toggleOverlay', description: 'Toggle heatmap overlay', category: 'View', preventDefault: true },
        'v': { action: 'toggle3D', description: 'Toggle 3D visualization', category: 'View' },
        'h': { action: 'toggleHeatmap', description: 'Toggle heatmap view', category: 'View' },

        // Navigation
        'ArrowLeft': { action: 'previousResult', description: 'Previous result', category: 'Navigation', preventDefault: true },
        'ArrowRight': { action: 'nextResult', description: 'Next result', category: 'Navigation', preventDefault: true },
        'ArrowUp': { action: 'scrollUp', description: 'Scroll up', category: 'Navigation', preventDefault: true },
        'ArrowDown': { action: 'scrollDown', description: 'Scroll down', category: 'Navigation', preventDefault: true },

        // Actions
        '/': { action: 'showHelp', description: 'Show help', category: 'Help', preventDefault: true },
        '?': { action: 'showHelp', description: 'Show help', category: 'Help', shift: true },
        'Escape': { action: 'closeAll', description: 'Close panels/dialogs', category: 'General' },

        // With modifiers
        'ctrl+c': { action: 'copyResult', description: 'Copy result to clipboard', category: 'Actions' },
        'ctrl+s': { action: 'saveResult', description: 'Save/export result', category: 'Actions' },
        'ctrl+p': { action: 'print', description: 'Print result', category: 'Actions' },
        'ctrl+f': { action: 'search', description: 'Search/filter', category: 'Actions' },

        // Quick actions
        'r': { action: 'refresh', description: 'Refresh view', category: 'Actions' },
        'e': { action: 'export', description: 'Export current view', category: 'Actions' },
        'c': { action: 'clearSelection', description: 'Clear selection', category: 'Actions' },

        // Advanced
        'ctrl+shift+d': { action: 'toggleDebugMode', description: 'Toggle debug mode', category: 'Debug' },
        'ctrl+shift+l': { action: 'toggleLogs', description: 'Toggle console logs', category: 'Debug' }
    };

    /**
     * Initialize keyboard shortcuts
     *
     * @param {Object} customShortcuts - Custom shortcuts to override defaults
     */
    v7.keyboard.init = function(customShortcuts = {}) {
        state.shortcuts = { ...DEFAULT_SHORTCUTS, ...customShortcuts };

        // Attach global key listener
        document.addEventListener('keydown', handleKeyDown);

        console.log('[v7.keyboard] Keyboard shortcuts initialized');
        console.log('Press "/" or "?" to see all available shortcuts');

        return true;
    };

    /**
     * Handle keydown events
     */
    function handleKeyDown(e) {
        if (!state.enabled) return;

        // Ignore if typing in input/textarea (except Escape)
        if (e.key !== 'Escape' && isInputFocused()) {
            return;
        }

        // Build key combination string
        const keyCombo = buildKeyCombo(e);

        // Find matching shortcut
        const shortcut = state.shortcuts[keyCombo];

        if (shortcut) {
            if (shortcut.preventDefault) {
                e.preventDefault();
            }

            // Execute action
            executeAction(shortcut.action, e);

            // Track key sequence
            trackKeySequence(e.key);
        }
    }

    /**
     * Build key combination string from event
     */
    function buildKeyCombo(e) {
        const parts = [];

        if (e.ctrlKey || e.metaKey) parts.push('ctrl');
        if (e.shiftKey && e.key.length > 1) parts.push('shift');  // Only for special keys
        if (e.altKey) parts.push('alt');

        parts.push(e.key.toLowerCase());

        return parts.join('+');
    }

    /**
     * Check if input element is focused
     */
    function isInputFocused() {
        const active = document.activeElement;
        const tagName = active.tagName.toLowerCase();
        return tagName === 'input' || tagName === 'textarea' || active.isContentEditable;
    }

    /**
     * Execute shortcut action
     */
    function executeAction(action, event) {
        console.log('[v7.keyboard] Action:', action);

        switch (action) {
            case 'switchToGate':
                if (v7.inspection && v7.inspection.applyInspectionMode) {
                    v7.inspection.applyInspectionMode('gate');
                }
                break;

            case 'switchToSignature':
                if (v7.inspection && v7.inspection.applyInspectionMode) {
                    v7.inspection.applyInspectionMode('signature');
                }
                break;

            case 'switchToInk':
                if (v7.inspection && v7.inspection.applyInspectionMode) {
                    v7.inspection.applyInspectionMode('ink');
                }
                break;

            case 'switchToAll':
                if (v7.inspection && v7.inspection.applyInspectionMode) {
                    v7.inspection.applyInspectionMode('all');
                }
                break;

            case 'toggleOverlay':
                toggleOverlay();
                break;

            case 'toggle3D':
                toggle3DView();
                break;

            case 'toggleHeatmap':
                toggleHeatmapView();
                break;

            case 'showHelp':
                v7.keyboard.showHelp();
                break;

            case 'closeAll':
                closeAllPanels();
                break;

            case 'copyResult':
                copyResultToClipboard();
                break;

            case 'saveResult':
                event.preventDefault();
                saveCurrentResult();
                break;

            case 'print':
                event.preventDefault();
                window.print();
                break;

            case 'previousResult':
                navigateResults(-1);
                break;

            case 'nextResult':
                navigateResults(1);
                break;

            case 'refresh':
                refreshCurrentView();
                break;

            case 'export':
                exportCurrentView();
                break;

            case 'clearSelection':
                clearCurrentSelection();
                break;

            case 'toggleDebugMode':
                toggleDebugMode();
                break;

            default:
                console.warn('[v7.keyboard] Unknown action:', action);
        }
    }

    /**
     * Show help modal with all shortcuts
     */
    v7.keyboard.showHelp = function() {
        if (state.helpVisible) {
            v7.keyboard.hideHelp();
            return;
        }

        // Create modal if doesn't exist
        let modal = document.getElementById('keyboardHelpModal');

        if (!modal) {
            modal = createHelpModal();
            document.body.appendChild(modal);
        }

        // Populate shortcuts
        populateHelpContent(modal);

        // Show modal
        modal.style.display = 'flex';
        state.helpVisible = true;
    };

    /**
     * Hide help modal
     */
    v7.keyboard.hideHelp = function() {
        const modal = document.getElementById('keyboardHelpModal');
        if (modal) {
            modal.style.display = 'none';
        }
        state.helpVisible = false;
    };

    /**
     * Create help modal element
     */
    function createHelpModal() {
        const modal = document.createElement('div');
        modal.id = 'keyboardHelpModal';
        modal.style.cssText = `
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10000;
            align-items: center;
            justify-content: center;
            font-family: 'Courier New', monospace;
        `;

        modal.innerHTML = `
            <div style="
                background: #1e293b;
                border: 2px solid #475569;
                border-radius: 8px;
                max-width: 800px;
                max-height: 80vh;
                overflow-y: auto;
                padding: 0;
            ">
                <div style="
                    background: #334155;
                    padding: 15px 20px;
                    border-bottom: 1px solid #475569;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <h2 style="margin: 0; color: #60a5fa; font-size: 18px;">⌨️ Keyboard Shortcuts</h2>
                    <button onclick="v7.keyboard.hideHelp()" style="
                        background: transparent;
                        border: none;
                        color: #94a3b8;
                        font-size: 24px;
                        cursor: pointer;
                        padding: 0;
                        width: 30px;
                        height: 30px;
                    ">×</button>
                </div>
                <div id="shortcutsContent" style="padding: 20px;"></div>
            </div>
        `;

        // Click outside to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                v7.keyboard.hideHelp();
            }
        });

        return modal;
    }

    /**
     * Populate help content with shortcuts grouped by category
     */
    function populateHelpContent(modal) {
        const content = modal.querySelector('#shortcutsContent');

        // Group shortcuts by category
        const categories = {};

        Object.entries(state.shortcuts).forEach(([key, shortcut]) => {
            const category = shortcut.category || 'Other';
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push({ key, ...shortcut });
        });

        // Build HTML
        let html = '';

        const categoryOrder = ['Mode', 'View', 'Navigation', 'Actions', 'Help', 'Debug', 'Other'];

        categoryOrder.forEach(category => {
            if (!categories[category]) return;

            html += `
                <div style="margin-bottom: 25px;">
                    <h3 style="
                        color: #94a3b8;
                        font-size: 14px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        margin: 0 0 10px 0;
                        border-bottom: 1px solid #334155;
                        padding-bottom: 5px;
                    ">${category}</h3>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 8px;">
            `;

            categories[category].forEach(shortcut => {
                html += `
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 8px 0;
                        border-bottom: 1px solid #334155;
                    ">
                        <span style="color: #cbd5e1; font-size: 13px;">${shortcut.description}</span>
                        <kbd style="
                            background: #0f172a;
                            border: 1px solid #475569;
                            border-radius: 3px;
                            padding: 4px 8px;
                            font-size: 12px;
                            color: #60a5fa;
                            min-width: 40px;
                            text-align: center;
                        ">${formatKey(shortcut.key)}</kbd>
                    </div>
                `;
            });

            html += '</div></div>';
        });

        content.innerHTML = html;
    }

    /**
     * Format key for display
     */
    function formatKey(key) {
        return key
            .replace('ctrl+', 'Ctrl+')
            .replace('shift+', 'Shift+')
            .replace('alt+', 'Alt+')
            .replace('ArrowLeft', '←')
            .replace('ArrowRight', '→')
            .replace('ArrowUp', '↑')
            .replace('ArrowDown', '↓')
            .replace(' ', 'Space');
    }

    /**
     * Helper functions for actions
     */
    function toggleOverlay() {
        const overlayCanvas = document.getElementById('hotspotOverlay');
        if (overlayCanvas) {
            overlayCanvas.style.display = overlayCanvas.style.display === 'none' ? 'block' : 'none';
        }
    }

    function toggle3DView() {
        console.log('[v7.keyboard] Toggle 3D view');
        // Implementation depends on UI structure
    }

    function toggleHeatmapView() {
        console.log('[v7.keyboard] Toggle heatmap view');
        // Implementation depends on UI structure
    }

    function closeAllPanels() {
        // Close drill-down panel
        if (v7.drilldown && v7.drilldown.hide) {
            v7.drilldown.hide();
        }

        // Close help
        v7.keyboard.hideHelp();

        // Clear selections
        clearCurrentSelection();
    }

    function copyResultToClipboard() {
        console.log('[v7.keyboard] Copy result to clipboard');
        // Implementation: get current result data and copy to clipboard
    }

    function saveCurrentResult() {
        console.log('[v7.keyboard] Save current result');
        // Implementation: trigger download of current result
    }

    function navigateResults(direction) {
        console.log('[v7.keyboard] Navigate results:', direction);
        // Implementation: go to previous/next result
    }

    function refreshCurrentView() {
        console.log('[v7.keyboard] Refresh view');
        window.location.reload();
    }

    function exportCurrentView() {
        console.log('[v7.keyboard] Export view');
        // Implementation: export current visualization
    }

    function clearCurrentSelection() {
        console.log('[v7.keyboard] Clear selection');
        // Clear heatmap brush selection
        if (v7.heatmap) {
            document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        }
    }

    function toggleDebugMode() {
        console.log('[v7.keyboard] Toggle debug mode');
        document.body.classList.toggle('debug-mode');
    }

    function trackKeySequence(key) {
        const now = Date.now();

        // Reset sequence if more than 1 second since last key
        if (now - state.lastKeyTime > 1000) {
            state.keySequence = [];
        }

        state.keySequence.push(key);
        state.lastKeyTime = now;

        // Easter egg: Konami code
        const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
        if (state.keySequence.length >= 10) {
            const recent = state.keySequence.slice(-10).join(',');
            if (recent === konamiCode.join(',')) {
                console.log('🎮 Konami Code activated!');
                state.keySequence = [];
            }
        }
    }

    /**
     * Enable/disable keyboard shortcuts
     */
    v7.keyboard.setEnabled = function(enabled) {
        state.enabled = enabled;
        console.log('[v7.keyboard] Shortcuts', enabled ? 'enabled' : 'disabled');
    };

    /**
     * Register custom shortcut
     */
    v7.keyboard.register = function(key, action, description, category = 'Custom') {
        state.shortcuts[key] = { action, description, category };
        console.log('[v7.keyboard] Registered shortcut:', key, '→', action);
    };

    /**
     * Unregister shortcut
     */
    v7.keyboard.unregister = function(key) {
        delete state.shortcuts[key];
        console.log('[v7.keyboard] Unregistered shortcut:', key);
    };

    console.log('[v7.keyboard] Keyboard shortcuts module loaded');
})();
