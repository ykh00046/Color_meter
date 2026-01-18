/**
 * Color Meter Web UI - Main Entry Point
 *
 * This is the unified ES6 module entry point for all JavaScript modules.
 * Import from this file for a clean, organized dependency structure.
 *
 * Usage in HTML:
 *   <script type="module">
 *     import { appState, apiClient } from '/static/js/index.js';
 *   </script>
 *
 * Or import specific modules:
 *   import { showNotification } from '/static/js/index.js';
 */

// =============================================================================
// Core Infrastructure
// =============================================================================
export { appState } from './core/state.js';
export { apiClient, v7Api } from './core/api.js';

// =============================================================================
// Utilities
// =============================================================================
export { byId, safeText, getReasonInfo, formatNumber } from './utils/helpers.js';
export { showNotification, hideNotification } from './utils/notifications.js';
export { copyToClipboard } from './utils/clipboard.js';
export { t, UI_TEXT } from './utils/i18n.js';

// =============================================================================
// Components
// =============================================================================
export { Components } from './components/base.js';
export { ImageViewer } from './components/viewer.js';
export { initResultTabs } from './components/tabs.js';
export * as visuals from './components/visuals.js';
export * as uiImprovements from './components/ui_improvements.js';

// =============================================================================
// Features - Inspection
// =============================================================================
export {
    initInspection,
    applyInspectionMode,
    runInspection,
} from './features/inspection/inspection.js';

export {
    loadProducts,
    setupProductListeners,
    ensureProductActive,
    setInspInkCountDisplay,
} from './features/inspection/products.js';

export * as diagnosticsVisuals from './features/inspection/diagnostics_visuals.js';
export * as gateVisual from './features/inspection/gate_visual.js';

// =============================================================================
// Features - Analysis
// =============================================================================
export { initSingleAnalysis } from './features/analysis/single.js';
export * as inkVisuals from './features/analysis/ink_visuals.js';
export * as simulationUI from './features/analysis/simulation_ui.js';

// =============================================================================
// Features - Registration
// =============================================================================
export { initRegistration } from './features/registration/registration.js';

// =============================================================================
// Features - History / STD Admin
// =============================================================================
export { initStdAdmin } from './features/history/std_admin.js';

// =============================================================================
// Features - Test Lab
// =============================================================================
export { initTestTab } from './features/test/test.js';

// =============================================================================
// Default Export - Full Module Object
// =============================================================================
import { appState } from './core/state.js';
import { apiClient, v7Api } from './core/api.js';
import * as helpers from './utils/helpers.js';
import * as notifications from './utils/notifications.js';
import * as clipboard from './utils/clipboard.js';
import * as i18n from './utils/i18n.js';
import * as visuals from './components/visuals.js';
import * as uiImprovements from './components/ui_improvements.js';

/**
 * Color Meter Module - All-in-one namespace
 * For backward compatibility and debugging
 */
const ColorMeter = {
    // Core
    appState,
    apiClient,
    v7Api,

    // Utils
    helpers,
    notifications,
    clipboard,
    i18n,

    // Components
    visuals,
    uiImprovements,

    // Version
    version: '7.0.0',
    moduleSystem: 'ES6',
};

// Expose to window for debugging (development only)
if (typeof window !== 'undefined') {
    window.ColorMeter = ColorMeter;
    window.CM = ColorMeter; // Short alias
}

export default ColorMeter;
