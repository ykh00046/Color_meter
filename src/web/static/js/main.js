import { LensViewer } from './viewer.js';
import { AnalysisCharts } from './charts.js';
import { UIControls } from './controls.js';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Modules
    const viewer = new LensViewer('viewer-canvas', 'viewer-container');
    const charts = new AnalysisCharts();
    const controls = new UIControls(viewer, charts);

    // 2. Start
    viewer.initPanzoom();
    charts.init();
    controls.init();

    // 3. Initialize Tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })
});
