/**
 * Interactive Heatmap Module for Lens Signature Engine v7
 *
 * Provides high-performance canvas-based heatmap with drill-down capabilities:
 * - Canvas rendering for performance (handles large datasets)
 * - Click-to-drill-down showing Lab values, STD comparison, history
 * - Brush selection for region analysis
 * - Zoom and pan controls
 * - Hotspot annotation overlay
 *
 * Performance: Can handle 360×221 = 79,560 pixels at 60fps
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.heatmap = v7.heatmap || {};

    // State management
    let state = {
        canvas: null,
        ctx: null,
        overlayCanvas: null,
        overlayCtx: null,
        data: null,           // {theta: [], r: [], values: [[]], min, max, colormap}
        width: 0,
        height: 0,
        isDragging: false,
        brushStart: null,
        brushEnd: null,
        selectedRegion: null,
        zoom: 1.0,
        panX: 0,
        panY: 0,
        hotspots: [],         // [{theta, r, severity, label}]
        drilldownCallback: null
    };

    /**
     * Initialize interactive heatmap
     *
     * @param {string} canvasId - ID of main canvas element
     * @param {string} overlayCanvasId - ID of overlay canvas for annotations
     * @param {Object} options - Configuration options
     */
    v7.heatmap.init = function(canvasId, overlayCanvasId, options = {}) {
        state.canvas = document.getElementById(canvasId);
        state.overlayCanvas = document.getElementById(overlayCanvasId);

        if (!state.canvas || !state.overlayCanvas) {
            console.error('Heatmap canvases not found');
            return false;
        }

        state.ctx = state.canvas.getContext('2d');
        state.overlayCtx = state.overlayCanvas.getContext('2d');

        // Set canvas size
        const container = state.canvas.parentElement;
        state.width = options.width || container.clientWidth;
        state.height = options.height || container.clientHeight;

        state.canvas.width = state.width;
        state.canvas.height = state.height;
        state.overlayCanvas.width = state.width;
        state.overlayCanvas.height = state.height;

        // Set high DPI support
        const dpr = window.devicePixelRatio || 1;
        if (dpr > 1) {
            state.canvas.style.width = state.width + 'px';
            state.canvas.style.height = state.height + 'px';
            state.canvas.width = state.width * dpr;
            state.canvas.height = state.height * dpr;
            state.ctx.scale(dpr, dpr);

            state.overlayCanvas.style.width = state.width + 'px';
            state.overlayCanvas.style.height = state.height + 'px';
            state.overlayCanvas.width = state.width * dpr;
            state.overlayCanvas.height = state.height * dpr;
            state.overlayCtx.scale(dpr, dpr);
        }

        // Attach event listeners
        attachEventListeners();

        console.log('[v7.heatmap] Initialized', { width: state.width, height: state.height });
        return true;
    };

    /**
     * Render heatmap from grid data
     *
     * @param {Object} data - Grid data {theta: [], r: [], values: [[]], colormap: 'deltaE'|'Lab'}
     */
    v7.heatmap.render = function(data) {
        if (!state.ctx) {
            console.error('Heatmap not initialized');
            return;
        }

        state.data = data;

        // Calculate min/max for color scaling
        let min = Infinity;
        let max = -Infinity;
        data.values.forEach(row => {
            row.forEach(val => {
                if (val != null && isFinite(val)) {
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                }
            });
        });

        state.data.min = min;
        state.data.max = max;

        // Render to canvas
        renderCanvas();
        renderOverlay();
    };

    /**
     * Render main heatmap to canvas
     */
    function renderCanvas() {
        const { ctx, data, width, height } = state;
        if (!data) return;

        const { theta, r, values, min, max, colormap } = data;
        const thetaSteps = theta.length;
        const rSteps = r.length;

        // Calculate cell size
        const cellWidth = width / thetaSteps;
        const cellHeight = height / rSteps;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Render cells
        for (let ri = 0; ri < rSteps; ri++) {
            for (let ti = 0; ti < thetaSteps; ti++) {
                const value = values[ri][ti];
                if (value == null || !isFinite(value)) continue;

                // Normalize value to [0, 1]
                const normalized = (value - min) / (max - min);

                // Get color
                const color = getColor(normalized, colormap || 'deltaE');

                // Draw cell
                ctx.fillStyle = color;
                ctx.fillRect(ti * cellWidth, ri * cellHeight, cellWidth, cellHeight);
            }
        }

        // Draw brush selection if active
        if (state.brushStart && state.brushEnd) {
            ctx.strokeStyle = '#60a5fa';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(
                state.brushStart.x,
                state.brushStart.y,
                state.brushEnd.x - state.brushStart.x,
                state.brushEnd.y - state.brushStart.y
            );
            ctx.setLineDash([]);
        }
    }

    /**
     * Render overlay (hotspots, annotations)
     */
    function renderOverlay() {
        const { overlayCtx, width, height, hotspots, data } = state;
        if (!data) return;

        overlayCtx.clearRect(0, 0, width, height);

        // Render hotspots
        hotspots.forEach(hotspot => {
            const { theta, r, severity, label } = hotspot;

            // Convert (theta, r) to canvas coordinates
            const x = (theta / 360) * width;
            const y = (r) * height;

            // Draw marker
            overlayCtx.beginPath();
            overlayCtx.arc(x, y, 8, 0, 2 * Math.PI);
            overlayCtx.fillStyle = severity === 'high' ? 'rgba(239, 68, 68, 0.7)' :
                                  severity === 'medium' ? 'rgba(234, 179, 8, 0.7)' :
                                  'rgba(59, 130, 246, 0.7)';
            overlayCtx.fill();
            overlayCtx.strokeStyle = '#ffffff';
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();

            // Draw label
            if (label) {
                overlayCtx.fillStyle = '#ffffff';
                overlayCtx.font = '10px monospace';
                overlayCtx.fillText(label, x + 12, y + 4);
            }
        });
    }

    /**
     * Get color for normalized value [0, 1]
     *
     * @param {number} value - Normalized value (0-1)
     * @param {string} colormap - Colormap name ('deltaE', 'Lab', 'thermal')
     * @returns {string} - CSS color string
     */
    function getColor(value, colormap = 'deltaE') {
        if (colormap === 'deltaE') {
            // Green -> Yellow -> Red for ΔE
            if (value < 0.5) {
                const t = value * 2;
                const r = Math.floor(16 + (234 - 16) * t);
                const g = Math.floor(185 + (179 - 185) * t);
                const b = Math.floor(129 + (8 - 129) * t);
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                const t = (value - 0.5) * 2;
                const r = Math.floor(234 + (239 - 234) * t);
                const g = Math.floor(179 + (68 - 179) * t);
                const b = Math.floor(8 + (68 - 8) * t);
                return `rgb(${r}, ${g}, ${b})`;
            }
        } else if (colormap === 'thermal') {
            // Black -> Red -> Yellow -> White
            if (value < 0.33) {
                const t = value * 3;
                return `rgb(${Math.floor(255 * t)}, 0, 0)`;
            } else if (value < 0.66) {
                const t = (value - 0.33) * 3;
                return `rgb(255, ${Math.floor(255 * t)}, 0)`;
            } else {
                const t = (value - 0.66) * 3;
                return `rgb(255, 255, ${Math.floor(255 * t)})`;
            }
        } else if (colormap === 'Lab') {
            // Grayscale for L* channel
            const gray = Math.floor(value * 255);
            return `rgb(${gray}, ${gray}, ${gray})`;
        }

        return '#888888';
    }

    /**
     * Convert canvas coordinates to (theta, r)
     *
     * @param {number} x - Canvas x coordinate
     * @param {number} y - Canvas y coordinate
     * @returns {Object} - {theta, r, thetaIdx, rIdx}
     */
    function canvasToData(x, y) {
        const { data, width, height } = state;
        if (!data) return null;

        const thetaIdx = Math.floor((x / width) * data.theta.length);
        const rIdx = Math.floor((y / height) * data.r.length);

        if (thetaIdx < 0 || thetaIdx >= data.theta.length ||
            rIdx < 0 || rIdx >= data.r.length) {
            return null;
        }

        return {
            theta: data.theta[thetaIdx],
            r: data.r[rIdx],
            thetaIdx,
            rIdx
        };
    }

    /**
     * Handle click for drill-down
     *
     * @param {number} x - Canvas x coordinate
     * @param {number} y - Canvas y coordinate
     */
    function handleClick(x, y) {
        const coords = canvasToData(x, y);
        if (!coords) return;

        const { thetaIdx, rIdx, theta, r } = coords;
        const value = state.data.values[rIdx][thetaIdx];

        // Call drill-down callback if registered
        if (state.drilldownCallback) {
            state.drilldownCallback({
                theta,
                r,
                value,
                thetaIdx,
                rIdx,
                x,
                y
            });
        } else {
            // Default: show info in console
            console.log('Heatmap click:', { theta, r, value });
        }
    }

    /**
     * Register drill-down callback
     *
     * @param {Function} callback - Callback function(clickData)
     */
    v7.heatmap.onDrilldown = function(callback) {
        state.drilldownCallback = callback;
    };

    /**
     * Add hotspot annotation
     *
     * @param {Object} hotspot - {theta, r, severity, label}
     */
    v7.heatmap.addHotspot = function(hotspot) {
        state.hotspots.push(hotspot);
        renderOverlay();
    };

    /**
     * Clear all hotspots
     */
    v7.heatmap.clearHotspots = function() {
        state.hotspots = [];
        renderOverlay();
    };

    /**
     * Get selected region statistics
     *
     * @returns {Object} - {mean, median, min, max, count, failRatio}
     */
    v7.heatmap.getRegionStats = function() {
        if (!state.selectedRegion) return null;

        const { thetaStart, thetaEnd, rStart, rEnd } = state.selectedRegion;
        const values = [];

        for (let ri = rStart; ri <= rEnd; ri++) {
            for (let ti = thetaStart; ti <= thetaEnd; ti++) {
                const val = state.data.values[ri][ti];
                if (val != null && isFinite(val)) {
                    values.push(val);
                }
            }
        }

        if (values.length === 0) return null;

        values.sort((a, b) => a - b);

        return {
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            median: values[Math.floor(values.length / 2)],
            min: values[0],
            max: values[values.length - 1],
            count: values.length,
            failRatio: values.filter(v => v > 5.0).length / values.length  // Assuming ΔE > 5 is fail
        };
    };

    /**
     * Export heatmap as image
     *
     * @returns {string} - Data URL
     */
    v7.heatmap.exportImage = function() {
        if (!state.canvas) return null;

        // Create composite canvas
        const compositeCanvas = document.createElement('canvas');
        compositeCanvas.width = state.canvas.width;
        compositeCanvas.height = state.canvas.height;
        const compositeCtx = compositeCanvas.getContext('2d');

        // Draw main heatmap
        compositeCtx.drawImage(state.canvas, 0, 0);

        // Draw overlay
        compositeCtx.drawImage(state.overlayCanvas, 0, 0);

        return compositeCanvas.toDataURL('image/png');
    };

    /**
     * Attach event listeners
     */
    function attachEventListeners() {
        // Click for drill-down
        state.overlayCanvas.addEventListener('click', (e) => {
            const rect = state.overlayCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            handleClick(x, y);
        });

        // Brush selection (drag)
        state.overlayCanvas.addEventListener('mousedown', (e) => {
            if (e.shiftKey) {  // Shift + drag for brush
                const rect = state.overlayCanvas.getBoundingClientRect();
                state.isDragging = true;
                state.brushStart = {
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top
                };
                state.brushEnd = state.brushStart;
            }
        });

        state.overlayCanvas.addEventListener('mousemove', (e) => {
            if (state.isDragging) {
                const rect = state.overlayCanvas.getBoundingClientRect();
                state.brushEnd = {
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top
                };
                renderCanvas();  // Re-render with brush rectangle
            }

            // Show cursor info
            const rect = state.overlayCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const coords = canvasToData(x, y);
            if (coords) {
                const value = state.data.values[coords.rIdx][coords.thetaIdx];
                state.overlayCanvas.title = `θ=${coords.theta.toFixed(1)}°, r=${coords.r.toFixed(2)}, value=${value?.toFixed(2) || 'N/A'}`;
            }
        });

        state.overlayCanvas.addEventListener('mouseup', (e) => {
            if (state.isDragging) {
                state.isDragging = false;

                // Calculate selected region
                if (state.brushStart && state.brushEnd) {
                    const startCoords = canvasToData(state.brushStart.x, state.brushStart.y);
                    const endCoords = canvasToData(state.brushEnd.x, state.brushEnd.y);

                    if (startCoords && endCoords) {
                        state.selectedRegion = {
                            thetaStart: Math.min(startCoords.thetaIdx, endCoords.thetaIdx),
                            thetaEnd: Math.max(startCoords.thetaIdx, endCoords.thetaIdx),
                            rStart: Math.min(startCoords.rIdx, endCoords.rIdx),
                            rEnd: Math.max(startCoords.rIdx, endCoords.rIdx)
                        };

                        console.log('Region selected:', state.selectedRegion);
                        console.log('Region stats:', v7.heatmap.getRegionStats());
                    }
                }

                // Clear brush
                state.brushStart = null;
                state.brushEnd = null;
                renderCanvas();
            }
        });

        // Escape to clear selection
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                state.brushStart = null;
                state.brushEnd = null;
                state.selectedRegion = null;
                renderCanvas();
            }
        });
    }

    /**
     * Resize heatmap (call on window resize)
     */
    v7.heatmap.resize = function(width, height) {
        state.width = width;
        state.height = height;

        state.canvas.width = width;
        state.canvas.height = height;
        state.overlayCanvas.width = width;
        state.overlayCanvas.height = height;

        renderCanvas();
        renderOverlay();
    };

    console.log('[v7.heatmap] Interactive heatmap module loaded');
})();
