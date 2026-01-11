/**
 * Drill-Down Panel Module for Lens Signature Engine v7
 *
 * Provides detailed inspection data when user clicks on heatmap:
 * - Lab values at clicked location
 * - Comparison to STD reference at same location
 * - Historical trend for that pixel (if available)
 * - Nearby hotspots and anomalies
 * - Region statistics (when brush selection used)
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.drilldown = v7.drilldown || {};

    let state = {
        panelElement: null,
        isVisible: false,
        currentData: null,
        history: []  // Historical data for clicked location
    };

    /**
     * Initialize drill-down panel
     *
     * @param {string} panelId - ID of panel container element
     */
    v7.drilldown.init = function(panelId) {
        state.panelElement = document.getElementById(panelId);

        if (!state.panelElement) {
            console.error('Drill-down panel element not found:', panelId);
            return false;
        }

        // Create panel structure if empty
        if (!state.panelElement.innerHTML.trim()) {
            createPanelStructure();
        }

        // Attach close button handler
        const closeBtn = state.panelElement.querySelector('.drilldown-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', v7.drilldown.hide);
        }

        console.log('[v7.drilldown] Drill-down panel initialized');
        return true;
    };

    /**
     * Create panel HTML structure
     */
    function createPanelStructure() {
        state.panelElement.innerHTML = `
            <div class="drilldown-header">
                <h3 class="drilldown-title">Pixel Details</h3>
                <button class="drilldown-close" title="Close (Esc)">×</button>
            </div>
            <div class="drilldown-body">
                <div class="drilldown-section">
                    <h4>Location</h4>
                    <div id="drilldown-location" class="drilldown-content"></div>
                </div>
                <div class="drilldown-section">
                    <h4>Lab Values</h4>
                    <div id="drilldown-lab" class="drilldown-content"></div>
                </div>
                <div class="drilldown-section">
                    <h4>STD Comparison</h4>
                    <div id="drilldown-std" class="drilldown-content"></div>
                </div>
                <div class="drilldown-section">
                    <h4>Analysis</h4>
                    <div id="drilldown-analysis" class="drilldown-content"></div>
                </div>
                <div class="drilldown-section drilldown-history hidden">
                    <h4>Historical Trend</h4>
                    <canvas id="drilldown-trend-chart" width="280" height="120"></canvas>
                </div>
            </div>
        `;
    }

    /**
     * Show drill-down panel with data
     *
     * @param {Object} data - Click data from heatmap
     *   - theta: Angular position (0-360)
     *   - r: Normalized radius (0-1)
     *   - value: ΔE or other metric
     *   - labTest: {L, a, b} - Test sample Lab values
     *   - labStd: {L, a, b} - STD reference Lab values
     *   - nearby: Array of nearby hotspots
     */
    v7.drilldown.show = function(data) {
        if (!state.panelElement) return;

        state.currentData = data;
        state.isVisible = true;

        // Populate sections
        populateLocation(data);
        populateLab(data);
        populateStdComparison(data);
        populateAnalysis(data);

        // Show history if available
        if (state.history.length > 0) {
            populateHistory(data);
            state.panelElement.querySelector('.drilldown-history').classList.remove('hidden');
        }

        // Show panel
        state.panelElement.classList.add('visible');
    };

    /**
     * Hide drill-down panel
     */
    v7.drilldown.hide = function() {
        if (!state.panelElement) return;

        state.isVisible = false;
        state.panelElement.classList.remove('visible');
    };

    /**
     * Toggle panel visibility
     */
    v7.drilldown.toggle = function() {
        if (state.isVisible) {
            v7.drilldown.hide();
        } else if (state.currentData) {
            v7.drilldown.show(state.currentData);
        }
    };

    /**
     * Populate location section
     */
    function populateLocation(data) {
        const container = document.getElementById('drilldown-location');
        if (!container) return;

        const { theta, r, thetaIdx, rIdx } = data;

        container.innerHTML = `
            <div class="drilldown-grid">
                <div class="drilldown-item">
                    <span class="label">Theta (θ):</span>
                    <span class="value">${theta.toFixed(1)}°</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">Radius (r):</span>
                    <span class="value">${r.toFixed(3)}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">Index:</span>
                    <span class="value">[${thetaIdx}, ${rIdx}]</span>
                </div>
            </div>
        `;
    }

    /**
     * Populate Lab values section
     */
    function populateLab(data) {
        const container = document.getElementById('drilldown-lab');
        if (!container) return;

        const { labTest, labStd } = data;

        if (!labTest) {
            container.innerHTML = '<p class="text-muted">Lab data not available</p>';
            return;
        }

        const labHtml = `
            <div class="drilldown-grid">
                <div class="drilldown-item">
                    <span class="label">L* (Test):</span>
                    <span class="value">${labTest.L?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">a* (Test):</span>
                    <span class="value">${labTest.a?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">b* (Test):</span>
                    <span class="value">${labTest.b?.toFixed(2) || 'N/A'}</span>
                </div>
            </div>
        `;

        container.innerHTML = labHtml;
    }

    /**
     * Populate STD comparison section
     */
    function populateStdComparison(data) {
        const container = document.getElementById('drilldown-std');
        if (!container) return;

        const { labTest, labStd, value } = data;

        if (!labStd) {
            container.innerHTML = '<p class="text-muted">STD reference not available</p>';
            return;
        }

        // Calculate component-wise differences
        const deltaL = labTest && labTest.L != null ? (labTest.L - labStd.L) : null;
        const deltaA = labTest && labTest.a != null ? (labTest.a - labStd.a) : null;
        const deltaB = labTest && labTest.b != null ? (labTest.b - labStd.b) : null;

        const stdHtml = `
            <div class="drilldown-grid">
                <div class="drilldown-item">
                    <span class="label">ΔE (Total):</span>
                    <span class="value ${getValueClass(value)}">${value?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">ΔL*:</span>
                    <span class="value">${deltaL?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">Δa*:</span>
                    <span class="value">${deltaA?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="drilldown-item">
                    <span class="label">Δb*:</span>
                    <span class="value">${deltaB?.toFixed(2) || 'N/A'}</span>
                </div>
            </div>
            <div class="drilldown-ref">
                <p class="text-muted">STD Reference: L*=${labStd.L?.toFixed(1)}, a*=${labStd.a?.toFixed(1)}, b*=${labStd.b?.toFixed(1)}</p>
            </div>
        `;

        container.innerHTML = stdHtml;
    }

    /**
     * Populate analysis section
     */
    function populateAnalysis(data) {
        const container = document.getElementById('drilldown-analysis');
        if (!container) return;

        const { value, nearby, theta, r } = data;

        let status = 'PASS';
        let statusClass = 'status-ok';
        let message = 'Within tolerance';

        if (value != null) {
            if (value > 5.0) {
                status = 'FAIL';
                statusClass = 'status-fail';
                message = 'Exceeds ΔE threshold (>5.0)';
            } else if (value > 3.0) {
                status = 'WARN';
                statusClass = 'status-warn';
                message = 'Approaching threshold (>3.0)';
            }
        }

        // Check for nearby hotspots
        let nearbyHtml = '';
        if (nearby && nearby.length > 0) {
            nearbyHtml = '<div class="drilldown-nearby"><strong>Nearby Issues:</strong><ul>';
            nearby.forEach(h => {
                nearbyHtml += `<li>${h.label || 'Hotspot'} (${h.severity})</li>`;
            });
            nearbyHtml += '</ul></div>';
        }

        const analysisHtml = `
            <div class="drilldown-status ${statusClass}">
                <span class="status-badge">${status}</span>
                <span class="status-message">${message}</span>
            </div>
            ${nearbyHtml}
            <div class="drilldown-actions">
                <button class="btn-drilldown" onclick="v7.drilldown.markAsHotspot()">Mark as Hotspot</button>
                <button class="btn-drilldown" onclick="v7.drilldown.copyCoords()">Copy Coords</button>
            </div>
        `;

        container.innerHTML = analysisHtml;
    }

    /**
     * Populate history trend chart
     */
    function populateHistory(data) {
        const canvas = document.getElementById('drilldown-trend-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const { theta, r } = data;

        // Filter history for this location (within tolerance)
        const locationHistory = state.history.filter(h => {
            return Math.abs(h.theta - theta) < 5 && Math.abs(h.r - r) < 0.05;
        });

        if (locationHistory.length === 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#94a3b8';
            ctx.font = '12px monospace';
            ctx.fillText('No historical data', 10, 60);
            return;
        }

        // Simple line chart
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const padding = 20;
        const chartWidth = canvas.width - 2 * padding;
        const chartHeight = canvas.height - 2 * padding;

        // Find min/max
        const values = locationHistory.map(h => h.value);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;

        // Draw axes
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();

        // Draw threshold line (ΔE = 5.0)
        if (maxVal > 5.0) {
            const thresholdY = canvas.height - padding - ((5.0 - minVal) / range) * chartHeight;
            ctx.strokeStyle = '#ef4444';
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(padding, thresholdY);
            ctx.lineTo(canvas.width - padding, thresholdY);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw line
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        locationHistory.forEach((point, idx) => {
            const x = padding + (idx / (locationHistory.length - 1)) * chartWidth;
            const y = canvas.height - padding - ((point.value - minVal) / range) * chartHeight;

            if (idx === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw points
        ctx.fillStyle = '#60a5fa';
        locationHistory.forEach((point, idx) => {
            const x = padding + (idx / (locationHistory.length - 1)) * chartWidth;
            const y = canvas.height - padding - ((point.value - minVal) / range) * chartHeight;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px monospace';
        ctx.fillText(`Min: ${minVal.toFixed(1)}`, padding, canvas.height - 5);
        ctx.fillText(`Max: ${maxVal.toFixed(1)}`, canvas.width - 60, canvas.height - 5);
        ctx.fillText(`n=${locationHistory.length}`, canvas.width - 30, padding + 10);
    }

    /**
     * Get CSS class based on value
     */
    function getValueClass(value) {
        if (value == null) return '';
        if (value > 5.0) return 'text-danger';
        if (value > 3.0) return 'text-warning';
        return 'text-success';
    }

    /**
     * Mark current location as hotspot
     */
    v7.drilldown.markAsHotspot = function() {
        if (!state.currentData) return;

        const { theta, r, value } = state.currentData;
        const severity = value > 5.0 ? 'high' : value > 3.0 ? 'medium' : 'low';

        if (v7.heatmap && v7.heatmap.addHotspot) {
            v7.heatmap.addHotspot({
                theta,
                r,
                severity,
                label: `ΔE=${value.toFixed(1)}`
            });

            console.log('Hotspot added:', { theta, r, severity });
            alert(`Hotspot marked at θ=${theta.toFixed(1)}°, r=${r.toFixed(2)}`);
        }
    };

    /**
     * Copy coordinates to clipboard
     */
    v7.drilldown.copyCoords = function() {
        if (!state.currentData) return;

        const { theta, r, value } = state.currentData;
        const text = `θ=${theta.toFixed(1)}°, r=${r.toFixed(3)}, ΔE=${value?.toFixed(2) || 'N/A'}`;

        navigator.clipboard.writeText(text).then(() => {
            console.log('Coordinates copied:', text);
            alert('Coordinates copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy:', err);
        });
    };

    /**
     * Add historical data point
     *
     * @param {Object} point - {theta, r, value, timestamp}
     */
    v7.drilldown.addHistory = function(point) {
        state.history.push({
            ...point,
            timestamp: point.timestamp || new Date().toISOString()
        });

        // Limit history size
        if (state.history.length > 100) {
            state.history.shift();
        }
    };

    /**
     * Clear historical data
     */
    v7.drilldown.clearHistory = function() {
        state.history = [];
    };

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.isVisible) {
            v7.drilldown.hide();
        }
    });

    console.log('[v7.drilldown] Drill-down panel module loaded');
})();
