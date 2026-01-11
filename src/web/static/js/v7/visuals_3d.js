/**
 * 3D Visualization Module for Lens Signature Engine v7
 *
 * Provides advanced 3D visualization using Plotly.js:
 * - Cylindrical Mode: Wrap theta around cylinder, r as radius, Lab/ΔE as height
 * - Surface Mode: 3D surface plot of ΔE(theta, r)
 * - Scatter Mode: Test points colored by ΔE
 *
 * Requires: Plotly.js (loaded via CDN)
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.visuals3d = v7.visuals3d || {};

    // Configuration
    const CONFIG = {
        cylindrical: {
            layout: {
                scene: {
                    xaxis: { title: 'X' },
                    yaxis: { title: 'Y' },
                    zaxis: { title: 'ΔE' },
                    camera: {
                        eye: { x: 1.5, y: 1.5, z: 1.2 }
                    },
                    aspectmode: 'auto'
                },
                margin: { l: 0, r: 0, b: 0, t: 40 },
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#1e293b',
                font: { color: '#94a3b8', family: 'monospace', size: 10 }
            },
            config: {
                displayModeBar: true,
                displaylogo: false,
                responsive: true
            }
        },
        surface: {
            layout: {
                scene: {
                    xaxis: { title: 'Theta (°)' },
                    yaxis: { title: 'Radius (norm)' },
                    zaxis: { title: 'ΔE' },
                    camera: {
                        eye: { x: 1.5, y: 1.5, z: 1.2 }
                    }
                },
                margin: { l: 0, r: 0, b: 0, t: 40 },
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#1e293b',
                font: { color: '#94a3b8', family: 'monospace', size: 10 }
            },
            config: {
                displayModeBar: true,
                displaylogo: false,
                responsive: true
            }
        },
        scatter: {
            layout: {
                scene: {
                    xaxis: { title: 'a*' },
                    yaxis: { title: 'b*' },
                    zaxis: { title: 'L*' },
                    camera: {
                        eye: { x: 1.5, y: 1.5, z: 1.2 }
                    }
                },
                margin: { l: 0, r: 0, b: 0, t: 40 },
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#1e293b',
                font: { color: '#94a3b8', family: 'monospace', size: 10 }
            },
            config: {
                displayModeBar: true,
                displaylogo: false,
                responsive: true
            }
        }
    };

    /**
     * Check if Plotly is loaded
     */
    function checkPlotly() {
        if (typeof Plotly === 'undefined') {
            console.error('Plotly.js is not loaded. Include it in your HTML: <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>');
            return false;
        }
        return true;
    }

    /**
     * Render cylindrical visualization
     * Wraps theta around cylinder, r as radius, ΔE as height
     *
     * @param {string} containerId - ID of div element
     * @param {Object} polarData - Polar coordinate data with theta, r, deltaE arrays
     * @param {string} title - Plot title
     */
    v7.visuals3d.renderCylindrical = function(containerId, polarData, title = '3D Cylindrical View') {
        if (!checkPlotly()) return;

        const { theta, r, deltaE } = polarData;

        if (!theta || !r || !deltaE) {
            console.error('Invalid polar data: missing theta, r, or deltaE');
            return;
        }

        // Convert polar to cylindrical coordinates (x, y, z)
        // theta wraps around cylinder, r determines radial distance from center
        // z = deltaE (height)
        const x = [];
        const y = [];
        const z = [];
        const colors = [];

        for (let i = 0; i < theta.length; i++) {
            const angle = theta[i] * Math.PI / 180;  // Convert to radians
            const radius = r[i];
            const height = deltaE[i];

            // Cylindrical: x = r*cos(theta), y = r*sin(theta), z = deltaE
            x.push(radius * Math.cos(angle));
            y.push(radius * Math.sin(angle));
            z.push(height);
            colors.push(height);
        }

        const trace = {
            type: 'scatter3d',
            mode: 'markers',
            x: x,
            y: y,
            z: z,
            marker: {
                size: 3,
                color: colors,
                colorscale: [
                    [0, '#10b981'],      // Green (low ΔE)
                    [0.5, '#eab308'],    // Yellow
                    [1, '#ef4444']       // Red (high ΔE)
                ],
                colorbar: {
                    title: 'ΔE',
                    titleside: 'right',
                    tickfont: { color: '#94a3b8' },
                    bgcolor: '#1e293b',
                    outlinecolor: '#475569'
                },
                showscale: true
            },
            text: theta.map((t, i) => `θ=${t.toFixed(1)}°, r=${r[i].toFixed(2)}, ΔE=${deltaE[i].toFixed(2)}`),
            hoverinfo: 'text'
        };

        const layout = JSON.parse(JSON.stringify(CONFIG.cylindrical.layout));
        layout.title = { text: title, font: { color: '#cbd5e1' } };

        Plotly.newPlot(containerId, [trace], layout, CONFIG.cylindrical.config);
    };

    /**
     * Render surface plot of ΔE(theta, r)
     *
     * @param {string} containerId - ID of div element
     * @param {Object} gridData - Grid data with theta array, r array, and deltaE 2D array
     * @param {string} title - Plot title
     */
    v7.visuals3d.renderSurface = function(containerId, gridData, title = '3D Surface Plot - ΔE(θ, r)') {
        if (!checkPlotly()) return;

        const { theta, r, deltaE2D } = gridData;

        if (!theta || !r || !deltaE2D) {
            console.error('Invalid grid data: missing theta, r, or deltaE2D');
            return;
        }

        const trace = {
            type: 'surface',
            x: theta,  // Theta values (0-360)
            y: r,      // Normalized radius values (0-1)
            z: deltaE2D,  // 2D array: deltaE2D[r_idx][theta_idx]
            colorscale: [
                [0, '#10b981'],      // Green (low ΔE)
                [0.5, '#eab308'],    // Yellow
                [1, '#ef4444']       // Red (high ΔE)
            ],
            colorbar: {
                title: 'ΔE',
                titleside: 'right',
                tickfont: { color: '#94a3b8' },
                bgcolor: '#1e293b',
                outlinecolor: '#475569'
            },
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: '#ffffff',
                    project: { z: true }
                }
            },
            hovertemplate: 'θ=%{x:.1f}°<br>r=%{y:.2f}<br>ΔE=%{z:.2f}<extra></extra>'
        };

        const layout = JSON.parse(JSON.stringify(CONFIG.surface.layout));
        layout.title = { text: title, font: { color: '#cbd5e1' } };

        Plotly.newPlot(containerId, [trace], layout, CONFIG.surface.config);
    };

    /**
     * Render scatter plot in Lab color space
     * Shows test points vs STD reference with trajectory
     *
     * @param {string} containerId - ID of div element
     * @param {Object} labData - Lab space data with test and reference points
     * @param {string} title - Plot title
     */
    v7.visuals3d.renderLabScatter = function(containerId, labData, title = '3D Lab Space Scatter') {
        if (!checkPlotly()) return;

        const { test, reference, trajectory } = labData;

        const traces = [];

        // STD reference points (gold diamonds)
        if (reference && reference.L && reference.a && reference.b) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                name: 'STD Reference',
                x: reference.a,
                y: reference.b,
                z: reference.L,
                marker: {
                    size: 6,
                    color: '#fbbf24',  // Gold
                    symbol: 'diamond',
                    line: { color: '#f59e0b', width: 1 }
                },
                hovertemplate: 'STD<br>L*=%{z:.1f}<br>a*=%{x:.1f}<br>b*=%{y:.1f}<extra></extra>'
            });
        }

        // Test sample points (colored by ΔE)
        if (test && test.L && test.a && test.b && test.deltaE) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                name: 'Test Sample',
                x: test.a,
                y: test.b,
                z: test.L,
                marker: {
                    size: 4,
                    color: test.deltaE,
                    colorscale: [
                        [0, '#10b981'],      // Green
                        [0.5, '#eab308'],    // Yellow
                        [1, '#ef4444']       // Red
                    ],
                    colorbar: {
                        title: 'ΔE',
                        titleside: 'right',
                        tickfont: { color: '#94a3b8' },
                        bgcolor: '#1e293b',
                        outlinecolor: '#475569'
                    },
                    showscale: true
                },
                hovertemplate: 'Test<br>L*=%{z:.1f}<br>a*=%{x:.1f}<br>b*=%{y:.1f}<br>ΔE=%{marker.color:.2f}<extra></extra>'
            });
        }

        // Trajectory path (for per-color models showing LOW/MID/HIGH path)
        if (trajectory && trajectory.length > 0) {
            trajectory.forEach((traj, idx) => {
                if (traj.L && traj.a && traj.b) {
                    traces.push({
                        type: 'scatter3d',
                        mode: 'lines+markers',
                        name: traj.name || `Path ${idx}`,
                        x: traj.a,
                        y: traj.b,
                        z: traj.L,
                        line: {
                            color: traj.color || '#60a5fa',
                            width: 3
                        },
                        marker: {
                            size: 5,
                            color: traj.color || '#60a5fa'
                        },
                        hovertemplate: `${traj.name || 'Path'}<br>L*=%{z:.1f}<br>a*=%{x:.1f}<br>b*=%{y:.1f}<extra></extra>`
                    });
                }
            });
        }

        const layout = JSON.parse(JSON.stringify(CONFIG.scatter.layout));
        layout.title = { text: title, font: { color: '#cbd5e1' } };
        layout.legend = {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: '#475569',
            borderwidth: 1,
            font: { color: '#94a3b8' }
        };

        Plotly.newPlot(containerId, traces, layout, CONFIG.scatter.config);
    };

    /**
     * Render per-color comparison in 3D
     * Shows each ink color as a separate trace
     *
     * @param {string} containerId - ID of div element
     * @param {Object} perColorData - Per-color Lab data
     * @param {string} title - Plot title
     */
    v7.visuals3d.renderPerColorComparison = function(containerId, perColorData, title = 'Per-Color 3D Comparison') {
        if (!checkPlotly()) return;

        const { colors } = perColorData;  // Array of {color_id, test, reference, color_hex}

        if (!colors || colors.length === 0) {
            console.warn('No color data provided for per-color comparison');
            return;
        }

        const traces = [];

        colors.forEach((colorData, idx) => {
            const { color_id, test, reference, color_hex, role } = colorData;

            // Only show ink colors (skip gaps)
            if (role !== 'ink') return;

            // Reference centroid (larger marker)
            if (reference && reference.L != null) {
                traces.push({
                    type: 'scatter3d',
                    mode: 'markers',
                    name: `${color_id} STD`,
                    x: [reference.a],
                    y: [reference.b],
                    z: [reference.L],
                    marker: {
                        size: 10,
                        color: color_hex || '#fbbf24',
                        symbol: 'diamond',
                        line: { color: '#ffffff', width: 2 }
                    },
                    hovertemplate: `${color_id} STD<br>L*=%{z:.1f}<br>a*=%{x:.1f}<br>b*=%{y:.1f}<extra></extra>`
                });
            }

            // Test samples (colored by actual Lab color)
            if (test && test.L && test.a && test.b) {
                traces.push({
                    type: 'scatter3d',
                    mode: 'markers',
                    name: `${color_id} Test`,
                    x: test.a,
                    y: test.b,
                    z: test.L,
                    marker: {
                        size: 3,
                        color: color_hex || `hsl(${idx * 60}, 70%, 50%)`,
                        opacity: 0.6
                    },
                    hovertemplate: `${color_id} Test<br>L*=%{z:.1f}<br>a*=%{x:.1f}<br>b*=%{y:.1f}<extra></extra>`
                });
            }
        });

        const layout = JSON.parse(JSON.stringify(CONFIG.scatter.layout));
        layout.title = { text: title, font: { color: '#cbd5e1' } };
        layout.legend = {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: '#475569',
            borderwidth: 1,
            font: { color: '#94a3b8', size: 9 }
        };

        Plotly.newPlot(containerId, traces, layout, CONFIG.scatter.config);
    };

    /**
     * Utility: Convert polar grid data to format needed for surface plot
     *
     * @param {Array} polarMap - Flattened array with {theta, r, deltaE} objects
     * @param {number} thetaSteps - Number of theta bins (e.g., 360)
     * @param {number} rSteps - Number of radial bins (e.g., 221)
     * @returns {Object} - {theta: [], r: [], deltaE2D: [[]]}
     */
    v7.visuals3d.convertPolarToGrid = function(polarMap, thetaSteps = 360, rSteps = 221) {
        const theta = Array.from({ length: thetaSteps }, (_, i) => i);
        const r = Array.from({ length: rSteps }, (_, i) => i / (rSteps - 1));
        const deltaE2D = Array(rSteps).fill(null).map(() => Array(thetaSteps).fill(0));

        polarMap.forEach(point => {
            const tIdx = Math.floor(point.theta) % thetaSteps;
            const rIdx = Math.floor(point.r * (rSteps - 1));
            if (tIdx >= 0 && tIdx < thetaSteps && rIdx >= 0 && rIdx < rSteps) {
                deltaE2D[rIdx][tIdx] = point.deltaE || 0;
            }
        });

        return { theta, r, deltaE2D };
    };

    /**
     * Utility: Resize all Plotly plots on window resize
     */
    v7.visuals3d.resizeAll = function() {
        if (typeof Plotly !== 'undefined') {
            document.querySelectorAll('.plotly-graph-div').forEach(el => {
                Plotly.Plots.resize(el);
            });
        }
    };

    // Auto-resize on window resize
    window.addEventListener('resize', v7.visuals3d.resizeAll);

    console.log('[v7.visuals3d] 3D visualization module loaded');
})();
