/**
 * Charts Visualization Module
 * Handles Chart.js rendering for Radial Profiles and Delta E Trends
 */

export class AnalysisCharts {
    constructor() {
        this.profileChart = null;
        this.deltaEChart = null;
    }

    init() {
        // Initialize empty charts or wait for data
        // We'll initialize on data update
    }

    /**
     * Render the Radial Profile Chart
     * @param {Object} analysisData Data from AnalysisService (ProfileAnalysisResult)
     */
    updateProfileChart(analysisData) {
        const ctx = document.getElementById('profile-chart').getContext('2d');

        // Data mapping: The API returns keys directly in analysisData
        // e.g. radius, L_smoothed, a_smoothed, b_smoothed

        if (!analysisData || !analysisData.radius) return;

        if (this.profileChart) {
            this.profileChart.destroy();
        }

        this.profileChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: analysisData.radius.map(r => r.toFixed(2)), // Reduce label precision
                datasets: [
                    {
                        label: 'L*',
                        data: analysisData.L_smoothed,
                        borderColor: '#ef4444', // Red
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'a*',
                        data: analysisData.a_smoothed,
                        borderColor: '#10b981', // Green
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'b*',
                        data: analysisData.b_smoothed,
                        borderColor: '#3b82f6', // Blue
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Radial Color Profile (Smoothed)'
                    },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Normalized Radius'
                        },
                        ticks: {
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Lab Value'
                        }
                    }
                }
            }
        });
    }

    /**
     * Render the Delta E Bar Chart
     * @param {Array} zoneResults Array of ZoneResult objects
     */
    updateDeltaEChart(zoneResults) {
        const ctx = document.getElementById('delta-e-chart').getContext('2d');

        if (!zoneResults || zoneResults.length === 0) return;

        const labels = zoneResults.map(z => z.zone_name);
        const values = zoneResults.map(z => z.delta_e);
        const backgrounds = values.map(v => {
            if (v <= 2.0) return 'rgba(16, 185, 129, 0.7)'; // Green
            if (v <= 4.0) return 'rgba(245, 158, 11, 0.7)'; // Orange
            return 'rgba(239, 68, 68, 0.7)'; // Red
        });

        if (this.deltaEChart) {
            this.deltaEChart.destroy();
        }

        this.deltaEChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'ΔE (vs Target)',
                    data: values,
                    backgroundColor: backgrounds,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Zone Delta E Analysis'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'ΔE (CIEDE2000)'
                        }
                    }
                }
            }
        });
    }

    /**
     * Render the Sector Heatmap using Canvas API
     * @param {Array} cells RingSectorCell data
     * @param {Object} uniformity Uniformity analysis data
     */
    updateHeatmap(cells, uniformity) {
        const canvas = document.getElementById('heatmap-canvas');
        const placeholder = document.getElementById('heatmap-placeholder');

        if (!cells || cells.length === 0) {
            if (placeholder) placeholder.style.display = 'block';
            return;
        }

        if (placeholder) placeholder.style.display = 'none';

        const ctx = canvas.getContext('2d');
        const container = canvas.parentElement;

        // Resize canvas to fit container
        const size = Math.min(container.clientWidth, container.clientHeight) * 0.95;
        canvas.width = size;
        canvas.height = size;

        const cx = size / 2;
        const cy = size / 2;
        const maxRadius = size / 2; // Outer radius of the heatmap

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Calculate Delta E for each cell relative to global mean
        const globalMean = uniformity ? uniformity.global_mean_lab : [50, 0, 0];

        cells.forEach(cell => {
            // Parse angle range "0-30°"
            const angleParts = cell.angle.replace('°', '').split('-');
            const startAngleDeg = parseFloat(angleParts[0]);
            const endAngleDeg = parseFloat(angleParts[1]);

            // Convert to radians (0 deg = 3 o'clock, clockwise)
            const startRad = (startAngleDeg * Math.PI) / 180;
            const endRad = (endAngleDeg * Math.PI) / 180;

            // Map normalized radius (0.0-1.0) to canvas radius
            // We assume r_range is normalized.
            const innerR = cell.r_range[0] * maxRadius;
            const outerR = cell.r_range[1] * maxRadius;

            // Calculate Delta E
            const dE = this.calculateSimpleDeltaE(cell.lab, globalMean);

            // Determine Color
            let fillStyle;
            if (dE < 2.0) fillStyle = '#10b981'; // Green
            else if (dE < 4.0) fillStyle = '#f59e0b'; // Orange
            else fillStyle = '#ef4444'; // Red

            // Draw Sector
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, startRad, endRad);
            ctx.arc(cx, cy, innerR, endRad, startRad, true); // Draw inner arc in reverse
            ctx.closePath();
            ctx.fillStyle = fillStyle;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Highlight outliers
            if (uniformity && uniformity.outlier_cells) {
                const isOutlier = uniformity.outlier_cells.some(
                    o => o.ring === cell.ring && o.sector === cell.sector
                );
                if (isOutlier) {
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#dc2626'; // Dark Red
                    ctx.stroke();
                    ctx.lineWidth = 1; // Reset
                }
            }
        });
    }

    calculateSimpleDeltaE(lab1, lab2) {
        // Simple Euclidean distance for visualization (CIE76)
        return Math.sqrt(
            Math.pow(lab1[0] - lab2[0], 2) +
            Math.pow(lab1[1] - lab2[1], 2) +
            Math.pow(lab1[2] - lab2[2], 2)
        );
    }
}
