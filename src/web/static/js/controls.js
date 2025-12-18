/**
 * UI Controls Module
 * Handles user input, settings updates, and API calls
 */

export class UIControls {
    constructor(viewer, charts) {
        this.viewer = viewer;
        this.charts = charts;

        // State
        this.currentFile = null;
        this.currentSku = 'SKU001'; // Default, should load from API
        this.currentRunId = null;
    }

    init() {
        this.bindEvents();
        this.loadSkuList();
    }

    bindEvents() {
        // 1. Image Upload
        const fileInput = document.getElementById('file-upload');
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.currentFile = e.target.files[0];
                const url = URL.createObjectURL(this.currentFile);
                this.viewer.loadImage(url);

                // Reset result tabs
                this.resetResults();
            }
        });

        // 2. Viewer Controls
        document.getElementById('inner-radius').addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            document.getElementById('inner-val').textContent = val.toFixed(2);
            this.viewer.updateSettings({ innerRadius: val });
        });

        document.getElementById('outer-radius').addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            document.getElementById('outer-val').textContent = val.toFixed(2);
            this.viewer.updateSettings({ outerRadius: val });
        });

        document.getElementById('ring-count').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('ring-val').textContent = val;
            this.viewer.updateSettings({ ringCount: val });
        });

        document.getElementById('sector-count').addEventListener('change', (e) => {
            const val = parseInt(e.target.value);
            this.viewer.updateSettings({ sectorCount: val });
        });

        document.querySelectorAll('input[name="ring-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.viewer.updateSettings({ ringMode: e.target.value });
            });
        });

        // 3. Analysis Action
        document.getElementById('btn-analyze').addEventListener('click', () => {
            this.runInspection();
        });

        // 4. Export Actions
        document.getElementById('btn-download-json').addEventListener('click', (e) => {
            e.preventDefault();
            if (this.currentRunId) window.open(`/results/${this.currentRunId}/result.json`, '_blank');
        });

        document.getElementById('btn-download-overlay').addEventListener('click', (e) => {
            e.preventDefault();
            if (this.currentRunId) window.open(`/results/${this.currentRunId}/overlay.png`, '_blank');
        });
    }

    async loadSkuList() {
        // TODO: Implement /sku/list API if available, or hardcode for now
        // For now, populate with sample
        const select = document.getElementById('sku-select');
        select.innerHTML = `
            <option value="SKU001">SKU001</option>
            <option value="SKU_EXAMPLE">SKU_EXAMPLE</option>
        `;
    }

    async runInspection() {
        if (!this.currentFile) {
            alert("Please upload an image first.");
            return;
        }

        const btn = document.getElementById('btn-analyze');
        btn.disabled = true;
        btn.textContent = "Analyzing...";

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('sku', document.getElementById('sku-select').value);

            // Pass the expected zones hint from UI
            const rings = document.getElementById('ring-count').value;
            formData.append('expected_zones', rings);
            formData.append('run_judgment', 'true');

            // Enable 2D zone analysis (AI template)
            formData.append('use_2d_analysis', 'true');

            // Advanced Options
            const options = {
                do_normalize: document.getElementById('check-normalize').checked,
                qc_only: document.getElementById('check-qc-only').checked,
                illumination_correction: document.getElementById('check-illumination').checked,
                scale_normalize: true, // Default fixed per specs
                return_preview: false  // Default fixed per specs
            };
            formData.append('options', JSON.stringify(options));

            const response = await fetch('/inspect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Inspection failed");
            }

            const result = await response.json();

            this.handleAnalysisResult(result);

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = "Run Analysis";
        }
    }

    handleAnalysisResult(result) {
        console.log("Analysis Result:", result);

        // 1. Update Viewer Detection
        if (result.lens_detection) {
            this.viewer.setLensDetection(result.lens_detection);
        }

        // 2. Update Summary Table & Uniformity Status
        this.renderSummaryTable(result.zone_results, result.judgment);
        if (result.uniformity) {
            this.renderUniformityStatus(result.uniformity);
        }

        // 3. Update Charts & Heatmap
        if (result.analysis) {
            this.charts.updateProfileChart(result.analysis);
        }
        if (result.zone_results) {
            this.charts.updateDeltaEChart(result.zone_results);
        }
        if (result.ring_sector_cells) {
            this.charts.updateHeatmap(result.ring_sector_cells, result.uniformity);
        }

        // 4. Update Header Badge
        const badge = document.getElementById('judgment-badge');
        if (result.judgment) {
            badge.textContent = result.judgment.result;
            badge.className = `badge ${result.judgment.result === 'OK' ? 'bg-success' : 'bg-danger'}`;
        } else {
            // result.judgment might be null if run_judgment was false,
            // but we forced it to true. Check structure.
            // If direct result object:
            if (result.judgment === 'OK' || result.judgment === 'NG') {
                 badge.textContent = result.judgment;
                 badge.className = `badge ${result.judgment === 'OK' ? 'bg-success' : 'bg-danger'}`;
            }
        }

        // 5. Enable Export
        if (result.run_id) {
            this.currentRunId = result.run_id;
            document.getElementById('dropdownExport').disabled = false;
        }
    }

    renderUniformityStatus(uniformity) {
        const card = document.getElementById('uniformity-card');
        const badge = document.getElementById('uniformity-badge');
        const bar = document.getElementById('uniformity-bar');
        const maxDe = document.getElementById('uni-max-de');
        const outliers = document.getElementById('uni-outliers');

        if (!uniformity) {
            card.style.display = 'none';
            return;
        }

        card.style.display = 'block';

        // Badge
        badge.textContent = uniformity.is_uniform ? 'UNIFORM' : 'NON-UNIFORM';
        badge.className = `badge ${uniformity.is_uniform ? 'bg-success' : 'bg-danger'}`;

        // Confidence Bar
        const confPercent = (uniformity.confidence * 100).toFixed(0);
        bar.style.width = `${confPercent}%`;
        bar.className = `progress-bar ${uniformity.confidence > 0.8 ? 'bg-success' : uniformity.confidence > 0.6 ? 'bg-warning' : 'bg-danger'}`;

        // Stats
        maxDe.textContent = uniformity.max_delta_e.toFixed(2);
        outliers.textContent = uniformity.outlier_cells ? uniformity.outlier_cells.length : 0;
    }

    renderSummaryTable(zoneResults, judgmentInfo) {
        const tbody = document.getElementById('summary-table-body');
        tbody.innerHTML = '';

        if (!zoneResults) return;

        zoneResults.forEach(zone => {
            const tr = document.createElement('tr');

            const labStr = `L:${zone.measured_lab[0].toFixed(1)} a:${zone.measured_lab[1].toFixed(1)} b:${zone.measured_lab[2].toFixed(1)}`;
            const deltaE = zone.delta_e.toFixed(2);

            // Determine row color based on threshold
            const isOk = zone.is_ok;
            const statusClass = isOk ? 'text-success' : 'text-danger fw-bold';

            tr.innerHTML = `
                <td>${zone.zone_name}</td>
                <td>${labStr}</td>
                <td>${zone.threshold}</td>
                <td class="${statusClass}">${deltaE}</td>
                <td><span class="badge ${isOk ? 'bg-success' : 'bg-danger'}">${isOk ? 'OK' : 'NG'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    resetResults() {
        document.getElementById('summary-table-body').innerHTML = '';
        document.getElementById('judgment-badge').textContent = 'READY';
        document.getElementById('judgment-badge').className = 'badge bg-secondary';
        document.getElementById('uniformity-card').style.display = 'none';

        // Disable Export
        this.currentRunId = null;
        document.getElementById('dropdownExport').disabled = true;

        // Charts reset logic could go here
    }
}
