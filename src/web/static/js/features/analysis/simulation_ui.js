/**
 * Color Simulation UI Module
 *
 * Provides UI components for color simulation with method selection,
 * supporting both area_ratio and mask_based simulation methods.
 *
 * Part of Phase 6 Engine Integration.
 */

import { appState } from '../../core/state.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Initialize simulation method selector
 * @param {HTMLElement} container - Container element for the selector
 * @param {Function} onMethodChange - Callback when method changes
 */
export function initSimulationMethodSelector(container, onMethodChange = null) {
    const html = `
        <div class="simulation-method-selector bg-surface-elevated p-4 rounded-lg border border-white/10">
            <div class="flex items-center justify-between mb-3">
                <label class="text-sm font-bold text-dim uppercase">Simulation Method</label>
                <button id="simMethodInfo" class="btn-icon text-xs" title="Simulation method help">
                    <i class="fa-solid fa-circle-info"></i>
                </button>
            </div>

            <select id="simMethodSelect" class="terminal-input w-full">
                <option value="area_ratio">Area Ratio (Fast, Recommended)</option>
                <option value="mask_based">Mask-based (Precise, Experimental)</option>
            </select>

            <!-- Help tooltip -->
            <div id="simMethodTooltip" class="hidden mt-3 p-3 bg-brand-500/10 border border-brand-500/30 rounded text-xs space-y-2">
                <p><b>Area Ratio:</b> Fast scalar mixing based on ink coverage ratios. Suitable for most use cases.</p>
                <p><b>Mask-based:</b> Pixel-level synthesis using segmentation masks. More accurate but requires additional analysis data.</p>
            </div>
        </div>
    `;

    container.innerHTML = html;

    const select = container.querySelector('#simMethodSelect');
    const tooltip = container.querySelector('#simMethodTooltip');
    const infoBtn = container.querySelector('#simMethodInfo');

    // Method change handler
    select.addEventListener('change', (e) => {
        const method = e.target.value;
        appState.setState('analysis.simulationMethod', method);

        if (method === 'mask_based') {
            showNotification(
                'Precise Mode Selected',
                'Mask-based simulation may take longer. Requires full analysis data.',
                'info'
            );
        }

        if (onMethodChange) {
            onMethodChange(method);
        }
    });

    // Toggle help tooltip
    infoBtn.addEventListener('click', () => {
        tooltip.classList.toggle('hidden');
    });

    // Set initial state
    const savedMethod = appState.getState('analysis.simulationMethod') || 'area_ratio';
    select.value = savedMethod;

    return {
        select,
        tooltip,
        getMethod: () => select.value,
        setMethod: (method) => { select.value = method; },
    };
}

/**
 * Run color simulation via API
 * @param {Array} inkLabs - Array of ink Lab colors [[L, a, b], ...]
 * @param {Array} areaRatios - Array of coverage ratios [0.1, 0.2, ...]
 * @param {Object} options - Additional options
 * @returns {Promise<Object>} Simulation result
 */
export async function runSimulation(inkLabs, areaRatios, options = {}) {
    const {
        method = 'area_ratio',
        bgType = 'white',
        bgLab = null,
    } = options;

    const formData = new FormData();
    formData.append('method', method);
    formData.append('ink_labs', JSON.stringify(inkLabs));
    formData.append('area_ratios', JSON.stringify(areaRatios));
    formData.append('bg_type', bgType);

    if (bgLab) {
        formData.append('bg_lab', JSON.stringify(bgLab));
    }

    try {
        const response = await fetch('/api/v7/simulation', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Simulation failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Simulation failed:', error);
        showNotification('Simulation Failed', error.message, 'error');
        throw error;
    }
}

/**
 * Get available simulation methods from API
 * @returns {Promise<Object>} Methods list
 */
export async function getSimulationMethods() {
    try {
        const response = await fetch('/api/v7/simulation/methods');
        if (!response.ok) {
            throw new Error('Failed to fetch simulation methods');
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to get simulation methods:', error);
        return {
            methods: [
                { id: 'area_ratio', name: 'Area Ratio', recommended: true },
                { id: 'mask_based', name: 'Mask-based', recommended: false },
            ],
            default: 'area_ratio',
        };
    }
}

/**
 * Render simulation result
 * @param {HTMLElement} container - Container element
 * @param {Object} result - Simulation result from API
 */
export function renderSimulationResult(container, result) {
    const { method, background, composite, per_ink, total_coverage, model } = result;

    let inkHtml = '';
    per_ink.forEach((ink, idx) => {
        inkHtml += `
            <div class="flex items-center gap-4 p-3 bg-black/20 rounded-lg">
                <div class="w-10 h-10 rounded-lg border border-white/20"
                     style="background-color: ${ink.perceived.hex}"></div>
                <div class="flex-1">
                    <div class="text-sm font-mono">Ink ${idx + 1}</div>
                    <div class="text-xs text-dim">
                        Coverage: ${(ink.coverage * 100).toFixed(1)}% |
                        Lab: [${ink.perceived.lab.join(', ')}]
                    </div>
                </div>
                <div class="text-sm font-mono">${ink.perceived.hex}</div>
            </div>
        `;
    });

    container.innerHTML = `
        <div class="space-y-4">
            <!-- Method & Model Info -->
            <div class="flex items-center gap-2 text-xs text-dim">
                <span class="px-2 py-1 bg-brand-500/20 rounded">${method}</span>
                <span>${model.name} ${model.version}</span>
            </div>

            <!-- Composite Result -->
            <div class="terminal-panel p-4">
                <div class="flex items-center gap-4">
                    <div class="w-16 h-16 rounded-xl border-2 border-white/20 shadow-lg"
                         style="background-color: ${composite.hex}"></div>
                    <div>
                        <div class="text-lg font-bold">Composite Color</div>
                        <div class="text-sm font-mono">${composite.hex}</div>
                        <div class="text-xs text-dim">
                            Lab: [${composite.lab.join(', ')}] |
                            Total Coverage: ${(total_coverage * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>

            <!-- Background Info -->
            <div class="flex items-center gap-3 text-sm">
                <div class="w-6 h-6 rounded border border-white/20"
                     style="background-color: ${background.type === 'white' ? '#fff' : '#111'}"></div>
                <span class="text-dim">Background:</span>
                <span>${background.type} [${background.lab.join(', ')}]</span>
            </div>

            <!-- Per-Ink Results -->
            <div class="space-y-2">
                <div class="text-sm font-bold text-dim uppercase">Per-Ink Simulation</div>
                ${inkHtml}
            </div>

            <!-- Model Assumptions -->
            <details class="text-xs text-dim">
                <summary class="cursor-pointer hover:text-secondary">Model Assumptions</summary>
                <ul class="mt-2 ml-4 space-y-1 list-disc">
                    ${model.assumptions.map(a => `<li>${a.replace(/_/g, ' ')}</li>`).join('')}
                </ul>
            </details>
        </div>
    `;
}

/**
 * Create a complete simulation panel
 * @param {HTMLElement} parentContainer - Parent element
 * @returns {Object} Panel interface
 */
export function createSimulationPanel(parentContainer) {
    const html = `
        <div class="simulation-panel terminal-panel p-6 space-y-4">
            <h3 class="text-lg font-bold">Color Simulation</h3>

            <!-- Method Selector -->
            <div id="simMethodContainer"></div>

            <!-- Input Section -->
            <div class="space-y-3">
                <div>
                    <label class="text-xs text-dim uppercase">Ink Colors (Lab)</label>
                    <textarea id="simInkLabs" class="terminal-input w-full h-20 font-mono text-sm"
                              placeholder='[[50, 10, -20], [60, -5, 15]]'></textarea>
                </div>
                <div>
                    <label class="text-xs text-dim uppercase">Coverage Ratios</label>
                    <input id="simAreaRatios" type="text" class="terminal-input w-full font-mono"
                           placeholder="[0.15, 0.10]">
                </div>
                <div class="flex gap-4">
                    <div class="flex-1">
                        <label class="text-xs text-dim uppercase">Background</label>
                        <select id="simBgType" class="terminal-input w-full">
                            <option value="white">White</option>
                            <option value="black">Black</option>
                        </select>
                    </div>
                    <div class="flex items-end">
                        <button id="btnRunSimulation" class="btn btn-primary">
                            <i class="fa-solid fa-play mr-2"></i>Simulate
                        </button>
                    </div>
                </div>
            </div>

            <!-- Result Section -->
            <div id="simResultContainer" class="hidden"></div>
        </div>
    `;

    parentContainer.innerHTML = html;

    // Initialize method selector
    const methodSelector = initSimulationMethodSelector(
        parentContainer.querySelector('#simMethodContainer')
    );

    // Get elements
    const inkLabsInput = parentContainer.querySelector('#simInkLabs');
    const areaRatiosInput = parentContainer.querySelector('#simAreaRatios');
    const bgTypeSelect = parentContainer.querySelector('#simBgType');
    const runBtn = parentContainer.querySelector('#btnRunSimulation');
    const resultContainer = parentContainer.querySelector('#simResultContainer');

    // Run simulation handler
    runBtn.addEventListener('click', async () => {
        try {
            const inkLabs = JSON.parse(inkLabsInput.value);
            const areaRatios = JSON.parse(areaRatiosInput.value);

            runBtn.disabled = true;
            runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Processing...';

            const result = await runSimulation(inkLabs, areaRatios, {
                method: methodSelector.getMethod(),
                bgType: bgTypeSelect.value,
            });

            resultContainer.classList.remove('hidden');
            renderSimulationResult(resultContainer, result);

        } catch (error) {
            if (error instanceof SyntaxError) {
                showNotification('Invalid Input', 'Please check your JSON format', 'error');
            }
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="fa-solid fa-play mr-2"></i>Simulate';
        }
    });

    return {
        methodSelector,
        setInkLabs: (labs) => { inkLabsInput.value = JSON.stringify(labs); },
        setAreaRatios: (ratios) => { areaRatiosInput.value = JSON.stringify(ratios); },
        setBgType: (type) => { bgTypeSelect.value = type; },
        run: () => runBtn.click(),
    };
}

export default {
    initSimulationMethodSelector,
    runSimulation,
    getSimulationMethods,
    renderSimulationResult,
    createSimulationPanel,
};
