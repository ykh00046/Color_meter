/**
 * Plate Gate Visualization Module
 *
 * Provides UI components for visualizing plate gate extraction results,
 * including ink mask rendering and quality indicators.
 *
 * Part of Phase 6 Engine Integration.
 */

import { apiClient } from '../../core/api.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Render Gate quality indicator
 * @param {HTMLElement} container - Container element
 * @param {Object} gateQuality - Gate quality result from API
 */
export function renderGateQuality(container, gateQuality) {
    const { usable, artifact_ratio, reason } = gateQuality;

    const statusClass = usable ? 'border-green-500 bg-green-500/10' : 'border-red-500 bg-red-500/10';
    const statusText = usable ? '✓ USABLE' : '✗ UNUSABLE';
    const statusColor = usable ? 'text-green-400' : 'text-red-400';

    container.innerHTML = `
        <div class="metric-card ${statusClass} border-l-4 p-4">
            <div class="text-xs text-dim uppercase mb-2">Gate Quality</div>
            <div class="text-2xl font-bold ${statusColor}">${statusText}</div>
            <div class="text-xs text-secondary mt-2">
                Artifact Ratio: ${(artifact_ratio * 100).toFixed(1)}%
            </div>
            ${reason ? `<div class="text-xs text-warning mt-1">Reason: ${reason}</div>` : ''}
        </div>
    `;
}

/**
 * Render ink mask on canvas
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {string} base64Image - Base64 encoded PNG mask
 * @param {Object} geom - Geometry info {cx, cy, r}
 */
export function renderInkMask(canvas, base64Image, geom) {
    if (!base64Image) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No mask data available', canvas.width / 2, canvas.height / 2);
        return;
    }

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        // Clear and set canvas size
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw mask image
        ctx.drawImage(img, 0, 0);

        // Apply color tint (make it more visible)
        ctx.globalCompositeOperation = 'source-atop';
        ctx.fillStyle = 'rgba(59, 130, 246, 0.6)';  // Blue tint
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = 'source-over';

        // Draw geometry overlay if available
        if (geom && geom.cx && geom.cy && geom.r) {
            // Scale geometry to polar coordinates
            const scaleX = canvas.width / 720;  // Default polar T
            const scaleY = canvas.height / 260; // Default polar R

            ctx.strokeStyle = '#22d3ee';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);

            // Draw center marker
            ctx.beginPath();
            ctx.arc(canvas.width / 2, 0, 5, 0, Math.PI * 2);
            ctx.stroke();
        }
    };

    img.onerror = () => {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ef4444';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Failed to load mask image', canvas.width / 2, canvas.height / 2);
    };

    img.src = `data:image/png;base64,${base64Image}`;
}

/**
 * Render complete gate visualization
 * @param {Object} plateGateResult - Full result from /api/v7/plate_gate
 * @param {Object} containers - {quality: HTMLElement, mask: HTMLCanvasElement, info: HTMLElement}
 */
export function renderGateVisualization(plateGateResult, containers) {
    const { usable, artifact_ratio, reason, registration, geom, ink_mask_polar_image, mask_shape } = plateGateResult;

    // Render quality indicator
    if (containers.quality) {
        renderGateQuality(containers.quality, { usable, artifact_ratio, reason });
    }

    // Render mask
    if (containers.mask) {
        renderInkMask(containers.mask, ink_mask_polar_image, geom);
    }

    // Render info panel
    if (containers.info) {
        containers.info.innerHTML = `
            <div class="space-y-3">
                <div class="flex items-center justify-between">
                    <span class="text-xs text-dim">Registration Method</span>
                    <span class="text-sm font-mono">${registration?.method || 'unknown'}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-xs text-dim">Images Swapped</span>
                    <span class="text-sm ${registration?.swapped ? 'text-warning' : 'text-secondary'}">
                        ${registration?.swapped ? 'Yes (Auto-corrected)' : 'No'}
                    </span>
                </div>
                ${geom ? `
                <div class="flex items-center justify-between">
                    <span class="text-xs text-dim">Lens Center</span>
                    <span class="text-sm font-mono">(${geom.cx?.toFixed(1)}, ${geom.cy?.toFixed(1)})</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-xs text-dim">Lens Radius</span>
                    <span class="text-sm font-mono">${geom.r?.toFixed(1)} px</span>
                </div>
                ` : ''}
                ${mask_shape ? `
                <div class="flex items-center justify-between">
                    <span class="text-xs text-dim">Polar Mask Size</span>
                    <span class="text-sm font-mono">${mask_shape[0]} × ${mask_shape[1]}</span>
                </div>
                ` : ''}
            </div>
        `;
    }

    // Show notification if swapped
    if (registration?.swapped) {
        showNotification(
            'Images Swapped',
            'White/Black images were automatically swapped for correct analysis.',
            'warning'
        );
    }
}

/**
 * Run plate gate extraction via API
 * @param {File} whiteFile - White backlight image file
 * @param {File} blackFile - Black backlight image file
 * @param {string} sku - Optional SKU for configuration
 * @returns {Promise<Object>} Gate extraction result
 */
export async function runPlateGateExtraction(whiteFile, blackFile, sku = null) {
    const formData = new FormData();
    formData.append('white_file', whiteFile);
    formData.append('black_file', blackFile);
    if (sku) {
        formData.append('sku', sku);
    }

    try {
        const result = await fetch('/api/v7/plate_gate', {
            method: 'POST',
            body: formData,
        });

        if (!result.ok) {
            const error = await result.json();
            throw new Error(error.detail || 'Gate extraction failed');
        }

        return await result.json();
    } catch (error) {
        console.error('Plate Gate extraction failed:', error);
        showNotification('Gate Extraction Failed', error.message, 'error');
        throw error;
    }
}

/**
 * Initialize gate visualization UI in a container
 * @param {HTMLElement} parentContainer - Parent element to add UI
 * @returns {Object} References to created elements
 */
export function initGateVisualizationUI(parentContainer) {
    const html = `
        <div class="gate-visualization terminal-panel p-6 space-y-4">
            <div class="flex items-center justify-between">
                <h3 class="text-lg font-bold">Gate Mask Visualization</h3>
                <button id="btnRunGateExtraction" class="btn btn-secondary text-xs">
                    <i class="fa-solid fa-play mr-2"></i>Extract Gate
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Quality Indicator -->
                <div id="gateQualityContainer">
                    <div class="metric-card bg-surface border border-default p-4">
                        <div class="text-xs text-dim uppercase mb-2">Gate Quality</div>
                        <div class="text-lg text-secondary">Upload images to extract gate</div>
                    </div>
                </div>

                <!-- Info Panel -->
                <div id="gateInfoContainer">
                    <div class="text-sm text-secondary">
                        <p>Upload white and black backlight images to extract ink gate masks.</p>
                        <p class="mt-2 text-xs text-dim">This is a lightweight extraction for quick validation.</p>
                    </div>
                </div>
            </div>

            <!-- Mask Canvas -->
            <div class="terminal-panel p-2 bg-black/50 rounded-lg">
                <canvas id="gateMaskCanvas" class="w-full border border-white/10 rounded" style="height: 200px;"></canvas>
            </div>
        </div>
    `;

    parentContainer.innerHTML = html;

    return {
        quality: parentContainer.querySelector('#gateQualityContainer'),
        mask: parentContainer.querySelector('#gateMaskCanvas'),
        info: parentContainer.querySelector('#gateInfoContainer'),
        extractBtn: parentContainer.querySelector('#btnRunGateExtraction'),
    };
}

export default {
    renderGateQuality,
    renderInkMask,
    renderGateVisualization,
    runPlateGateExtraction,
    initGateVisualizationUI,
};
