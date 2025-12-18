/**
 * Interactive Lens Viewer Module
 * Handles Canvas rendering, Panzoom, and Grid Overlay
 */

export class LensViewer {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.container = document.getElementById(containerId);

        this.image = null;
        this.lensDetection = null;

        // View settings
        this.settings = {
            innerRadius: 0.2,
            outerRadius: 1.0,
            ringCount: 3,
            sectorCount: 12,
            ringMode: 'uniform', // 'uniform' or 'center-focus'
            showRings: true,
            showSectors: true,
            showHeatmap: false
        };

        // Initialize Panzoom
        // We'll initialize it only after image load to set correct bounds if needed
        this.panzoom = null;
    }

    // Initialize Panzoom logic
    initPanzoom() {
        if (typeof Panzoom === 'undefined') {
            console.error("Panzoom library not loaded");
            return;
        }

        this.panzoom = Panzoom(this.canvas, {
            maxScale: 20,
            minScale: 0.1,
            contain: 'outside',
            startScale: 1
        });

        // Enable mouse wheel zoom
        this.container.addEventListener('wheel', this.panzoom.zoomWithWheel);
    }

    /**
     * Load an image from a URL (blob or static)
     */
    loadImage(src) {
        return new Promise((resolve, reject) => {
            this.image = new Image();
            this.image.onload = () => {
                // Resize canvas to match image dimensions
                this.canvas.width = this.image.width;
                this.canvas.height = this.image.height;

                // Initialize Panzoom if needed
                if (!this.panzoom) {
                    this.initPanzoom();
                }

                // --- Fit to Screen Logic ---
                const containerW = this.container.clientWidth || 800;
                const containerH = this.container.clientHeight || 600;

                // Calculate scale to fit container with padding
                const scaleX = containerW / this.image.width;
                const scaleY = containerH / this.image.height;
                const scale = Math.min(scaleX, scaleY) * 0.95; // 95% fit

                // Calculate centering offset
                // The canvas has 0,0 origin. We scale it, so its visual size is w*scale, h*scale.
                // We want to center this visual rect in the container.
                const centerX = (containerW - this.image.width * scale) / 2;
                const centerY = (containerH - this.image.height * scale) / 2;

                // Reset and Apply
                // Note: panzoom.zoom() usually zooms relative to current, or to a focal point.
                // panzoom.zoom(scale) sets the scale directly if using Panzoom 4.x options properly
                // but usually it's safer to use setOptions or reset then pan/zoom.

                // Force reset first to clear any previous state
                this.panzoom.reset({ animate: false });

                // Apply calculated zoom and pan
                // We use setTimeout to ensure the DOM has updated layout if needed, though usually synchronous here.
                this.panzoom.zoom(scale, { animate: false });
                this.panzoom.pan(centerX, centerY, { animate: false });

                this.render();
                resolve();
            };
            this.image.onerror = reject;
            this.image.src = src;
        });
    }

    /**
     * Set lens detection data
     * @param {Object} detection { center_x, center_y, radius }
     */
    setLensDetection(detection) {
        this.lensDetection = detection;
        this.render();
    }

    /**
     * Update display settings
     * @param {Object} newSettings Partial settings object
     */
    updateSettings(newSettings) {
        Object.assign(this.settings, newSettings);
        this.render();
    }

    /**
     * Main render loop
     */
    render() {
        if (!this.image) return;

        // 1. Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 2. Draw original image
        this.ctx.drawImage(this.image, 0, 0);

        // 3. Draw Grid Overlay
        if (this.lensDetection) {
            this.drawGridOverlay();
        }
    }

    drawGridOverlay() {
        const { center_x, center_y, radius } = this.lensDetection;
        const s = this.settings;

        this.ctx.save();

        // Common styles
        this.ctx.lineWidth = 2; // Fixed line width (could scale with zoom inverse if desired)

        // 3.1 Inner & Outer Circles (Boundary)
        this.ctx.strokeStyle = '#00ff00'; // Green for Inner
        this.ctx.beginPath();
        this.ctx.arc(center_x, center_y, radius * s.innerRadius, 0, 2 * Math.PI);
        this.ctx.stroke();

        this.ctx.strokeStyle = '#ff0000'; // Red for Outer
        this.ctx.beginPath();
        this.ctx.arc(center_x, center_y, radius * s.outerRadius, 0, 2 * Math.PI);
        this.ctx.stroke();

        // 3.2 Rings
        if (s.showRings && s.ringCount > 0) {
            this.ctx.strokeStyle = 'rgba(255, 255, 0, 0.7)'; // Yellow transparent
            this.ctx.lineWidth = 1;

            for (let i = 1; i < s.ringCount; i++) {
                let r_norm;
                if (s.ringMode === 'uniform') {
                    // Uniform distribution between inner and outer
                    r_norm = s.innerRadius + (s.outerRadius - s.innerRadius) * (i / s.ringCount);
                } else {
                    // Center-focus (example logic, can be refined)
                    // Not implemented in backend yet, using uniform for now
                     r_norm = s.innerRadius + (s.outerRadius - s.innerRadius) * (i / s.ringCount);
                }

                this.ctx.beginPath();
                this.ctx.arc(center_x, center_y, radius * r_norm, 0, 2 * Math.PI);
                this.ctx.stroke();
            }
        }

        // 3.3 Sectors
        if (s.showSectors && s.sectorCount > 0) {
            this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)'; // Cyan transparent
            this.ctx.lineWidth = 1;

            const angleStep = (2 * Math.PI) / s.sectorCount;

            for (let i = 0; i < s.sectorCount; i++) {
                const angle = i * angleStep;

                // Draw line from Inner Radius to Outer Radius
                const startX = center_x + (radius * s.innerRadius) * Math.cos(angle);
                const startY = center_y + (radius * s.innerRadius) * Math.sin(angle);
                const endX = center_x + (radius * s.outerRadius) * Math.cos(angle);
                const endY = center_y + (radius * s.outerRadius) * Math.sin(angle);

                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
                this.ctx.stroke();
            }
        }

        this.ctx.restore();
    }
}
