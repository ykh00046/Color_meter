/**
 * Image Viewer Component
 * Migrated from v7/viewer.js
 */

import { appState } from '../core/state.js';

export class ImageViewer {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element not found: ${canvasId}`);
        }

        this.ctx = this.canvas.getContext('2d');
        this.image = null;
        this.overlayData = null;

        // Options
        this.options = {
            enableZoom: options.enableZoom ?? true,
            enablePan: options.enablePan ?? true,
            ...options
        };

        // Transform state
        this.transform = {
            scale: 1,
            offsetX: 0,
            offsetY: 0
        };

        if (this.options.enableZoom || this.options.enablePan) {
            this.setupInteraction();
        }
    }

    /**
     * Load image from URL
     * @param {string} src - Image source URL
     * @returns {Promise<void>}
     */
    async loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.fitImageToCanvas();
                this.render();
                resolve();
            };
            img.onerror = reject;
            img.src = src;
        });
    }

    /**
     * Set overlay data
     * @param {ImageData|HTMLCanvasElement} data - Overlay data
     */
    setOverlay(data) {
        this.overlayData = data;
        this.render();
    }

    /**
     * Clear overlay
     */
    clearOverlay() {
        this.overlayData = null;
        this.render();
    }

    /**
     * Fit image to canvas
     * @private
     */
    fitImageToCanvas() {
        if (!this.image) return;

        const scaleX = this.canvas.width / this.image.width;
        const scaleY = this.canvas.height / this.image.height;
        this.transform.scale = Math.min(scaleX, scaleY);
        this.transform.offsetX = (this.canvas.width - this.image.width * this.transform.scale) / 2;
        this.transform.offsetY = (this.canvas.height - this.image.height * this.transform.scale) / 2;
    }

    /**
     * Render image and overlay
     * @private
     */
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.image) return;

        // Draw image
        this.ctx.save();
        this.ctx.translate(this.transform.offsetX, this.transform.offsetY);
        this.ctx.scale(this.transform.scale, this.transform.scale);
        this.ctx.drawImage(this.image, 0, 0);
        this.ctx.restore();

        // Draw overlay
        if (this.overlayData) {
            this.ctx.save();
            this.ctx.globalAlpha = 0.5;
            if (this.overlayData instanceof HTMLCanvasElement) {
                this.ctx.drawImage(
                    this.overlayData,
                    this.transform.offsetX,
                    this.transform.offsetY,
                    this.image.width * this.transform.scale,
                    this.image.height * this.transform.scale
                );
            }
            this.ctx.restore();
        }
    }

    /**
     * Setup mouse/touch interaction
     * @private
     */
    setupInteraction() {
        // Zoom with mouse wheel
        if (this.options.enableZoom) {
            this.canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                this.transform.scale *= delta;
                this.render();
            });
        }

        // Pan with mouse drag
        if (this.options.enablePan) {
            let isDragging = false;
            let lastX, lastY;

            this.canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
            });

            this.canvas.addEventListener('mousemove', (e) => {
                if (!isDragging) return;

                const dx = e.clientX - lastX;
                const dy = e.clientY - lastY;
                this.transform.offsetX += dx;
                this.transform.offsetY += dy;

                lastX = e.clientX;
                lastY = e.clientY;
                this.render();
            });

            this.canvas.addEventListener('mouseup', () => {
                isDragging = false;
            });

            this.canvas.addEventListener('mouseleave', () => {
                isDragging = false;
            });
        }
    }

    /**
     * Reset view
     */
    reset() {
        this.fitImageToCanvas();
        this.render();
    }
}

export default ImageViewer;
