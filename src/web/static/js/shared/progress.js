/**
 * ProgressTracker - Simulated multi-stage progress indicator.
 *
 * Since the backend does not yet support SSE, this provides a
 * deterministic stage-based progress animation that gives users
 * visual feedback during long-running analysis operations.
 *
 * When SSE is available, replace _simulateProgress with EventSource.
 */

const DEFAULT_STAGES = [
    { name: 'gate', label: 'Running gate analysis...', target: 25 },
    { name: 'color', label: 'Extracting colors...', target: 50 },
    { name: 'ink', label: 'Analyzing ink patterns...', target: 75 },
    { name: 'signature', label: 'Computing signature...', target: 90 },
];

export class ProgressTracker {
    /**
     * @param {Object} options
     * @param {string} options.containerId   - Progress container ID
     * @param {string} options.fillId        - Progress fill bar ID
     * @param {string} options.textId        - Progress text ID
     * @param {Array}  [options.stages]      - Custom stages
     * @param {Function} [options.onComplete] - Called when progress reaches 100
     */
    constructor(options) {
        this.container = document.getElementById(options.containerId);
        this.fill = document.getElementById(options.fillId);
        this.text = document.getElementById(options.textId);
        this.stages = options.stages || DEFAULT_STAGES;
        this.onComplete = options.onComplete;
        this._timer = null;
        this._currentProgress = 0;
    }

    /**
     * Show and start the progress animation.
     */
    start() {
        if (!this.container) return;

        this._currentProgress = 0;
        this.container.classList.remove('hidden');
        this._updateUI(0, 'Preparing...');
        this._simulateProgress();
    }

    /**
     * Complete the progress (jump to 100%).
     */
    complete() {
        this._stopTimer();
        this._updateUI(100, 'Complete');
        if (this.onComplete) this.onComplete();
    }

    /**
     * Stop and hide the progress indicator.
     */
    stop() {
        this._stopTimer();
        if (this.container) {
            this.container.classList.add('hidden');
        }
    }

    _simulateProgress() {
        let stageIndex = 0;
        const interval = 800;

        this._timer = setInterval(() => {
            if (stageIndex < this.stages.length) {
                const stage = this.stages[stageIndex];
                this._currentProgress = stage.target;
                this._updateUI(stage.target, stage.label);
                stageIndex++;
            } else {
                // Hold at 90% until complete() is called
                this._stopTimer();
            }
        }, interval);
    }

    _updateUI(progress, label) {
        if (this.fill) {
            this.fill.style.width = `${progress}%`;
        }
        if (this.text) {
            this.text.textContent = label;
        }
        if (this.container) {
            this.container.setAttribute('aria-valuenow', String(progress));
        }
    }

    _stopTimer() {
        if (this._timer) {
            clearInterval(this._timer);
            this._timer = null;
        }
    }
}
