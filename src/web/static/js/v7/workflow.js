/**
 * Workflow Stepper Module for Lens Signature Engine v7
 *
 * Provides guided step-by-step workflow with visual progress indicators:
 * 1. Upload → 2. Gate → 3. Signature → 4. Ink → 5. Decision
 *
 * Features:
 * - Visual progress stepper
 * - Step validation
 * - Auto-advance on completion
 * - Help tooltips per step
 * - Mobile-responsive design
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.workflow = v7.workflow || {};

    // Workflow definition
    const WORKFLOW_STEPS = [
        {
            id: 'upload',
            label: 'Upload',
            description: 'Upload lens image for inspection',
            icon: '📤',
            help: 'Select a clear, well-lit image of the contact lens. Ensure the lens is centered and in focus.',
            validation: (state) => state.hasImage,
            nextAction: 'processImage'
        },
        {
            id: 'gate',
            label: 'Gate Check',
            description: 'Verify geometry and quality gates',
            icon: '🚪',
            help: 'Gate check validates lens geometry (center, size, circularity) and basic quality metrics.',
            validation: (state) => state.gateComplete,
            nextAction: 'analyzeSignature'
        },
        {
            id: 'signature',
            label: 'Signature',
            description: 'Compare radial signature to STD',
            icon: '📊',
            help: 'Radial signature analysis compares color distribution patterns against the reference standard.',
            validation: (state) => state.signatureComplete,
            nextAction: 'analyzeInk'
        },
        {
            id: 'ink',
            label: 'Ink Analysis',
            description: 'Analyze ink distribution and color',
            icon: '🎨',
            help: 'Ink analysis evaluates color uniformity, ink count, and Lab color space metrics.',
            validation: (state) => state.inkComplete,
            nextAction: 'makeDecision'
        },
        {
            id: 'decision',
            label: 'Decision',
            description: 'Final inspection verdict',
            icon: '✅',
            help: 'Final decision aggregates all analysis results to determine PASS/FAIL/RETAKE status.',
            validation: (state) => state.decisionComplete,
            nextAction: null
        }
    ];

    // State
    let state = {
        currentStep: 0,
        stepStates: {},
        containerElement: null,
        onStepChange: null,
        autoAdvance: true,
        hasImage: false,
        gateComplete: false,
        signatureComplete: false,
        inkComplete: false,
        decisionComplete: false
    };

    /**
     * Initialize workflow stepper
     *
     * @param {string} containerId - ID of container element
     * @param {Object} options - Configuration options
     */
    v7.workflow.init = function(containerId, options = {}) {
        state.containerElement = document.getElementById(containerId);

        if (!state.containerElement) {
            console.error('Workflow container not found:', containerId);
            return false;
        }

        state.autoAdvance = options.autoAdvance !== false;
        state.onStepChange = options.onStepChange || null;

        // Create stepper UI
        createStepperUI();

        // Set initial step
        v7.workflow.goToStep(0);

        console.log('[v7.workflow] Workflow stepper initialized');
        return true;
    };

    /**
     * Create stepper UI elements
     */
    function createStepperUI() {
        const html = `
            <div class="workflow-stepper">
                <div class="workflow-steps">
                    ${WORKFLOW_STEPS.map((step, idx) => `
                        <div class="workflow-step" data-step="${idx}" onclick="v7.workflow.goToStep(${idx})">
                            <div class="workflow-step-indicator">
                                <div class="workflow-step-number">${idx + 1}</div>
                                <div class="workflow-step-icon">${step.icon}</div>
                            </div>
                            <div class="workflow-step-content">
                                <div class="workflow-step-label">${step.label}</div>
                                <div class="workflow-step-description">${step.description}</div>
                            </div>
                            ${idx < WORKFLOW_STEPS.length - 1 ? '<div class="workflow-step-connector"></div>' : ''}
                        </div>
                    `).join('')}
                </div>
                <div class="workflow-help-panel">
                    <div class="workflow-help-icon">💡</div>
                    <div class="workflow-help-text" id="workflowHelpText"></div>
                </div>
                <div class="workflow-actions">
                    <button class="workflow-btn workflow-btn-secondary" onclick="v7.workflow.previousStep()">
                        ← Previous
                    </button>
                    <button class="workflow-btn workflow-btn-primary" onclick="v7.workflow.nextStep()">
                        Next →
                    </button>
                </div>
            </div>
        `;

        state.containerElement.innerHTML = html;

        // Add styles if not already present
        if (!document.getElementById('workflow-stepper-styles')) {
            addStyles();
        }
    }

    /**
     * Add workflow stepper styles
     */
    function addStyles() {
        const style = document.createElement('style');
        style.id = 'workflow-stepper-styles';
        style.textContent = `
            .workflow-stepper {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }

            .workflow-steps {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
                position: relative;
            }

            .workflow-step {
                flex: 1;
                display: flex;
                align-items: flex-start;
                cursor: pointer;
                position: relative;
                transition: all 0.3s;
            }

            .workflow-step:hover {
                opacity: 0.8;
            }

            .workflow-step-indicator {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-right: 10px;
                z-index: 2;
            }

            .workflow-step-number {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: #334155;
                color: #94a3b8;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
                border: 2px solid #475569;
                transition: all 0.3s;
            }

            .workflow-step.active .workflow-step-number {
                background: #3b82f6;
                color: white;
                border-color: #2563eb;
                box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
            }

            .workflow-step.completed .workflow-step-number {
                background: #10b981;
                color: white;
                border-color: #059669;
            }

            .workflow-step.error .workflow-step-number {
                background: #ef4444;
                color: white;
                border-color: #dc2626;
            }

            .workflow-step-icon {
                font-size: 20px;
            }

            .workflow-step-content {
                flex: 1;
            }

            .workflow-step-label {
                color: #cbd5e1;
                font-weight: bold;
                font-size: 13px;
                margin-bottom: 3px;
            }

            .workflow-step.active .workflow-step-label {
                color: #60a5fa;
            }

            .workflow-step-description {
                color: #64748b;
                font-size: 11px;
                line-height: 1.4;
            }

            .workflow-step-connector {
                position: absolute;
                top: 16px;
                left: 50%;
                right: -50%;
                height: 2px;
                background: #334155;
                z-index: 1;
            }

            .workflow-step.completed .workflow-step-connector {
                background: #10b981;
            }

            .workflow-help-panel {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 15px;
                margin: 20px 0;
                display: flex;
                gap: 10px;
            }

            .workflow-help-icon {
                font-size: 24px;
                flex-shrink: 0;
            }

            .workflow-help-text {
                color: #94a3b8;
                font-size: 13px;
                line-height: 1.6;
            }

            .workflow-actions {
                display: flex;
                justify-content: space-between;
                gap: 10px;
            }

            .workflow-btn {
                flex: 1;
                padding: 10px 20px;
                border: 1px solid #475569;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.2s;
            }

            .workflow-btn-primary {
                background: #3b82f6;
                color: white;
                border-color: #2563eb;
            }

            .workflow-btn-primary:hover {
                background: #2563eb;
            }

            .workflow-btn-primary:disabled {
                background: #334155;
                color: #64748b;
                border-color: #475569;
                cursor: not-allowed;
            }

            .workflow-btn-secondary {
                background: #334155;
                color: #cbd5e1;
            }

            .workflow-btn-secondary:hover {
                background: #475569;
            }

            .workflow-btn-secondary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            @media (max-width: 768px) {
                .workflow-steps {
                    flex-direction: column;
                }

                .workflow-step {
                    margin-bottom: 15px;
                }

                .workflow-step-connector {
                    display: none;
                }
            }
        `;

        document.head.appendChild(style);
    }

    /**
     * Go to specific step
     *
     * @param {number} stepIndex - Step index to navigate to
     */
    v7.workflow.goToStep = function(stepIndex) {
        if (stepIndex < 0 || stepIndex >= WORKFLOW_STEPS.length) {
            console.warn('[v7.workflow] Invalid step index:', stepIndex);
            return;
        }

        const previousStep = state.currentStep;
        state.currentStep = stepIndex;

        updateStepUI(stepIndex);
        updateHelpText(stepIndex);
        updateButtons();

        // Callback
        if (state.onStepChange) {
            state.onStepChange(stepIndex, WORKFLOW_STEPS[stepIndex], previousStep);
        }

        console.log('[v7.workflow] Step changed:', WORKFLOW_STEPS[stepIndex].label);
    };

    /**
     * Go to next step
     */
    v7.workflow.nextStep = function() {
        if (state.currentStep < WORKFLOW_STEPS.length - 1) {
            v7.workflow.goToStep(state.currentStep + 1);
        }
    };

    /**
     * Go to previous step
     */
    v7.workflow.previousStep = function() {
        if (state.currentStep > 0) {
            v7.workflow.goToStep(state.currentStep - 1);
        }
    };

    /**
     * Mark current step as completed
     */
    v7.workflow.completeStep = function(stepIndex = state.currentStep) {
        const step = WORKFLOW_STEPS[stepIndex];
        if (!step) return;

        state.stepStates[step.id] = 'completed';

        // Update state flags
        switch (step.id) {
            case 'upload':
                state.hasImage = true;
                break;
            case 'gate':
                state.gateComplete = true;
                break;
            case 'signature':
                state.signatureComplete = true;
                break;
            case 'ink':
                state.inkComplete = true;
                break;
            case 'decision':
                state.decisionComplete = true;
                break;
        }

        // Update UI
        const stepElement = state.containerElement.querySelector(`.workflow-step[data-step="${stepIndex}"]`);
        if (stepElement) {
            stepElement.classList.add('completed');
            stepElement.classList.remove('error');
        }

        // Auto-advance
        if (state.autoAdvance && stepIndex === state.currentStep && stepIndex < WORKFLOW_STEPS.length - 1) {
            setTimeout(() => v7.workflow.nextStep(), 500);
        }

        console.log('[v7.workflow] Step completed:', step.label);
    };

    /**
     * Mark step as error
     */
    v7.workflow.markStepError = function(stepIndex = state.currentStep, errorMessage) {
        const step = WORKFLOW_STEPS[stepIndex];
        if (!step) return;

        state.stepStates[step.id] = 'error';

        const stepElement = state.containerElement.querySelector(`.workflow-step[data-step="${stepIndex}"]`);
        if (stepElement) {
            stepElement.classList.add('error');
            stepElement.classList.remove('completed');
        }

        if (errorMessage) {
            updateHelpText(stepIndex, `❌ Error: ${errorMessage}`);
        }

        console.error('[v7.workflow] Step error:', step.label, errorMessage);
    };

    /**
     * Update step UI
     */
    function updateStepUI(stepIndex) {
        const steps = state.containerElement.querySelectorAll('.workflow-step');

        steps.forEach((stepEl, idx) => {
            stepEl.classList.remove('active');

            if (idx === stepIndex) {
                stepEl.classList.add('active');
            }

            if (idx < stepIndex || state.stepStates[WORKFLOW_STEPS[idx].id] === 'completed') {
                stepEl.classList.add('completed');
            }
        });
    }

    /**
     * Update help text
     */
    function updateHelpText(stepIndex, customText) {
        const helpTextEl = document.getElementById('workflowHelpText');
        if (!helpTextEl) return;

        const step = WORKFLOW_STEPS[stepIndex];
        helpTextEl.textContent = customText || step.help;
    }

    /**
     * Update button states
     */
    function updateButtons() {
        const prevBtn = state.containerElement.querySelector('.workflow-btn-secondary');
        const nextBtn = state.containerElement.querySelector('.workflow-btn-primary');

        if (prevBtn) {
            prevBtn.disabled = state.currentStep === 0;
        }

        if (nextBtn) {
            if (state.currentStep === WORKFLOW_STEPS.length - 1) {
                nextBtn.textContent = 'Finish ✓';
                nextBtn.onclick = () => v7.workflow.finish();
            } else {
                nextBtn.textContent = 'Next →';
                nextBtn.onclick = () => v7.workflow.nextStep();
            }

            // Disable if validation fails
            const currentStep = WORKFLOW_STEPS[state.currentStep];
            if (currentStep.validation && !currentStep.validation(state)) {
                nextBtn.disabled = true;
            } else {
                nextBtn.disabled = false;
            }
        }
    }

    /**
     * Finish workflow
     */
    v7.workflow.finish = function() {
        console.log('[v7.workflow] Workflow finished');
        v7.workflow.completeStep(state.currentStep);

        // Optional callback
        if (state.onStepChange) {
            state.onStepChange(-1, null, state.currentStep);  // -1 indicates finished
        }
    };

    /**
     * Reset workflow to beginning
     */
    v7.workflow.reset = function() {
        state.currentStep = 0;
        state.stepStates = {};
        state.hasImage = false;
        state.gateComplete = false;
        state.signatureComplete = false;
        state.inkComplete = false;
        state.decisionComplete = false;

        v7.workflow.goToStep(0);

        // Clear completed states
        const steps = state.containerElement.querySelectorAll('.workflow-step');
        steps.forEach(step => {
            step.classList.remove('completed', 'error');
        });

        console.log('[v7.workflow] Workflow reset');
    };

    /**
     * Get current step info
     */
    v7.workflow.getCurrentStep = function() {
        return {
            index: state.currentStep,
            step: WORKFLOW_STEPS[state.currentStep],
            state: state.stepStates[WORKFLOW_STEPS[state.currentStep].id] || 'pending'
        };
    };

    console.log('[v7.workflow] Workflow stepper module loaded');
})();
