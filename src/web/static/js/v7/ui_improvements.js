/**
 * UI Improvements Module for Lens Signature Engine v7
 *
 * Implements 8 key UI enhancements:
 * 1. Unified summary card (3-second operator decision)
 * 2. Core/Ink same-scale display
 * 3. Direction dual-mode clarity (ROI vs Global)
 * 4. Pattern_color score policy labels
 * 5. Ink "forced_to_expected" status display
 * 6. Radial profile: summary→expand
 * 7. Standardized data sparsity warnings
 * 8. Reason→Evidence auto-scroll
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.uiImprovements = v7.uiImprovements || {};

    // ============================================================================
    // 1. UNIFIED SUMMARY CARD (3-second operator decision)
    // ============================================================================

    /**
     * Render unified summary card with all critical info in one place
     *
     * @param {Object} decision - Decision object
     * @param {Object} v2Diag - V2 diagnostics
     * @param {Object} ops - Operator signals
     */
    v7.uiImprovements.renderUnifiedSummary = function(decision, v2Diag, ops = {}) {
        const container = v7.utils.byId('unifiedSummaryCard');
        if (!container) {
            console.warn('[ui_improvements] unifiedSummaryCard not found');
            return;
        }

        const finalJudgment = ops.judgment || decision?.label || 'UNKNOWN';
        const label = decision?.label || 'UNKNOWN';
        const bestMode = decision?.best_mode || '-';

        // Core metrics
        const sig = decision?.signature || {};
        const corr = sig.score_corr != null ? sig.score_corr.toFixed(3) : '-';
        const deMean = sig.delta_e_mean != null ? sig.delta_e_mean.toFixed(2) : '-';
        const deP95 = sig.delta_e_p95 != null ? sig.delta_e_p95.toFixed(2) : '-';

        // Ink metrics
        const inkSummary = v2Diag?.ink_match?.trajectory_summary || {};
        const offTrack = inkSummary.max_off_track != null ? inkSummary.max_off_track.toFixed(2) : '-';
        const inkUncertain = Boolean(v2Diag?.ink_match?.warning || (v2Diag?.warnings && v2Diag.warnings.length));

        // Top signals (1-2 only)
        const topSignals = (ops.top_signals || []).slice(0, 2);
        const moreSignalsCount = (ops.top_signals || []).length - 2;

        // Status badge color
        let statusClass = 'status-unknown';
        if (finalJudgment === 'PASS' || finalJudgment === 'OK') {
            statusClass = 'status-ok';
        } else if (finalJudgment === 'RETAKE' || finalJudgment === 'MANUAL_REVIEW') {
            statusClass = 'status-review';
        } else if (finalJudgment.startsWith('NG_') || finalJudgment === 'FAIL') {
            statusClass = 'status-ng';
        }

        // Build HTML
        let html = `
            <div class="unified-summary-header">
                <div class="unified-summary-verdict ${statusClass}">
                    <span class="verdict-label">${finalJudgment}</span>
                    ${inkUncertain ? '<span class="verdict-uncertain">⚠ UNCERTAIN</span>' : ''}
                </div>
            </div>

            <div class="unified-summary-body">
                <div class="summary-line summary-line-primary">
                    <span class="summary-core">Core: <strong>${label}</strong> (${bestMode})</span>
                    <span class="summary-divider">·</span>
                    <span class="summary-ink">Ink: ${inkUncertain ? '<strong class="text-amber-400">UNCERTAIN</strong>' : 'OK'}</span>
                    <span class="summary-divider">·</span>
                    <span class="summary-metrics">Corr <strong>${corr}</strong> · off-track <strong>${offTrack}</strong></span>
                </div>

                ${topSignals.length > 0 ? `
                    <div class="summary-line summary-line-reasons">
                        <span class="reasons-label">왜?</span>
                        <div class="reasons-list">
                            ${topSignals.map(signal => {
                                const info = v7.utils.getReasonInfo(signal.code);
                                return `
                                    <div class="reason-item" data-code="${signal.code}" onclick="v7.uiImprovements.scrollToEvidence('${signal.code}')">
                                        <span class="reason-code">[${signal.code}]</span>
                                        <span class="reason-text">${info.title || signal.code}</span>
                                    </div>
                                `;
                            }).join('')}
                            ${moreSignalsCount > 0 ? `
                                <button class="reason-expand" onclick="v7.uiImprovements.expandAllReasons()">
                                    +${moreSignalsCount} more
                                </button>
                            ` : ''}
                        </div>
                    </div>
                ` : `
                    <div class="summary-line summary-line-ok">
                        ✓ No issues detected
                    </div>
                `}
            </div>
        `;

        container.innerHTML = html;

        // Add CSS if not present
        if (!document.getElementById('unified-summary-styles')) {
            addUnifiedSummaryStyles();
        }
    };

    /**
     * Add unified summary styles
     */
    function addUnifiedSummaryStyles() {
        const style = document.createElement('style');
        style.id = 'unified-summary-styles';
        style.textContent = `
            .unified-summary-header {
                margin-bottom: 12px;
            }

            .unified-summary-verdict {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .verdict-label {
                font-size: 24px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }

            .status-ok .verdict-label { color: #10b981; }
            .status-review .verdict-label { color: #eab308; }
            .status-ng .verdict-label { color: #ef4444; }

            .verdict-uncertain {
                background: #f59e0b;
                color: #0f172a;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }

            .unified-summary-body {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .summary-line {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 6px;
                font-size: 13px;
                line-height: 1.6;
            }

            .summary-line-primary {
                color: #cbd5e1;
            }

            .summary-divider {
                color: #64748b;
            }

            .summary-metrics strong {
                color: #60a5fa;
            }

            .summary-line-reasons {
                border-top: 1px solid #334155;
                padding-top: 8px;
            }

            .reasons-label {
                color: #94a3b8;
                font-weight: bold;
                margin-right: 5px;
            }

            .reasons-list {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                flex: 1;
            }

            .reason-item {
                display: flex;
                align-items: center;
                gap: 4px;
                background: #1e293b;
                border: 1px solid #334155;
                padding: 3px 8px;
                border-radius: 3px;
                cursor: pointer;
                transition: all 0.2s;
            }

            .reason-item:hover {
                background: #334155;
                border-color: #60a5fa;
            }

            .reason-code {
                color: #ef4444;
                font-size: 10px;
                font-weight: bold;
            }

            .reason-text {
                color: #cbd5e1;
                font-size: 11px;
            }

            .reason-expand {
                background: #334155;
                border: 1px solid #475569;
                color: #94a3b8;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 10px;
                cursor: pointer;
            }

            .reason-expand:hover {
                background: #475569;
                color: #cbd5e1;
            }

            .summary-line-ok {
                color: #10b981;
                font-weight: bold;
            }
        `;
        document.head.appendChild(style);
    }

    // ============================================================================
    // 2. CORE/INK SAME-SCALE DISPLAY
    // ============================================================================

    /**
     * Render Core and Ink panels with consistent badge system
     *
     * @param {Object} decision - Decision object
     * @param {Object} v2Diag - V2 diagnostics
     */
    v7.uiImprovements.renderSameScalePanels = function(decision, v2Diag) {
        // Core panel
        const coreContainer = v7.utils.byId('corePanelMetrics');
        if (coreContainer) {
            const sig = decision?.signature || {};
            const corePassed = sig.passed !== false;

            const html = `
                <div class="panel-badge ${corePassed ? 'badge-ok' : 'badge-ng'}">
                    ${corePassed ? 'OK' : 'NG'}
                </div>
                <div class="panel-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Correlation:</span>
                        <span class="metric-value">${(sig.score_corr || 0).toFixed(3)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">ΔE mean:</span>
                        <span class="metric-value">${(sig.delta_e_mean || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">ΔE p95:</span>
                        <span class="metric-value">${(sig.delta_e_p95 || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Best mode:</span>
                        <span class="metric-value">${decision?.best_mode || '-'}</span>
                    </div>
                </div>
            `;
            coreContainer.innerHTML = html;
        }

        // Ink panel
        const inkContainer = v7.utils.byId('inkPanelMetrics');
        if (inkContainer) {
            const inkSummary = v2Diag?.ink_match?.trajectory_summary || {};
            const inkWarning = v2Diag?.ink_match?.warning;
            const inkUncertain = Boolean(inkWarning || (v2Diag?.warnings && v2Diag.warnings.length));

            const badge = inkUncertain ? 'WARN' : 'OK';
            const badgeClass = inkUncertain ? 'badge-warn' : 'badge-ok';

            const html = `
                <div class="panel-badge ${badgeClass}">
                    ${badge}${inkUncertain ? ' (uncertain)' : ''}
                </div>
                <div class="panel-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Off-track max:</span>
                        <span class="metric-value">${(inkSummary.max_off_track || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Max ΔE:</span>
                        <span class="metric-value">${(inkSummary.max_de || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">k expected/used:</span>
                        <span class="metric-value">${v2Diag?.expected_ink_count || '-'}/${v2Diag?.auto_estimation?.auto_k_best || '-'}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">${(v2Diag?.auto_estimation?.confidence || 0).toFixed(2)}</span>
                    </div>
                </div>
            `;
            inkContainer.innerHTML = html;
        }

        // Add panel styles if needed
        if (!document.getElementById('same-scale-panel-styles')) {
            addSameScalePanelStyles();
        }
    };

    function addSameScalePanelStyles() {
        const style = document.createElement('style');
        style.id = 'same-scale-panel-styles';
        style.textContent = `
            .panel-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 12px;
            }

            .badge-ok {
                background: #10b981;
                color: #0f172a;
            }

            .badge-warn {
                background: #eab308;
                color: #0f172a;
            }

            .badge-ng {
                background: #ef4444;
                color: white;
            }

            .panel-metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }

            .metric-item {
                display: flex;
                justify-content: space-between;
                padding: 6px 0;
                border-bottom: 1px solid #334155;
            }

            .metric-label {
                color: #94a3b8;
                font-size: 12px;
            }

            .metric-value {
                color: #cbd5e1;
                font-weight: bold;
                font-size: 12px;
            }
        `;
        document.head.appendChild(style);
    }

    // ============================================================================
    // 3. DIRECTION DUAL-MODE CLARITY (ROI vs Global)
    // ============================================================================

    /**
     * Render direction with clear ROI vs Global distinction
     */
    v7.uiImprovements.renderDirectionClarified = function(v2Diag) {
        const container = v7.utils.byId('directionDisplay');
        if (!container) return;

        const direction = v2Diag?.ink_match?.direction || {};
        const roiDirection = direction.roi || {};
        const globalDirection = direction.global || {};

        const html = `
            <div class="direction-section">
                <div class="direction-primary">
                    <h4>색상 변화 (ROI)
                        <span class="tooltip-icon" title="실제 패턴 영역 중심 (판정/설명용)">ⓘ</span>
                    </h4>
                    <div class="direction-values">
                        <div class="direction-item">
                            <span class="label">ΔL:</span>
                            <span class="value">${(roiDirection.delta_L || 0).toFixed(2)}</span>
                        </div>
                        <div class="direction-item">
                            <span class="label">Δa:</span>
                            <span class="value">${(roiDirection.delta_a || 0).toFixed(2)}</span>
                        </div>
                        <div class="direction-item">
                            <span class="label">Δb:</span>
                            <span class="value">${(roiDirection.delta_b || 0).toFixed(2)}</span>
                        </div>
                    </div>
                </div>

                <details class="direction-secondary">
                    <summary>색상 변화 (전체) - 참고용</summary>
                    <div class="direction-values">
                        <div class="direction-item">
                            <span class="label">ΔL:</span>
                            <span class="value">${(globalDirection.delta_L || 0).toFixed(2)}</span>
                        </div>
                        <div class="direction-item">
                            <span class="label">Δa:</span>
                            <span class="value">${(globalDirection.delta_a || 0).toFixed(2)}</span>
                        </div>
                        <div class="direction-item">
                            <span class="label">Δb:</span>
                            <span class="value">${(globalDirection.delta_b || 0).toFixed(2)}</span>
                        </div>
                    </div>
                    <p class="direction-note">전체 polar 평균 (조명/배경 영향 포함)</p>
                </details>
            </div>
        `;

        container.innerHTML = html;
    };

    // ============================================================================
    // 4. PATTERN_COLOR SCORE POLICY LABELS
    // ============================================================================

    /**
     * Display pattern_color score with policy label
     */
    v7.uiImprovements.renderPatternColorScore = function(ops) {
        const container = v7.utils.byId('patternColorScoreDisplay');
        if (!container) return;

        const patternColor = ops?.pattern_color || {};
        const score = patternColor.score != null ? patternColor.score.toFixed(2) : '-';
        const policy = patternColor.policy || 'heuristic_v1';
        const uncertain = patternColor.uncertain || false;

        const html = `
            <div class="score-display">
                <div class="score-main">
                    <span class="score-label">Pattern & Color Score:</span>
                    <span class="score-value">${score}</span>
                    <span class="score-policy">policy: ${policy}</span>
                </div>
                ${uncertain ? `
                    <div class="score-warning">
                        ⚠ UNCERTAIN → score capped (0.70 max)
                    </div>
                ` : ''}
            </div>
        `;

        container.innerHTML = html;
    };

    // ============================================================================
    // 5. INK "FORCED_TO_EXPECTED" STATUS
    // ============================================================================

    /**
     * Display forced_to_expected badge for ink k
     */
    v7.uiImprovements.renderForcedKBadge = function(v2Diag) {
        const container = v7.utils.byId('inkKDisplay');
        if (!container) return;

        const autoEst = v2Diag?.auto_estimation || {};
        const autoK = autoEst.auto_k_best;
        const expectedK = v2Diag?.expected_ink_count;
        const forced = autoEst.forced_to_expected || false;

        const html = `
            <div class="k-display">
                <span class="k-label">k =</span>
                <span class="k-value">${autoK || '-'}</span>
                ${forced ? `
                    <span class="k-forced-badge" title="클러스터 품질이 애매해 expected k로 강제 적용">
                        (forced)
                    </span>
                ` : ''}
                ${expectedK ? `
                    <span class="k-expected">expected: ${expectedK}</span>
                ` : ''}
            </div>
        `;

        container.innerHTML = html;

        if (!document.getElementById('forced-k-styles')) {
            const style = document.createElement('style');
            style.id = 'forced-k-styles';
            style.textContent = `
                .k-display {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 14px;
                }

                .k-label {
                    color: #94a3b8;
                }

                .k-value {
                    color: #cbd5e1;
                    font-weight: bold;
                    font-size: 16px;
                }

                .k-forced-badge {
                    background: #f59e0b;
                    color: #0f172a;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: bold;
                    cursor: help;
                }

                .k-expected {
                    color: #64748b;
                    font-size: 11px;
                }
            `;
            document.head.appendChild(style);
        }
    };

    // ============================================================================
    // 6. RADIAL PROFILE: SUMMARY → EXPAND
    // ============================================================================

    /**
     * Render radial profile with summary-first approach
     */
    v7.uiImprovements.renderRadialSummary = function(radialData) {
        const container = v7.utils.byId('radialProfileDisplay');
        if (!container) return;

        const summary = radialData?.summary || {};

        const html = `
            <div class="radial-summary">
                <h4>Radial Profile Summary</h4>
                <div class="radial-metrics">
                    <div class="radial-metric">
                        <span class="label">knee_r_de:</span>
                        <span class="value">${(summary.knee_r_de || 0).toFixed(2)}</span>
                    </div>
                    <div class="radial-metric">
                        <span class="label">fade_slope_outer_de:</span>
                        <span class="value">${(summary.fade_slope_outer_de || 0).toFixed(3)}</span>
                    </div>
                    <div class="radial-metric">
                        <span class="label">inner_mean_de:</span>
                        <span class="value">${(summary.inner_mean_de || 0).toFixed(2)}</span>
                    </div>
                    <div class="radial-metric">
                        <span class="label">outer_mean_de:</span>
                        <span class="value">${(summary.outer_mean_de || 0).toFixed(2)}</span>
                    </div>
                </div>

                <details class="radial-expand">
                    <summary>📊 View Full Profile</summary>
                    <div class="radial-charts">
                        <canvas id="radialDeProfileChart" width="600" height="300"></canvas>
                        <div class="radial-tabs">
                            <button onclick="v7.uiImprovements.showRadialTab('L')">L* Profile</button>
                            <button onclick="v7.uiImprovements.showRadialTab('a')">a* Profile</button>
                            <button onclick="v7.uiImprovements.showRadialTab('b')">b* Profile</button>
                        </div>
                        <canvas id="radialLabProfileChart" width="600" height="300" class="hidden"></canvas>
                    </div>
                </details>
            </div>
        `;

        container.innerHTML = html;
    };

    v7.uiImprovements.showRadialTab = function(channel) {
        console.log('[ui_improvements] Show radial tab:', channel);
        // Implementation: render specific channel profile
    };

    // ============================================================================
    // 7. STANDARDIZED DATA SPARSITY WARNING
    // ============================================================================

    /**
     * Render standardized data sparsity warning component
     */
    v7.uiImprovements.renderSparsityWarning = function(v3Summary) {
        const dataSparsity = v3Summary?.data_sparsity;
        if (!dataSparsity || dataSparsity === 'sufficient') return '';

        const confidence = v3Summary?.confidence || 'low';
        const windowEffective = v3Summary?.window_effective || 0;
        const windowRequested = v3Summary?.window_requested || 0;

        return `
            <div class="sparsity-warning" onclick="v7.uiImprovements.showSparsityDetails('${JSON.stringify({ dataSparsity, confidence, windowEffective, windowRequested }).replace(/"/g, '&quot;')}')">
                <span class="sparsity-icon">ⓘ</span>
                <span class="sparsity-text">참고용 (데이터 부족)</span>
            </div>
        `;
    };

    v7.uiImprovements.showSparsityDetails = function(dataStr) {
        const data = JSON.parse(dataStr.replace(/&quot;/g, '"'));
        alert(`Data Sparsity Details:\n\nStatus: ${data.dataSparsity}\nConfidence: ${data.confidence}\nWindow: ${data.windowEffective}/${data.windowRequested}`);
    };

    // Add sparsity warning styles
    if (!document.getElementById('sparsity-warning-styles')) {
        const style = document.createElement('style');
        style.id = 'sparsity-warning-styles';
        style.textContent = `
            .sparsity-warning {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                background: #f59e0b;
                color: #0f172a;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }

            .sparsity-warning:hover {
                background: #fbbf24;
            }

            .sparsity-icon {
                font-size: 14px;
            }
        `;
        document.head.appendChild(style);
    }

    // ============================================================================
    // 8. REASON → EVIDENCE AUTO-SCROLL
    // ============================================================================

    /**
     * Scroll to evidence section based on reason code
     */
    v7.uiImprovements.scrollToEvidence = function(reasonCode) {
        console.log('[ui_improvements] Scroll to evidence:', reasonCode);

        // Map reason codes to section IDs
        const codeToSection = {
            'V2_INK_SHIFT_SUMMARY': 'inkTrajectorySection',
            'V2_INK_UNEXPECTED_K': 'inkKSection',
            'DELTAE_P95_HIGH': 'signatureDeltaESection',
            'DELTAE_MEAN_HIGH': 'signatureDeltaESection',
            'CORR_LOW': 'signatureProfileSection',
            'GATE_CENTER_OFFSET': 'gateGeometrySection',
            'GATE_SIZE_MISMATCH': 'gateGeometrySection'
            // Add more mappings as needed
        };

        const sectionId = codeToSection[reasonCode];

        if (sectionId) {
            const section = document.getElementById(sectionId);
            if (section) {
                // Switch to appropriate mode first
                if (reasonCode.startsWith('V2_INK_')) {
                    if (v7.inspection && v7.inspection.applyInspectionMode) {
                        v7.inspection.applyInspectionMode('ink');
                    }
                } else if (reasonCode.startsWith('DELTAE_') || reasonCode.startsWith('CORR_')) {
                    if (v7.inspection && v7.inspection.applyInspectionMode) {
                        v7.inspection.applyInspectionMode('signature');
                    }
                } else if (reasonCode.startsWith('GATE_')) {
                    if (v7.inspection && v7.inspection.applyInspectionMode) {
                        v7.inspection.applyInspectionMode('gate');
                    }
                }

                // Scroll to section with smooth animation
                setTimeout(() => {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

                    // Highlight section briefly
                    section.classList.add('evidence-highlight');
                    setTimeout(() => {
                        section.classList.remove('evidence-highlight');
                    }, 2000);
                }, 300);
            } else {
                console.warn('[ui_improvements] Section not found:', sectionId);
            }
        } else {
            console.warn('[ui_improvements] No section mapping for code:', reasonCode);
        }
    };

    // Add evidence highlight styles
    if (!document.getElementById('evidence-highlight-styles')) {
        const style = document.createElement('style');
        style.id = 'evidence-highlight-styles';
        style.textContent = `
            .evidence-highlight {
                animation: evidencePulse 2s ease-in-out;
                border: 2px solid #60a5fa;
                border-radius: 4px;
            }

            @keyframes evidencePulse {
                0%, 100% {
                    box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.7);
                }
                50% {
                    box-shadow: 0 0 20px 10px rgba(96, 165, 250, 0);
                }
            }
        `;
        document.head.appendChild(style);
    }

    // ============================================================================
    // UTILITY: Expand all reasons
    // ============================================================================

    v7.uiImprovements.expandAllReasons = function() {
        console.log('[ui_improvements] Expand all reasons');
        // Implementation: show modal with all top_signals
        alert('Feature: Show all reasons in modal - To be implemented');
    };

    console.log('[v7.uiImprovements] UI improvements module loaded');
})();
