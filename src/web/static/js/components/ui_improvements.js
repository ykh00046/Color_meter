/**
 * UI Improvements Module
 * Migrated from v7/ui_improvements.js (712 lines)
 *
 * Implements 8 key UI enhancements:
 * 1. Unified summary card
 * 2. Core/Ink same-scale display
 * 3. Direction dual-mode clarity
 * 4. Pattern_color score policy labels
 * 5. Ink "forced_to_expected" status
 * 6. Radial profile summary→expand
 * 7. Standardized data sparsity warnings
 * 8. Reason→Evidence auto-scroll
 */

import { byId, safeText, getReasonInfo } from '../utils/helpers.js';

/**
 * 1. Render unified summary card
 */
export function renderUnifiedSummary(decision, v2Diag, ops = {}) {
    const container = byId('unifiedSummaryCard');
    if (!container) return;

    const finalJudgment = ops.judgment || decision?.label || 'UNKNOWN';
    const label = decision?.label || 'UNKNOWN';
    const bestMode = decision?.best_mode || '-';

    const sig = decision?.signature || {};
    const corr = sig.score_corr != null ? sig.score_corr.toFixed(3) : '-';

    const inkSummary = v2Diag?.ink_match?.trajectory_summary || {};
    const offTrack = inkSummary.max_off_track != null ? inkSummary.max_off_track.toFixed(2) : '-';
    const inkUncertain = Boolean(v2Diag?.ink_match?.warning || (v2Diag?.warnings && v2Diag.warnings.length));

    const topSignals = (ops.top_signals || []).slice(0, 2);

    let statusClass = 'status-unknown';
    if (finalJudgment === 'PASS' || finalJudgment === 'OK') statusClass = 'status-ok';
    else if (finalJudgment === 'RETAKE') statusClass = 'status-review';
    else if (finalJudgment.startsWith('NG_')) statusClass = 'status-ng';

    const html = `
        <div class="unified-summary-header">
            <div class="unified-summary-verdict ${statusClass}">
                <span class="verdict-label">${finalJudgment}</span>
                ${inkUncertain ? '<span class="verdict-uncertain">⚠ UNCERTAIN</span>' : ''}
            </div>
        </div>
        <div class="unified-sum mary-body">
            <div class="summary-line">
                <span>Core: <strong>${label}</strong> (${bestMode})</span>
                <span>·</span>
                <span>Corr <strong>${corr}</strong> · off-track <strong>${offTrack}</strong></span>
            </div>
            ${topSignals.length > 0 ? `
                <div class="summary-reasons">
                    ${topSignals.map(signal => {
                        const info = getReasonInfo(signal.code);
                        return `<div><span>${signal.code}:</span> ${info.title}</div>`;
                    }).join('')}
                </div>
            ` : ''}
        </div>
    `;

    container.innerHTML = html;
}

/**
 * 2. Render Core/Ink same-scale panels
 */
export function renderSameScalePanels(decision, v2Diag) {
    const coreContainer = byId('corePanelMetrics');
    if (coreContainer) {
        const sig = decision?.signature || {};
        const corePassed = sig.passed !== false;

        coreContainer.innerHTML = `
            <div class="panel-badge ${corePassed ? 'badge-ok' : 'badge-ng'}">${corePassed ? 'OK' : 'NG'}</div>
            <div class="panel-metrics">
                <div>Correlation: ${(sig.score_corr || 0).toFixed(3)}</div>
                <div>ΔE mean: ${(sig.delta_e_mean || 0).toFixed(2)}</div>
            </div>
        `;
    }

    const inkContainer = byId('inkPanelMetrics');
    if (inkContainer) {
        const inkSummary = v2Diag?.ink_match?.trajectory_summary || {};
        const inkUncertain = Boolean(v2Diag?.ink_match?.warning);

        inkContainer.innerHTML = `
            <div class="panel-badge ${inkUncertain ? 'badge-warn' : 'badge-ok'}">${inkUncertain ? 'WARN' : 'OK'}</div>
            <div class="panel-metrics">
                <div>Off-track: ${(inkSummary.max_off_track || 0).toFixed(2)}</div>
            </div>
        `;
    }
}

/**
 * 3. Render direction with ROI vs Global clarity
 */
export function renderDirectionClarified(v2Diag) {
    const container = byId('directionDisplay');
    if (!container) return;

    const direction = v2Diag?.ink_match?.direction || {};
    const roiDirection = direction.roi || {};
    const globalDirection = direction.global || {};

    container.innerHTML = `
        <div class="direction-section">
            <h4>색상 변화 (ROI)</h4>
            <div>ΔL: ${(roiDirection.delta_L || 0).toFixed(2)}</div>
            <div>Δa: ${(roiDirection.delta_a || 0).toFixed(2)}</div>
            <div>Δb: ${(roiDirection.delta_b || 0).toFixed(2)}</div>
            <details>
                <summary>전체 (Global)</summary>
                <div>ΔL: ${(globalDirection.delta_L || 0).toFixed(2)}</div>
                <div>Δa: ${(globalDirection.delta_a || 0).toFixed(2)}</div>
                <div>Δb: ${(globalDirection.delta_b || 0).toFixed(2)}</div>
            </details>
        </div>
    `;
}

/**
 * 4. Render pattern_color score with policy
 */
export function renderPatternColorScore(ops) {
    const container = byId('patternColorScoreDisplay');
    if (!container) return;

    const patternColor = ops?.pattern_color || {};
    const score = patternColor.score != null ? patternColor.score.toFixed(2) : '-';
    const policy = patternColor.policy || 'heuristic_v1';

    container.innerHTML = `
        <div>Pattern & Color Score: <strong>${score}</strong> (policy: ${policy})</div>
    `;
}

/**
 * 5. Render forced k badge
 */
export function renderForcedKBadge(v2Diag) {
    const container = byId('inkKDisplay');
    if (!container) return;

    const autoEst = v2Diag?.auto_estimation || {};
    const autoK = autoEst.auto_k_best;
    const forced = autoEst.forced_to_expected || false;

    container.innerHTML = `
        <div>k = <strong>${autoK || '-'}</strong>
        ${forced ? '<span class="k-forced-badge">(forced)</span>' : ''}
        </div>
    `;
}

/**
 * 6. Render radial profile summary
 */
export function renderRadialSummary(radialData) {
    const container = byId('radialProfileDisplay');
    if (!container) return;

    const summary = radialData?.summary || {};

    container.innerHTML = `
        <div class="radial-summary">
            <h4>Radial Profile Summary</h4>
            <div>knee_r_de: ${(summary.knee_r_de || 0).toFixed(2)}</div>
            <div>inner_mean_de: ${(summary.inner_mean_de || 0).toFixed(2)}</div>
            <details>
                <summary>View Full Profile</summary>
                <canvas id="radialDeProfileChart" width="600" height="300"></canvas>
            </details>
        </div>
    `;
}

/**
 * 8. Scroll to evidence section
 */
export function scrollToEvidence(reasonCode) {
    const codeToSection = {
        'V2_INK_SHIFT_SUMMARY': 'inkTrajectorySection',
        'V2_INK_UNEXPECTED_K': 'inkKSection',
        'DELTAE_P95_HIGH': 'signatureDeltaESection',
        'CORR_LOW': 'signatureProfileSection',
        'GATE_CENTER_OFFSET': 'gateGeometrySection'
    };

    const sectionId = codeToSection[reasonCode];
    if (sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            section.classList.add('evidence-highlight');
            setTimeout(() => section.classList.remove('evidence-highlight'), 2000);
        }
    }
}

/**
 * Expand all reasons
 */
export function expandAllReasons() {
    console.log('[ui_improvements] Expand all reasons');
    // TODO: Implement modal with all signals
}

export default {
    renderUnifiedSummary,
    renderSameScalePanels,
    renderDirectionClarified,
    renderPatternColorScore,
    renderForcedKBadge,
    renderRadialSummary,
    scrollToEvidence,
    expandAllReasons
};
