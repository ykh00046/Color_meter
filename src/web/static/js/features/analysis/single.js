/**
 * Single Analysis Feature Module
 * Simplified from v7/single_analysis.js
 */

import { apiClient } from '../../core/api.js';
import { appState } from '../../core/state.js';
import { byId } from '../../utils/helpers.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Run single analysis
 * @returns {Promise<void>}
 */
export async function runSingleAnalysis() {
    const whiteFileInput = byId('fileWhite');
    const blackFileInput = byId('fileBlack');
    const inkCountInput = byId('inkCountInput');
    const btnAnalyze = byId('btnAnalyze');

    try {
        const whiteFile = whiteFileInput?.files?.[0];
        if (!whiteFile) {
            showNotification('오류', 'White 배경 이미지를 선택하세요', 'error');
            return;
        }

        const blackFile = blackFileInput?.files?.[0];
        const inkCount = inkCountInput?.value || '3';

        if (btnAnalyze) {
            btnAnalyze.disabled = true;
            btnAnalyze.textContent = '분석 중...';
        }

        appState.setState('analysis.isProcessing', true);

        const fd = new FormData();
        fd.append('white_file', whiteFile);
        if (blackFile) fd.append('black_file', blackFile);
        fd.append('expected_ink_count', inkCount);
        fd.append('analysis_scope', byId('analysisScope')?.value || 'full');

        const data = await apiClient.post('/v7/analyze_single', fd, 'multipart');

        appState.setState('analysis.result', data);

        // Show results
        byId('resultsSection')?.classList.remove('hidden');
        renderAnalysisResults(data);

        showNotification('완료', '분석이 완료되었습니다', 'success');

    } catch (err) {
        console.error('Analysis failed:', err);
        showNotification('오류', `분석 실패: ${err.message}`, 'error');
    } finally {
        if (btnAnalyze) {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = '분석 시작';
        }
        appState.setState('analysis.isProcessing', false);
    }
}

/**
 * Render analysis results (placeholder)
 * @param {Object} data - Analysis result data
 * @private
 */
function renderAnalysisResults(data) {
    // TODO: Implement result rendering
    console.log('Rendering analysis results:', data);

    // Example: Update quality score
    const qualityScore = data.quality_score || 0;
    const scoreEl = byId('qualityScore');
    if (scoreEl) {
        scoreEl.textContent = qualityScore.toFixed(1);
    }
}

/**
 * Initialize single analysis UI
 */
export function initSingleAnalysis() {
    const btnAnalyze = byId('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', runSingleAnalysis);
    }

    // File input displays
    const whiteFileInput = byId('fileWhite');
    const whiteFileName = byId('fileWhiteName');
    if (whiteFileInput && whiteFileName) {
        whiteFileInput.addEventListener('change', () => {
            const file = whiteFileInput.files?.[0];
            whiteFileName.textContent = file ? file.name : '파일 선택...';
        });
    }

    const blackFileInput = byId('fileBlack');
    const blackFileName = byId('fileBlackName');
    if (blackFileInput && blackFileName) {
        blackFileInput.addEventListener('change', () => {
            const file = blackFileInput.files?.[0];
            blackFileName.textContent = file ? file.name : '파일 선택...';
        });
    }
}

export default { initSingleAnalysis, runSingleAnalysis };
