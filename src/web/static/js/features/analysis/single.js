/**
 * Single Analysis Feature Module
 * Simplified from v7/single_analysis.js
 */

import { apiClient } from '../../core/api.js';
import { appState } from '../../core/state.js';
import { byId } from '../../utils/helpers.js';
import { showNotification } from '../../utils/notifications.js';

/**
 * Escape HTML to prevent XSS attacks
 * @param {string|number|null|undefined} str - Value to escape
 * @returns {string} Escaped string safe for HTML insertion
 */
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const s = String(str);
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return s.replace(/[&<>"']/g, c => map[c]);
}

/**
 * Validate and sanitize hex color
 * @param {string} hex - Hex color string
 * @returns {string} Safe hex color or fallback
 */
function sanitizeHexColor(hex) {
    if (!hex || typeof hex !== 'string') return '#808080';
    // Only allow valid hex colors
    const match = hex.match(/^#?([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})$/);
    if (!match) return '#808080';
    return hex.startsWith('#') ? hex : `#${hex}`;
}

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
        const inkCountValue = inkCountInput?.value;
        const inkCount = inkCountValue ? parseInt(inkCountValue, 10) : null;

        if (btnAnalyze) {
            btnAnalyze.disabled = true;
            btnAnalyze.textContent = '분석 중...';
        }

        appState.setState('analysis.isProcessing', true);

        // Show loading overlay with progress
        byId('loadingOverlay')?.classList.remove('hidden');
        if (window.saProgress) window.saProgress.start();

        // Build FormData with required and optional parameters
        const fd = new FormData();

        // Required: files (must be first for multipart/form-data)
        fd.append('files', whiteFile);

        // Optional: black_files
        if (blackFile) {
            fd.append('black_files', blackFile);
        }

        // Always send analysis_modes (default to 'all')
        const analysisMode = byId('analysisScope')?.value || 'all';
        fd.append('analysis_modes', analysisMode);

        // Optional: expected_ink_count (only if valid integer 1-8)
        if (inkCount && Number.isInteger(inkCount) && inkCount > 0 && inkCount <= 8) {
            fd.append('expected_ink_count', String(inkCount));
        }

        // Debug logging - enumerate all FormData entries
        console.log('[SingleAnalysis] Request details:');
        console.log('  Endpoint: /api/v7/analyze_single');
        console.log('  FormData contents:');
        for (const [key, value] of fd.entries()) {
            if (value instanceof File) {
                console.log(`    ${key}: File("${value.name}", ${value.size} bytes, type="${value.type}")`);
            } else {
                console.log(`    ${key}: "${value}"`);
            }
        }

        const data = await apiClient.post('/v7/analyze_single', fd, 'multipart');

        appState.setState('analysis.result', data);

        // Show results
        byId('resultsSection')?.classList.remove('hidden');
        renderAnalysisResults(data);

        // Complete progress and hide overlay
        if (window.saProgress) window.saProgress.complete();
        setTimeout(() => {
            byId('loadingOverlay')?.classList.add('hidden');
            if (window.saProgress) window.saProgress.stop();
        }, 600);

        showNotification('완료', '분석이 완료되었습니다', 'success');

    } catch (err) {
        console.error('[SingleAnalysis] Analysis failed:', err);
        // Show detailed error info for 422 errors
        let errorMsg = err.message;
        if (err.data?.detail) {
            const detail = err.data.detail;
            if (Array.isArray(detail)) {
                errorMsg = detail.map(d => `${d.loc?.join('.')}: ${d.msg}`).join(', ');
            } else {
                errorMsg = String(detail);
            }
            console.error('[SingleAnalysis] Validation error details:', detail);
        }
        showNotification('오류', `분석 실패: ${errorMsg}`, 'error');

        // Hide overlay on error
        if (window.saProgress) window.saProgress.stop();
        byId('loadingOverlay')?.classList.add('hidden');
    } finally {
        if (btnAnalyze) {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = '분석 시작';
        }
        appState.setState('analysis.isProcessing', false);
    }
}

// Store last result for download
let lastAnalysisResult = null;
let currentPlateSource = null;
let lastInkData = null;

const simPresets = [
    { key: 'dark', label: 'Dark Brown', rgb: [60, 40, 20] },
    { key: 'hazel', label: 'Hazel', rgb: [120, 90, 40] },
    { key: 'black', label: 'Black', rgb: [20, 20, 20] },
];

const simState = {
    bgKey: 'dark',
    thickness: 1.0,
    debounceId: null,
};

function resolvePlateSource(analysis) {
    if (analysis?.plate) return 'plate';
    if (analysis?.plate_lite) return 'plate_lite';
    return null;
}

function getPlateDataBySource(analysis, source) {
    if (!analysis) return null;
    if (source === 'plate') return analysis.plate || null;
    if (source === 'plate_lite') return analysis.plate_lite || null;
    return null;
}

function updatePlateSourceToggle(analysis) {
    const container = byId('plateSourceToggle');
    const btnPlate = byId('plateSourcePlate');
    const btnLite = byId('plateSourceLite');
    const label = byId('plateSourceLabel');
    if (!container || !btnPlate || !btnLite) return;

    const hasPlate = !!analysis?.plate;
    const hasLite = !!analysis?.plate_lite;
    if (hasPlate && hasLite) {
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
    }

    const active = currentPlateSource || resolvePlateSource(analysis);
    btnPlate.classList.toggle('btn-primary', active === 'plate');
    btnPlate.classList.toggle('btn-secondary', active !== 'plate');
    btnLite.classList.toggle('btn-primary', active === 'plate_lite');
    btnLite.classList.toggle('btn-secondary', active !== 'plate_lite');

    if (label) {
        const data = getPlateDataBySource(analysis, active);
        const schema = data?.schema_version || '-';
        label.textContent = active ? `Source: ${active} (${schema})` : '';
    }
}

/**
 * Render analysis results
 * @param {Object} data - Analysis result data from API
 * @private
 */
function renderAnalysisResults(data) {
    console.log('[Render] Starting render with data:', data);

    // Store for download
    lastAnalysisResult = data;

    const result = data.results?.[0];
    if (!result) {
        console.warn('[Render] No results to render');
        return;
    }

    const analysis = result.analysis || {};
    console.log('[Render] Analysis object:', analysis);
    console.log('[Render] quality_score:', analysis.quality_score);

    renderOriginalPair(data, result);

    // 1. Quality Score
    const qualityScore = analysis.quality_score ?? 0;
    const scoreEl = byId('qualityScore');
    if (scoreEl) {
        scoreEl.textContent = qualityScore.toFixed(1);
    }

    // 2. Quality Badge
    const badgeEl = byId('qualityBadge');
    if (badgeEl) {
        let badgeClass = 'quality-poor';
        let badgeText = '불량';
        if (qualityScore >= 90) {
            badgeClass = 'quality-excellent';
            badgeText = '우수';
        } else if (qualityScore >= 75) {
            badgeClass = 'quality-good';
            badgeText = '양호';
        } else if (qualityScore >= 50) {
            badgeClass = 'quality-fair';
            badgeText = '보통';
        }
        badgeEl.innerHTML = `<span class="quality-badge ${badgeClass}">${badgeText} (${qualityScore.toFixed(1)})</span>`;
    }

    // 3. Detected Inks
    const inkData = analysis.ink || {};
    const detectedInks = inkData.k ?? inkData.k_detected ?? inkData.k_used ?? '-';
    const detectedInksEl = byId('detectedInks');
    if (detectedInksEl) {
        detectedInksEl.textContent = detectedInks;
    }
    console.log('[Render] Ink data:', inkData);

    // 4. Average Coverage (sum of all cluster area_ratios)
    const clusters = inkData.clusters || [];
    const totalCoverage = clusters.reduce((sum, c) => sum + (c.area_ratio || 0), 0);
    const avgCoverageEl = byId('avgCoverage');
    if (avgCoverageEl) {
        avgCoverageEl.textContent = `${(totalCoverage * 100).toFixed(1)}%`;
    }

    // 5. Color Uniformity (use L_cie.std as uniformity metric)
    const colorData = analysis.color || {};
    const uniformity = colorData.uniformity_score ?? colorData.L_cie?.std ?? colorData.L_cv8?.std ?? 0;
    const uniformityEl = byId('colorUniformity');
    if (uniformityEl) {
        uniformityEl.textContent = typeof uniformity === 'number' ? uniformity.toFixed(2) : '-';
    }

      // 6. Ink Palette (cluster + plate measurement)
      renderInkPalette(inkData, analysis.plate || null, analysis.plate_lite || null);
      renderInkSimulation(inkData);

    // 7. Radial Chart (if Chart.js available)
    renderRadialChart(analysis.radial);

    // 8. Plate Analysis (plate/plate_lite selector)
    if (!currentPlateSource || !getPlateDataBySource(analysis, currentPlateSource)) {
        currentPlateSource = resolvePlateSource(analysis);
    }
    updatePlateSourceToggle(analysis);
    const plateData = getPlateDataBySource(analysis, currentPlateSource);
      renderPlateAnalysis(plateData);

    // 9. Details
    renderDetails(analysis);
}

function renderOriginalPair(data, result) {
    const whiteImg = byId('whiteOriginalImage');
    const blackImg = byId('blackOriginalImage');
    const whiteMeta = byId('whiteOriginalMeta');
    const blackMeta = byId('blackOriginalMeta');

    const artifacts = result.artifacts || {};
    const whiteSrc = artifacts.original || '';
    const whiteName = result.filename || '';

    const runId = data.run_id;
    const blackName = result.black_filename || '';
    const blackSrc = runId && blackName ? `/v7_results/${runId}/black/${blackName}` : '';

    if (whiteImg) {
        if (whiteSrc) {
            whiteImg.src = whiteSrc;
            whiteImg.classList.remove('hidden');
        } else {
            whiteImg.removeAttribute('src');
            whiteImg.classList.add('hidden');
        }
    }
    if (whiteMeta) {
        whiteMeta.textContent = whiteName || (whiteSrc ? 'white_original' : 'No white image');
    }

    if (blackImg) {
        if (blackSrc) {
            blackImg.src = blackSrc;
            blackImg.classList.remove('hidden');
        } else {
            blackImg.removeAttribute('src');
            blackImg.classList.add('hidden');
        }
    }
    if (blackMeta) {
        blackMeta.textContent = blackName || 'No black image';
    }
}

  /**
   * Render ink palette with cluster and plate measurement colors
   * @param {Object} inkData - Ink analysis data with clusters
   * @param {Object|null} plateData - Plate analysis data (optional)
   * @param {Object|null} plateLiteData - Plate-lite analysis data (optional)
   */
  /**
   * P2-UI: Get phenomenon explanation based on alpha value
   */
  function getAlphaPhenomenon(alpha) {
      if (alpha == null) return { text: "-", color: "#94a3b8", icon: "" };
      if (alpha >= 0.8) return { text: "진함", color: "#22d3ee", icon: "●", detail: "높은 농도 → 색이 선명하게 표현됨" };
      if (alpha >= 0.6) return { text: "보통", color: "#34d399", icon: "◐", detail: "적정 농도 → 자연스러운 색상" };
      if (alpha >= 0.4) return { text: "연함", color: "#fbbf24", icon: "○", detail: "낮은 농도 → 색이 밝게 보임" };
      return { text: "투명", color: "#f87171", icon: "◯", detail: "매우 낮은 농도 → 하얗게/투명하게 보임" };
  }

  function renderInkPalette(inkData, plateData = null, plateLiteData = null) {
      const container = byId('inkPaletteContainer');
      if (!container) {
          console.warn('[Render] inkPaletteContainer not found');
          return;
      }

      const inks = inkData.clusters || inkData.inks || [];
      const { plateMap, plateLabel } = buildPlateMeasurementMap(plateData, plateLiteData);
      console.log('[Render] Rendering ink palette with', inks.length, 'inks');

      if (!inks.length) {
          container.innerHTML = '<p class="text-dim">잉크 정보 없음</p>';
          return;
      }

      container.innerHTML = inks.map((ink, idx) => {
          // 1. Lens clustering (raw extracted)
          const lensHex = sanitizeHexColor(
              ink.mean_hex || ink.hex_ref || rgbToHex(labToRgb(ink.centroid_lab || [50, 0, 0]))
          );
          // 2. Plate measurement
          const inkId = ink.id ?? ink.ink_id ?? idx;
          const plateHex = plateMap.get(inkId) || plateMap.get(idx) || null;

          const coverage = ink.area_ratio ?? ink.coverage_ratio ?? 0;
          const role = escapeHtml(ink.role || 'ink');
          const coverageVal = typeof coverage === 'number' ? (coverage * 100).toFixed(1) : '-';

          // P2-UI: Effective density and alpha
          const alpha = ink.alpha_used;
          const effectiveDensity = ink.effective_density;
          const hasEffDensity = effectiveDensity != null && alpha != null;
          const phenomenon = getAlphaPhenomenon(alpha);
          const alphaVal = alpha != null ? (alpha * 100).toFixed(0) : '-';
          const effDensityVal = effectiveDensity != null ? (effectiveDensity * 100).toFixed(1) : '-';

          const plateSwatch = plateHex
              ? `<div class="ink-swatch-box" style="background:${plateHex}" title="${plateLabel}: ${plateHex}"></div>`
              : `<div class="ink-swatch-box" style="background:#2a2a2a;display:flex;align-items:center;justify-content:center" title="No Plate Data"><span style="font-size:10px;color:#666">N/A</span></div>`;

          // P2-UI: Build effective density display
          const effDensityHtml = hasEffDensity ? `
              <div class="ink-card-meta" style="margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.1)">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                      <span style="color:${phenomenon.color};font-size:11px" title="${phenomenon.detail}">${phenomenon.icon} ${phenomenon.text}</span>
                      <span style="font-size:10px;color:#888">Alpha ${alphaVal}%</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:2px">
                      <span style="font-size:10px;color:#888">실효 커버리지</span>
                      <span style="font-size:11px;font-weight:600;color:white">${effDensityVal}%</span>
                  </div>
              </div>
          ` : '';

          return `
              <div class="ink-card">
                  <div class="ink-card-header">
                      <div class="ink-card-title">Ink ${idx + 1}</div>
                      <div class="ink-card-role">${role}</div>
                  </div>
                  <div class="ink-swatch-stack">
                      <div class="ink-swatch-row">
                          <div class="ink-swatch-label" style="color:#22d3ee">Cluster</div>
                          <div class="ink-swatch-box" style="background:${lensHex}" title="Cluster: ${lensHex}"></div>
                      </div>
                      <div class="ink-swatch-row">
                          <div class="ink-swatch-label" style="color:#fbbf24">${plateLabel}</div>
                          ${plateSwatch}
                      </div>
                  </div>
                  <div class="ink-card-meta">Coverage: ${coverageVal}%</div>
                  ${effDensityHtml}
              </div>
          `;
      }).join('');
  }

  function classifyKValue(kMean) {
      if (kMean < 0.3) return { label: 'Very Clear', className: 'clear' };
      if (kMean < 1.0) return { label: 'Translucent', className: 'translucent' };
      return { label: 'Deep/Solid', className: 'deep' };
  }

  function updateTrustIndicator(isCalibrated, warnings = []) {
      const badge = byId('trustBadge');
      const warnBox = byId('simWarningBox');
      const warnText = byId('simWarningText');
      if (!badge || !warnBox || !warnText) return;

      const relevant = (warnings || []).filter(w =>
          String(w).includes('K_VALUE') ||
          String(w).includes('FALLBACK') ||
          String(w).includes('FILTER')
      );

      if (!isCalibrated) {
          badge.className = 'badge badge-danger';
          badge.textContent = 'Calibration Needed';
          warnBox.style.display = 'flex';
          warnText.textContent = '캘리브레이션이 필요합니다.';
          return;
      }

      if (relevant.length) {
          badge.className = 'badge badge-warning';
          badge.textContent = 'Low Confidence';
          warnBox.style.display = 'flex';
          let msg = '측정 신뢰도가 낮습니다.';
          if (relevant.includes('FALLBACK_WHITE_ONLY_MODEL')) {
              msg = '검은판 데이터가 불안정해 흰판만 사용했습니다.';
          } else if (relevant.includes('K_VALUE_TOO_HIGH')) {
              msg = '렌즈가 매우 진하거나 이물질 가능성이 있습니다.';
          } else if (relevant.includes('INTRINSIC_BRIGHTNESS_FILTER_EMPTY_FALLBACK')) {
              msg = '렌즈 영역 감지에 실패했습니다.';
          }
          warnText.textContent = msg;
          return;
      }

      badge.className = 'badge badge-success';
      badge.textContent = 'High Confidence';
      warnBox.style.display = 'none';
  }

  function updateComparisonView(idx, resultRgb, bgRgb) {
      const original = byId(`previewOriginal-${idx}`);
      const result = byId(`previewResult-${idx}`);
      if (original) {
          original.style.backgroundColor = formatRgb(bgRgb);
      }
      if (result && resultRgb) {
          result.style.backgroundColor = formatRgb(resultRgb);
      }
  }

  function updateKValueInfo(idx, kRgb) {
      if (!Array.isArray(kRgb)) return;
      const raw = byId(`kValueRaw-${idx}`);
      const grade = byId(`kGrade-${idx}`);
      if (!raw || !grade) return;

      raw.textContent = `[${kRgb.map(v => Number(v).toFixed(3)).join(', ')}]`;
      const avg = kRgb.reduce((a, b) => a + b, 0) / kRgb.length;
      if (avg < 0.3) {
          grade.textContent = 'Very Clear (투명)';
          grade.style.color = '#6ee7b7';
      } else if (avg < 1.0) {
          grade.textContent = 'Translucent (반투명)';
          grade.style.color = '#93c5fd';
      } else {
          grade.textContent = 'Deep/Solid (진함)';
          grade.style.color = '#fca5a5';
      }
  }

  function calculateClientSideColor(kRgb, bgRgb, thickness) {
      const gamma = 2.2;
      const bgLin = bgRgb.map(c => Math.pow(c / 255.0, gamma));
      const tNew = kRgb.map(k => Math.exp(-k * thickness));
      const resLin = bgLin.map((b, i) => b * tNew[i]);
      return resLin.map(c => Math.round(Math.pow(Math.max(0, c), 1.0 / gamma) * 255));
  }

  function formatRgb(rgb) {
      if (!Array.isArray(rgb) || rgb.length !== 3) return 'rgb(42,42,42)';
      return `rgb(${rgb.map(v => Math.max(0, Math.min(255, Math.round(v)))).join(', ')})`;
  }

  function updateSimStatus(msg) {
      const el = byId('simStatus');
      if (el) {
          el.textContent = msg || '';
      }
  }

  function renderInkSimulation(inkData) {
      const container = byId('inkSimulationGrid');
      const empty = byId('inkSimulationEmpty');
      if (!container || !empty) return;

      const inks = inkData?.clusters || [];
      lastInkData = inkData;
      const intrinsicMeta = inkData?.intrinsic_color || {};
      updateTrustIndicator(!!intrinsicMeta.calibrated, intrinsicMeta.warnings || []);

      if (!inks.length) {
          container.innerHTML = '';
          empty.classList.remove('hidden');
          updateSimStatus('');
          return;
      }
      empty.classList.add('hidden');

      const bg = simPresets.find(p => p.key === simState.bgKey)?.rgb || simPresets[0].rgb;

      container.innerHTML = inks.map((ink, idx) => {
          const kRgb = ink.intrinsic_k_rgb;
          const kMean = Array.isArray(kRgb) ? kRgb.reduce((a, b) => a + b, 0) / kRgb.length : null;
          const kInfo = kMean != null ? classifyKValue(kMean) : { label: 'No k', className: '' };
          const warnings = Array.isArray(ink.intrinsic_warnings) ? ink.intrinsic_warnings.join(', ') : '-';
          const kText = kMean != null ? kMean.toFixed(2) : '-';

          return `
              <div class="sim-card" data-ink="${idx}">
                  <div class="sim-card-header">
                      <div class="sim-card-title">Ink ${idx + 1}</div>
                      <span class="sim-k-tag ${kInfo.className}">${kInfo.label}</span>
                  </div>
                  <div class="comparison-container">
                      <div class="preview-card">
                          <div id="previewOriginal-${idx}" class="circle-preview"></div>
                          <span class="label">Original Eye</span>
                      </div>
                      <div class="arrow-divider">→</div>
                      <div class="preview-card">
                          <div id="previewResult-${idx}" class="circle-preview with-overlay"></div>
                          <span class="label">Simulated</span>
                      </div>
                  </div>
                  <div class="k-value-panel">
                      <div class="k-row">
                          <span class="k-label">Absorbance (K):</span>
                          <code id="kValueRaw-${idx}">[--, --, --]</code>
                      </div>
                      <div class="k-row">
                          <span class="k-label">Grade:</span>
                          <span id="kGrade-${idx}" class="grade-badge">-</span>
                      </div>
                  </div>
                  <div class="sim-meta">
                      <div>k mean: ${kText}</div>
                      <div>Warnings: ${escapeHtml(warnings)}</div>
                  </div>
              </div>
          `;
      }).join('');

      if (!inks.some(ink => Array.isArray(ink.intrinsic_k_rgb))) {
          updateSimStatus('No intrinsic k data.');
          return;
      }

      updateSimStatus('Live');

      inks.forEach((ink, idx) => {
          const kRgb = ink.intrinsic_k_rgb;
          if (!Array.isArray(kRgb)) return;
          updateComparisonView(idx, null, bg);
          updateKValueInfo(idx, kRgb);
          const rgb = calculateClientSideColor(kRgb, bg, simState.thickness);
          updateComparisonView(idx, rgb, bg);
      });

      updateSimStatus('');
  }

  function buildPlateMeasurementMap(plateData, plateLiteData) {
      const plateMap = new Map();
      let plateLabel = 'Plate';

      const liteInks = plateLiteData?.inks;
      if (Array.isArray(liteInks) && liteInks.length) {
          plateLabel = 'P-Lite';
          liteInks.forEach((entry, idx) => {
              const inkId = entry?.ink_id ?? idx;
              const hex = entry?.obs_hex || entry?.ink_hex;
              if (hex) {
                  plateMap.set(inkId, sanitizeHexColor(hex));
              }
          });
          return { plateMap, plateLabel };
      }

      const fromWhite = plateData?.inks?.by_source?.from_white;
      if (fromWhite && typeof fromWhite === 'object') {
          const keys = Object.keys(fromWhite).sort();
          keys.forEach((inkKey, idx) => {
              const data = fromWhite[inkKey] || {};
              if (data.empty) return;
              const hex = data.hex_ref || (data.lab?.mean ? rgbToHex(labToRgb(data.lab.mean)) : null);
              if (hex) {
                  plateMap.set(idx, sanitizeHexColor(hex));
              }
          });
      }

      return { plateMap, plateLabel };
  }

/**
 * Render radial chart
 */
function renderRadialChart(radialData) {
    const canvas = byId('radialChart');
    if (!canvas || !radialData) {
        console.warn('[Render] No radial canvas or data');
        return;
    }

    // Get L* profile data - could be in different formats
    let profile = [];
    if (radialData.profile?.L_mean) {
        profile = radialData.profile.L_mean;
    } else if (Array.isArray(radialData.profile)) {
        profile = radialData.profile;
    } else if (radialData.l_profile) {
        profile = radialData.l_profile;
    }

    console.log('[Render] Radial profile length:', profile.length);

    if (!profile.length) {
        canvas.parentElement.innerHTML = '<p class="text-dim text-center py-8">방사형 프로파일 데이터 없음</p>';
        return;
    }

    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('[Render] Chart.js not loaded');
        canvas.parentElement.innerHTML = '<p class="text-dim text-center py-8">Chart.js 로드 실패</p>';
        return;
    }

    // Destroy existing chart
    if (canvas._chart) {
        canvas._chart.destroy();
    }

    // Generate x-axis labels (normalized radius 0-1)
    const labels = profile.map((_, i) => (i / (profile.length - 1)).toFixed(2));

    const ctx = canvas.getContext('2d');
    canvas._chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'L* Profile',
                data: profile,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: 'Radius (normalized)' },
                    ticks: { maxTicksLimit: 10 }
                },
                y: { title: { display: true, text: 'L*' }, min: 0, max: 100 }
            }
        }
    });
}

/**
 * Render plate analysis
 */
function renderPlateAnalysis(plateData) {
    const container = byId('plateAnalysisContainer');
    if (!container) return;

    if (!plateData) {
        container.innerHTML = '<p class="text-dim">Plate analysis data not available.</p>';
        return;
    }

    const isLite = !!plateData.zones && !plateData.plates;
    if (isLite) {
        const zones = plateData.zones || {};
        const schema = plateData.schema_version || '-';
        const paperLab = plateData.paper_color_used?.lab;
        const warnings = Array.isArray(plateData.warnings) ? plateData.warnings : [];
        const metaHtml = `
            <div class="metric-card mb-3">
                <div class="metric-label">PLATE-LITE</div>
                <div class="text-sm">
                    <p>Schema: ${escapeHtml(schema)}</p>
                    <p>Paper Lab: ${paperLab ? escapeHtml(JSON.stringify(paperLab)) : '-'}</p>
                    <p>Warnings: ${warnings.length ? escapeHtml(warnings.join(', ')) : '-'}</p>
                </div>
            </div>
        `;
        const zoneHtml = Object.entries(zones).map(([name, data]) => {
            if (data?.empty) {
                return `
                    <div class="metric-card mb-3">
                        <div class="metric-label">${escapeHtml(name).toUpperCase()}</div>
                        <div class="text-sm">
                            <p>Empty zone</p>
                        </div>
                    </div>
                `;
            }
            const lMean = data?.ink_lab?.[0];
            const alphaMean = data?.alpha_mean;
            const inkHex = data?.ink_hex;
            const obsLab = data?.obs_lab;
            return `
                <div class="metric-card mb-3">
                    <div class="metric-label">${escapeHtml(name).toUpperCase()}</div>
                    <div class="text-sm">
                        <p>L* Mean (ink): ${typeof lMean === 'number' ? lMean.toFixed(2) : '-'}</p>
                        <p>Alpha Mean: ${typeof alphaMean === 'number' ? alphaMean.toFixed(3) : '-'}</p>
                        <p>Ink Hex: ${inkHex ? escapeHtml(inkHex) : '-'}</p>
                        <p>Obs Lab: ${obsLab ? escapeHtml(JSON.stringify(obsLab)) : '-'}</p>
                    </div>
                </div>
            `;
        }).join('');
        container.innerHTML = metaHtml + (zoneHtml || '<p class="text-dim">No plate-lite data.</p>');
        return;
    }

    const plates = plateData.plates || {};
    const html = Object.entries(plates).map(([name, data]) => {
        const core = data.core || {};
        const lMean = core.lab?.mean?.[0];
        const alphaMean = core.alpha?.mean;
        const areaRatio = data.geometry?.area_ratio;
        const safeName = escapeHtml(name).toUpperCase();

        return `
            <div class="metric-card mb-3">
                <div class="metric-label">${safeName} ???</div>
                <div class="text-sm">
                    <p>L* Mean: ${typeof lMean === 'number' ? lMean.toFixed(2) : '-'}</p>
                    <p>Alpha Mean: ${typeof alphaMean === 'number' ? alphaMean.toFixed(3) : '-'}</p>
                    <p>Area: ${typeof areaRatio === 'number' ? (areaRatio * 100).toFixed(1) : '-'}%</p>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = html || '<p class="text-dim">No plate data.</p>';
}

/**
 * Render plate black analysis
 */
/**
 * Render detailed metrics
 */
function renderDetails(analysis) {
    const container = byId('detailsContainer');
    if (!container) return;

    const sections = [
        { key: 'gate', title: 'Gate 검사' },
        { key: 'pattern', title: '패턴 분석' },
        { key: 'zones', title: '존 분석' }
    ];

    const html = sections.map(({ key, title }) => {
        const data = analysis[key];
        if (!data) return '';
        // Security: escape JSON output to prevent XSS
        const safeJson = escapeHtml(JSON.stringify(data, null, 2));

        return `
            <div class="metric-card mb-3">
                <div class="metric-label">${escapeHtml(title)}</div>
                <pre class="text-xs text-dim overflow-auto max-h-40">${safeJson}</pre>
            </div>
        `;
    }).join('');

    container.innerHTML = html || '<p class="text-dim">상세 데이터 없음</p>';
}

/**
 * Convert Lab to RGB (approximate)
 */
function labToRgb(lab) {
    const [L, a, b] = lab;
    // Simplified Lab to RGB conversion
    let y = (L + 16) / 116;
    let x = a / 500 + y;
    let z = y - b / 200;

    const xyz = [x, y, z].map(v => {
        const v3 = v * v * v;
        return v3 > 0.008856 ? v3 : (v - 16 / 116) / 7.787;
    });

    // XYZ to RGB
    let r = xyz[0] * 3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986;
    let g = xyz[0] * -0.9689 + xyz[1] * 1.8758 + xyz[2] * 0.0415;
    let bl = xyz[0] * 0.0557 + xyz[1] * -0.2040 + xyz[2] * 1.0570;

    [r, g, bl] = [r, g, bl].map(v => {
        v = v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v;
        return Math.max(0, Math.min(255, Math.round(v * 255)));
    });

    return [r, g, bl];
}

/**
 * Convert RGB to Hex
 */
function rgbToHex([r, g, b]) {
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

/**
 * Initialize single analysis UI
 */
export function initSingleAnalysis() {
    console.log('[Init] Initializing single analysis UI');

    const btnAnalyze = byId('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', runSingleAnalysis);
        console.log('[Init] Analyze button handler attached');
    }

    const plateSourcePlate = byId('plateSourcePlate');
    const plateSourceLite = byId('plateSourceLite');
    if (plateSourcePlate) {
        plateSourcePlate.addEventListener('click', () => {
            currentPlateSource = 'plate';
            if (lastAnalysisResult) {
                renderAnalysisResults(lastAnalysisResult);
            }
        });
    }
    if (plateSourceLite) {
        plateSourceLite.addEventListener('click', () => {
            currentPlateSource = 'plate_lite';
            if (lastAnalysisResult) {
                renderAnalysisResults(lastAnalysisResult);
            }
        });
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

    // Download button
    const btnDownload = byId('btnDownloadSpec');
    if (btnDownload) {
        btnDownload.addEventListener('click', downloadAnalysisResult);
        console.log('[Init] Download button handler attached');
    }

    initSimulationControls();
}

function initSimulationControls() {
    const bgWrap = byId('simBgButtons');
    const thicknessInput = byId('simThickness');
    const thicknessValue = byId('simThicknessValue');
    if (bgWrap) {
        bgWrap.innerHTML = simPresets.map(preset => `
            <button class="sim-bg-btn" data-bg="${preset.key}" style="background: rgba(${preset.rgb.join(', ')}, 0.2);">
                ${preset.label}
            </button>
        `).join('');
        bgWrap.querySelectorAll('.sim-bg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                simState.bgKey = btn.dataset.bg || 'dark';
                bgWrap.querySelectorAll('.sim-bg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                scheduleSimUpdate();
            });
        });
        const active = bgWrap.querySelector(`[data-bg="${simState.bgKey}"]`);
        if (active) active.classList.add('active');
    }

    if (thicknessInput) {
        thicknessInput.addEventListener('input', () => {
            const val = parseFloat(thicknessInput.value);
            simState.thickness = Number.isFinite(val) ? val : 1.0;
            if (thicknessValue) {
                thicknessValue.textContent = `${simState.thickness.toFixed(1)}x`;
            }
            scheduleSimUpdate();
        });
    }
}

function scheduleSimUpdate() {
    if (simState.debounceId) {
        clearTimeout(simState.debounceId);
    }
    simState.debounceId = setTimeout(() => {
        if (lastInkData) {
            renderInkSimulation(lastInkData);
        }
    }, 300);
}

/**
 * Download analysis result as JSON
 */
function downloadAnalysisResult() {
    if (!lastAnalysisResult) {
        showNotification('오류', '다운로드할 분석 결과가 없습니다', 'error');
        return;
    }

    const blob = new Blob([JSON.stringify(lastAnalysisResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${lastAnalysisResult.run_id || 'result'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('완료', 'JSON 파일이 다운로드되었습니다', 'success');
}

export default { initSingleAnalysis, runSingleAnalysis };
