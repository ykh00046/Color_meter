/**
 * Single Sample Analysis Module
 *
 * Handles rendering of single sample analysis results without STD comparison
 */

(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.singleAnalysis = v7.singleAnalysis || {};

    /**
     * Main render function - orchestrates all sub-renders
     */
    v7.singleAnalysis.renderResults = function(result) {
        const analysis = result.analysis;
        const artifacts = result.artifacts;

        console.log('Rendering single analysis results:', result);

        // Check if analysis failed
        if (!analysis) {
            const errorMsg = result.error || 'Unknown error occurred during analysis';
            alert('분석 실패: ' + errorMsg);
            console.error('Analysis failed:', errorMsg);
            return;
        }

        // Store artifacts for image switching
        if (window.setArtifacts) {
            window.setArtifacts(artifacts);
        }

        // Render operator summary and engineer KPI first (NEW!)
        v7.singleAnalysis.renderOperatorSummary(analysis.operator_summary);
        v7.singleAnalysis.renderEngineerKPI(analysis.engineer_kpi);

        // Render each section (복제 정보 중심)
        v7.singleAnalysis.renderQCScore(analysis.gate);
        v7.singleAnalysis.renderCopySummary(analysis.ink, analysis.radial);
        v7.singleAnalysis.renderInkClusters(analysis.ink);
        v7.singleAnalysis.renderRadialProfile(analysis.radial);
        v7.singleAnalysis.renderPatternQuality(analysis.pattern, artifacts);
        v7.singleAnalysis.renderColorDistribution(analysis.color);
        v7.singleAnalysis.renderZoneAnalysis(analysis.zones, analysis.gate);
        v7.singleAnalysis.renderWarnings(analysis.warnings);
    };

    /**
     * Render QC Score (측정 신뢰도)
     */
    v7.singleAnalysis.renderQCScore = function(gate) {
        if (!gate) return;

        const statusEl = document.getElementById('qcGateStatus');
        const centerEl = document.getElementById('qcCenterOffset');
        const sharpnessEl = document.getElementById('qcSharpness');
        const illumEl = document.getElementById('qcIllumination');
        const failReasonsDiv = document.getElementById('qcFailReasons');
        const failReasonsList = document.getElementById('qcFailReasonsList');
        const guidanceDiv = document.getElementById('qcGuidance');
        const guidanceContent = document.getElementById('qcGuidanceContent');

        if (statusEl) {
            statusEl.textContent = gate.passed ? '✅ Valid' : '❌ Failed';
            statusEl.style.color = gate.passed ? '#10b981' : '#ef4444';
        }

        if (centerEl && gate.scores) {
            const offset = gate.scores.center_offset_mm || 0;
            centerEl.textContent = offset.toFixed(2) + ' mm';

            // Use threshold from scores if available
            const threshold = gate.scores._thresholds?.center_off_max || 0.12;
            const thresholdMm = threshold * 10; // rough conversion
            centerEl.style.color = offset < thresholdMm ? '#10b981' : '#f59e0b';
        }

        if (sharpnessEl && gate.scores) {
            const blur = gate.scores.sharpness_score || 0;
            sharpnessEl.textContent = blur.toFixed(1);

            const threshold = gate.scores._thresholds?.blur_min || 40;
            sharpnessEl.style.color = blur > threshold ? '#10b981' : '#f59e0b';
        }

        if (illumEl && gate.scores) {
            const illum = gate.scores.illumination_asymmetry || 0;
            illumEl.textContent = illum.toFixed(3);

            const threshold = gate.scores._thresholds?.illum_max || 0.1;
            illumEl.style.color = illum < threshold ? '#10b981' : '#f59e0b';
        }

        // Show FAIL reasons prominently (크고 명확하게)
        if (!gate.passed && failReasonsDiv && failReasonsList && gate.reasons) {
            failReasonsList.innerHTML = '';

            // Display reasons (최대 3개만)
            const reasons = gate.reasons.slice(0, 3);
            reasons.forEach(reason => {
                const item = document.createElement('div');
                item.className = 'font-mono text-sm text-red-error font-semibold';
                item.textContent = '• ' + reason;
                failReasonsList.appendChild(item);
            });

            failReasonsDiv.classList.remove('hidden');
        } else if (failReasonsDiv) {
            failReasonsDiv.classList.add('hidden');
        }

        // Show detailed guidance (상세 가이드)
        if (!gate.passed && guidanceDiv && guidanceContent && gate.scores && gate.scores._guidance) {
            guidanceContent.innerHTML = '';
            const guidance = gate.scores._guidance;

            Object.entries(guidance).forEach(([key, message]) => {
                const item = document.createElement('div');
                item.className = 'p-2 bg-yellow-warning/10 border border-yellow-warning/30 rounded';
                item.textContent = message;
                guidanceContent.appendChild(item);
            });

            guidanceDiv.classList.remove('hidden');
        } else if (guidanceDiv) {
            guidanceDiv.classList.add('hidden');
        }
    };

    /**
     * Render Copy Summary (복제 정보 요약)
     */
    v7.singleAnalysis.renderCopySummary = function(ink, radial) {
        const inkCountEl = document.getElementById('copySummaryInkCount');
        const confidenceEl = document.getElementById('copySummaryConfidence');
        const patternEl = document.getElementById('copySummaryRadialPattern');

        if (inkCountEl && ink) {
            inkCountEl.textContent = ink.k || 0;
        }

        if (confidenceEl && ink) {
            const conf = (ink.confidence * 100).toFixed(0);
            confidenceEl.textContent = conf + '%';
            confidenceEl.style.color = conf > 70 ? '#10b981' : '#f59e0b';
        }

        if (patternEl && radial && radial.summary) {
            const inner = radial.summary.inner_mean_L;
            const outer = radial.summary.outer_mean_L;
            if (inner > outer) {
                patternEl.textContent = 'Center Darker';
            } else if (outer > inner) {
                patternEl.textContent = 'Edge Darker';
            } else {
                patternEl.textContent = 'Uniform';
            }
        }
    };

    /**
     * Render gate check results
     */
    v7.singleAnalysis.renderGate = function(gate) {
        if (!gate) return;

        const badgeEl = document.getElementById('gatePassedBadge');
        const textEl = document.getElementById('gatePassedText');
        const centerEl = document.getElementById('gateCenter');
        const blurEl = document.getElementById('gateBlur');

        if (badgeEl) {
            badgeEl.textContent = gate.passed ? '✅' : '❌';
        }

        if (textEl) {
            textEl.textContent = gate.passed ? 'PASSED' : 'FAILED';
            textEl.style.color = gate.passed ? '#10b981' : '#ef4444';
        }

        if (centerEl && gate.scores) {
            const centerOffset = gate.scores.center_offset_mm || 0;
            centerEl.textContent = centerOffset.toFixed(2) + 'mm';
        }

        if (blurEl && gate.scores) {
            const blur = gate.scores.sharpness_score || 0;
            blurEl.textContent = blur.toFixed(2);
        }
    };

    /**
     * Render color distribution histogram (using CIE L*a*b* scale)
     */
    v7.singleAnalysis.renderColorDistribution = function(color) {
        if (!color) return;

        // Update summary stats (use CIE scale)
        const colorL = document.getElementById('colorL');
        const colorA = document.getElementById('colorA');
        const colorB = document.getElementById('colorB');

        // Prefer CIE scale, fallback to cv8 for backward compatibility
        const L = color.L_cie || color.L;
        const a = color.a_cie || color.a;
        const b = color.b_cie || color.b;

        if (colorL && L) {
            colorL.textContent = `${L.mean.toFixed(1)} ± ${L.std.toFixed(1)} (${L.p05.toFixed(1)} ~ ${L.p95.toFixed(1)}) [CIE]`;
        }

        if (colorA && a) {
            colorA.textContent = `${a.mean.toFixed(1)} ± ${a.std.toFixed(1)} (${a.p05.toFixed(1)} ~ ${a.p95.toFixed(1)}) [CIE]`;
        }

        if (colorB && b) {
            colorB.textContent = `${b.mean.toFixed(1)} ± ${b.std.toFixed(1)} (${b.p05.toFixed(1)} ~ ${b.p95.toFixed(1)}) [CIE]`;
        }

        // Render histogram chart
        const ctx = document.getElementById('colorHistogramChart');
        if (!ctx) return;

        // Destroy existing chart if any
        if (v7.singleAnalysis._colorChart) {
            v7.singleAnalysis._colorChart.destroy();
        }

        const bins = Array.from({length: 50}, (_, i) => i);

        v7.singleAnalysis._colorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: bins,
                datasets: [
                    {
                        label: 'L*',
                        data: color.histogram_L || [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'a*',
                        data: color.histogram_a || [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'b*',
                        data: color.histogram_b || [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#cbd5e1'
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Frequency',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Bin',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8',
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                }
            }
        });
    };

    /**
     * Render radial profile chart
     */
    v7.singleAnalysis.renderRadialProfile = function(radial) {
        if (!radial || !radial.profile) return;

        // Update summary stats
        const innerEl = document.getElementById('radialInner');
        const outerEl = document.getElementById('radialOuter');
        const uniformityEl = document.getElementById('radialUniformity');
        const patternTypeEl = document.getElementById('radialPatternType');

        if (innerEl && radial.summary) {
            innerEl.textContent = radial.summary.inner_mean_L.toFixed(1);
        }

        if (outerEl && radial.summary) {
            outerEl.textContent = radial.summary.outer_mean_L.toFixed(1);
        }

        if (uniformityEl && radial.summary) {
            uniformityEl.textContent = radial.summary.uniformity.toFixed(2);
        }

        // Pattern type detection
        if (patternTypeEl && radial.summary) {
            const inner = radial.summary.inner_mean_L;
            const outer = radial.summary.outer_mean_L;
            const diff = Math.abs(inner - outer);

            if (diff < 2) {
                patternTypeEl.textContent = 'Uniform';
            } else if (inner > outer) {
                patternTypeEl.textContent = 'Fade Out';
            } else {
                patternTypeEl.textContent = 'Fade In';
            }
        }

        // Render profile chart
        const ctx = document.getElementById('radialProfileChart');
        if (!ctx) return;

        // Destroy existing chart if any
        if (v7.singleAnalysis._radialChart) {
            v7.singleAnalysis._radialChart.destroy();
        }

        const numPoints = radial.profile.L_mean.length;
        const radii = Array.from({length: numPoints}, (_, i) => (i / numPoints).toFixed(2));

        v7.singleAnalysis._radialChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: radii,
                datasets: [
                    {
                        label: 'L*',
                        data: radial.profile.L_mean,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    },
                    {
                        label: 'a*',
                        data: radial.profile.a_mean,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    },
                    {
                        label: 'b*',
                        data: radial.profile.b_mean,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Lab Value',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Radius (normalized)',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8',
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                }
            }
        });
    };

    /**
     * Render ink clusters (복제 핵심 정보)
     */
    v7.singleAnalysis.renderInkClusters = function(ink) {
        if (!ink) return;

        const countEl = document.getElementById('inkClusterCount');
        const confidenceEl = document.getElementById('inkConfidence');
        const sortPolicyEl = document.getElementById('inkSortPolicy');
        const gridEl = document.getElementById('inkClustersGrid');

        if (countEl) {
            countEl.textContent = ink.k || 0;
        }

        if (confidenceEl) {
            const conf = (ink.confidence * 100).toFixed(0);
            confidenceEl.textContent = conf + '%';
        }

        if (sortPolicyEl) {
            sortPolicyEl.textContent = 'L_asc (dark → light)';
        }

        if (!gridEl) return;

        gridEl.innerHTML = '';

        const clusters = ink.clusters || [];
        if (clusters.length === 0) {
            gridEl.innerHTML = '<div class="text-text-dim text-sm col-span-3">클러스터가 검출되지 않았습니다.</div>';
            return;
        }

        clusters.forEach((cluster, idx) => {
            const card = document.createElement('div');
            card.className = 'cluster-card';

            const colorDiv = document.createElement('div');
            colorDiv.className = 'cluster-color';
            colorDiv.style.backgroundColor = cluster.mean_hex || '#666';

            // Use CIE values (preferred)
            const lab_cie = cluster.centroid_lab_cie || cluster.centroid_lab;

            const infoDiv = document.createElement('div');
            infoDiv.className = 'flex-1';
            infoDiv.innerHTML = `
                <div class="font-semibold mb-1">Ink #${cluster.id}</div>
                <div class="text-xs text-text-dim font-mono mb-2">
                    Lab* (CIE): (${lab_cie[0].toFixed(1)}, ${lab_cie[1].toFixed(1)}, ${lab_cie[2].toFixed(1)})
                </div>
                <div class="text-xs text-text-secondary">
                    Area: <span class="font-semibold">${(cluster.area_ratio * 100).toFixed(1)}%</span>
                </div>
                <div class="text-xs text-text-secondary mt-1">
                    Hex: <span class="font-mono">${cluster.mean_hex || '#000000'}</span>
                </div>
            `;

            card.appendChild(colorDiv);
            card.appendChild(infoDiv);
            gridEl.appendChild(card);
        });
    };

    /**
     * Render pattern quality metrics
     */
    v7.singleAnalysis.renderPatternQuality = function(pattern, artifacts) {
        if (!pattern) return;

        const angularEl = document.getElementById('patternAngular');
        const blobsEl = document.getElementById('patternBlobs');
        const contrastEl = document.getElementById('patternContrast');
        const edgeEl = document.getElementById('patternEdge');

        if (angularEl) {
            angularEl.textContent = pattern.angular_uniformity.toFixed(3);
        }

        if (blobsEl && pattern.center_blobs) {
            blobsEl.textContent = pattern.center_blobs.blob_count || 0;
        }

        if (contrastEl) {
            contrastEl.textContent = pattern.contrast.toFixed(1);
        }

        if (edgeEl) {
            edgeEl.textContent = pattern.edge_density.toFixed(3);
        }

        // Display pattern images if available
        if (artifacts) {
            const patternImagesDiv = document.getElementById('patternImages');
            const overlayImg = document.getElementById('patternOverlayImg');
            const heatmapImg = document.getElementById('patternHeatmapImg');
            const heatmapContainer = document.getElementById('patternHeatmapContainer');

            let hasImages = false;

            if (artifacts.overlay && overlayImg) {
                overlayImg.src = artifacts.overlay;
                hasImages = true;
            }

            if (artifacts.heatmap && heatmapImg && heatmapContainer) {
                heatmapImg.src = artifacts.heatmap;
                heatmapContainer.classList.remove('hidden');
                hasImages = true;
            } else if (heatmapContainer) {
                heatmapContainer.classList.add('hidden');
            }

            if (hasImages && patternImagesDiv) {
                patternImagesDiv.classList.remove('hidden');
            }
        }
    };

    /**
     * Render zone analysis on canvas
     */
    v7.singleAnalysis.renderZoneAnalysis = function(zones, gate) {
        if (!zones) return;

        const uniformityEl = document.getElementById('zoneUniformity');
        if (uniformityEl) {
            uniformityEl.textContent = zones.zone_uniformity.toFixed(2);
        }

        const canvas = document.getElementById('zoneCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2 - 20;

        // Clear canvas
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        // Draw zones
        const zoneList = zones.zones || [];
        zoneList.forEach((zone) => {
            const angleStart = zone.angle_range[0] * Math.PI / 180;
            const angleEnd = zone.angle_range[1] * Math.PI / 180;

            // Get L value for coloring
            const L = zone.mean_lab[0];
            const a = zone.mean_lab[1];
            const b = zone.mean_lab[2];

            // Simple Lab to RGB conversion (approximate)
            const color = labToRgbApprox(L, a, b);

            // Draw sector
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, angleStart, angleEnd);
            ctx.closePath();

            ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.7)`;
            ctx.fill();

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw zone number
            const midAngle = (angleStart + angleEnd) / 2;
            const textX = centerX + (radius * 0.7) * Math.cos(midAngle);
            const textY = centerY + (radius * 0.7) * Math.sin(midAngle);

            ctx.fillStyle = '#fff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(zone.zone_id.toString(), textX, textY);
        });

        // Draw center circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#10b981';
        ctx.fill();
    };

    /**
     * Render warnings
     */
    v7.singleAnalysis.renderWarnings = function(warnings) {
        const section = document.getElementById('warningsSection');
        const listEl = document.getElementById('warningsList');

        if (!warnings || warnings.length === 0) {
            if (section) section.classList.add('hidden');
            return;
        }

        if (section) section.classList.remove('hidden');
        if (!listEl) return;

        listEl.innerHTML = '';

        warnings.forEach(warning => {
            const badge = document.createElement('span');
            badge.className = 'warning-badge';
            badge.textContent = warning;
            listEl.appendChild(badge);
        });
    };

    /**
     * Render Operator Summary (최상단 요약)
     */
    v7.singleAnalysis.renderOperatorSummary = function(operatorSummary) {
        if (!operatorSummary) return;

        const summaryDiv = document.getElementById('operatorSummary');
        if (!summaryDiv) return;

        const decision = operatorSummary.decision || 'UNKNOWN';
        const grade = operatorSummary.quality_grade || 'F';
        const severity = operatorSummary.severity || 'LOW';
        const reasons = operatorSummary.decision_reason_top2 || [];
        const action = operatorSummary.action || '';

        // Style decision badge
        const decisionDiv = document.getElementById('operatorDecision');
        if (decisionDiv) {
            decisionDiv.textContent = decision;

            if (decision === 'PASS') {
                summaryDiv.className = 'mb-6 p-6 rounded-lg border-2 border-green-success bg-green-success/10';
                decisionDiv.className = 'text-3xl font-bold px-6 py-3 rounded-lg bg-green-success text-white';
            } else if (decision === 'RECAPTURE') {
                summaryDiv.className = 'mb-6 p-6 rounded-lg border-2 border-red-error bg-red-error/10';
                decisionDiv.className = 'text-3xl font-bold px-6 py-3 rounded-lg bg-red-error text-white';
            } else { // HOLD
                summaryDiv.className = 'mb-6 p-6 rounded-lg border-2 border-yellow-warning bg-yellow-warning/10';
                decisionDiv.className = 'text-3xl font-bold px-6 py-3 rounded-lg bg-yellow-warning text-black';
            }
        }

        // Grade
        const gradeDiv = document.getElementById('operatorGrade');
        if (gradeDiv) {
            gradeDiv.textContent = 'Grade: ' + grade;
        }

        // Severity
        const severityDiv = document.getElementById('operatorSeverity');
        if (severityDiv) {
            severityDiv.textContent = severity;
            severityDiv.className = 'text-sm font-semibold px-3 py-1 rounded ' +
                (severity === 'HIGH' ? 'bg-red-error text-white' :
                 severity === 'MEDIUM' ? 'bg-yellow-warning text-black' :
                 'bg-green-success text-white');
        }

        // Reasons
        const reasonsDiv = document.getElementById('operatorReasons');
        if (reasonsDiv) {
            reasonsDiv.innerHTML = '';
            reasons.forEach((reason, idx) => {
                const reasonItem = document.createElement('div');
                reasonItem.className = 'text-sm font-mono p-2 bg-white/5 rounded';
                reasonItem.textContent = `${idx + 1}. ${reason}`;
                reasonsDiv.appendChild(reasonItem);
            });
        }

        // Action
        const actionText = document.getElementById('operatorActionText');
        if (actionText) {
            actionText.textContent = action;
        }
    };

    /**
     * Render Engineer KPI (핵심 지표 10개)
     */
    v7.singleAnalysis.renderEngineerKPI = function(engineerKPI) {
        if (!engineerKPI) return;

        // QC
        const kpiQC = document.getElementById('kpiQC');
        if (kpiQC && engineerKPI.qc) {
            kpiQC.innerHTML = `
                <div>Sharpness: <span class="text-blue-primary">${engineerKPI.qc.sharpness}</span></div>
                <div>Center Offset: <span class="text-blue-primary">${engineerKPI.qc.center_offset_mm} mm</span></div>
                <div>Illum Asymmetry: <span class="text-blue-primary">${engineerKPI.qc.illumination_asymmetry}</span></div>
            `;
        }

        // Ink
        const kpiInk = document.getElementById('kpiInk');
        if (kpiInk && engineerKPI.ink) {
            const inkRatios = engineerKPI.ink.ink_area_ratios.map(r => r.toFixed(2)).join(', ');
            kpiInk.innerHTML = `
                <div>Detected Count: <span class="text-green-success">${engineerKPI.ink.detected_count}</span></div>
                <div>Clustering Conf: <span class="text-green-success">${engineerKPI.ink.clustering_confidence}</span></div>
                <div>Area Ratios: <span class="text-green-success">[${inkRatios}]</span></div>
            `;
        }

        // Pattern
        const kpiPattern = document.getElementById('kpiPattern');
        if (kpiPattern && engineerKPI.pattern) {
            kpiPattern.innerHTML = `
                <div>Angular Uniformity: <span class="text-purple-500">${engineerKPI.pattern.angular_uniformity}</span></div>
                <div>Zone Uniformity: <span class="text-purple-500">${engineerKPI.pattern.zone_uniformity}</span></div>
                <div>Radial Uniformity: <span class="text-purple-500">${engineerKPI.pattern.radial_uniformity}</span></div>
                <div>Radial Slope: <span class="text-purple-500">${engineerKPI.pattern.radial_slope}</span></div>
                <div>Ring Contrast: <span class="text-purple-500">${engineerKPI.pattern.ring_contrast}</span></div>
            `;
        }

        // Defect
        const kpiDefect = document.getElementById('kpiDefect');
        if (kpiDefect && engineerKPI.defect) {
            kpiDefect.innerHTML = `
                <div>Blob Count: <span class="text-red-error">${engineerKPI.defect.blob_count}</span></div>
                <div>Total Area: <span class="text-red-error">${engineerKPI.defect.blob_total_area} px²</span></div>
            `;
        }
    };

    /**
     * Helper: Approximate Lab to RGB conversion
     */
    function labToRgbApprox(L, a, b) {
        // Very rough approximation - just for visualization
        const r = Math.max(0, Math.min(255, L + 1.5 * a));
        const g = Math.max(0, Math.min(255, L - 0.5 * a + 0.5 * b));
        const b_val = Math.max(0, Math.min(255, L - 2.0 * b));

        return [Math.round(r), Math.round(g), Math.round(b_val)];
    }

    // Store chart instances for cleanup
    v7.singleAnalysis._colorChart = null;
    v7.singleAnalysis._radialChart = null;

})();
