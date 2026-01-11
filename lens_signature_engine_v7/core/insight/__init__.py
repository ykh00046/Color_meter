"""
Batch analytics and trend analysis (v3).

This module provides post-processing analysis and aggregation over decision history.
Operates in batch mode with time flexibility (can take seconds).

Modules:
- summary.py: Single-result operator summary with severity classification
- trend.py: Multi-result time-series trend analysis (20+ samples)

Characteristics:
- Time axis: Batch/post-processing (multiple samples, historical analysis)
- Scope: Aggregation and pattern detection
- Dependencies: Reads decision module outputs (one-way dependency)
- Threshold integration: Uses centralized threshold_policy for ΔE gates

Usage:
    from core.insight.summary import build_v3_summary
    from core.insight.trend import build_v3_trend

    summary = build_v3_summary(decision_dict)
    trend = build_v3_trend(decision_history, window_requested=20)
"""
