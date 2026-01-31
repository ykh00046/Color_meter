"""
Real-time decision making for single samples.

This module handles immediate judgment for individual lens inspection results.
Operates in real-time (< 0.1s) during production flow.

Modules:
- decision_engine.py: V1 simple rules (gate → sig → anom)
- decision_builder.py: V2 uncertainty + evidence synthesis with robust statistics
- uncertainty.py: Confidence and uncertainty metrics calculation

Characteristics:
- Time axis: Real-time (single sample, immediate decision)
- Scope: Individual sample judgment
- Dependencies: None on insight module (can run independently)
"""
