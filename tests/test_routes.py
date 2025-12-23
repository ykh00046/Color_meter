#!/usr/bin/env python3
"""Test script to verify routes are loaded"""

from src.web.app import app

print("=== All routes ===")
for route in app.routes:
    if hasattr(route, "path"):
        methods = getattr(route, "methods", set())
        print(f"{' '.join(sorted(methods)):10} {route.path}")

print("\n=== Looking for /stats and /history ===")
stats_found = any(r.path == "/stats" for r in app.routes if hasattr(r, "path"))
history_found = any(r.path == "/history" for r in app.routes if hasattr(r, "path"))
print(f"/stats found: {stats_found}")
print(f"/history found: {history_found}")
