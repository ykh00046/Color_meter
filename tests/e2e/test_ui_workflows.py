"""
E2E Tests for Web UI Workflows

Tests navigation, page accessibility, and key user workflows.
Note: Tests are limited to avoid server timeout issues on Windows.
"""

import re

import pytest
from playwright.sync_api import expect


class TestNavigation:
    """Test site navigation and page accessibility."""

    def test_home_page_loads(self, page, app_server):
        """Home page should load successfully."""
        page.goto(f"{app_server}/", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile(r"Color Meter"))
        expect(page.locator("nav")).to_be_visible()

    def test_v7_console_loads(self, page, app_server):
        """V7 Console page should load successfully."""
        page.goto(f"{app_server}/v7", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile(r"Color Meter v7"))

        # Check sub-navigation exists
        expect(page.locator(".sub-nav")).to_be_visible()

        # Check all tabs exist
        expect(page.locator("#tab-inspection")).to_be_visible()
        expect(page.locator("#tab-registration")).to_be_visible()
        expect(page.locator("#tab-std-admin")).to_be_visible()
        expect(page.locator("#tab-test")).to_be_visible()

    def test_single_analysis_page_loads(self, page, app_server):
        """Single Analysis page should load successfully."""
        page.goto(f"{app_server}/single_analysis", wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile(r"Color Meter"))

    def test_other_pages_load(self, page, app_server):
        """Calibration, History, Stats pages should load successfully."""
        for path in ["/calibration", "/history", "/stats"]:
            page.goto(f"{app_server}{path}", wait_until="domcontentloaded")
            expect(page).to_have_title(re.compile(r"Color Meter"))


class TestV7Console:
    """Test V7 Console specific functionality."""

    def test_tab_switching(self, page, app_server):
        """Tab switching should show/hide correct content."""
        page.goto(f"{app_server}/v7", wait_until="domcontentloaded")

        # Initial state: inspection tab active
        expect(page.locator("#tab-inspection")).to_have_class(re.compile(r"active"))
        expect(page.locator("#view-inspection")).to_have_class(re.compile(r"active"))

        # Click registration tab
        page.locator("#tab-registration").click()
        expect(page.locator("#tab-registration")).to_have_class(re.compile(r"active"))
        expect(page.locator("#view-registration")).to_have_class(re.compile(r"active"))
        expect(page.locator("#view-inspection")).not_to_have_class(re.compile(r"active"))

        # Click history tab
        page.locator("#tab-std-admin").click()
        expect(page.locator("#tab-std-admin")).to_have_class(re.compile(r"active"))
        expect(page.locator("#view-std-admin")).to_have_class(re.compile(r"active"))

        # Click test tab
        page.locator("#tab-test").click()
        expect(page.locator("#tab-test")).to_have_class(re.compile(r"active"))
        expect(page.locator("#view-test")).to_have_class(re.compile(r"active"))

    def test_inspection_ui_elements(self, page, app_server):
        """Inspection UI elements should be present."""
        page.goto(f"{app_server}/v7", wait_until="domcontentloaded")

        # Check product selector exists
        expect(page.locator("#inspProductSelect")).to_be_visible()

        # Check file input exists
        expect(page.locator("#inspFiles")).to_be_attached()

        # Check Run Inspection button
        btn = page.locator("#btnInspect")
        expect(btn).to_be_visible()
        expect(btn).to_have_text("Run Inspection")

        # Check visualization toggle buttons
        expect(page.locator("#viz-overlay")).to_be_attached()
        expect(page.locator("#viz-heatmap")).to_be_attached()
