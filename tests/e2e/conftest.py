"""
E2E Test Configuration for Playwright

Provides fixtures for browser testing with the web application.
"""

import os
import socket
import subprocess
import time
from contextlib import closing

import pytest


def find_free_port():
    """Find an available port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_port():
    """Get a free port for the test server."""
    return find_free_port()


@pytest.fixture(scope="session")
def app_server(server_port):
    """
    Start the FastAPI server for testing.

    Yields the base URL for the test server.
    """
    # Set environment to suppress unnecessary logging
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Start uvicorn server in subprocess (no workers on Windows)
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "src.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--timeout-keep-alive",
            "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="C:/X/Color_meter",
        env=env,
    )

    # Wait for server to start
    base_url = f"http://127.0.0.1:{server_port}"
    max_retries = 60
    for i in range(max_retries):
        try:
            import urllib.request

            urllib.request.urlopen(f"{base_url}/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        process.kill()
        raise RuntimeError("Server failed to start")

    # Give server a moment to fully initialize
    time.sleep(1)

    yield base_url

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="session")
def browser_context(browser, app_server):
    """
    Create a shared browser context for all tests.
    """
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(browser_context, app_server):
    """
    Create a new browser page for each test.

    Uses shared browser context for better performance.
    """
    page = browser_context.new_page()
    page.set_default_timeout(30000)  # 30 seconds
    page.set_default_navigation_timeout(30000)

    yield page

    page.close()
