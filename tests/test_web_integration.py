import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import app directly from src.web.app
from src.web.app import app

client = TestClient(app)


@pytest.fixture
def mock_sku_config(mocker):
    """Mock the load_sku_config function to return a sample config."""
    sample_config = {
        "sku_code": "TEST_SKU",
        "zones": {"A": {"L": 50.0, "a": 10.0, "b": 10.0, "threshold": 5.0}},
        "params": {"expected_zones": 1},
    }
    # Mock where it is imported in src.web.app
    mocker.patch("src.web.app.load_sku_config", return_value=sample_config)
    return sample_config


def create_dummy_image_bytes():
    """Create a dummy JPEG image in memory."""
    # Create a 200x200 image with a circle (lens)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)  # White circle
    _, encoded = cv2.imencode(".jpg", img)
    return io.BytesIO(encoded.tobytes())


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inspect_endpoint_success(mock_sku_config):
    """Test successful inspection via /inspect endpoint."""
    img_bytes = create_dummy_image_bytes()

    # Mocking pipeline to avoid complex processing logic errors during integration test
    # We want to test the WEB layer integration, not the core pipeline here.
    # However, if we want to test full integration, we need a valid image that passes detection.
    # The dummy image has a clear circle, so detection should pass.

    # We need to ensure sku_db path exists or is mocked.
    # Since we mocked load_sku_config, we don't need actual file.

    response = client.post(
        "/inspect",
        files={"file": ("test_lens.jpg", img_bytes, "image/jpeg")},
        data={"sku": "TEST_SKU", "expected_zones": "1"},
    )

    # Note: If pipeline fails (e.g. lens detection), it returns 400.
    # Our dummy image is simple, but might fail strict detection parameters.
    # Let's see the result. If it fails, we might need to mock Pipeline.process.

    if response.status_code != 200:
        print(f"Response error: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    assert data["sku"] == "TEST_SKU"
    assert "analysis" in data
    assert "lens_info" in data
    assert data["analysis"] is not None  # AnalysisService should return data


def test_inspect_endpoint_missing_file():
    """Test /inspect without file."""
    response = client.post("/inspect", data={"sku": "TEST_SKU"})
    assert response.status_code == 422  # Validation error (missing field)


def test_batch_endpoint_missing_params():
    """Test /batch without dir or zip."""
    response = client.post("/batch", data={"sku": "TEST_SKU"})
    assert response.status_code == 400
    assert "batch_dir or batch_zip" in response.json()["detail"]


def test_batch_endpoint_with_zip(mock_sku_config, tmp_path):
    """Test /batch with ZIP file."""
    # Create a zip file containing dummy images
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        img_bytes = create_dummy_image_bytes()
        zf.writestr("img1.jpg", img_bytes.getvalue())
        zf.writestr("img2.jpg", img_bytes.getvalue())

    zip_buffer.seek(0)

    response = client.post(
        "/batch", data={"sku": "TEST_SKU"}, files={"batch_zip": ("images.zip", zip_buffer, "application/zip")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "summary" in data
