"""
Security Tests

Tests for security utilities and input validation.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.utils.security import (
    SecurityError,
    safe_sku_path,
    validate_file_extension,
    validate_file_size,
    validate_sku_identifier,
)


class TestSkuValidation:
    """Test SKU identifier validation"""

    def test_valid_sku_formats(self):
        """Valid SKU formats should pass"""
        valid_skus = ["SKU001", "SKU_TEST", "SKU-2024-01", "SKU_Test_123", "A1B2C3", "test-sku_123"]
        for sku in valid_skus:
            assert validate_sku_identifier(sku) == sku

    def test_reject_path_traversal(self):
        """Path traversal attempts should be rejected"""
        malicious_skus = [
            "../etc/passwd",
            "../../config",
            "..\\windows\\system32",
            "SKU/../../../secret",
            "test/../admin",
        ]
        for sku in malicious_skus:
            # Should raise SecurityError (either "Invalid SKU format" or "Path traversal")
            with pytest.raises(SecurityError):
                validate_sku_identifier(sku)

    def test_reject_invalid_characters(self):
        """Invalid characters should be rejected"""
        invalid_skus = [
            "SKU@001",
            "SKU 001",  # space
            "SKU#001",
            "SKU/001",
            "SKU\\001",
            "SKU;DROP TABLE",
            "SKU<script>",
            "SKU\x00NULL",
        ]
        for sku in invalid_skus:
            with pytest.raises(SecurityError, match="Invalid SKU format"):
                validate_sku_identifier(sku)

    def test_reject_empty_sku(self):
        """Empty SKU should be rejected"""
        with pytest.raises(SecurityError, match="cannot be empty"):
            validate_sku_identifier("")

    def test_reject_non_string(self):
        """Non-string SKU should be rejected"""
        with pytest.raises(SecurityError, match="must be a string"):
            validate_sku_identifier(123)

    def test_reject_too_long(self):
        """SKU longer than 50 chars should be rejected"""
        long_sku = "A" * 51
        with pytest.raises(SecurityError, match="Invalid SKU format"):
            validate_sku_identifier(long_sku)


class TestSafeSkuPath:
    """Test safe SKU path generation"""

    def test_valid_sku_path(self):
        """Valid SKU should generate correct path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            sku_path = safe_sku_path("SKU001", config_dir)

            # Should be inside config_dir
            assert sku_path.parent == config_dir.resolve()
            assert sku_path.name == "SKU001.json"

    def test_prevent_path_traversal_in_path(self):
        """Path traversal should be blocked at path level"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # This should fail at SKU validation level
            with pytest.raises(SecurityError):
                safe_sku_path("../etc/passwd", config_dir)

    def test_path_stays_within_config_dir(self):
        """Generated path must stay within config directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            sku_path = safe_sku_path("SKU001", config_dir)

            # Verify path is within config_dir
            try:
                sku_path.relative_to(config_dir)
            except ValueError:
                pytest.fail("Path escaped config directory")


class TestFileValidation:
    """Test file upload validation"""

    def test_valid_file_extensions(self):
        """Valid image extensions should pass"""
        valid_files = ["image.jpg", "photo.JPG", "picture.jpeg", "test.png", "sample.PNG"]
        for filename in valid_files:
            assert validate_file_extension(filename) is True

    def test_invalid_file_extensions(self):
        """Invalid extensions should fail"""
        invalid_files = ["script.exe", "malware.sh", "virus.bat", "exploit.php", "hack.js", "file.txt", "document.pdf"]
        for filename in invalid_files:
            assert validate_file_extension(filename) is False

    def test_custom_allowed_extensions(self):
        """Custom extension list should work"""
        assert validate_file_extension("test.bmp", [".bmp", ".tiff"]) is True
        assert validate_file_extension("test.jpg", [".bmp", ".tiff"]) is False

    def test_file_size_validation(self):
        """File size limits should work"""
        # 5MB should pass with 10MB limit
        assert validate_file_size(5 * 1024 * 1024, max_size_mb=10) is True

        # 15MB should fail with 10MB limit
        assert validate_file_size(15 * 1024 * 1024, max_size_mb=10) is False

        # Exactly at limit should pass
        assert validate_file_size(10 * 1024 * 1024, max_size_mb=10) is True

        # Just over limit should fail
        assert validate_file_size(10 * 1024 * 1024 + 1, max_size_mb=10) is False


class TestBaselineConsistency:
    """Test baseline generation zone consistency (Issue 3)"""

    def test_zone_structure_validation(self):
        """Baseline generation should enforce consistent zone structure"""
        # This test would require mocking the SKU manager
        # For now, documenting the expected behavior:
        # 1. First OK sample defines expected zone structure (e.g., [A, B, C])
        # 2. Subsequent samples must have exact same zone names
        # 3. Samples with different zone structures are skipped with warning
        # 4. Final baseline only includes consistent samples
        pass  # TODO: Implement when SKU manager is refactored


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
