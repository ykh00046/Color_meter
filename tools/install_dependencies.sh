#!/bin/bash
# ============================================
# Contact Lens Color Inspection System
# Dependency Installation Script (Linux/Mac)
# ============================================

set -e  # Exit on error

echo ""
echo "========================================"
echo "  Installing Dependencies"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

python3 --version

echo ""
echo "[1/4] Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "[2/4] Installing core dependencies..."
pip3 install -r requirements.txt

echo ""
echo "[3/4] Verifying installation..."
python3 tools/check_imports.py

echo ""
echo "[4/4] Running quick tests..."
pytest tests/test_ink_estimator.py::TestInkEstimatorSampling::test_sample_ink_pixels_basic -v

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run full test suite: pytest tests/ -v"
echo "  2. Start Web UI: python3 -m src.web.app"
echo "  3. Run pipeline: python3 -m src.pipeline data/raw_images/SKU001_OK_001.jpg"
echo ""

exit 0
