# Tests Directory

This directory contains unit tests, integration tests, and performance tests.

## Structure

```
tests/
├── test_image_loader.py
├── test_lens_detector.py
├── test_radial_profiler.py
├── test_zone_segmenter.py
├── test_color_evaluator.py
├── test_sku_manager.py
├── test_logger.py
└── integration/
    ├── test_pipeline.py
    └── test_performance.py
```

## Running Tests

### All Tests
```bash
pytest
```

### With Coverage
```bash
pytest --cov=src --cov-report=html
```

### Specific Module
```bash
pytest tests/test_lens_detector.py -v
```

### Performance Tests Only
```bash
pytest tests/integration/test_performance.py -v
```

## Test Guidelines

1. **Naming**: Test files must start with `test_`
2. **Structure**: One test file per module
3. **Coverage**: Aim for 80%+ code coverage
4. **Fixtures**: Use pytest fixtures for common setup
5. **Mocking**: Use `pytest-mock` for external dependencies

## Test Categories

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test complete pipeline workflows
- **Performance Tests**: Measure processing speed and memory usage
- **Accuracy Tests**: Validate detection accuracy with known samples

## Example Test

```python
import pytest
from src.core.lens_detector import LensDetector

def test_lens_detector_basic():
    detector = LensDetector()
    # Test code here
    assert detector is not None
```
