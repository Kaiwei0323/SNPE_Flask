# Test Suite for SNPE Flask Application

This directory contains test cases for the SNPE Flask application modules.

## Test Structure

- `test_app.py` - Tests for Flask application routes
- `test_camera.py` - Tests for camera module and model_map
- `test_mqtt.py` - Tests for MQTT client
- `test_snpe.py` - Tests for SNPE helper manager (PerfProfile, Runtime, SnpeContext)
- `test_model_handlers.py` - Tests for model handlers (YOLOV5, YOLOV8, DETR)
- `test_pipelines.py` - Tests for GStreamer pipelines
- `conftest.py` - Pytest configuration and fixtures

## Running Tests

### Install pytest (if not already installed)
```bash
pip install pytest pytest-cov
```

### Run all tests
```bash
cd Tutorials
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_mqtt.py
```

### Run with coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run with verbose output
```bash
pytest tests/ -v
```

## Test Categories

### Unit Tests
- `test_mqtt.py` - MQTT client functionality
- `test_snpe.py` - SNPE constants and configuration
- `test_camera.py` - Model map structure

### Integration Tests
- `test_app.py` - Flask routes and endpoints
- `test_model_handlers.py` - Model handler initialization (requires DLC files)
- `test_pipelines.py` - Pipeline initialization (requires GStreamer)

## Notes

- Some tests are marked with `@pytest.mark.skip` because they require:
  - SNPE SDK libraries (`libsnpehelper.so`)
  - DLC model files
  - GStreamer libraries
  - Hardware access (DSP runtime)

- To run tests that require these dependencies, ensure:
  1. SNPE SDK is properly installed
  2. Model files are available in `models/` directory
  3. GStreamer is installed and configured
  4. Required hardware is available

## Mocking

Tests use mocking where possible to avoid requiring actual hardware or libraries:
- MQTT client tests use mocks
- Some pipeline tests use mocks for GStreamer

## Continuous Integration

These tests can be integrated into CI/CD pipelines. Tests that don't require hardware can run in any environment.

