# GitHub Actions Workflows

This directory contains CI/CD workflows for the SNPE Flask application.

## Workflows

### 1. `ci.yml` - Main CI Pipeline
Runs on every push and pull request to main/master/develop branches.

**Jobs:**
- **test**: Runs tests on Python 3.10 and 3.12
  - Installs system dependencies
  - Installs Python dependencies
  - Verifies imports
  - Runs pytest test suite
  - Performs syntax checks
  
- **lint**: Code quality checks
  - Runs flake8 (if installed)
  - Checks project structure
  
- **docker-build**: Tests Docker build
  - Attempts to build Docker image
  - Tests image structure

### 2. `test.yml` - Test Suite
Focused on running the test suite with coverage.

**Jobs:**
- **unit-tests**: Runs unit tests with coverage
  - Tests on Python 3.10 and 3.12
  - Generates coverage reports
  - Uploads to Codecov
  
- **integration-tests**: Runs integration tests
  - Sets up Mosquitto MQTT broker
  - Tests Flask application routes

### 3. `docker.yml` - Docker Build
Builds Docker images for releases.

**Triggers:**
- Push to main/master branches
- Tagged releases (v*)
- Pull requests

## Usage

### Running Tests Locally

Before pushing, you can run tests locally:

```bash
cd Tutorials
pytest tests/ -v
```

### Viewing Workflow Results

1. Go to your GitHub repository
2. Click on "Actions" tab
3. View workflow runs and their results

## Notes

- Some tests are skipped in CI because they require:
  - SNPE SDK libraries
  - ARM64 hardware
  - GStreamer with Qualcomm plugins
  - DLC model files

- The workflows use `continue-on-error: true` for steps that may fail due to missing dependencies

- Docker builds may require ARM64 emulation or actual ARM64 runners

## Customization

You can customize these workflows by:
- Adding more Python versions to the matrix
- Adding more test jobs
- Configuring deployment steps
- Adding code quality checks (black, isort, etc.)

