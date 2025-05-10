# zfit Development Guidelines

This document provides essential information for developers working on the zfit project. It includes build/configuration instructions, testing information, and additional development guidelines.

## Build/Configuration Instructions

### Installation

zfit can be installed with different sets of dependencies depending on your needs:

1. **Basic Installation**:
   ```bash
   pip install .
   ```

2. **Development Installation**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Full Installation with All Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Platform-Specific Dependencies

The project has different dependencies based on the operating system:

- **Linux**: Full support for all dependencies
- **macOS**: Most dependencies supported, with special handling for Apple Silicon
- **Windows**: Some limitations (e.g., jaxlib not available)

### Environment Setup

For development, it's recommended to use a conda/mamba environment:

```bash
# Create environment with micromamba
micromamba create -n zfit python=3.9 uv
micromamba activate zfit

# Install dependencies
uv pip install -e ".[dev]"
```

For testing with ROOT support (not available on Windows or Python 3.12):

```bash
micromamba create -n zfit python=3.9 uv root<6.32
micromamba activate zfit
uv pip install -e ".[dev]"
```

## Testing Information

### Running Tests

zfit uses pytest for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-config=pyproject.toml

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_pdf_normal.py

# Run specific test function
pytest tests/test_pdf_normal.py::test_gauss1
```

### Test Options

The project supports several pytest options:

- `--longtests`: Run longer tests that take more time
- `--longtests-kde`: Run longer tests specifically for KDE
- `--recreate-truth`: Recreate truth values for tests

### Creating New Tests

When creating new tests:

1. Place test files in the appropriate subdirectory of the `tests` directory
2. Follow the naming convention `test_*.py` for test files and `test_*` for test functions
3. Use pytest fixtures from `conftest.py` when needed
4. Set a fixed random seed for reproducible tests
5. Use `np.testing.assert_allclose` for comparing floating-point values
6. Use `pytest.approx` for approximate equality assertions

### Example Test

Here's a simple example of a test for a Gaussian PDF:

```python
import numpy as np
import pytest
import zfit
from zfit import Parameter
from zfit.models.dist_tfp import Gauss

def test_simple_gauss():
    """Test that a simple Gaussian PDF works correctly."""
    # Create a Gaussian PDF
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    obs = zfit.Space("x", limits=(-10, 10))
    gauss = Gauss(mu=mu, sigma=sigma, obs=obs)
    
    # Generate some test points
    test_points = np.array([-1.0, 0.0, 1.0])
    
    # Calculate PDF values
    pdf_values = gauss.pdf(test_points)
    
    # Expected values for a standard normal distribution
    expected_values = np.array([0.24197072, 0.39894228, 0.24197072])
    
    # Check that the values are close to the expected ones
    np.testing.assert_allclose(pdf_values, expected_values, rtol=1e-5)
    
    # Test that the PDF integrates to 1
    integral = gauss.integrate(limits=obs)
    assert pytest.approx(integral, rel=1e-5) == 1.0
```

## Additional Development Information

### Code Style

The project uses several tools to enforce code style and quality:

1. **Ruff**: For linting and formatting with a line length of 120 characters
2. **Pre-commit hooks**: For various code quality checks
3. **Type annotations**: The project uses Python type annotations

To set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Project Structure

- `zfit/`: Main package directory
  - `core/`: Core functionality
  - `models/`: PDF models
  - `minimizers/`: Optimization algorithms
  - `_mcmc/`: MCMC samplers
  - `_bayesian/`: Bayesian inference
  - `util/`: Utility functions
- `tests/`: Test directory
- `examples/`: Example scripts
- `docs/`: Documentation

### Environment Variables

- `ZFIT_DO_JIT`: Enable/disable JIT compilation (1 or 0)
- `ZFIT_DISABLE_TF_WARNINGS`: Control TensorFlow warnings (0, 1, or 2)

### Debugging Tips

1. **TensorFlow Debugging**:
   - Set `ZFIT_DISABLE_TF_WARNINGS=0` to see TensorFlow warnings
   - Use `tf.debugging.enable_check_numerics()` to catch NaN/Inf values

2. **Performance Profiling**:
   - Use `pytest-benchmark` for performance testing
   - Control chunking with `zfit.run.chunking.max_n_points` and `zfit.run.chunking.active`

3. **Common Issues**:
   - GPU errors: The library works without GPU acceleration, but will use it if available
   - Memory issues: Large datasets may require chunking

### Continuous Integration

The project uses GitHub Actions for CI with the following jobs:

1. `unittests`: Run tests on different OS and Python versions
2. `docs`: Build and test documentation
3. `tutorials`: Run tutorial notebooks
4. `examples`: Run example scripts

### Documentation

To build the documentation locally:

```bash
cd docs
make html
```

The documentation will be available in `docs/_build/html/index.html`.
