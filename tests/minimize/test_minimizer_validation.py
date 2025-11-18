"""Test input validation for minimizers."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import pytest

import zfit
from zfit.minimizers.baseminimizer import BaseMinimizer


@pytest.mark.parametrize("tol", [0.001, 1e-5, 1])
def test_tol_valid_values(tol):
    """Test that valid tol values are accepted."""
    minimizer = BaseMinimizer(tol=tol)
    assert minimizer.tol == float(tol)


@pytest.mark.parametrize("tol,error_type,error_match", [
    (0, ValueError, "tol must be positive"),
    (-0.001, ValueError, "tol must be positive"),
    ("0.001", TypeError, "tol must be numeric"),
    ([0.001], TypeError, "tol must be numeric"),
])
def test_tol_invalid_values(tol, error_type, error_match):
    """Test that invalid tol values raise appropriate errors."""
    with pytest.raises(error_type, match=error_match):
        BaseMinimizer(tol=tol)


@pytest.mark.parametrize("verbosity", [0, 5, 10])
def test_verbosity_valid_values(verbosity):
    """Test that valid verbosity values are accepted."""
    minimizer = BaseMinimizer(verbosity=verbosity)
    assert minimizer.verbosity == verbosity


@pytest.mark.parametrize("verbosity,error_type,error_match", [
    (-1, ValueError, "verbosity must be between 0 and 10"),
    (11, ValueError, "verbosity must be between 0 and 10"),
    (5.5, TypeError, "verbosity must be an integer"),
    ("5", TypeError, "verbosity must be an integer"),
])
def test_verbosity_invalid_values(verbosity, error_type, error_match):
    """Test that invalid verbosity values raise appropriate errors."""
    with pytest.raises(error_type, match=error_match):
        BaseMinimizer(verbosity=verbosity)


@pytest.mark.parametrize("maxiter", [
    1000,
    1,
    "auto",
    100.0,  # float should be accepted
    1e6,    # scientific notation
    1e20,   # very large number (used by Minuit)
])
def test_maxiter_valid_values(maxiter):
    """Test that valid maxiter values are accepted."""
    minimizer = BaseMinimizer(maxiter=maxiter)
    if maxiter == "auto":
        assert minimizer._maxiter == "auto"
    else:
        # Float values are converted to int
        assert isinstance(minimizer._maxiter, (int, str))


@pytest.mark.parametrize("maxiter,error_type,error_match", [
    (0, ValueError, "maxiter must be positive"),
    (-100, ValueError, "maxiter must be positive"),
    (-100.5, ValueError, "maxiter must be positive"),
    ("100", TypeError, "maxiter must be numeric or 'auto'"),
    ([100], TypeError, "maxiter must be numeric or 'auto'"),
])
def test_maxiter_invalid_values(maxiter, error_type, error_match):
    """Test that invalid maxiter values raise appropriate errors."""
    with pytest.raises(error_type, match=error_match):
        BaseMinimizer(maxiter=maxiter)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_minuit_mode_valid_values(mode):
    """Test that valid Minuit mode values are accepted."""
    minimizer = zfit.minimize.Minuit(mode=mode)
    assert minimizer is not None


@pytest.mark.parametrize("mode", [3, -1])
def test_minuit_mode_invalid_values(mode):
    """Test that invalid Minuit mode values raise errors."""
    with pytest.raises(ValueError, match="mode has to be 0, 1 or 2"):
        zfit.minimize.Minuit(mode=mode)


@pytest.mark.parametrize("tol,verbosity,maxiter", [
    (0.01, 3, 500),     # All valid
    (1e-3, 0, "auto"),  # Mix of different valid values
    (1.0, 10, 1e6),     # Edge cases but valid
])
def test_combined_valid_parameters(tol, verbosity, maxiter):
    """Test that valid parameter combinations work."""
    minimizer = BaseMinimizer(tol=tol, verbosity=verbosity, maxiter=maxiter)
    assert minimizer.tol == float(tol)
    assert minimizer.verbosity == verbosity


@pytest.mark.parametrize("params,error_type,error_match", [
    ({"tol": -0.01, "verbosity": 3, "maxiter": 500}, ValueError, "tol must be positive"),
    ({"tol": 0.01, "verbosity": 15, "maxiter": 500}, ValueError, "verbosity must be between 0 and 10"),
    ({"tol": 0.01, "verbosity": 3, "maxiter": -500}, ValueError, "maxiter must be positive"),
])
def test_combined_invalid_parameters(params, error_type, error_match):
    """Test that invalid parameter combinations raise appropriate errors."""
    with pytest.raises(error_type, match=error_match):
        BaseMinimizer(**params)


def test_defaults_are_valid():
    """Test that default values pass validation."""
    minimizer = BaseMinimizer()
    assert minimizer.tol > 0
    assert 0 <= minimizer.verbosity <= 10
    assert minimizer._maxiter == "auto" or minimizer._maxiter > 0


@pytest.fixture
def base_minimizer():
    """Fixture providing a default BaseMinimizer instance."""
    return BaseMinimizer()


def test_minimizer_has_expected_attributes(base_minimizer):
    """Test that minimizer has expected attributes after initialization."""
    assert hasattr(base_minimizer, 'tol')
    assert hasattr(base_minimizer, 'verbosity')
    assert hasattr(base_minimizer, '_maxiter')
    assert hasattr(base_minimizer, 'name')
