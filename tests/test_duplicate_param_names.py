#  Copyright (c) 2025 zfit

"""Test that duplicate parameter names are detected and raise appropriate errors."""

from __future__ import annotations

import pytest

import zfit


def test_simpleloss_duplicate_param_names():
    """Test that SimpleLoss raises error for parameters with duplicate names."""
    param1 = zfit.Parameter('duplicate_name', 5.0)
    param2 = zfit.Parameter('duplicate_name', 3.0)

    with pytest.raises(ValueError, match="Parameters with duplicate names"):
        zfit.loss.SimpleLoss(
            lambda x: x[0]**2 + x[1]**2,
            params=[param1, param2],
            errordef=0.5
        )


def test_simpleloss_unique_param_names():
    """Test that SimpleLoss works correctly with unique parameter names."""
    param1 = zfit.Parameter('param1', 5.0)
    param2 = zfit.Parameter('param2', 3.0)

    # This should work fine
    loss = zfit.loss.SimpleLoss(
        lambda x: x[0]**2 + x[1]**2,
        params=[param1, param2],
        errordef=0.5
    )
    assert loss is not None
    params = loss.get_params()
    assert len(params) == 2


def test_minimizer_duplicate_param_names():
    """Test that minimizer raises error for parameters with duplicate names.
    
    Since SimpleLoss now validates parameters at construction, this tests that
    the check happens early and provides a clear error message.
    """
    param1 = zfit.Parameter('dup_param', 5.0)
    param2 = zfit.Parameter('dup_param', 3.0)

    # SimpleLoss should now catch this during construction
    with pytest.raises(ValueError, match="Parameters with duplicate names"):
        zfit.loss.SimpleLoss(
            lambda x: x[0]**2 + x[1]**2,
            params=[param1, param2],
            errordef=0.5
        )


def test_minimizer_with_explicit_duplicate_params():
    """Test that minimizer raises error when given explicit params with duplicate names."""
    param1 = zfit.Parameter('param1', 1.0)
    param2 = zfit.Parameter('param2', 2.0)
    param3 = zfit.Parameter('param3', 3.0)
    param_dup = zfit.Parameter('param1', 1.5)  # Duplicate name

    loss = zfit.loss.SimpleLoss(
        lambda x: x[0]**2 + x[1]**2 + x[2]**2,
        params=[param1, param2, param3],
        errordef=0.5
    )

    minimizer = zfit.minimize.Minuit()

    # Even when passing params explicitly with duplicates, should catch duplicates
    with pytest.raises(ValueError, match="Parameters with duplicate names"):
        minimizer.minimize(loss, params=[param1, param_dup, param2])
