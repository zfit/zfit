#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from .core.constraint import (
    GaussianConstraint,
    LogNormalConstraint,
    PoissonConstraint,
    SimpleConstraint,
)
from .util import ztyping
from .util.deprecation import deprecated

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = [
    "GaussianConstraint",
    "LogNormalConstraint",
    "PoissonConstraint",
    "SimpleConstraint",
]


@deprecated(None, "Use `GaussianConstraint` directly.")
def nll_gaussian(
    params: ztyping.ParamTypeInput,
    observation: ztyping.NumericalScalarType,
    uncertainty: ztyping.NumericalScalarType,
) -> GaussianConstraint:
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.

    Args:
        params: The parameters to constraint.
        observation: observed values of the parameter.
        uncertainty: Uncertainties or covariance/error.
            matrix of the observed values. Can either be a single value, a list of values, an array or a tensor.
    Returns:
        The constraint object.
    Raises:
        ShapeIncompatibleError: if params, mu and sigma don't have the same size.
    """
    return GaussianConstraint(params=params, observation=observation, sigma=uncertainty)
