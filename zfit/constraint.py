#  Copyright (c) 2019 zfit

from .util import ztyping
from .core.constraint import SimpleConstraint, GaussianConstraint
import tensorflow as tf



__all__ = ["nll_gaussian", "SimpleConstraint", "GaussianConstraint"]


def nll_gaussian(params: ztyping.ParamTypeInput, observation: ztyping.NumericalScalarType,
                 uncertainty: ztyping.NumericalScalarType) -> tf.Tensor:
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

    return GaussianConstraint(params=params, observation=observation, uncertainty=uncertainty)
