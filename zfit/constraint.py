#  Copyright (c) 2019 zfit

from .util import ztyping
from .core.constraint import SimpleConstraint, GaussianConstraint
import tensorflow as tf



__all__ = ["nll_gaussian", "SimpleConstraint", "GaussianConstraint"]


def nll_gaussian(params: ztyping.ParamTypeInput, observation: ztyping.NumericalScalarType,
                 uncertainty: ztyping.NumericalScalarType) -> tf.Tensor:
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.
    Args:
        params (list(zfit.Parameter)): The parameters to constraint
        observation (numerical, list(numerical)): observed values of the parameter
        uncertainty (numerical, list(numerical) or array/tensor): Uncertainties or covariance/error
            matrix of the observed values. Can either be a single value, a list of values, an array or a tensor
    Returns:
        `GaussianConstraint`: the constraint object
    Raises:
        ShapeIncompatibleError: if params, mu and sigma don't have the same size
    """

    return GaussianConstraint(params=params, observation=observation, uncertainty=uncertainty)
