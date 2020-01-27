#  Copyright (c) 2019 zfit

from .util import ztyping
from .core.constraint import SimpleConstraint, GaussianConstraint
import tensorflow as tf



__all__ = ["nll_gaussian", "SimpleConstraint", "GaussianConstraint"]


def nll_gaussian(x: ztyping.NumericalScalarType, mu: ztyping.ParamTypeInput,
                 sigma: ztyping.NumericalScalarType) -> tf.Tensor:
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.

    Args:
        x (numerical, list(numerical) or list(zfit.Parameter)): Observed values of the parameter
            to constraint obtained from auxiliary measurements
        mu (list(zfit.Parameter)): The parameters to constraint
        sigma (numerical, list(numerical) or array/tensor): The standard deviations or covariance
            matrix of the constraint. Can either be a single value, a list of values, an array or a tensor
    Returns:
        `GaussianConstraint`: the constraint object
    Raises:
        ShapeIncompatibleError: if params, mu and sigma don't have the same size
    """

    return GaussianConstraint(x=x, mu=mu, sigma=sigma)

# def nll_pdf(constraints: dict):
#     if not constraints:
#         return z.constant(0.)  # adding 0 to nll
#     probs = []
#     for param, dist in constraints.items():
#         probs.append(dist.pdf(param))
#     # probs = [dist.pdf(param) for param, dist in constraints.items()]
#     constraints_neg_log_prob = -tf.reduce_sum(tf.log(probs))
#     return constraints_neg_log_prob
