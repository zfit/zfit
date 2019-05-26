#  Copyright (c) 2019 zfit

from .util.exception import ShapeIncompatibleError
from .util import ztyping
from .util.container import convert_to_container
from zfit import ztf
import tensorflow as tf
import numpy as np

__all__ = ["nll_gaussian"]


def nll_gaussian(params: ztyping.ParamTypeInput, mu: ztyping.NumericalScalarType,
                 sigma: ztyping.NumericalScalarType) -> tf.Tensor:
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.

    Args:
        params (list(zfit.Parameter)): The parameters to constraint
        mu (numerical, list(numerical)): The central value of the constraint
        sigma (numerical, list(numerical) or array/tensor): The standard deviations or covariance
            matrix of the constraint. Can either be a single value or
    Returns:
        `tf.Tensor`: the nll of the constraint
    Raises:
        ShapeIncompatibleError: if params, mu and sigma don't have the same size
    """

    params = convert_to_container(params, tuple)
    mu = convert_to_container(mu, container=tuple, non_containers=[np.ndarray])

    params = ztf.convert_to_tensor(params)
    mu = ztf.convert_to_tensor(mu)
    sigma = ztf.convert_to_tensor(sigma)

    def covfunc(s):
        return tf.diag(ztf.pow(s, 2.))

    if sigma.shape.ndims > 1:
        covariance = sigma
    elif sigma.shape.ndims == 1:
        covariance = covfunc(sigma)
    else:
        sigma = tf.reshape(sigma, [1])
        covariance = covfunc(sigma)

    if not params.shape[0] == mu.shape[0] == covariance.shape[0] == covariance.shape[1]:
        raise ShapeIncompatibleError(f"params, mu and sigma have to have the same length. Currently"
                                     f"param: {params.shape[0]}, mu: {mu.shape[0]}, "
                                     f"covariance (from sigma): {covariance.shape[0:2]}")

    x = (params - mu)
    xt = tf.transpose(x)

    constraint = tf.tensordot(tf.linalg.inv(covariance), x, 1)
    constraint = 0.5 * tf.tensordot(xt, constraint, 1)

    return constraint

# def nll_pdf(constraints: dict):
#     if not constraints:
#         return ztf.constant(0.)  # adding 0 to nll
#     probs = []
#     for param, dist in constraints.items():
#         probs.append(dist.pdf(param))
#     # probs = [dist.pdf(param) for param, dist in constraints.items()]
#     constraints_neg_log_prob = -tf.reduce_sum(tf.log(probs))
#     return constraints_neg_log_prob
