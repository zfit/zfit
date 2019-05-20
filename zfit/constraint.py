from zfit import ztf
from .util.container import convert_to_container
import tensorflow as tf

__all__ = ["nll_gaussian"]


def nll_gaussian(params, mu, sigma):
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.

    Args:
        params (list(zfit.Parameter)): The parameters to constraint
        mu (float, list(float) or numpy array): The central value of the constraint
        sigma (float, list(float), numpy array or a tensor): The standard deviations or covariance
        matrix of the constraint
    Returns:
        graph: the nll of the constraint
    Raises:
        ValueError: if params, mu and sigma don't have the same size
    """

    params = convert_to_container(params, tuple)
    mu = convert_to_container(mu, tuple)
    sigma = ztf.convert_to_tensor(sigma)

    def covfunc(s):
        return tf.diag(ztf.pow(s, 2.))

    if len(sigma.shape) > 1:
        covariance = sigma
    elif len(sigma.shape) == 1:
        covariance = covfunc(sigma)
    else:
        sigma = tf.reshape(sigma, [1])
        covariance = covfunc(sigma)

    if not len(params) == len(mu) == covariance.shape[0] == covariance.shape[1]:
        raise ValueError("params, mu and sigma have to have the same length.")

    params = ztf.convert_to_tensor(params)
    mu = ztf.convert_to_tensor(mu)

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
