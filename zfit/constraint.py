from zfit import ztf
from .util.container import convert_to_container
import numpy as np
import tensorflow as tf

__all__ = ["nll_gaussian"]

#def nll_gaussian(params, mu, sigma):
#    params = convert_to_container(params, container=tuple)
#    mu = convert_to_container(mu, container=tuple)
#    sigma = convert_to_container(sigma, container=tuple)
#    constraint = ztf.constant(0.)
#    if not len(params) == len(mu) == len(sigma):
#        raise ValueError("params, mu and sigma have to have the same length.")
#    for param, mean, sig in zip(params, mu, sigma):
#        mean = ztf.convert_to_tensor(mean)
#        sig = ztf.convert_to_tensor(sig)
#        constraint += ztf.reduce_sum(ztf.square(param - mean) / (2. * ztf.square(sig)))
#
#    return constraint
    
def nll_gaussian(params, mu, sigma):
    """Return negative log likelihood graph for gaussian constraints on a list of parameters.
    
    Args:
        params (list(zfit.Parameter)): The parameters to constraint
        mu (float, list(float) or numpy array): The central value of the constraint
        sigma (float, list(float) or numpy array): The standard deviations of correlations matrix of the constraint
    Returns:
        graph: the nll of the constraint
    Raises:
        ValueError: if params, mu and sigma don't have the same size
    """
    
    params = convert_to_container(params, tuple)
    mu = convert_to_container(mu, tuple)
    
    iscontainer = isinstance(sigma, (list, tuple))
    isarray = isinstance(sigma, (np.ndarray))
    is1darray = isarray and sigma.ndim == 1
    isnumber = lambda x: isinstance(x, (float, int))
    
    if iscontainer or is1darray:
        covariance = np.diag(sigma)
    elif isnumber(sigma):
        covariance = np.diag([sigma])
        
    covariance = ztf.convert_to_tensor(covariance)
                
    if not len(params) == len(mu) == covariance.shape[0] == covariance.shape[1]:
        raise ValueError("params, mu and sigma have to have the same length.")
        
    params = ztf.convert_to_tensor(params)
    mu = ztf.convert_to_tensor(mu)
        
    X = (params - mu)
    Xt = tf.transpose(X)

    constraint = tf.tensordot(tf.linalg.inv(covariance), X, 1)
    constraint = 0.5 * tf.tensordot(Xt, constraint, 1)

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
