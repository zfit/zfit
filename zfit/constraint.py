from zfit import ztf
from .util.container import convert_to_container

__all__ = ["nll_gaussian"]

def nll_gaussian(params, mu, sigma):
    params = convert_to_container(params, container=tuple)
    mu = convert_to_container(mu, container=tuple)
    sigma = convert_to_container(sigma, container=tuple)
    constraint = ztf.constant(0.)
    if not len(params) == len(mu) == len(sigma):
        raise ValueError("params, mu and sigma have to have the same length.")
    for param, mean, sig in zip(params, mu, sigma):
        mean = ztf.convert_to_tensor(mean)
        sig = ztf.convert_to_tensor(sig)
        constraint += ztf.reduce_sum(ztf.square(param - mean) / (2. * ztf.square(sig)))

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
