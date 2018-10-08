from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability as tfp

from zfit.core.utils import dotdict


def auto_integrate(func, limits, n_dims, method="AUTO", dtype=None, mc_sampler="TODO",
                   mc_options=None):
    if method == "AUTO":  # TODO
        method = "mc"
    # TODO get n dims
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options['draws_per_dim']
        integral = mc_integrate(func=func, limits=limits, n_dims=n_dims, method=method, dtype=dtype,
                                mc_sampler=mc_sampler, draws_per_dim=draws_per_dim,
                                importance_sampling=None)
    return integral


# TODO
def numeric_integrate():
    """Integrate `func` using numerical and/or MC methods."""
    integral = None
    return integral


def mc_integrate(func, limits, n_dims=None, draws_per_dim=10000, method=None, dtype=None,
                 mc_sampler="TODO", importance_sampling=None):
    lower, upper = limits
    # TODO: get dimensions properly
    n_samples = n_dims * draws_per_dim
    # TODO: add times dim or so
    samples_normed = mc_sampler(dim=n_dims, num_results=n_samples, dtype=dtype)
    samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
    avg = tfp.monte_carlo.expectation(f=func, samples=samples)
    integral = avg * (upper - lower)
    return tf.cast(integral, dtype=dtype)


class AnalyticIntegral(object):
    def __init__(self, *args, **kwargs):
        super(AnalyticIntegral, self).__init__(*args, **kwargs)
        self.max_dims = []
        self._integrals = {}

    def register(self, func, dims):
        """Register an analytic integral."""
        if len(dims) > len(self.max_dims):
            self.max_dims = dims
        self._integrals[dims] = func

    def integrate(self, value, limits, dims):
        """Integrate analytically over the dims if available."""
        integral_fn = self._integrals.get(dims)
        if integral_fn is None:
            raise NotImplementedError("This integral is not available for dims {}".format(dims))
        integral = integral_fn(value=value, limits=limits)
        return integral
