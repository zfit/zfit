from __future__ import print_function, division, absolute_import

from zfit.core.utils import dotdict


def auto_integrate(dist, limits=None, sampler=None, ):

    return


# TODO
def numeric_integrate(func, limits, n_dims, method="AUTO", mc_sampler="TODO"):
    """Integrate `func` using numerical and/or MC methods."""
    pass


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
