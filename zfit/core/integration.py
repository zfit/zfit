"""
This module contains functions for the numeric as well as the analytic (partial) integration.
"""

import collections
import numpy as np
import typing

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Optional

from zfit import ztf
from zfit.util import ztyping
from .limits import convert_to_range, Range, supports
from ..settings import types as ztypes


@supports()
def auto_integrate(func, limits, n_dims, x=None, method="AUTO", dtype=ztypes.float,
                   mc_sampler=tfp.mcmc.sample_halton_sequence,
                   mc_options=None):
    if method == "AUTO":  # TODO unfinished, other methods?
        method = "mc"
    # TODO get n axes
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options['draws_per_dim']
        integral = mc_integrate(x=x, func=func, limits=limits, n_dims=n_dims, method=method, dtype=dtype,
                                mc_sampler=mc_sampler, draws_per_dim=draws_per_dim,
                                importance_sampling=None)
    return integral


# TODO implement numerical integration method
def numeric_integrate():
    """Integrate `func` using numerical methods."""
    integral = None
    return integral


def mc_integrate(func: Callable, limits: ztyping.LimitsType, axes: Optional[ztyping.DimsType] = None,
                 x: Optional[ztyping.XType] = None, n_dims: Optional[int] = None, draws_per_dim: int = 10000,
                 method: str = None,
                 dtype: typing.Type = ztypes.float,
                 mc_sampler: Callable = tfp.mcmc.sample_halton_sequence,
                 importance_sampling: Optional[Callable] = None) -> tf.Tensor:
    """Monte Carlo integration of `func` over `limits`.

    Args:
        func (callable): The function to be integrated over
        limits (`Range`): The limits of the integral
        axes (tuple(int)): The row to integrate over. None means integration over all values
        x (numeric): If a partial integration is performed, this are the values where x will be evaluated.
        n_dims (int): the number of total dimensions (old?)
        draws_per_dim (int): How many random points to draw per dimensions
        method (str): Which integration method to use
        dtype (dtype): |dtype_arg_descr|
        mc_sampler (callable): A function that takes one argument (`n_draws` or similar) and returns
            random values between 0 and 1.
        importance_sampling ():

    Returns:
        numerical: the integral
    """
    if axes is not None and n_dims is not None:
        raise ValueError("Either specify axes or n_dims")
    limits = convert_to_range(limits, axes)

    axes = limits.axes
    partial = (axes is not None) and (x is not None)  # axes, value can be tensors

    if axes is not None and n_dims is None:
        n_dims = len(axes)
    if n_dims is not None and axes is None:
        axes = tuple(range(n_dims))

    lower, upper = limits.limits()
    if np.infty in upper[0] or -np.infty in lower[0]:
        raise ValueError("MC integration does (currently) not support unbound limits (np.infty) as given here:"
                         "\nlower: {}, upper: {}".format(lower, upper))

    lower = ztf.convert_to_tensor(lower, dtype=dtype)
    upper = ztf.convert_to_tensor(upper, dtype=dtype)

    n_samples = draws_per_dim ** n_dims
    if partial:
        n_vals = x.get_shape()[0].value
        n_samples *= n_vals  # each entry wants it's mc
    else:
        n_vals = 1
    # TODO: deal with n_dims properly?
    samples_normed = mc_sampler(dim=n_dims, num_results=n_samples, dtype=dtype)
    samples_normed = tf.reshape(samples_normed, shape=(n_vals, int(n_samples / n_vals), n_dims))
    samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
    samples = tf.transpose(samples, perm=[2, 0, 1])

    if partial:
        value_list = []
        index_samples = 0
        index_values = 0
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=1)
        for i in range(n_dims + x.shape[1].value):
            if i in axes:
                value_list.append(samples[index_samples, :, :])
                index_samples += 1
            else:
                value_list.append(tf.expand_dims(x[:, index_values], axis=1))
                index_values += 1
        value_list = [tf.cast(val, dtype=dtype) for val in value_list]
        x = value_list
    else:
        x = samples

    # convert rnd samples with values to feedable vector
    reduce_axis = 1 if partial else None
    avg = tfp.monte_carlo.expectation(f=func, samples=x, axis=reduce_axis)
    # TODO: importance sampling?
    # avg = tfb.monte_carlo.expectation_importance_sampler(f=func, samples=value,axis=reduce_axis)
    integral = avg * limits.area()
    return ztf.to_real(integral, dtype=dtype)


class AnalyticIntegral(object):
    def __init__(self, *args, **kwargs):
        """Hold analytic integrals and manage their dimensions, limits etc."""
        super(AnalyticIntegral, self).__init__(*args, **kwargs)
        self._integrals = collections.defaultdict(dict)

    def get_max_dims(self, limits: ztyping.LimitsType, dims: ztyping.DimsType = None) -> typing.Tuple[int]:
        """Return the maximal available axes to integrate over analytically for given limits

        Args:
            limits (Range): The integral function will be able to integrate over this limits
            dims (tuple): The dimensions it has to integrate over (or a subset!)

        Returns:
            Tuple[int]:
        """
        limits = convert_to_range(limits=limits, dims=dims)

        return self._get_max_dims_limits(limits, out_of_dims=dims)[0]  # only axes

    def _get_max_dims_limits(self, limits, out_of_dims):  # TODO: automatic caching? but most probably not relevant
        if out_of_dims:
            out_of_dims = frozenset(out_of_dims)
            implemented_dims = set(d for d in self._integrals.keys() if d <= out_of_dims)
        else:
            implemented_dims = set(self._integrals.keys())
        implemented_dims = sorted(implemented_dims, key=len, reverse=True)  # iter through biggest first
        for dims in implemented_dims:
            limits_matched = []
            for lim, integ in self._integrals[dims].items():
                if integ.limits >= limits:
                    limits_matched.append(lim)

            if limits_matched:  # one or more integrals available
                return tuple(sorted(dims)), limits_matched
        return (), ()  # no integral available for this axes

    def get_max_integral(self, limits: ztyping.LimitsType,
                         dims: ztyping.DimsType = None) -> typing.Union[None, "Integral"]:
        """Return the integral over the `limits` with `axes` (or a subset of them).

        Args:
            limits (`zfit.Range`):
            dims (Tuple[int]):

        Returns:
            Union[None, Integral]: Return a callable that integrated over the given limits.
        """
        limits = convert_to_range(limits=limits, dims=dims)

        dims, limits = self._get_max_dims_limits(limits=limits, out_of_dims=dims)
        dims = frozenset(dims)
        integrals = [self._integrals[dims][lim] for lim in limits]
        integral_fn = max(integrals, key=lambda l: l.priority, default=None)
        return integral_fn

    def register(self, func: Callable, dims: ztyping.DimsType, limits: ztyping.LimitsType = None, priority: int = 50, *,
                 supports_norm_range: bool = False, supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral.

        Args:
            func (callable): The integral function. Takes 1 argument.
            dims (tuple): |dims_arg_descr|
            limits (Range): |limits_arg_descr| `Limits` can be None if `func` works for any possible limits
            priority (int): If two or more integrals can integrate over certain limits, the one with the higher
                priority is taken (usually around 0-100).
            supports_norm_range (bool): If True, norm_range will (if needed) be given to `func` as an argument.
            supports_multiple_limits (bool): If True, multiple limits may be given as an argument to `func`.
        """
        if limits is False:
            raise ValueError("Limits for the analytical integral have to be specified or None (for any limits).")
        if limits is None:
            limits = tuple((None, None) for _ in range(len(dims)))
            limits = convert_to_range(limits=limits, dims=dims, convert_none=True)
        else:
            limits = convert_to_range(limits=limits, dims=dims, convert_none=True)
            # limits = limits.get_limits()
        dims = frozenset(limits.axes)

        # add catching everything unsupported:
        func = supports(norm_range=supports_norm_range, multiple_limits=supports_multiple_limits)(func)

        self._integrals[dims][limits.limits()] = Integral(func=func, limits=limits, dims=dims,
                                                          priority=priority)  # TODO improve with
        # database-like access

    def integrate(self, x: Optional[ztyping.XType], limits: ztyping.LimitsType, dims: ztyping.DimsType = None,
                  norm_range: ztyping.LimitsType = None, params: dict = None) -> ztyping.XType:
        """Integrate analytically over the axes if available.


        Args:
            x (numeric): If a partial integration is made, x are the values to be evaluated for the partial
                integrated function. If a full integration is performed, this should be `None`.
            limits (zfit.Range): The limits to integrate
            dims (Tuple[int]): The dimensions to integrate over
            norm_range (bool): |norm_range_arg_descr|
            params (dict): The parameters of the function


        Returns:
            Union[tf.Tensor, float]:

        Raises:
            NotImplementedError: If the requested integral is not available.
        """
        if dims is None:
            dims = limits.dims
        dims = frozenset(dims)
        integral_holder = self._integrals.get(dims)
        if integral_holder is None:
            raise NotImplementedError("Integral is not available for axes {}".format(dims))
        integral_fn = self.get_max_integral(limits=limits, dims=dims)
        if integral_fn is None:
            raise NotImplementedError(
                "Integral is available for axes {}, but not for limits {}".format(dims, limits))

        if x is None:
            integral = integral_fn(limits=limits, norm_range=norm_range, params=params)
        else:
            integral = integral_fn(x=x, limits=limits, norm_range=norm_range, params=params)
        return integral


class Integral(object):  # TODO analytic integral
    def __init__(self, func: Callable, limits: ztyping.LimitsType, dims: ztyping.DimsType, priority: int):
        """A lightweight holder for the integral function."""
        self.limits = convert_to_range(limits=limits, dims=dims)
        self.integrate = func
        self.dims = limits.dims
        self.priority = priority

    def __call__(self, *args, **kwargs):
        return self.integrate(*args, **kwargs)


if __name__ == '__main__':
    def my_fn1(x):
        if isinstance(x, tf.Tensor):
            x = tf.unstack(x)
        w, x, y, z, l = x
        # return x ** 2 + 0.1 * y ** 2 + 0.01 * z ** 2 + 0.001 * w ** 2 + 0.0001 * l ** 2
        return w + x


    import tensorflow_probability as tfp

    res = mc_integrate(func=my_fn1, limits=(0., 5.), axes=(1, 3), draws_per_dim=1000,
                       x=tf.constant([[i, i, i] for i in range(1, 6)]),
                       mc_sampler=tfp.mcmc.sample_halton_sequence)

    with tf.Session() as sess:
        res = sess.run(res)
        print(res)
