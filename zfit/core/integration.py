"""This module contains functions for the numeric as well as the analytic (partial) integration."""

#  Copyright (c) 2021 zfit

import collections
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit.core.dimension import BaseDimensional
from zfit.core.interfaces import ZfitData, ZfitModel, ZfitSpace
from zfit.util.container import convert_to_container
from zfit.util.temporary import TemporarilySet

from ..settings import ztypes
from ..util import ztyping
from ..util.exception import (AnalyticIntegralNotImplemented,
                              WorkInProgressError)
from .space import Space, convert_to_space, supports


@supports()
def auto_integrate(func, limits, n_axes=None, x=None, method="AUTO", dtype=ztypes.float,
                   mc_sampler=tfp.mcmc.sample_halton_sequence,
                   mc_options=None):
    if method == "AUTO":  # TODO unfinished, other methods?
        method = "mc"
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options['draws_per_dim']
        integral = mc_integrate(x=x, func=func, limits=limits, n_axes=n_axes, method=method, dtype=dtype,
                                mc_sampler=mc_sampler, draws_per_dim=draws_per_dim,
                                importance_sampling=None)
    return integral


# TODO implement numerical integration method
def numeric_integrate():
    """Integrate `func` using numerical methods."""
    integral = None
    return integral


# @z.function
def mc_integrate(func: Callable, limits: ztyping.LimitsType, axes: Optional[ztyping.AxesTypeInput] = None,
                 x: Optional[ztyping.XType] = None, n_axes: Optional[int] = None, draws_per_dim: int = 20000,
                 method: str = None,
                 dtype: Type = ztypes.float,
                 mc_sampler: Callable = tfp.mcmc.sample_halton_sequence,
                 importance_sampling: Optional[Callable] = None) -> tf.Tensor:
    """Monte Carlo integration of `func` over `limits`.

    Args:
        func: The function to be integrated over
        limits: The limits of the integral
        axes: The row to integrate over. None means integration over all value
        x: If a partial integration is performed, this are the value where x will be evaluated.
        n_axes: the number of total dimensions (old?)
        draws_per_dim: How many random points to draw per dimensions
        method: Which integration method to use
        dtype: |dtype_arg_descr|
        mc_sampler: A function that takes one argument (`n_draws` or similar) and returns
            random value between 0 and 1.
        importance_sampling:

    Returns:
        The integral
    """
    if axes is not None and n_axes is not None:
        raise ValueError("Either specify axes or n_axes")
    limits = convert_to_space(limits)

    axes = limits.axes
    partial = (axes is not None) and (x is not None)  # axes, value can be tensors

    if axes is not None and n_axes is None:
        n_axes = len(axes)
    if n_axes is not None and axes is None:
        axes = tuple(range(n_axes))

    integrals = []
    for space in limits:
        lower, upper = space._rect_limits_tf
        tf.debugging.assert_all_finite((lower, upper),
                                       "MC integration does (currently) not support unbound limits (np.infty) as given here:"
                                       "\nlower: {}, upper: {}".format(lower, upper)
                                       )

        n_samples = draws_per_dim

        chunked_normalization = zfit.run.chunksize < n_samples
        # chunked_normalization = True
        if chunked_normalization and partial:
            print("NOT SUPPORTED! partial and chunked not working, auto switch back to not-chunked.")
        if chunked_normalization and not partial:
            n_chunks = int(np.ceil(n_samples / zfit.run.chunksize))
            chunksize = int(np.ceil(n_samples / n_chunks))
            # print("starting normalization with {} chunks and a chunksize of {}".format(n_chunks, chunksize))
            avg = normalization_chunked(func=func, n_axes=n_axes, dtype=dtype, x=x,
                                        num_batches=n_chunks, batch_size=chunksize, space=space)

        else:
            # TODO: deal with n_obs properly?

            samples_normed = mc_sampler(dim=n_axes, num_results=n_samples / 2,  # to decrease integration size
                                        dtype=dtype)
            samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it

            if partial:  # TODO(Mayou36): shape of partial integral?
                data_obs = x.obs
                new_obs = []
                x = x.value()
                value_list = []
                index_samples = 0
                index_values = 0
                if len(x.shape) == 1:
                    x = znp.expand_dims(x, axis=1)
                for i in range(n_axes + x.shape[-1]):
                    if i in axes:
                        new_obs.append(space.obs[index_samples])
                        value_list.append(samples[:, index_samples])
                        index_samples += 1
                    else:
                        new_obs.append(data_obs[index_values])
                        value_list.append(znp.expand_dims(x[:, index_values], axis=1))
                        index_values += 1
                value_list = [tf.cast(val, dtype=dtype) for val in value_list]
                x = PartialIntegralSampleData(sample=value_list,
                                              space=Space(obs=new_obs))
            else:
                x = samples

            # convert rnd samples with value to feedable vector
            reduce_axis = 1 if partial else None
            avg = znp.mean(func(x), axis=reduce_axis)
            # avg = tfp.monte_carlo.expectation(f=func, samples=x, axis=reduce_axis)
            # TODO: importance sampling?
            # avg = tfb.monte_carlo.expectation_importance_sampler(f=func, samples=value,axis=reduce_axis)
        integral = avg * tf.cast(z.convert_to_tensor(space.rect_area()), dtype=avg.dtype)
        integrals.append(integral)
    return z.reduce_sum(integrals, axis=0)
    # return z.to_real(integral, dtype=dtype)


# TODO(Mayou36): Make more flexible for sampling
# @z.function
def normalization_nograd(func, n_axes, batch_size, num_batches, dtype, space, x=None, shape_after=()):
    upper, lower = space.rect_limits
    lower = z.convert_to_tensor(lower, dtype=dtype)
    upper = z.convert_to_tensor(upper, dtype=dtype)

    def body(batch_num, mean):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        indices = tf.range(start_idx, end_idx, dtype=tf.int32)
        samples_normed = tfp.mcmc.sample_halton_sequence(n_axes,
                                                         # num_results=batch_size,
                                                         sequence_indices=indices,
                                                         dtype=dtype,
                                                         randomized=False)
        # halton_sample = tf.random_uniform(shape=(n_axes, batch_size), dtype=dtype)
        samples_normed.set_shape((batch_size, n_axes))
        samples_normed = znp.expand_dims(samples_normed, axis=0)
        samples = samples_normed * (upper - lower) + lower
        func_vals = func(samples)
        if shape_after == ():
            reduce_axis = None
        else:
            reduce_axis = 1
            if len(func_vals.shape) == 1:
                func_vals = znp.expand_dims(func_vals, -1)
        batch_mean = znp.mean(func_vals, axis=reduce_axis)  # if there are gradients
        err_weight = 1 / tf.cast(batch_num + 1, dtype=tf.float64)

        do_print = False
        if do_print:
            tf.print(batch_num + 1)
        return batch_num + 1, mean + err_weight * (batch_mean - mean)

    cond = lambda batch_num, _: batch_num < num_batches

    initial_mean = tf.constant(0, shape=shape_after, dtype=dtype)
    initial_body_args = (0, initial_mean)
    _, final_mean = tf.while_loop(cond=cond, body=body, loop_vars=initial_body_args, parallel_iterations=1,
                                  swap_memory=False, back_prop=True)
    # def normalization_grad(x):
    return final_mean


# @z.function
def normalization_chunked(func, n_axes, batch_size, num_batches, dtype, space, x=None, shape_after=()):
    x_is_none = x is None

    @tf.custom_gradient
    def normalization_func(x):
        if x_is_none:
            x = None
        value = normalization_nograd(func=func, n_axes=n_axes, batch_size=batch_size, num_batches=num_batches,
                                     dtype=dtype,
                                     space=space, x=x, shape_after=shape_after)

        def grad_fn(dy, variables=None):
            if variables is None:
                return dy, None
            with tf.GradientTape() as tape:
                value = normalization_nograd(func=func, n_axes=n_axes,
                                             batch_size=batch_size, num_batches=num_batches,
                                             dtype=dtype,
                                             space=space, x=x, shape_after=shape_after)


            return dy, tape.gradient(value, variables)

        return value, grad_fn

    fake_x = 1 if x_is_none else x
    return normalization_func(fake_x)


# @z.function
def chunked_average(func, x, num_batches, batch_size, space, mc_sampler):
    lower, upper = space.limits

    fake_resource_var = tf.Variable("fake_hack_ResVar_for_custom_gradient",
                                    initializer=z.constant(4242.))
    fake_x = z.constant(42.) * fake_resource_var

    @tf.custom_gradient
    def dummy_func(fake_x):  # to make working with custom_gradient
        if x is not None:
            raise WorkInProgressError("partial not yet implemented")

        def body(batch_num, mean):
            if mc_sampler == tfp.mcmc.sample_halton_sequence:
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                indices = tf.range(start_idx, end_idx, dtype=tf.int32)
                sample = mc_sampler(space.n_obs, sequence_indices=indices,
                                    dtype=ztypes.float, randomized=False)
            else:
                sample = mc_sampler(shape=(batch_size, space.n_obs), dtype=ztypes.float)
            sample = tf.guarantee_const(sample)
            sample = (np.array(upper[0]) - np.array(lower[0])) * sample + lower[0]
            sample = znp.transpose(a=sample)
            sample = func(sample)
            sample = tf.guarantee_const(sample)

            batch_mean = znp.mean(sample)
            batch_mean = tf.guarantee_const(batch_mean)
            err_weight = 1 / tf.cast(batch_num + 1, dtype=tf.float64)
            # err_weight /= err_weight + 1
            # print_op = tf.print(batch_mean)
            do_print = False
            if do_print:
                tf.print(batch_num + 1, mean, err_weight * (batch_mean - mean))

            return batch_num + 1, mean + err_weight * (batch_mean - mean)

        cond = lambda batch_num, _: batch_num < num_batches

        initial_mean = tf.convert_to_tensor(value=0, dtype=ztypes.float)
        _, final_mean = tf.while_loop(cond=cond, body=body, loop_vars=(0, initial_mean), parallel_iterations=1,
                                      swap_memory=False, back_prop=False, maximum_iterations=num_batches)

        def dummy_grad_with_var(dy, variables=None):
            raise WorkInProgressError("Who called me? Mayou36")
            if variables is None:
                raise WorkInProgressError("Is this needed? Why? It's not a NN. Please make an issue.")

            def dummy_grad_func(x):
                values = func(x)
                if variables:
                    gradients = tf.gradients(ys=values, xs=variables, grad_ys=dy)
                else:
                    gradients = None
                return gradients

            return chunked_average(func=dummy_grad_func, x=x, num_batches=num_batches, batch_size=batch_size,
                                   space=space, mc_sampler=mc_sampler)

        def dummy_grad_without_var(dy):
            return dummy_grad_with_var(dy=dy, variables=None)

        do_print = False
        if do_print:
            tf.print("Total mean calculated = ", final_mean)

        return final_mean, dummy_grad_with_var

    try:
        return dummy_func(fake_x)
    except TypeError:
        return dummy_func(fake_x)


class PartialIntegralSampleData(BaseDimensional, ZfitData):

    def __init__(self, sample: List[tf.Tensor], space: ZfitSpace):
        """Takes a list of tensors and "fakes" a dataset. Useful for tensors with non-matching shapes.

        Args:
            sample:
            space:
        """
        if not isinstance(sample, list):
            raise TypeError("Sample has to be a list of tf.Tensors")
        super().__init__()
        self._space = space
        self._sample = sample
        self._reorder_indices_list = list(range(len(sample)))

    @property
    def weights(self):
        raise NotImplementedError("Weights for PartialIntegralsampleData are not implemented. Are they needed?")

    @property
    def space(self) -> "zfit.Space":
        return self._space

    def sort_by_axes(self, axes, allow_superset: bool = True):
        axes = convert_to_container(axes)
        new_reorder_list = [self._reorder_indices_list[self.space.axes.index(ax)] for ax in axes]
        value = self.space.with_axes(axes=axes), new_reorder_list

        getter = lambda: (self.space, self._reorder_indices_list)

        def setter(value):
            self._space, self._reorder_indices_list = value

        return TemporarilySet(value=value, getter=getter, setter=setter)

    def sort_by_obs(self, obs, allow_superset: bool = True):
        obs = convert_to_container(obs)
        new_reorder_list = [self._reorder_indices_list[self.space.obs.index(ob)] for ob in obs]

        value = self.space.with_obs(obs=obs), new_reorder_list

        getter = lambda: (self.space, self._reorder_indices_list)

        def setter(value):
            self._space, self._reorder_indices_list = value

        return TemporarilySet(value=value, getter=getter, setter=setter)

    def value(self, obs: List[str] = None):
        return self

    def unstack_x(self, always_list=False):
        unstacked_x = [self._sample[i] for i in self._reorder_indices_list]
        if len(unstacked_x) == 1 and not always_list:
            unstacked_x = unstacked_x[0]
        return unstacked_x

    def __hash__(self) -> int:
        return id(self)


class AnalyticIntegral:
    def __init__(self, *args, **kwargs):
        """Hold analytic integrals and manage their dimensions, limits etc."""
        super().__init__(*args, **kwargs)
        self._integrals = collections.defaultdict(dict)

    def get_max_axes(self, limits: ztyping.LimitsType, axes: ztyping.AxesTypeInput = None) -> Tuple[int]:
        """Return the maximal available axes to integrate over analytically for given limits.

        Args:
            limits: The integral function will be able to integrate over this limits
            axes: The axes over which (or over a subset) it will integrate

        Returns:
            Tuple[int]:
        """
        if not isinstance(limits, ZfitSpace):
            raise TypeError("`limits` have to be a `ZfitSpace`")
        # limits = convert_to_space(limits=limits)

        return self._get_max_axes_limits(limits, out_of_axes=limits.axes)[0]  # only axes

    def _get_max_axes_limits(self, limits, out_of_axes):  # TODO: automatic caching? but most probably not relevant
        if out_of_axes:
            out_of_axes = frozenset(out_of_axes)
            implemented_axes = frozenset(d for d in self._integrals.keys() if d <= out_of_axes)
        else:
            implemented_axes = set(self._integrals.keys())
        implemented_axes = sorted(implemented_axes, key=len, reverse=True)  # iter through biggest first
        for axes in implemented_axes:
            limits_matched = []
            for lim, integ in self._integrals[axes].items():
                if integ.limits >= limits:
                    limits_matched.append(lim)

            if limits_matched:  # one or more integrals available
                return tuple(sorted(axes)), limits_matched
        return (), ()  # no integral available for this axes

    def get_max_integral(self, limits: ztyping.LimitsType,
                         axes: ztyping.AxesTypeInput = None) -> Union[None, "Integral"]:
        """Return the integral over the `limits` with `axes` (or a subset of them).

        Args:
            limits:
            axes:

        Returns:
            Return a callable that integrated over the given limits.
        """
        limits = convert_to_space(limits=limits, axes=axes)

        axes, limits = self._get_max_axes_limits(limits=limits, out_of_axes=axes)
        axes = frozenset(axes)
        integrals = [self._integrals[axes][lim] for lim in limits]
        integral_fn = max(integrals, key=lambda l: l.priority, default=None)
        return integral_fn

    def register(self, func: Callable, limits: ztyping.LimitsType,
                 priority: int = 50, *,
                 supports_norm_range: bool = False, supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral.

        Args:
            func: The integral function. Takes 1 argument.
            axes: |dims_arg_descr|
            limits: |limits_arg_descr| `Limits` can be None if `func` works for any
            possible limits
            priority: If two or more integrals can integrate over certain limits, the one with the higher
                priority is taken (usually around 0-100).
            supports_norm_range: If True, norm_range will (if needed) be given to `func` as an argument.
            supports_multiple_limits: If True, multiple limits may be given as an argument to `func`.
        """

        # if limits is False:
        #     raise ValueError("Limits for the analytical integral have to be specified or None (for any limits).")
        # if limits is None:
        #     limits = tuple((Space.ANY_LOWER, Space.ANY_UPPER) for _ in range(len(axes)))
        #     limits = convert_to_space(axes=axes, limits=limits)
        # else:
        #     limits = convert_to_space(axes=self.axes, limits=limits)
        # limits = limits.get_limits()
        if not isinstance(limits, ZfitSpace):
            raise TypeError("Limits for registering an integral have to be `ZfitSpace`")
        axes = frozenset(limits.axes)

        # add catching everything unsupported:
        func = supports(norm_range=supports_norm_range, multiple_limits=supports_multiple_limits)(func)
        limits = limits.with_axes(axes=tuple(sorted(limits.axes)))
        self._integrals[axes][limits] = Integral(func=func, limits=limits,
                                                 priority=priority)  # TODO improve with
        # database-like access

    def integrate(self, x: Optional[ztyping.XType], limits: ztyping.LimitsType, axes: ztyping.AxesTypeInput = None,
                  norm_range: ztyping.LimitsType = None, model: ZfitModel = None, params: dict = None) -> ztyping.XType:
        """Integrate analytically over the axes if available.

        Args:
            x: If a partial integration is made, x are the value to be evaluated for the partial
                integrated function. If a full integration is performed, this should be `None`.
            limits: The limits to integrate
            axes: The dimensions to integrate over
            norm_range: |norm_range_arg_descr|
            params: The parameters of the function


        Returns:
            Union[tf.Tensor, float]:

        Raises:
            AnalyticIntegralNotImplementedError: If the requested integral is not available.
        """
        if axes is None:
            axes = limits.axes
        axes = frozenset(axes)
        integral_holder = self._integrals.get(axes)
        # limits = convert_to_space(axes=self.axes, limits=limits)
        if integral_holder is None:
            raise AnalyticIntegralNotImplemented(f"Analytic integral is not available for axes {axes}")
        integral_fn = self.get_max_integral(limits=limits)
        if integral_fn is None:
            raise AnalyticIntegralNotImplemented(
                f"Integral is available for axes {axes}, but not for limits {limits}")

        try:
            integral = integral_fn(x=x, limits=limits, norm_range=norm_range, params=params, model=model)
        except TypeError:
            integral = integral_fn(limits=limits, norm_range=norm_range, params=params, model=model)
        return integral


class Integral:  # TODO analytic integral
    def __init__(self, func: Callable, limits: "ZfitSpace", priority: Union[int, float]):
        """A lightweight holder for the integral function."""
        self.limits = limits
        self.integrate = func
        self.axes = limits.axes
        self.priority = priority

    def __call__(self, *args, **kwargs):
        return self.integrate(*args, **kwargs)


# to be "the" future integral class
class Integration:

    def __init__(self, mc_sampler, draws_per_dim):
        self.mc_sampler = mc_sampler
        self.draws_per_dim = draws_per_dim
