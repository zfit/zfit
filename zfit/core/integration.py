"""This module contains functions for the numeric as well as the analytic (partial) integration."""
#  Copyright (c) 2023 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

from collections.abc import Callable

import collections
from contextlib import suppress
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from zfit import z
from .dimension import BaseDimensional
from .interfaces import ZfitData, ZfitModel, ZfitSpace
from ..util.container import convert_to_container
from ..util.temporary import TemporarilySet

from ..settings import ztypes
from ..util import ztyping
from ..util.exception import AnalyticIntegralNotImplemented, WorkInProgressError
from .space import Space, convert_to_space, supports


def auto_integrate(
    func,
    limits,
    n_axes=None,
    x=None,
    method="AUTO",
    dtype=ztypes.float,
    mc_sampler=tfp.mcmc.sample_halton_sequence,
    max_draws=None,
    tol=None,
    vectorizable=None,
    mc_options=None,
    simpsons_options=None,
):
    if vectorizable is None:
        vectorizable = False
    limits = convert_to_space(limits)

    if n_axes is None:
        n_axes = limits.n_obs
    if method == "AUTO":  # TODO unfinished, other methods?
        if n_axes == 1 and x is None:
            method = "simpson"
        else:
            method = "mc"
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options["draws_per_dim"]
        max_draws = mc_options.get("max_draws")
        integral = mc_integrate(
            x=x,
            func=func,
            limits=limits,
            n_axes=n_axes,
            method=method,
            dtype=dtype,
            mc_sampler=mc_sampler,
            draws_per_dim=draws_per_dim,
            max_draws=max_draws,
            tol=tol,
            importance_sampling=None,
            vectorizable=vectorizable,
        )
    elif method.lower() == "simpson":
        num_points = simpsons_options["draws_simpson"]
        integral = simpson_integrate(func=func, limits=limits, num_points=num_points)
    else:
        raise ValueError(f"Method {method} not a legal choice for integration method.")
    return integral


# TODO implement numerical integration method
def numeric_integrate():
    """Integrate ``func`` using numerical methods."""
    return None


# COPIED FROM tf_quant_finance/math/integrate.py which is deprecated
def simpson(func, lower, upper, num_points=1001, dtype=None, name=None):
    """Evaluates definite integral using composite Simpson's 1/3 rule.

    Integrates `func` using composite Simpson's 1/3 rule [1].

    Evaluates function at points of evenly spaced grid of `num_points` points,
    then uses obtained values to interpolate `func` with quadratic polynomials
    and integrates these polynomials.

    #### References
    [1] Weisstein, Eric W. "Simpson's Rule." From MathWorld - A Wolfram Web
        Resource. http://mathworld.wolfram.com/SimpsonsRule.html

    #### Example
    ```python
      f = lambda x: x*x
      a = tf.constant(0.0)
      b = tf.constant(3.0)
      integrate(f, a, b, num_points=1001) # 9.0
    ```

    Args:
      func: Python callable representing a function to be integrated. It must be a
        callable of a single `Tensor` parameter and return a `Tensor` of the same
        shape and dtype as its input. It will be called with a `Tesnor` of shape
        `lower.shape + [n]` (where n is integer number of points) and of the same
        `dtype` as `lower`.
      lower: `Tensor` or Python float representing the lower limits of
        integration. `func` will be integrated between each pair of points defined
        by `lower` and `upper`.
      upper: `Tensor` of the same shape and dtype as `lower` or Python float
        representing the upper limits of intergation.
      num_points: Scalar int32 `Tensor`. Number of points at which function `func`
        will be evaluated. Must be odd and at least 3.
        Default value: 1001.
      dtype: Optional `tf.Dtype`. If supplied, the dtype for the `lower` and
        `upper`. Result will have the same dtype.
        Default value: None which maps to dtype of `lower`.
      name: Python str. The name to give to the ops created by this function.
        Default value: None which maps to 'integrate_simpson_composite'.

    Returns:
      `Tensor` of shape `func_batch_shape + limits_batch_shape`, containing
        value of the definite integral.
    """
    with tf.compat.v1.name_scope(
        name, default_name="integrate_simpson_composite", values=[lower, upper]
    ):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name="lower")
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name="upper")
        num_points = tf.convert_to_tensor(num_points, dtype=tf.int32, name="num_points")

        assertions = [
            tf.debugging.assert_greater_equal(num_points, 3),
            tf.debugging.assert_equal(num_points % 2, 1),
        ]

        with tf.compat.v1.control_dependencies(assertions):
            dx = (upper - lower) / (tf.cast(num_points, dtype=dtype) - 1)
            dx_expand = tf.expand_dims(dx, -1)
            lower_exp = tf.expand_dims(lower, -1)
            grid = lower_exp + dx_expand * tf.cast(tf.range(num_points), dtype=dtype)
            weights_first = tf.constant([1.0], dtype=dtype)
            weights_mid = tf.tile(
                tf.constant([4.0, 2.0], dtype=dtype), [(num_points - 3) // 2]
            )
            weights_last = tf.constant([4.0, 1.0], dtype=dtype)
            weights = tf.concat([weights_first, weights_mid, weights_last], axis=0)

        return tf.reduce_sum(func(grid) * weights, axis=-1) * dx / 3.0


def simpson_integrate(func, limits, num_points):  # currently not vectorized
    integrals = []
    num_points = tf.cast(num_points, znp.int32)
    num_points += num_points % 2 + 1  # sanitize number of points
    for space in limits:
        lower, upper = space.rect_limits
        if lower.shape[0] > 1:
            raise ValueError(
                "Vectorized spaces in integration currently not supported."
            )
        lower = znp.array(lower)[0, 0]
        upper = znp.array(upper)[0, 0]
        tf.debugging.assert_all_finite(
            (lower, upper),
            "MC integration does (currently) not support unbound limits (np.infty) as given here:"
            "\nlower: {}, upper: {}".format(lower, upper),
        )
        integrals.append(
            simpson(
                func=func,
                lower=lower,
                upper=upper,
                num_points=num_points,
                dtype=znp.float64,
            )
        )
    return znp.sum(integrals, axis=0)


# @z.function
def mc_integrate(
    func: Callable,
    limits: ztyping.LimitsType,
    axes: ztyping.AxesTypeInput | None = None,
    x: ztyping.XType | None = None,
    n_axes: int | None = None,
    draws_per_dim: int = 40000,
    max_draws=800_000,
    tol: float = 1e-6,
    method: str = None,
    dtype: type = ztypes.float,
    mc_sampler: Callable = tfp.mcmc.sample_halton_sequence,
    importance_sampling: Callable | None = None,
    vectorizable=None,
) -> tf.Tensor:
    """Monte Carlo integration of ``func`` over ``limits``.

    Args:
        vectorizable ():
        func: The function to be integrated over
        limits: The limits of the integral
        axes: The row to integrate over. None means integration over all value
        x: If a partial integration is performed, this are the value where x will be evaluated.
        n_axes: the number of total dimensions (old?)
        draws_per_dim: How many random points to draw per dimensions
        method: Which integration method to use
        dtype: |dtype_arg_descr|
        mc_sampler: A function that takes one argument (``n_draws`` or similar) and returns
            random value between 0 and 1.
        importance_sampling:

    Returns:
        The integral
    """
    import zfit

    if vectorizable is None:
        vectorizable = False
    tol = znp.array(tol, dtype=znp.float64)
    if axes is not None and n_axes is not None:
        raise ValueError("Either specify axes or n_axes")

    axes = limits.axes
    partial = (axes is not None) and (x is not None)  # axes, value can be tensors

    if axes is not None and n_axes is None:
        n_axes = len(axes)
    if n_axes is not None and axes is None:
        axes = tuple(range(n_axes))

    integrals = []
    for space in limits:
        lower, upper = space._rect_limits_tf
        tf.debugging.assert_all_finite(
            (lower, upper),
            "MC integration does (currently) not support unbound limits (np.infty) as given here:"
            "\nlower: {}, upper: {}".format(lower, upper),
        )

        n_samples = draws_per_dim * n_axes

        chunked_normalization = zfit.run.chunksize < n_samples
        # chunked_normalization = True
        if chunked_normalization and partial:
            print(
                "NOT SUPPORTED! partial and chunked not working, auto switch back to not-chunked."
            )
        if chunked_normalization and not partial:
            n_chunks = int(np.ceil(n_samples / zfit.run.chunksize))
            chunksize = int(np.ceil(n_samples / n_chunks))
            # print("starting normalization with {} chunks and a chunksize of {}".format(n_chunks, chunksize))
            avg = normalization_chunked(
                func=func,
                n_axes=n_axes,
                dtype=dtype,
                x=x,
                num_batches=n_chunks,
                batch_size=chunksize,
                space=space,
            )

        else:
            # TODO: deal with n_obs properly?
            @z.function(wraps="tensor")
            def cond(avg, error, std, ntot, i):
                # return i < 3
                return znp.logical_and(error > tol, ntot < max_draws)

            def body_integrate(avg, error, std, ntot, i):
                ntot_old = ntot
                ntot += n_samples
                if partial:
                    samples_normed = tfp.mcmc.sample_halton_sequence(
                        dim=n_axes,
                        # sequence_indices=tf.range(ntot_old, ntot),
                        num_results=n_samples / 10,
                        # reduce, it explodes otherwise easily
                        # as we don't do adaptive now
                        # to decrease integration size
                        dtype=dtype,
                        randomized=False,
                    )
                else:
                    samples_normed = tfp.mcmc.sample_halton_sequence(
                        dim=n_axes,
                        sequence_indices=tf.range(ntot_old, ntot),
                        # num_results=n_samples,  # to decrease integration size
                        dtype=dtype,
                        randomized=False,
                    )
                samples = (
                    samples_normed * (upper - lower) + lower
                )  # samples is [0, 1], stretch it
                if partial:  # TODO(Mayou36): shape of partial integral?
                    data_obs = x.obs
                    new_obs = []
                    xval = x.value()
                    value_list = []
                    index_samples = 0
                    index_values = 0
                    if len(xval.shape) == 1:
                        xval = znp.expand_dims(xval, axis=1)
                    for i in range(n_axes + xval.shape[-1]):
                        if i in axes:
                            new_obs.append(space.obs[index_samples])
                            value_list.append(samples[:, index_samples])
                            index_samples += 1
                        else:
                            new_obs.append(data_obs[index_values])
                            value_list.append(
                                znp.expand_dims(xval[:, index_values], axis=1)
                            )
                            index_values += 1
                    value_list = [tf.cast(val, dtype=dtype) for val in value_list]
                    xval = PartialIntegralSampleData(
                        sample=value_list, space=Space(obs=new_obs)
                    )
                else:
                    xval = samples
                # convert rnd samples with value to feedable vector
                reduce_axis = 1 if partial else None
                y = func(xval)
                # avg = znp.mean(y)
                ifloat = tf.cast(i, dtype=tf.float64)
                if partial:
                    avg = znp.mean(y, axis=reduce_axis)
                else:
                    avg = avg / (ifloat + 1.0) * ifloat + znp.mean(
                        y, axis=reduce_axis
                    ) / (ifloat + 1.0)
                std = std / (ifloat + 1.0) * ifloat + znp.std(y) / (ifloat + 1.0)
                ntot_float = znp.asarray(ntot, dtype=znp.float64)

                # estimating the error of QMC is non-trivial
                # (https://www.degruyter.com/document/doi/10.1515/mcma-2020-2067/html or
                # https://stats.stackexchange.com/questions/533725/how-to-calculate-quasi-monte-carlo-integration-error-when-sampling-with-sobols)
                # However, we use here just something that is in the right direction for the moment being. Therefore,
                # we use the MC error as an upper bound as QMC is better/equal to MC (for our cases).
                error_sobol = std * znp.log(ntot_float) ** n_axes / ntot_float
                error_random = std / znp.sqrt(ntot_float)
                error = (
                    znp.minimum(error_sobol, error_random) * 0.1
                )  # heuristic factor from using QMC
                return avg, error, std, ntot, i + 1

            avg, error, std, ntot, i = [
                znp.array(0.0, dtype=znp.float64),
                znp.array(9999.0, dtype=znp.float64),  # init value large, irrelevant
                znp.array(0.0, dtype=znp.float64),
                0,
                0,
            ]
            if partial or vectorizable:
                avg, error, std, ntot, i = body_integrate(avg, error, std, ntot, i)
            else:
                avg, error, std, ntot, i = tf.while_loop(
                    cond=cond, body=body_integrate, loop_vars=[avg, error, std, ntot, i]
                )
                from zfit import settings

                if settings.get_verbosity() > 9:
                    tf.print("i:", i, "   ntot:", ntot)

            # avg = tfp.monte_carlo.expectation(f=func, samples=x, axis=reduce_axis)
            # TODO: importance sampling?
            # avg = tfb.monte_carlo.expectation_importance_sampler(f=func, samples=value,axis=reduce_axis)

            def print_none_return():
                from zfit import settings

                if settings.get_verbosity() >= 0:
                    tf.print(
                        "Estimated integral error (",
                        error,
                        ") larger than tolerance (",
                        tol,
                        "), which is maybe not enough (but maybe it's also fine)."
                        " You can (best solution) implement an anatytical integral (see examples in repo) or"
                        " manually set a higher number on the PDF with 'update_integration_options'"
                        " and increase the 'max_draws' (or adjust 'tol'). "
                        "If partial integration is chosen, this can lead to large memory consumption."
                        "This is a new warning checking the integral accuracy. It may warns too often as it is"
                        " Work In Progress. If you have any observation on it, please tell us about it:"
                        " https://github.com/zfit/zfit/issues/new/choose"
                        "To suppress this warning, use zfit.settings.set_verbosity(-1).",
                    )
                return

            if not vectorizable:
                tf.cond(error > tol, print_none_return, lambda: None)
        integral = avg * tf.cast(
            z.convert_to_tensor(space.rect_area()), dtype=avg.dtype
        )
        integrals.append(integral)
    return z.reduce_sum(integrals, axis=0)
    # return z.to_real(integral, dtype=dtype)


# TODO(Mayou36): Make more flexible for sampling
# @z.function
def normalization_nograd(
    func, n_axes, batch_size, num_batches, dtype, space, x=None, shape_after=()
):
    upper, lower = space.rect_limits
    lower = z.convert_to_tensor(lower, dtype=dtype)
    upper = z.convert_to_tensor(upper, dtype=dtype)

    def body(batch_num, mean):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        indices = tf.range(start_idx, end_idx, dtype=tf.int32)
        samples_normed = tfp.mcmc.sample_halton_sequence(
            n_axes,
            # num_results=batch_size,
            sequence_indices=indices,
            dtype=dtype,
            randomized=False,
        )
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
    _, final_mean = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=initial_body_args,
        parallel_iterations=1,
        swap_memory=False,
        back_prop=True,
    )
    # def normalization_grad(x):
    return final_mean


# @z.function
def normalization_chunked(
    func, n_axes, batch_size, num_batches, dtype, space, x=None, shape_after=()
):
    x_is_none = x is None

    @tf.custom_gradient
    def normalization_func(x):
        if x_is_none:
            x = None
        value = normalization_nograd(
            func=func,
            n_axes=n_axes,
            batch_size=batch_size,
            num_batches=num_batches,
            dtype=dtype,
            space=space,
            x=x,
            shape_after=shape_after,
        )

        def grad_fn(dy, variables=None):
            if variables is None:
                return dy, None
            with tf.GradientTape() as tape:
                value = normalization_nograd(
                    func=func,
                    n_axes=n_axes,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    dtype=dtype,
                    space=space,
                    x=x,
                    shape_after=shape_after,
                )

            return dy, tape.gradient(value, variables)

        return value, grad_fn

    fake_x = 1 if x_is_none else x
    return normalization_func(fake_x)


# @z.function
def chunked_average(func, x, num_batches, batch_size, space, mc_sampler):
    lower, upper = space.limits

    fake_resource_var = tf.Variable(
        "fake_hack_ResVar_for_custom_gradient", initializer=z.constant(4242.0)
    )
    fake_x = z.constant(42.0) * fake_resource_var

    @tf.custom_gradient
    def dummy_func(fake_x):  # to make working with custom_gradient
        if x is not None:
            raise WorkInProgressError("partial not yet implemented")

        def body(batch_num, mean):
            if mc_sampler == tfp.mcmc.sample_halton_sequence:
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                indices = tf.range(start_idx, end_idx, dtype=tf.int32)
                sample = mc_sampler(
                    space.n_obs,
                    sequence_indices=indices,
                    dtype=ztypes.float,
                    randomized=False,
                )
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
        _, final_mean = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(0, initial_mean),
            parallel_iterations=1,
            swap_memory=False,
            back_prop=False,
            maximum_iterations=num_batches,
        )

        def dummy_grad_with_var(dy, variables=None):
            raise WorkInProgressError("Who called me? Mayou36")
            if variables is None:
                raise WorkInProgressError(
                    "Is this needed? Why? It's not a NN. Please make an issue."
                )

            def dummy_grad_func(x):
                values = func(x)
                if variables:
                    gradients = tf.gradients(ys=values, xs=variables, grad_ys=dy)
                else:
                    gradients = None
                return gradients

            return chunked_average(
                func=dummy_grad_func,
                x=x,
                num_batches=num_batches,
                batch_size=batch_size,
                space=space,
                mc_sampler=mc_sampler,
            )

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
    def __init__(self, sample: list[tf.Tensor], space: ZfitSpace):
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
        raise NotImplementedError(
            "Weights for PartialIntegralsampleData are not implemented. Are they needed?"
        )

    @property
    def space(self) -> zfit.Space:
        return self._space

    def sort_by_axes(self, axes, allow_superset: bool = True):
        axes = convert_to_container(axes)
        new_reorder_list = [
            self._reorder_indices_list[self.space.axes.index(ax)] for ax in axes
        ]
        value = self.space.with_axes(axes=axes), new_reorder_list

        getter = lambda: (self.space, self._reorder_indices_list)

        def setter(value):
            self._space, self._reorder_indices_list = value

        return TemporarilySet(value=value, getter=getter, setter=setter)

    def sort_by_obs(self, obs, allow_superset: bool = True):
        obs = convert_to_container(obs)
        new_reorder_list = [
            self._reorder_indices_list[self.space.obs.index(ob)] for ob in obs
        ]

        value = self.space.with_obs(obs=obs), new_reorder_list

        getter = lambda: (self.space, self._reorder_indices_list)

        def setter(value):
            self._space, self._reorder_indices_list = value

        return TemporarilySet(value=value, getter=getter, setter=setter)

    def value(self, obs: list[str] = None):
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

    def get_max_axes(
        self, limits: ztyping.LimitsType, axes: ztyping.AxesTypeInput = None
    ) -> tuple[int]:
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

        return self._get_max_axes_limits(limits, out_of_axes=limits.axes)[
            0
        ]  # only axes

    def _get_max_axes_limits(
        self, limits, out_of_axes
    ):  # TODO: automatic caching? but most probably not relevant
        if out_of_axes:
            out_of_axes = frozenset(out_of_axes)
            implemented_axes = frozenset(
                d for d in self._integrals.keys() if d <= out_of_axes
            )
        else:
            implemented_axes = set(self._integrals.keys())
        implemented_axes = sorted(
            implemented_axes, key=len, reverse=True
        )  # iter through biggest first
        for axes in implemented_axes:
            limits_matched = [
                lim
                for lim, integ in self._integrals[axes].items()
                if integ.limits >= limits
            ]

            if limits_matched:  # one or more integrals available
                return tuple(sorted(axes)), limits_matched
        return (), ()  # no integral available for this axes

    def get_max_integral(
        self, limits: ztyping.LimitsType, axes: ztyping.AxesTypeInput = None
    ) -> None | Integral:
        """Return the integral over the ``limits`` with ``axes`` (or a subset of them).

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
        return max(integrals, key=lambda l: l.priority, default=None)

    def register(
        self,
        func: Callable,
        limits: ztyping.LimitsType,
        priority: int = 50,
        *,
        supports_norm: bool = False,
        supports_multiple_limits: bool = False,
    ) -> None:
        """Register an analytic integral.

        Args:
            func: The integral function. Takes 1 argument.
            axes: |dims_arg_descr|
            limits: |limits_arg_descr| ``Limits`` can be None if ``func`` works for any
            possible limits
            priority: If two or more integrals can integrate over certain limits, the one with the higher
                priority is taken (usually around 0-100).
            supports_norm: If True, norm_range will (if needed) be given to ``func`` as an argument.
            supports_multiple_limits: If True, multiple limits may be given as an argument to ``func``.
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
        func = supports(norm=supports_norm, multiple_limits=supports_multiple_limits)(
            func
        )
        limits = limits.with_axes(axes=tuple(sorted(limits.axes)))
        self._integrals[axes][limits] = Integral(
            func=func, limits=limits, priority=priority
        )  # TODO improve with
        # database-like access

    def integrate(
        self,
        x: ztyping.XType | None,
        limits: ztyping.LimitsType,
        axes: ztyping.AxesTypeInput = None,
        norm: ztyping.LimitsType = None,
        model: ZfitModel = None,
        params: dict = None,
    ) -> ztyping.XType:
        """Integrate analytically over the axes if available.

        Args:
            x: If a partial integration is made, x are the value to be evaluated for the partial
                integrated function. If a full integration is performed, this should be `None`.
            limits: The limits to integrate
            axes: The dimensions to integrate over
            norm: |norm_range_arg_descr|
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
            raise AnalyticIntegralNotImplemented(
                f"Analytic integral is not available for axes {axes}"
            )
        integral_fn = self.get_max_integral(limits=limits)
        if integral_fn is None:
            raise AnalyticIntegralNotImplemented(
                f"Integral is available for axes {axes}, but not for limits {limits}"
            )

        with suppress(TypeError):
            return integral_fn(
                x=x, limits=limits, norm=norm, params=params, model=model
            )
        with suppress(TypeError):
            return integral_fn(limits=limits, norm=norm, params=params, model=model)

        with suppress(TypeError):
            return integral_fn(
                x=x, limits=limits, norm_range=norm, params=params, model=model
            )
        with suppress(TypeError):
            return integral_fn(
                limits=limits, norm_range=norm, params=params, model=model
            )
        assert False, "Could not integrate, unknown reason. Please fill a bug report."

        # with suppress(TypeError):
        #     return integral_fn(x=x, limits=limits, norm=norm, params=params, model=model)
        # with suppress(TypeError):
        #     return integral_fn(limits=limits, norm=norm, params=params, model=model)


class Integral:  # TODO analytic integral
    def __init__(self, func: Callable, limits: ZfitSpace, priority: int | float):
        """A lightweight holder for the integral function."""
        self.limits = limits
        self.integrate = func
        self.axes = limits.axes
        self.priority = priority

    def __call__(self, *args, **kwargs):
        return self.integrate(*args, **kwargs)


# to be "the" future integral class
class Integration:
    def __init__(self, mc_sampler, draws_per_dim, tol, max_draws, draws_simpson):
        self.tol = tol
        self.max_draws = max_draws
        self.mc_sampler = mc_sampler
        self.draws_per_dim = draws_per_dim
        self.draws_simpson = draws_simpson
