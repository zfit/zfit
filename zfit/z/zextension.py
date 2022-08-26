#  Copyright (c) 2022 zfit

from __future__ import annotations

import contextlib
import functools
import logging
from weakref import WeakSet
import math as _mt
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp
from ..settings import ztypes
from ..util.exception import BreakingAPIChangeError
from ..util.warnings import warn_advanced_feature


def constant(value, dtype=ztypes.float, shape=None, name="Const", verify_shape=None):
    # TODO(tf2): remove this legacy thing below
    if verify_shape is not None:
        raise RuntimeError(
            "'verify_shape' is not a valid argument anymore. It's always true. Please remove."
        )
    return tf.constant(value, dtype=dtype, shape=shape, name=name)


pi = np.float64(_mt.pi)


def to_complex(number, dtype=ztypes.complex):
    return tf.cast(number, dtype=dtype)


def to_real(x, dtype=ztypes.float):
    return znp.asarray(tf.cast(x, dtype=dtype))


def abs_square(x):
    return znp.real(x * znp.conj(x))


def nth_pow(x, n):
    """Calculate the nth power of the complex Tensor x.

    Args:
        x:
        n: Power of x, has to be a positive int
        name: No effect, for API compatibility with tf.pow
    """
    if n < 0:
        raise ValueError(f"n (power) has to be >= 0. Currently, n={n}")

    power = to_complex(1.0)
    for _ in range(n):
        power *= x
    return power


def unstack_x(
    value: Any,
    num: Any = None,
    axis: int = -1,
    always_list: bool = False,
    name: str = "unstack_x",
):
    """Unstack a Data object and return a list of (or a single) tensors in the right order.

    Args:
        value:
        num:
        axis:
        always_list: If True, also return a list if only one element.
        name:

    Returns:
        Union[List[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor, None]:
    """
    if isinstance(value, list):
        if len(value) == 1 and not always_list:
            value = value[0]
        return value
    try:
        return value.unstack_x(always_list=always_list)
    except AttributeError:
        unstacked_x = tf.unstack(value=value, num=num, axis=axis, name=name)
    if len(unstacked_x) == 1 and not always_list:
        unstacked_x = unstacked_x[0]
    return unstacked_x


def stack_x(values, axis: int = -1, name: str = "stack_x"):
    return tf.stack(values=values, axis=axis, name=name)


# random sampling


def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    return tf.convert_to_tensor(
        value=value, dtype=dtype, name=name, dtype_hint=preferred_dtype
    )


def safe_where(
    condition: tf.Tensor,
    func: Callable,
    safe_func: Callable,
    values: tf.Tensor,
    value_safer: Callable = tf.ones_like,
) -> tf.Tensor:
    """Like :py:func:`tf.where` but fixes gradient `NaN` if func produces `NaN` with certain `values`.

    Args:
        condition: Same argument as to :py:func:`tf.where`, a boolean :py:class:`tf.Tensor`
        func: Function taking `values` as argument and returning the tensor _in case
            condition is True_. Equivalent `x` of :py:func:`tf.where` but as function.
        safe_func: Function taking `values` as argument and returning the tensor
            _in case the condition is False_, Equivalent `y` of :py:func:`tf.where` but as function.
        values: Values to be evaluated either by `func` or `safe_func` depending on
            `condition`.
        value_safer: Function taking `values` as arguments and returns "safe" values
            that won't cause troubles when given to`func` or by taking the gradient with respect
            to `func(value_safer(values))`.

    Returns:
        :py:class:`tf.Tensor`:
    """
    safe_x = tf.where(condition=condition, x=values, y=value_safer(values))
    result = tf.where(condition=condition, x=func(safe_x), y=safe_func(values))
    return result


def run_no_nan(func, x):
    from zfit.core.data import Data

    value_with_nans = func(x=x)
    if value_with_nans.dtype in (tf.complex128, tf.complex64):
        value_with_nans = znp.real(value_with_nans) + znp.imag(
            value_with_nans
        )  # we care only about NaN or not
    finite_bools = znp.isfinite(tf.cast(value_with_nans, dtype=tf.float64))
    finite_indices = tf.where(finite_bools)
    new_x = tf.gather_nd(params=x, indices=finite_indices)
    new_x = Data.from_tensor(obs=x.obs, tensor=new_x)
    vals_no_nan = func(x=new_x)
    result = tf.scatter_nd(
        indices=finite_indices,
        updates=vals_no_nan,
        shape=tf.shape(input=value_with_nans, out_type=finite_indices.dtype),
    )
    return result


class FunctionWrapperRegistry:
    registries = WeakSet()
    _debug_all_tf_functions = WeakSet()
    allow_jit = True
    _DEFAULT_DO_JIT_TYPES = defaultdict(lambda: True)
    _DEFAULT_DO_JIT_TYPES.update(
        {
            None: True,
            "model": False,
            "loss": True,
            "sample": True,
            "model_sampling": True,
            "zfit_tensor": True,
            "tensor": True,
        }
    )

    do_jit_types = _DEFAULT_DO_JIT_TYPES.copy()

    def __init__(self, wraps=None, stateless_args=None, **kwargs_user) -> None:
        """`tf.function`-like decorator with additional cache-invalidation functionality.

        Args:
            **kwargs_user: arguments to `tf.function`
        """
        super().__init__()
        if stateless_args is None:
            stateless_args = False
        self._initial_user_kwargs = kwargs_user

        self.registries.add(self)  # TODO: remove?
        self.python_func = None
        self.wrapped_func = None

        if wraps not in self.do_jit_types:
            self.do_jit_types[wraps] = True
        self.wraps = wraps
        self.stateless_args = stateless_args
        self.function_cache = []
        self.reset(**self._initial_user_kwargs)
        self.currently_traced = set()

    @property
    def do_jit(self):
        return self.do_jit_types[self.wraps] and self.allow_jit

    def reset(self, **kwargs_user):
        kwargs = dict(autograph=False, reduce_retracing=False)
        kwargs.update(self._initial_user_kwargs)
        kwargs.update(kwargs_user)
        self.tf_function_kwargs = kwargs
        self.function_cache.clear()

    @property
    def tf_function(self):
        function = tf.function(**self.tf_function_kwargs)

        return function

    def __call__(self, func):
        wrapped_func = self.tf_function(func)
        cache = self.function_cache
        from ..util.cache import FunctionCacheHolder

        def concrete_func(*args, **kwargs):
            # TODO: add limited cache side, maybe check if inside big function, grow cache.

            if not self.do_jit or func in self.currently_traced:
                return func(*args, **kwargs)

            self.currently_traced.add(func)
            nonlocal wrapped_func, cache

            def deleter(function_holder):
                with contextlib.suppress(ValueError):
                    cache.remove(function_holder)

            #
            function_holder = FunctionCacheHolder(
                func,
                wrapped_func,
                args,
                kwargs,
                deleter=deleter,
                stateless_args=self.stateless_args,
            )
            #
            try:
                func_holder_index = cache.index(function_holder)
            except ValueError:
                # cache = cache[-10:]
                wrapped_func = self.tf_function(func)
                func_to_run = wrapped_func
                function_holder = FunctionCacheHolder(
                    func,
                    wrapped_func,
                    args,
                    kwargs,
                    deleter=deleter,
                    stateless_args=self.stateless_args,
                )
                cache.append(function_holder)
            else:
                func_to_run = cache[func_holder_index].wrapped_func
            # if function_holder in cache:
            #     # print("DEBUG: using cached function")
            #
            #     func_holder_cached = cache[0]
            #     func_to_run = func_holder_cached.wrapped_func
            # else:
            #     cache.clear()

            #     if func_holder_cached.is_valid:
            #         function_holder = func_holder_cached
            #     else:
            #         cache.clear()
            #         self_tf_function = self.tf_function
            #         wrapped_func = self_tf_function(
            #             func
            #         )  # update nonlocal wrapped function
            #         self._debug_all_tf_functions.add(self_tf_function)
            #         print(f"Registered {self_tf_function}, {self._debug_all_tf_functions}")
            #         function_holder = FunctionCacheHolder(
            #             func, wrapped_func, args, kwargs
            #         )
            #         cache.append(function_holder)
            # else:
            #     # print(f"DEBUG: caching function {func} with args {args} and kwargs {kwargs}")
            #     cache.clear()
            #     cache.append(function_holder)
            # func_to_run = function_holder.wrapped_func
            # import gc
            #
            # gc.collect()
            try:
                result = func_to_run(*args, **kwargs)
            finally:
                self.currently_traced.remove(func)
            return result

        # concrete_func.zfit_graph_cache_registered = False
        return concrete_func


# equivalent to tf.function
def function(func=None, *, stateless_args=None, **kwargs):
    """JIT/Graph compilation of functions, `tf.function`-like with additional cache-invalidation functionality.

    Args:
        func: Function to be compiled.
        stateless_args: If True, the function is assumed to be stateless and *does not depend on the name of tf.Variables*
            but only needs the values of the variables. This is not the case for taking gradients, for example.
        **kwargs: arguments to `tf.function`

    Returns:
    """
    if stateless_args is None:
        stateless_args = False
    if callable(func):
        wrapper = FunctionWrapperRegistry()
        return wrapper(func)
    elif func:
        raise ValueError(
            "All argument have to be key-word only. `func` must not be used"
        )
    else:
        return FunctionWrapperRegistry(**kwargs, stateless_args=stateless_args)


# legacy, remove 0.6
def function_tf_input(*_, **__):
    raise BreakingAPIChangeError(
        "This function has been removed. Use `z.function(wraps='zfit_tensor') or your"
        "own category"
    )


# legacy, remove 0.6
def function_sampling(*_, **__):
    raise BreakingAPIChangeError(
        "This function has been removed. Use `z.function(wraps='zfit_sampling') or your"
        "own category"
    )


@functools.wraps(tf.py_function)
def py_function(func, inp, Tout, name=None):
    from .. import settings

    if not settings.options["numerical_grad"]:
        warn_advanced_feature(
            "Using py_function without numerical gradients. If the Python code does not contain any"
            " parametrization by `zfit.Parameter` or similar, this can work out. Otherwise, in case"
            " it depends on those, you may want to set `zfit.run.set_autograd_mode(=False)`.",
            identifier="py_func_autograd",
        )

    return tf.py_function(func=func, inp=inp, Tout=Tout, name=name)
