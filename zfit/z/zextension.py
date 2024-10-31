#  Copyright (c) 2024 zfit

from __future__ import annotations

import collections
import contextlib
import functools
import math as _mt
import typing
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from weakref import WeakSet

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp

from ..settings import run, ztypes
from ..util.warnings import warn_advanced_feature


def constant(value, dtype=ztypes.float, shape=None, name="Const", verify_shape=None):
    del verify_shape
    return tf.constant(value, dtype=dtype, shape=shape, name=name)


pi = np.float64(_mt.pi)


def to_complex(number, dtype=ztypes.complex):
    return znp.asarray(number, dtype=dtype)


def to_real(x, dtype=ztypes.float):
    return znp.asarray(x, dtype=dtype)


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
        msg = f"n (power) has to be >= 0. Currently, n={n}"
        raise ValueError(msg)

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
        assert isinstance(unstacked_x, list), (
            "unstacked_x has to be a list, otherwise this is a bug. Please report on github: "
            "https://github.com/zfit/zfit/issues/new?assignees=&labels=bug&projects=&template="
            "bug_report.md&title=[ASSERT]%20unstack_x%20does%20not%20provide%20a%20list,%20internal%20error"
        )
        unstacked_x = unstacked_x[0]
    return unstacked_x


def stack_x(values, axis: int = -1, name: str = "stack_x"):
    return tf.stack(values=values, axis=axis, name=name)


# random sampling


def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, dtype_hint=preferred_dtype)


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
    return tf.where(condition=condition, x=func(safe_x), y=safe_func(values))


def run_no_nan(func, x):
    from zfit.core.data import Data

    value_with_nans = func(x=x)
    if value_with_nans.dtype in (tf.complex128, tf.complex64):
        value_with_nans = znp.real(value_with_nans) + znp.imag(value_with_nans)  # we care only about NaN or not
    finite_bools = znp.isfinite(znp.asarray(value_with_nans, dtype=tf.float64))
    finite_indices = tf.where(finite_bools)
    new_x = tf.gather_nd(params=x, indices=finite_indices)
    new_x = Data.from_tensor(obs=x.obs, tensor=new_x)
    vals_no_nan = func(x=new_x)
    return tf.scatter_nd(
        indices=finite_indices,
        updates=vals_no_nan,
        shape=tf.shape(input=value_with_nans, out_type=finite_indices.dtype),
    )


class DoNotCompile(Exception):
    """Raise this error if the function is being jitted but should not be (yet)."""


class FunctionWrapperRegistry:
    registries = WeakSet()
    allow_jit = True
    DEFAULT_CACHE_SIZE = 40
    _DEFAULT_DO_JIT_TYPES: typing.ClassVar = defaultdict(lambda: True)
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

    def __init__(
        self,
        wraps=None,
        *,
        stateless_args=None,
        cachesize=None,
        keepalive=None,
        force_eager=None,
        **kwargs_user,
    ) -> None:
        """`tf.function`-like decorator with additional cache-invalidation functionality.

        Args:
            wraps: name of the function/type that is wrapped. Can be used to toggle specific types to be jitted or not.
            stateless_args: If true, assume that all arguments are stateless, i.e. aren't `zfit.Parameters` or similar.
            force_eager: Forces the execution of the function (and functions that have called this function) to be
                executed eagerly, that is, Python-like (and not jitted).
                This can significantly reduce the performance, however, normal
                Python syntax and control flow can be used also on values.
            cachesize: OLD
            keepalive: OLD
            **kwargs_user: arguments to `tf.function`
        """
        super().__init__()
        if cachesize is None:
            cachesize = self.DEFAULT_CACHE_SIZE
        if stateless_args is None:
            stateless_args = False
        if keepalive is None:
            keepalive = True
        self._initial_user_kwargs = kwargs_user
        self._deleted_cachers = collections.Counter()

        self.registries.add(self)  # TODO: remove?
        self.python_func = None
        self.wrapped_func = None
        self.force_eager = force_eager if force_eager is not None else False

        if wraps not in self.do_jit_types:
            from ..settings import run

            self.do_jit_types[wraps] = bool(run.get_graph_mode())
        self.wraps = wraps
        self.stateless_args = stateless_args

        self.function_cache = collections.deque()
        self.reset(**self._initial_user_kwargs)
        self.currently_traced = set()
        self.cachesize = cachesize
        self.keepalive = keepalive

    @property
    def do_jit(self):
        return self.do_jit_types[self.wraps] and self.allow_jit and not self.force_eager

    def reset(self, **kwargs_user):
        kwargs = {"autograph": False, "reduce_retracing": True}
        kwargs.update(self._initial_user_kwargs)
        kwargs.update(kwargs_user)
        self.tf_function_kwargs = kwargs
        self.function_cache.clear()

    def set_graph_cache_size(self, cachesize: int | None = None):
        """Set the size of the graph cache.

        Args:
            cachesize: Size of the cache. If None, the default size is used.
        """
        if cachesize is None:
            cachesize = self.DEFAULT_CACHE_SIZE
        self.cachesize = cachesize
        while len(self.function_cache) >= self.cachesize:
            self.function_cache.popleft()

    @property
    def tf_function(self):
        return tf.function(**self.tf_function_kwargs)

    def __call__(self, func):
        keepalive = self.keepalive
        wrapped_func = self.tf_function(func)
        cache = self.function_cache
        deleted_cachers = self._deleted_cachers
        from ..util.cache import FunctionCacheHolder

        def concrete_func(*args, **kwargs):
            if self.force_eager and not run.executing_eagerly():
                raise DoNotCompile
            if not self.do_jit or func in self.currently_traced or not run.executing_eagerly():
                return func(*args, **kwargs)

            self.currently_traced.add(func)
            nonlocal wrapped_func

            # todo: we could return the function here? Need still registry to avoid deadlock in TF
            # try:
            #     value = wrapped_func(*args, **kwargs)
            # finally:
            #     self.currently_traced.remove(func)
            # return value

            def deleter(proxy):
                del proxy
                with contextlib.suppress(ValueError):
                    cache.remove(function_holder)

            function_holder = FunctionCacheHolder(
                func,
                wrapped_func,
                args,
                kwargs,
                deleter=deleter,
                stateless_args=self.stateless_args,
            )
            try:
                func_holder_index = cache.index(function_holder)
            except ValueError:
                wrapped_func = self.tf_function(func)
                func_to_run = wrapped_func
                function_holder = FunctionCacheHolder(
                    func,
                    wrapped_func,
                    args,
                    kwargs,
                    deleter=deleter,
                    stateless_args=self.stateless_args,
                    keepalive=keepalive,
                )
                if len(cache) >= self.cachesize:
                    popped_holder = cache.popleft()
                    hash_popped_holder = hash(popped_holder)
                    deleted_cachers.update((hash_popped_holder,))
                    if self._deleted_cachers[hash_popped_holder] > 3:
                        warnings.warn(
                            f"Function {function_holder.python_func} was removed from the cache more than 3"
                            f" times (and getting recompiled). Maybe consider increasing the cache size"
                            f" using `zfit.run.set_graph_cache_size(...)`, the current size is {self.cachesize}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                        self._deleted_cachers - collections.Counter(
                            {hash(function_holder): int(-1e100)}
                        )  # won't be warned again
                    del popped_holder
                cache.append(function_holder)
            else:
                function_holder = cache[func_holder_index]
                func_to_run = function_holder.execute_func

            try:
                try:
                    result = func_to_run(*args, **kwargs)
                except KeyError as error:
                    warnings.warn(
                        f"An error occurred while running a jitted function. The error was: {error}."
                        f" The function will be recompiled",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    result = func_to_run(*args, **kwargs)
            except DoNotCompile:
                function_holder.do_jit = False
                if not run.executing_eagerly():
                    raise
                result = function_holder.execute_func(*args, **kwargs)
            finally:
                self.currently_traced.remove(func)
            return result

        return concrete_func


# equivalent to tf.function
def function(func=None, *, stateless_args=None, cachesize=None, **kwargs):
    """JIT/Graph compilation of functions, `tf.function`-like with additional cache-invalidation functionality.

    Args:
        func: Function to be compiled.
        stateless_args: If True, the function is assumed to be stateless and *does not depend on the name of tf.Variables*
            but only needs the values of the variables. This is not the case for taking gradients, for example.
        cachesize: Size of the cache. If None, the default size is used.
        **kwargs: arguments to `tf.function`

    Returns:
    """

    if stateless_args is None:
        stateless_args = False
    if callable(func):
        wrapper = FunctionWrapperRegistry(cachesize=cachesize, stateless_args=stateless_args, **kwargs)
        return wrapper(func)
    if func:
        msg = "All argument have to be key-word only. `func` must not be used"
        raise ValueError(msg)

    return FunctionWrapperRegistry(**kwargs, cachesize=cachesize, stateless_args=stateless_args)


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
