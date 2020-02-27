#  Copyright (c) 2020 zfit
import copy
import functools
import math as _mt
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from ..settings import ztypes


def constant(value, dtype=ztypes.float, shape=None, name="Const", verify_shape=None):
    # TODO(tf2): remove this legacy thing below
    if verify_shape is not None:
        raise RuntimeError("'verify_shape' is not a valid argument anymore. It's always true. Please remove.")
    return tf.constant(value, dtype=dtype, shape=shape, name=name)


pi = np.float64(_mt.pi)


def to_complex(number, dtype=ztypes.complex):
    return tf.cast(number, dtype=dtype)


def to_real(x, dtype=ztypes.float):
    return tf.cast(x, dtype=dtype)


def abs_square(x):
    return tf.math.real(x * tf.math.conj(x))


def nth_pow(x, n, name=None):
    """Calculate the nth power of the complex Tensor x.

    Args:
        x (tf.Tensor, complex):
        n (int >= 0): Power
        name (str): No effect, for API compatibility with tf.pow
    """
    if not n >= 0:
        raise ValueError("n (power) has to be >= 0. Currently, n={}".format(n))

    power = to_complex(1.)
    for _ in range(n):
        power *= x
    return power


def unstack_x(value: Any, num: Any = None, axis: int = -1, always_list: bool = False, name: str = "unstack_x"):
    """Unstack a Data object and return a list of (or a single) tensors in the right order.

    Args:
        value ():
        num (Union[]):
        axis (int):
        always_list (bool): If True, also return a list if only one element.
        name (str):

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
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, dtype_hint=preferred_dtype)


def safe_where(condition: tf.Tensor, func: Callable, safe_func: Callable, values: tf.Tensor,
               value_safer: Callable = tf.ones_like) -> tf.Tensor:
    """Like :py:func:`tf.where` but fixes gradient `NaN` if func produces `NaN` with certain `values`.

    Args:
        condition (:py:class:`tf.Tensor`): Same argument as to :py:func:`tf.where`, a boolean :py:class:`tf.Tensor`
        func (Callable): Function taking `values` as argument and returning the tensor _in case
            condition is True_. Equivalent `x` of :py:func:`tf.where` but as function.
        safe_func (Callable): Function taking `values` as argument and returning the tensor
            _in case the condition is False_, Equivalent `y` of :py:func:`tf.where` but as function.
        values (:py:class:`tf.Tensor`): Values to be evaluated either by `func` or `safe_func` depending on
            `condition`.
        value_safer (Callable): Function taking `values` as arguments and returns "safe" values
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
        value_with_nans = tf.math.real(value_with_nans) + tf.math.imag(value_with_nans)  # we care only about NaN or not
    finite_bools = tf.math.is_finite(tf.cast(value_with_nans, dtype=tf.float64))
    finite_indices = tf.compat.v1.where(finite_bools)
    new_x = tf.gather_nd(params=x, indices=finite_indices)
    new_x = Data.from_tensor(obs=x.obs, tensor=new_x)
    vals_no_nan = func(x=new_x)
    result = tf.scatter_nd(indices=finite_indices, updates=vals_no_nan,
                           shape=tf.shape(input=value_with_nans, out_type=finite_indices.dtype))
    return result


# tf_function_deco = tf.function(autograph=False, experimental_relax_shapes=True)


class FunctionWrapperRegistry:
    wrapped_functions = []
    registries = []
    do_jit = True

    @classmethod
    def check_wrapped_functions_registered(cls):
        return all((func.zfit_graph_cache_registered for func in cls.wrapped_functions))

    def __init__(self, **kwargs_user) -> None:
        """`tf.function`-like decorator with additional cache-invalidation functionality.

        Args:
            **kwargs_user: arguments to `tf.function`
        """
        super().__init__()
        self._initial_user_kwargs = kwargs_user
        self.registries.append(self)
        self.reset(**self._initial_user_kwargs)
        # self.inside_tracing = False
        self.currently_traced = set()

    def reset(self, **kwargs_user):
        kwargs = dict(autograph=False, experimental_relax_shapes=True)
        kwargs.update(self._initial_user_kwargs)
        kwargs.update(kwargs_user)

        self.tf_function = tf.function(**kwargs)
        self.function_cache = defaultdict(list)

    def __call__(self, func):
        wrapped_func = self.tf_function(func)
        cache = self.function_cache[func]
        from zfit.util.cache import FunctionCacheHolder

        def call_correct_signature(func, args, kwargs):
            if args == [] and kwargs != {}:
                return func(**kwargs)
            elif args != [] and kwargs == {}:
                return func(*args)
            elif args == [] and kwargs == {}:
                return func()
            elif args != [] and kwargs != {}:
                return func(*args, **kwargs)

        def concrete_func(*args, **kwargs):

            if not self.do_jit or func in self.currently_traced:
                return call_correct_signature(func, args, kwargs)

            # self.inside_tracing = True
            self.currently_traced.add(func)
            nonlocal wrapped_func
            function_holder = FunctionCacheHolder(func, wrapped_func, args, kwargs)

            try:
                func_holder_index = cache.index(function_holder)
            except ValueError:  # not in cache
                cache.append(function_holder)
            else:
                func_holder_cached = cache[func_holder_index]
                if func_holder_cached.is_valid:
                    function_holder = func_holder_cached
                else:
                    wrapped_func = self.tf_function(func)  # update nonlocal wrapped function
                    function_holder = FunctionCacheHolder(func, wrapped_func, args, kwargs)
                    cache[func_holder_index] = function_holder
            func_to_run = function_holder.wrapped_func
            result = call_correct_signature(func_to_run, args, kwargs)
            self.currently_traced.remove(func)
            return result

        return concrete_func

FunctionWrapperRegistry2 = copy.deepcopy(FunctionWrapperRegistry)
# FunctionWrapperRegistry2.do_jit = True
# FunctionWrapperRegistry.do_jit = False


tf_function = FunctionWrapperRegistry()

function_tf = FunctionWrapperRegistry2()  # for only tensorflow inside
function_sampling = tf_function


# py_function = tf.py_function

@functools.wraps(tf.py_function)
def py_function(func, inp, Tout, name=None):
    from .. import settings
    if not settings.options['numerical_grad']:
        raise RuntimeError("Running a py_function without using the numerical gradient will result in wrong gradient"
                           " calculation. Will be more fine-grained in the future. To switch to numerical calculation"
                           " (even if the gradients are not calculated at all), do"
                           " `zfit.settings.options['numerical_grad'] = True`")

    return tf.py_function(func=func, inp=inp, Tout=Tout, name=name)
