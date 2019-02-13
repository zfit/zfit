"""Baseclass for a Model. Handle integration and sampling"""

import abc
import builtins
from collections import OrderedDict
import contextlib
from contextlib import suppress
import typing
from typing import Dict, Type, Union
import warnings

import pep487
import tensorflow as tf
from tensorflow_probability.python import mcmc as mc

from zfit import ztf
from zfit.core.integration import Integration
from .data import Data, SampleData
from .dimension import BaseDimensional
from . import integration as zintegrate, sample as zsample
from .baseobject import BaseNumeric
from .interfaces import ZfitModel, ZfitParameter, ZfitData
from .limits import Space, convert_to_space, no_multiple_limits, no_norm_range, supports
from ..settings import ztypes
from ..util import container as zcontainer, ztyping
from ..util.exception import (BasePDFSubclassingError, MultipleLimitsNotImplementedError, NormRangeNotImplementedError,
                              ShapeIncompatibleError, SubclassingError, LimitsNotSpecifiedError, )

_BaseModel_USER_IMPL_METHODS_TO_CHECK = {}


def _BaseModel_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support (bool): If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.

    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func (function):

        Returns:
            function:
        """
        name = func.__name__
        _BaseModel_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BaseModel_register_check_support
        return func

    return register


class BaseModel(BaseNumeric, BaseDimensional, ZfitModel):
    """Base class for any generic model.

    # TODO instructions on how to use

    """
    _DEFAULTS_integration = zcontainer.DotDict()
    _DEFAULTS_integration.mc_sampler = lambda *args, **kwargs: mc.sample_halton_sequence(*args, randomized=False,
                                                                                         **kwargs)
    # _DEFAULTS_integration.mc_sampler = lambda dim, num_results, dtype: tf.random_uniform(maxval=1.,
    #                                                                                      shape=(num_results, dim),
    #                                                                                      dtype=dtype)
    _DEFAULTS_integration.draws_per_dim = 20000
    _DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(self, obs: ztyping.ObsTypeInput, params: Union[Dict[str, ZfitParameter], None] = None,
                 name: str = "BaseModel", dtype=ztypes.float,
                 **kwargs):
        """The base model to inherit from and overwrite `_unnormalized_pdf`.

        Args:
            dtype (Type): the dtype of the model
            name (str): the name of the model
            params (): the parameters the distribution depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, **kwargs)
        self._check_set_space(obs)

        self._integration = zcontainer.DotDict()
        # self._integration.mc_sampler = self._DEFAULTS_integration.mc_sampler
        # self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.auto_numeric_integrator = self._DEFAULTS_integration.auto_numeric_integrator
        self.integration = Integration(mc_sampler=self._DEFAULTS_integration.mc_sampler,
                                       draws_per_dim=self._DEFAULTS_integration.draws_per_dim)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if subclass has decorator if required
        cls._subclass_check_support(methods_to_check=_BaseModel_USER_IMPL_METHODS_TO_CHECK,
                                    wrapper_not_overwritten=_BaseModel_register_check_support)
        cls._analytic_integral = zintegrate.AnalyticIntegral()
        cls._inverse_analytic_integral = []
        cls._additional_repr = {}

    @classmethod
    def _subclass_check_support(cls, methods_to_check, wrapper_not_overwritten):
        for method_name, has_support in methods_to_check.items():
            method = getattr(cls, method_name)
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == wrapper_not_overwritten:
                    continue  # not overwritten, fine

            # here means: overwritten
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == supports:
                    if has_support:
                        continue  # needs support, has been wrapped
                    else:
                        raise BasePDFSubclassingError("Method {} has been wrapped with supports "
                                                      "but is not allowed to. Has to handle all "
                                                      "arguments.".format(method_name))
                elif has_support:
                    raise BasePDFSubclassingError("Method {} has been overwritten and *has to* be "
                                                  "wrapped by `supports` decorator (don't forget () )"
                                                  "to call the decorator as it takes arguments"
                                                  "".format(method_name))
                elif not has_support:
                    continue  # no support, has not been wrapped with
            else:
                if not has_support:
                    continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            raise BasePDFSubclassingError("Method {} has not been correctly wrapped with @supports "
                                          "OR has been wrapped but it should not be".format(method_name))

    @abc.abstractmethod
    def _func_to_integrate(self, x: ztyping.XType) -> tf.Tensor:  # TODO(Mayou36): return `Data`?
        raise NotImplementedError

    @abc.abstractmethod
    def _func_to_sample_from(self, x: ztyping.XType) -> tf.Tensor:  # TODO(Mayou36): return `Data`?
        raise NotImplementedError

    @property
    def space(self) -> "ZfitSpace":
        return self._space

    def _check_set_space(self, obs):
        if not isinstance(obs, Space):
            obs = Space(obs=obs)
        self._check_n_obs(space=obs)
        self._space = obs.with_autofill_axes(overwrite=True)

    @contextlib.contextmanager
    def _convert_sort_x(self, x: ztyping.XTypeInput) -> Data:
        if isinstance(x, ZfitData):
            if x.obs is not None:
                with x.sort_by_obs(obs=self.obs):
                    yield x
            elif x.axes is not None:
                with x.sort_by_axes(axes=self.axes):
                    yield x
            else:
                assert False, "Neither the `obs` nor the `axes` are specified in `Data`"
        else:
            if not isinstance(x, (tf.Tensor, tf.Variable)):
                try:
                    x = ztf.convert_to_tensor(value=x)
                except TypeError:
                    raise TypeError("Wrong type of x ({}). Has to be a `Data` or convertible to a tf.Tensor")
            # check dimension
            x = self._add_dim_to_x(x=x)
            x_shape = x.shape.as_list()[-1]
            if x_shape != self.n_obs:
                raise ShapeIncompatibleError("The shape of x (={}) (in the last dim) does not"
                                             "match the shape (={})of the model".format(x_shape, self.n_obs))
            x = Data.from_tensors(obs=self.obs, tensors=x)
            yield x

    def _add_dim_to_x(self, x):  # TODO(Mayou36): remove function? unnecessary? dealt with in `Data`?
        if self.n_obs == 1:
            if len(x.shape.as_list()) == 0:
                x = tf.expand_dims(x, -1)
            if len(x.shape.as_list()) == 1:
                x = tf.expand_dims(x, -1)
        return x

    def set_integration_options(self, draws_per_dim=None, mc_sampler=None):
        # mc_options = {} if mc_options is None else mc_options
        # numeric_options = {} if numeric_options is None else numeric_options
        # general_options = {} if general_options is None else general_options
        # analytic_options = {} if analytic_options is None else analytic_options
        # if analytic_options:
        #     raise NotImplementedError("analytic_options cannot be updated currently.")
        if draws_per_dim is not None:
            self.integration.draws_per_dim = draws_per_dim
        if mc_sampler is not None:
            self.integration.mc_sampler = mc_sampler

    @abc.abstractmethod
    def gradients(self, x: ztyping.XType, norm_range: ztyping.LimitsType, params: ztyping.ParamsTypeOpt = None):
        raise NotImplementedError

    def _check_input_norm_range(self, norm_range, caller_name="",
                                none_is_error=False) -> typing.Union[Space, bool]:
        """Convert to :py:class:`Space`.

        Args:
            norm_range (None or Space compatible):
            caller_name (str): name of the calling function. Used for exception message.
            none_is_error (bool): if both `norm_range` and `self.norm_range` are None, the default
                value is `False` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two `None` will raise a Value error.

        Returns:
            Union[Space, False]:

        """
        if norm_range is None or (isinstance(norm_range, Space) and norm_range.limits is None):
            if none_is_error:
                raise ValueError("Normalization range `norm_range` has to be specified when calling {name} or"
                                 "a default normalization range has to be set. Currently, both are None"
                                 "".format(name=caller_name))
            # else:
            #     norm_range = False
        # if norm_range is False and not convert_false:
        #     return False

        return self.convert_sort_space(limits=norm_range)

    def _check_input_limits(self, limits, caller_name="", none_is_error=False):
        if limits is None or (isinstance(limits, Space) and limits.limits is None):
            if none_is_error:
                raise ValueError("The `limits` have to be specified when calling {name} and not be None"
                                 "".format(name=caller_name))
            # else:
            #     limits = False

        return self.convert_sort_space(limits=limits)

    def convert_sort_space(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> Union[Space, None]:
        """Convert the inputs (using eventually `obs`, `axes`) to `Space` and sort them according to own `obs`.

        Args:
            obs ():
            axes ():
            limits ():

        Returns:

        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        self_space = self._space
        if self_space is not None:
            space = space.with_obs_axes(self_space.get_obs_axes(), ordered=True, allow_subset=True)
        return space

    # Integrals

    @_BaseModel_register_check_support(True)
    def _integrate(self, limits, norm_range):
        raise NotImplementedError()

    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, Space): the limits to integrate over
            norm_range (tuple, Space): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str):

        Returns:
            Tensor: the integral value
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits)
        return self._single_hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _single_hook_integrate(self, limits, norm_range, name):
        return self._hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        return self._norm_integrate(limits=limits, norm_range=norm_range, name=name)

    def _norm_integrate(self, limits, norm_range, name='_norm_integrate'):
        try:
            integral = self._limits_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            unnormalized_integral = self._limits_integrate(limits=limits, norm_range=False, name=name)
            normalization = self._limits_integrate(limits=limits, norm_range=norm_range, name=name)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)
        return integral

    def _call_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._integrate(limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._hook_analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_integrate(limits=limits, norm_range=norm_range)

    def _fallback_integrate(self, limits, norm_range):
        axes = limits.axes
        max_axes = self._analytic_integral.get_max_axes(limits=limits, axes=axes)

        integral = None
        if max_axes and integral:  # TODO improve handling of available analytic integrals
            with suppress(NotImplementedError):
                def part_int(x):
                    """Temporary partial integration function."""
                    return self._hook_partial_analytic_integrate(x, limits=limits, norm_range=norm_range)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._hook_numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    @classmethod
    def register_analytic_integral(cls, func: typing.Callable, limits: ztyping.LimitsType = None,
                                   priority: Union[int, float] = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral with the class.

        Args:
            func (callable):
            limits (): |limits_arg_descr|
            priority (int):
            supports_multiple_limits (bool):
            supports_norm_range (bool):

        """
        cls._analytic_integral.register(func=func, limits=limits, supports_norm_range=supports_norm_range,
                                        priority=priority, supports_multiple_limits=supports_multiple_limits)

    @classmethod
    def register_inverse_analytic_integral(cls, func: typing.Callable) -> None:
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func ():
        """
        if cls._inverse_analytic_integral:
            cls._inverse_analytic_integral[0] = func
        else:
            cls._inverse_analytic_integral.append(func)

    @_BaseModel_register_check_support(True)
    def _analytic_integrate(self, limits, norm_range):  # TODO: typing on overwriteable methods
        raise NotImplementedError

    def analytic_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                           name: str = "analytic_integrate") -> ztyping.XType:
        """Do analytical integration over function and raise Error if not possible.

        Args:
            limits (tuple, Space): the limits to integrate over
            norm_range (tuple, Space, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value
        Raises:
            NotImplementedError: If no analytical integral is available (for this limits).
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits)
        return self._single_hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _single_hook_analytic_integrate(self, limits, norm_range, name):
        return self._hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        return self._norm_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _norm_analytic_integrate(self, limits, norm_range, name='_norm_analytic_integrate'):
        try:
            integral = self._limits_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:

            unnormalized_integral = self._limits_analytic_integrate(limits, norm_range=None, name=name)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False, name=name)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any)"
                                                   "normalization range 'norm_range'."
                                                   " This usually means,that no analytic integral "
                                                   "is available for this function. Due to rule "
                                                   "safety, an analytical normalization has to "
                                                   "be available and no attempt of numerical "
                                                   "normalization was made.".format(name))
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_analytic_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_analytic_integrate(limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_analytic_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)
        return integral

    def _call_analytic_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_analytic_integrate(limits=limits, norm_range=norm_range)

    def _fallback_analytic_integrate(self, limits, norm_range):
        return self._analytic_integral.integrate(x=None, limits=limits, axes=limits.axes,
                                                 norm_range=norm_range, model=self, params=self.params)

    @_BaseModel_register_check_support(True)
    def _numeric_integrate(self, limits, norm_range):
        raise NotImplementedError

    def numeric_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                          name: str = "numeric_integrate") -> ztyping.XType:
        """Numerical integration over the model.

        Args:
            limits (tuple, Space): the limits to integrate over
            norm_range (tuple, Space, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value

        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits)

        return self._single_hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _single_hook_numeric_integrate(self, limits, norm_range, name):
        return self._hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        return self._norm_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _norm_numeric_integrate(self, limits, norm_range, name='_norm_numeric_integrate'):
        try:
            integral = self._limits_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range.limits is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_numeric_integrate(limits=limits, norm_range=False, name=name)
            normalization = self._limits_numeric_integrate(limits=norm_range, norm_range=False,
                                                           name=name + "_normalization")
            integral = unnormalized_integral / normalization
        return integral

    def _limits_numeric_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_numeric_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)

        return integral

    def _call_numeric_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._numeric_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_numeric_integrate(limits=limits, norm_range=norm_range)

    def _fallback_numeric_integrate(self, limits, norm_range):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm_range=norm_range)

    @_BaseModel_register_check_support(True)
    def _partial_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_integrate(self, x: ztyping.XTypeInput, limits: ztyping.LimitsType,
                          norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XTypeReturn:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The value at which the partially integrated function will be evaluated
            limits (tuple, Space): the limits to integrate over. Can contain only some axes
            norm_range (tuple, Space, False): the limits to normalize over. Has to have all axes
            name (str):

        Returns:
            Tensor: the value of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits, caller_name=name)
        with self._convert_sort_x(x) as x:
            return self._single_hook_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _single_hook_partial_integrate(self, x, limits, norm_range, name):
        return self._hook_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        integral = self._norm_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range.limits is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=False, name=name)
            normalization = self._hook_integrate(limits=norm_range, norm_range=False)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limit in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_partial_integrate(x=x, limits=sub_limit, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)

        return integral

    def _call_partial_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            with suppress(NotImplementedError):
                return self._partial_integrate(x=x, limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

            return self._fallback_partial_integrate(x=x, limits=limits, norm_range=norm_range)

    def _fallback_partial_integrate(self, x, limits, norm_range):
        max_axes = self._analytic_integral.get_max_axes(limits=limits, axes=limits.axes)
        if max_axes:
            sublimits = limits.subspace(max_axes)

            def part_int(x):  # change to partial integrate max axes?
                """Temporary partial integration function."""
                return self._hook_partial_analytic_integrate(x=x, limits=sublimits, norm_range=norm_range)

            axes = list(set(limits.axes) - set(max_axes))
        else:
            part_int = self._func_to_integrate
            axes = limits.axes

        if norm_range.limits is False:
            integral_vals = self._auto_numeric_integrate(func=part_int, limits=limits, axes=axes, x=x)
        else:
            raise NormRangeNotImplementedError
        return integral_vals

    @_BaseModel_register_check_support(True)
    def _partial_analytic_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_analytic_integrate(self, x: ztyping.XTypeInput, limits: ztyping.LimitsType,
                                   norm_range: ztyping.LimitsType = None,
                                   name: str = "partial_analytic_integrate") -> ztyping.XTypeReturn:
        """Do analytical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The value at which the partially integrated function will be evaluated
            limits (tuple, Space): the limits to integrate over. Can contain only some axes
            norm_range (tuple, Space, False): the limits to normalize over. Has to have all axes
            name (str):

        Returns:
            Tensor: the value of the partially integrated function evaluated at `x`.

        Raises:
            NotImplementedError: if the *analytic* integral (over this limits) is not implemented
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.

        """
        norm_range = self._check_input_norm_range(norm_range=norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits, caller_name=name)
        with self._convert_sort_x(x) as x:
            return self._single_hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range, name):
        return self._hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _hook_partial_analytic_integrate(self, x, limits, norm_range, name='_hook_partial_analytic_integrate'):
        return self._norm_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _norm_partial_analytic_integrate(self, x, limits, norm_range, name='_norm_partial_analytic_integrate'):
        try:
            integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range.limits is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=False,
                                                                            name=name)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False, name=name)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any) normalization range"
                                                   " 'norm_range'. This usually means,that no analytic integral "
                                                   "is available for this function. An analytical normalization has to "
                                                   "be available and no attempt of numerical normalization was made."
                                                   "".format(name))
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_analytic_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_partial_analytic_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                       name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)

        return integral

    def _call_partial_analytic_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)
            return self._fallback_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

    def _fallback_partial_analytic_integrate(self, x, limits, norm_range):
        return self._analytic_integral.integrate(x=x, limits=limits, axes=limits.axes,
                                                 norm_range=norm_range, model=self, params=self.params)

    @_BaseModel_register_check_support(True)
    def _partial_numeric_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_numeric_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType,
                                  norm_range: ztyping.LimitsType = None,
                                  name: str = "partial_numeric_integrate") -> ztyping.XType:
        """Force numerical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The value at which the partially integrated function will be evaluated
            limits (tuple, Space): the limits to integrate over. Can contain only some axes
            norm_range (tuple, Space, False): the limits to normalize over. Has to have all axes
            name (str):

        Returns:
            Tensor: the value of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits, caller_name=name)
        with self._convert_sort_x(x) as x:
            return self._single_hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range, name):
        return self._hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _hook_partial_numeric_integrate(self, x, limits, norm_range,
                                        name='_hook_partial_numeric_integrate'):
        integral = self._norm_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range.limits is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm_range=False,
                                                                           name=name)
            integral = unnormalized_integral / self._hook_numeric_integrate(limits=norm_range, norm_range=norm_range)
        return integral

    def _limits_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_partial_numeric_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                      name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)
        return integral

    def _call_partial_numeric_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            with suppress(NotImplementedError):
                return self._partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)
            return self._fallback_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)

    def _fallback_partial_numeric_integrate(self, x, limits, norm_range=False):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm_range=norm_range, x=x)

    @no_norm_range
    def _auto_numeric_integrate(self, func, limits, x=None, norm_range=False, **overwrite_options):
        integration_options = dict(func=func, limits=limits, n_axes=limits.n_obs, x=x, norm_range=norm_range,
                                   # auto from self
                                   dtype=self.dtype,
                                   mc_sampler=self.integration.mc_sampler,
                                   mc_options={
                                       "draws_per_dim": self.integration.draws_per_dim},
                                   **overwrite_options)
        return self._integration.auto_numeric_integrator(**integration_options)

    @no_norm_range
    def _inverse_analytic_integrate(self, x):
        if not self._inverse_analytic_integral:
            raise NotImplementedError
        else:
            return self._inverse_analytic_integral[0](x=x, params=self.params)

    @_BaseModel_register_check_support(True)
    def _sample(self, n, limits):
        raise NotImplementedError

    def sample(self, n: int, limits: ztyping.LimitsType = None, fixed_params=True, name: str = "sample") -> ztyping.XType:
        """Sample `n` points within `limits` from the model.

        If `limits` is not specified, `space` is used (if the space contains limits).

        Args:
            n (int): The number of samples to be generated
            limits (tuple, Space): In which region to sample in
            name (str):

        Returns:
            Tensor(n_obs, n_samples)
        """
        if limits is None:
            limits = self.space
        if fixed_params is True:
            fixed_params = list(self.get_dependents(only_floating=False))
        if fixed_params is False:
            fixed_params = []
        if not isinstance(fixed_params, (list, tuple)):
            raise TypeError("`Fixed_params` has to be a list, tuple or a boolean.")

        limits = self._check_input_limits(limits=limits, caller_name=name)
        sample = self._single_hook_sample(n=n, limits=limits, name=name)
        sample_data = SampleData.from_sample(sample=sample, obs=self.obs, fixed_params=fixed_params,
                                             name=name)
        return sample_data

    def _single_hook_sample(self, n, limits, name):
        return self._hook_sample(n=n, limits=limits, name=name)

    def _hook_sample(self, limits, n, name='_hook_sample'):
        return self._norm_sample(n=n, limits=limits, name=name)

    def _norm_sample(self, n, limits, name):
        """Dummy function"""
        return self._limits_sample(n=n, limits=limits, name=name)

    def _limits_sample(self, n, limits, name):
        try:
            return self._call_sample(n=n, limits=limits, name=name)
        except MultipleLimitsNotImplementedError:
            raise NotImplementedError("MultipleLimits auto handling in sample currently not supported.")

    def _call_sample(self, n, limits, name):
        with self._name_scope(name, values=[n, limits]):
            # n = ztf.convert_to_tensor(n, dtype=ztypes.int, name="n")

            with suppress(NotImplementedError):
                return self._sample(n=n, limits=limits)
            with suppress(NotImplementedError):
                return self._analytic_sample(n=n, limits=limits)
            return self._fallback_sample(n=n, limits=limits)

    @no_multiple_limits
    def _analytic_sample(self, n, limits: Space):

        (lower_bound,), (upper_bound,) = limits.limits
        neg_infinities = (tuple((-float("inf"),) * limits.n_obs),)  # py34 change float("inf") to math.inf
        # to the cdf to get the limits for the inverse analytic integral
        try:
            lower_prob_lim = self._norm_analytic_integrate(limits=Space.from_axes(limits=(neg_infinities,
                                                                                          (lower_bound,)),
                                                                                  axes=limits.axes),
                                                           norm_range=False)

            upper_prob_lim = self._norm_analytic_integrate(limits=Space.from_axes(limits=(neg_infinities,
                                                                                          (upper_bound,)),
                                                                                  axes=limits.axes),
                                                           norm_range=False)
        except NotImplementedError:
            raise NotImplementedError("analytic sampling not possible because the analytic integral is not"
                                      " implemented for the boundaries:".format(limits.limits))
        prob_sample = ztf.random_uniform(shape=(n, limits.n_obs), minval=lower_prob_lim,
                                         maxval=upper_prob_lim)
        # with self._convert_sort_x(prob_sample) as x:
        x = prob_sample
        sample = self._inverse_analytic_integrate(x=x)  # TODO(Mayou36): switch (n, n_obs) shape order
        return sample

    def _fallback_sample(self, n, limits):
        sample = zsample.accept_reject_sample(prob=self._func_to_sample_from, n=n, limits=limits,
                                              prob_max=None)  # None -> auto
        return sample

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with tf.name_scope(self.name):
            with tf.name_scope(name, values=([] if values is None else values)) as scope:
                yield scope

    @classmethod
    def register_additional_repr(cls, **kwargs):
        """Register an additional attribute to add to the repr.

        Args:
            any keyword argument. The value has to be gettable from the instance (has to be an
            attribute or callable method of self.
        """
        if cls._additional_repr is None:
            cls._additional_repr = {}
        overwritten_keys = set(kwargs).intersection(cls._additional_repr)
        if overwritten_keys:
            warnings.warn("The following keys have been overwritten while registering additional repr:"
                          "\n{}".format([str(k) for k in overwritten_keys]))
        cls._additional_repr = dict(cls._additional_repr, **kwargs)

    def _get_additional_repr(self, sorted=True):

        # nice name change
        sorted_ = sorted
        sorted = builtins.sorted
        # nice name change end

        additional_repr = OrderedDict()
        for key, val in self._additional_repr.items():
            try:
                new_obj = getattr(self, val)
            except AttributeError:
                raise AttributeError("The attribute {} is not a valid attribute of this class {}."
                                     "Cannot use it in __repr__. It was added using the"
                                     "`register_additional_repr` function".format(val, type(self)))
            else:
                if callable(new_obj):
                    new_obj = new_obj()
            additional_repr[key] = new_obj
        if sorted_:
            additional_repr = OrderedDict(sorted(additional_repr))
        return additional_repr

    def __repr__(self):  # TODO(mayou36):repr to baseobject with _repr

        return ("<zfit.{type_name} "
                " parameters=[{params}]"
                " dtype={dtype}>".format(type_name=type(self).__name__,
                                         params=", ".join(sorted(str(p.name) for p in self.params.values())),
                                         dtype=self.dtype.name) + str(sum(" {k}={v}".format(k=str(k), v=str(v))
                                                                          for k, v in
                                                                          self._get_additional_repr(
                                                                              sorted=True).items())))

    def _check_input_x_function(self, func):
        # TODO: signature etc?
        if not callable(func):
            raise TypeError("Function {} is not callable.")
        return func

    def _get_dependents(self) -> ztyping.DependentsType:
        return self._extract_dependents(self.get_params())

    def __add__(self, other):
        from . import operations
        return operations.add(self, other)

    def __radd__(self, other):
        from . import operations
        return operations.add(other, self)

    def __mul__(self, other):
        from . import operations
        return operations.multiply(self, other)

    def __rmul__(self, other):
        from . import operations
        return operations.multiply(other, self)


class SimpleModelSubclassMixin(pep487.ABC):
    """Subclass a model: implement the corresponding function and specify _PARAMS.

    In order to create a custom model, two things have to be implemented: the class attribute
    _PARAMS has to be a list containing the names of the parameters and the corresponding
    function (_unnormalized_pdf/_value) has to be overridden.

    Example:

    .. code:: python

        class MyPDF(zfit.pdf.ZPDF):
            _PARAMS = ['mu', 'sigma']

            def _unnormalized_pdf(self, x):
                mu = self.params['mu']
                sigma = self.params['sigma']
                x = ztf.unstack_x(x)
                return ztf.exp(-ztf.square((x - mu) / sigma))
        """

    def __init__(self, *args, **kwargs):
        try:
            params = OrderedDict((name, kwargs.pop(name)) for name in self._PARAMS)
        except KeyError:
            raise ValueError("The following parameters are not given (as keyword arguments): "
                             "".format([k for k in self._PARAMS if k not in kwargs]))
        super().__init__(params=params, *args, **kwargs)
        # super().__init__(params=params, *args, **kwargs)  # use if upper fails

    @classmethod
    def _check_simple_model_subclass(cls):
        try:
            params = cls._PARAMS
        except AttributeError:
            raise SubclassingError("Need to define `_PARAMS` in the definition of the subclass."
                                   "Example:"
                                   "class MyModel(ZPDF):"
                                   "    _PARAMS = ['mu', 'sigma']")
        not_str = [param for param in params if not isinstance(param, str)]
        if not_str:
            raise TypeError("The following parameters are not strings in `_PARAMS`: ".format(not_str))
