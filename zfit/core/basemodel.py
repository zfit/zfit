"""Baseclass for a Model.

Handle integration and sampling
"""

#  Copyright (c) 2021 zfit

import abc
import builtins
import contextlib
import inspect
import warnings
from collections import OrderedDict
from contextlib import suppress
from typing import Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from dotmap import DotMap
from tensorflow_probability.python import mcmc as mc

import zfit.z.numpy as znp

from .. import z
from ..core.integration import Integration
from ..settings import ztypes
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.exception import (AnalyticIntegralNotImplemented,
                              AnalyticSamplingNotImplemented,
                              BasePDFSubclassingError,
                              CannotConvertToNumpyError,
                              FunctionNotImplemented,
                              MultipleLimitsNotImplemented,
                              NormRangeNotImplemented, ShapeIncompatibleError,
                              SpaceIncompatibleError,
                              SpecificFunctionNotImplemented, SubclassingError,
                              WorkInProgressError)
from . import integration as zintegrate
from . import sample as zsample
from .baseobject import BaseNumeric
from .data import Data, SampleData, Sampler
from .dependents import _extract_dependencies
from .dimension import BaseDimensional
from .interfaces import ZfitData, ZfitModel, ZfitParameter, ZfitSpace
from .sample import UniformSampleAndWeights
from .space import Space, convert_to_space, no_norm_range, supports

_BaseModel_USER_IMPL_METHODS_TO_CHECK = {}


def _BaseModel_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support: If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.
    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func:

        Returns:
            Function:
        """
        name = func.__name__
        _BaseModel_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BaseModel_register_check_support
        return func

    return register


class BaseModel(BaseNumeric, GraphCachable, BaseDimensional, ZfitModel):
    """Base class for any generic model.

    # TODO instructions on how to use
    """
    _DEFAULTS_integration = DotMap()
    _DEFAULTS_integration.mc_sampler = lambda *args, **kwargs: mc.sample_halton_sequence(*args, randomized=False,
                                                                                         **kwargs)
    # _DEFAULTS_integration.mc_sampler = lambda dim, num_results, dtype: tf.random_uniform(maxval=1.,
    #                                                                                      shape=(num_results, dim),
    #                                                                                      dtype=dtype)
    _DEFAULTS_integration.draws_per_dim = 40000
    _DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(self, obs: ztyping.ObsTypeInput, params: Union[Dict[str, ZfitParameter], None] = None,
                 name: str = "BaseModel", dtype=ztypes.float,
                 **kwargs):
        """The base model to inherit from and overwrite `_unnormalized_pdf`.

        Args:
            dtype: the dtype of the model
            name: the name of the model
            params: A dictionary with the internal name of the parameter and
                the parameters itself the model depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, **kwargs)
        self._check_set_space(obs)

        self._integration = DotMap()
        self._integration.auto_numeric_integrator = self._DEFAULTS_integration.auto_numeric_integrator
        self.integration = Integration(mc_sampler=self._DEFAULTS_integration.mc_sampler,
                                       draws_per_dim=self._DEFAULTS_integration.draws_per_dim)

        self._sample_and_weights = UniformSampleAndWeights

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

    # since subclasses can be funcs of pdfs, we need to now what to sample/integrate from
    @abc.abstractmethod
    def _func_to_integrate(self, x: ztyping.XType) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _func_to_sample_from(self, x: ztyping.XType) -> Data:
        raise NotImplementedError

    @property
    def space(self) -> ZfitSpace:
        return self._space

    def _check_set_space(self, obs):
        if not isinstance(obs, ZfitSpace):
            obs = Space(obs=obs)
        self._check_n_obs(space=obs)
        self._space = obs.with_autofill_axes(overwrite=True)

    @contextlib.contextmanager
    def _convert_sort_x(self,
                        x: ztyping.XTypeInput,
                        partial: bool = False,
                        allow_none: bool = False) -> Data:
        if x is None:
            if not allow_none:
                raise ValueError(f"x {x} given to {self} must be non-empty (not None).")
            else:
                yield None

        elif isinstance(x, ZfitData):
            if x.obs is not None:
                with x.sort_by_obs(obs=self.obs, allow_superset=True):
                    yield x
            elif x.axes is not None:
                with x.sort_by_axes(axes=self.axes):
                    yield x
            else:
                assert False, "Neither the `obs` nor the `axes` are specified in `Data`"
        else:
            if not isinstance(x, (tf.Tensor, tf.Variable)):
                try:
                    x = z.convert_to_tensor(value=x)
                except TypeError:
                    raise TypeError(f"Wrong type of x ({type(x)}). Has to be a `Data` or convertible to a tf.Tensor")
            # check dimension
            x = self._add_dim_to_x(x=x)
            x_shape = x.shape.as_list()[-1]
            if not partial and x_shape != self.n_obs:
                raise ShapeIncompatibleError("The shape of x (={}) (in the last dim) does not"
                                             "match the shape (={})of the model".format(x_shape, self.n_obs))
            x = Data.from_tensor(obs=self.obs, tensor=x)
            yield x

    def _add_dim_to_x(self, x):  # TODO(Mayou36): remove function? unnecessary? dealt with in `Data`?
        if self.n_obs == 1:
            if len(x.shape.as_list()) == 0:
                x = znp.expand_dims(x, -1)
            if len(x.shape.as_list()) == 1:
                x = znp.expand_dims(x, -1)
        return x

    def update_integration_options(self, draws_per_dim=None, mc_sampler=None):
        """Set the integration options.

        Args:
            draws_per_dim: The draws for MC integration to do
            mc_sampler:
        """
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

    # TODO: remove below? or add "analytic gradients"?
    def gradient(self, x: ztyping.XType, norm_range: ztyping.LimitsType, params: ztyping.ParamsTypeOpt = None):
        raise NotImplementedError("Are the gradients needed?")

    def _check_input_norm_range(self, norm_range, none_is_error=False) -> Optional[ZfitSpace]:
        """Convert to :py:class:`~zfit.Space`.

        Args:
            norm_range:
            none_is_error: if both `norm_range` and `self.norm_range` are None, the default
                value is `False` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two `None` will raise a Value error.

        Returns:
            Union[:py:class:`~zfit.Space`, False]:
        """
        if norm_range is None or (isinstance(norm_range, ZfitSpace) and not norm_range.limits_are_set):
            if none_is_error:
                raise ValueError("Normalization range `norm_range` has to be specified or"
                                 "a default normalization range has to be set. Currently, both are None")

        return self.convert_sort_space(limits=norm_range)

    def _check_input_limits(self, limits, none_is_error=False):
        if limits is None or (isinstance(limits, ZfitSpace) and not limits.has_limits):
            if none_is_error:
                raise ValueError("The `limits` have to be specified and not be None")
            # else:
            #     limits = False

        return self.convert_sort_space(limits=limits)

    def convert_sort_space(self, obs: Union[ztyping.ObsTypeInput, ztyping.LimitsTypeInput] = None,
                           axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> Union[ZfitSpace, None]:
        """Convert the inputs (using eventually `obs`, `axes`) to
        :py:class:`~zfit.ZfitSpace` and sort them according to own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        elif not set(obs).intersection(self.obs):
            raise SpaceIncompatibleError("The given space {obs} is not compatible with the obs of the pdfs{self.obs};"
                                         " they are disjoint.")
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:  # e.g. not the first call
            space = space.with_coords(self.space, allow_superset=True, allow_subset=True)
        return space

    # Integrals

    @_BaseModel_register_check_support(True)
    def _integrate(self, limits, norm_range):
        raise SpecificFunctionNotImplemented

    @z.function(wraps='model')
    def integrate(self,
                  limits: ztyping.LimitsType,
                  norm_range: ztyping.LimitsType = None,
                  *,
                  x: Optional[ztyping.DataInputType] = None) -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits: the limits to integrate over
            norm_range: the limits to normalize over or False to integrate the
                unnormalized probability

        Returns:
            The integral value as a scalar with shape ()
        """
        norm_range = self._check_input_norm_range(norm_range)
        limits = self._check_input_limits(limits=limits)
        with self._convert_sort_x(x, allow_none=True) as x:
            integral = self._single_hook_integrate(limits=limits, norm_range=norm_range, x=x)
        # TODO: allow integral values as arrays?
        # if isinstance(integral, tf.Tensor):
        #     if not integral.shape.as_list() == []:
        #         raise ShapeIncompatibleError("Error in integral creation, should return an integral "
        #                                      "with shape () (resp. [] as list), current shape "
        #                                      "{}. If you registered an analytic integral which is used"
        #                                      "now, make sure to return a scalar and not a tensor "
        #                                      "(typically: shape is (1,) insead of () -> return tensor[0] "
        #                                      "instead of tensor)".format(integral.shape.as_list()))
        return integral

    def _single_hook_integrate(self, limits, norm_range, x):
        del x  # TODO HACK: how and what to pass through?
        return self._hook_integrate(limits=limits, norm_range=norm_range)

    def _hook_integrate(self, limits, norm_range):
        return self._norm_integrate(limits=limits, norm_range=norm_range)

    def _norm_integrate(self, limits, norm_range):
        try:
            integral = self._limits_integrate(limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:
            unnormalized_integral = self._limits_integrate(limits=limits, norm_range=False)
            normalization = self._limits_integrate(limits=norm_range, norm_range=False)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_integrate(self, limits, norm_range):
        try:
            integral = self._call_integrate(limits=limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_integrate(limits=sub_limits, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)  # TODO: remove stack?
        return integral

    def _call_integrate(self, limits, norm_range):

        with suppress(FunctionNotImplemented):
            return self._integrate(limits=limits, norm_range=norm_range)
        with suppress(AnalyticIntegralNotImplemented):
            return self._hook_analytic_integrate(limits=limits, norm_range=norm_range)
        return self._fallback_integrate(limits=limits, norm_range=norm_range)

    def _fallback_integrate(self, limits, norm_range):
        axes = limits.axes
        max_axes = self._analytic_integral.get_max_axes(limits=limits, axes=axes)

        integral = None
        if max_axes and integral is None:  # TODO improve handling of available analytic integrals
            with suppress(AnalyticIntegralNotImplemented):
                def part_int(x):
                    """Temporary partial integration function."""
                    return self._hook_partial_analytic_integrate(x, limits=limits, norm_range=norm_range)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._hook_numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    @classmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None,
                                   priority: Union[int, float] = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral with the class.

        Args:
            func: A function that calculates the (partial) integral over the axes `limits`.
                The signature has to be the following:

                    * x (:py:class:`~zfit.core.interfaces.ZfitData`, None): the data for the remaining axes in a partial
                        integral. If it is not a partial integral, this will be None.
                    * limits (:py:class:`~zfit.ZfitSpace`): the limits to integrate over.
                    * norm_range (:py:class:`~zfit.ZfitSpace`, None): Normalization range of the integral.
                        If not `supports_supports_norm_range`, this will be None.
                    * params (Dict[param_name, :py:class:`zfit.Parameters`]): The parameters of the model.
                    * model (:py:class:`~zfit.core.interfaces.ZfitModel`):The model that is being integrated.

            limits: |limits_arg_descr|
            priority: Priority of the function. If multiple functions cover the same space, the one with the
                highest priority will be used.
            supports_multiple_limits: If `True`, the `limits` given to the integration function can have
                multiple limits. If `False`, only simple limits will pass through and multiple limits will be
                auto-handled.
            supports_norm_range: If `True`, `norm_range` argument to the function may not be `None`.
                If `False`, `norm_range` will always be `None` and care is taken of the normalization automatically.
        """
        cls._analytic_integral.register(func=func, limits=limits, supports_norm_range=supports_norm_range,
                                        priority=priority, supports_multiple_limits=supports_multiple_limits)

    @classmethod
    def register_inverse_analytic_integral(cls, func: Callable) -> None:
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func: A function with the signature `func(x, params)`, where `x` is a Data object
                and `params` is a dict.
        """
        if cls._inverse_analytic_integral:
            cls._inverse_analytic_integral[0] = func
        else:
            cls._inverse_analytic_integral.append(func)

    @_BaseModel_register_check_support(True)
    def _analytic_integrate(self, limits, norm_range):
        raise SpecificFunctionNotImplemented

    def analytic_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        """Analytical integration over function and raise Error if not possible.

        Args:
            limits: the limits to integrate over
            norm_range: the limits to normalize over

        Returns:
            The integral value
        Raises:
            AnalyticIntegralNotImplementedError: If no analytical integral is available (for this limits).
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.
        """
        norm_range = self._check_input_norm_range(norm_range)
        limits = self._check_input_limits(limits=limits)
        return self._single_hook_analytic_integrate(limits=limits, norm_range=norm_range)

    def _single_hook_analytic_integrate(self, limits, norm_range):
        return self._hook_analytic_integrate(limits=limits, norm_range=norm_range)

    def _hook_analytic_integrate(self, limits, norm_range):
        return self._norm_analytic_integrate(limits=limits, norm_range=norm_range)

    def _norm_analytic_integrate(self, limits, norm_range):
        try:
            integral = self._limits_analytic_integrate(limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:

            unnormalized_integral = self._limits_analytic_integrate(limits, norm_range=False)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False)
            except (AnalyticIntegralNotImplemented):
                raise NormRangeNotImplemented("Function does not support this (or even any)"
                                              "normalization range 'norm_range'."
                                              " This usually means,that no analytic integral "
                                              "is available for this function. Due to rule "
                                              "safety, an analytical normalization has to "
                                              "be available and no attempt of numerical "
                                              "normalization was made.")
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_analytic_integrate(self, limits, norm_range):
        try:
            integral = self._call_analytic_integrate(limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_analytic_integrate(limits=sub_limits, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)
        return integral

    def _call_analytic_integrate(self, limits, norm_range):
        with suppress(FunctionNotImplemented, AnalyticIntegralNotImplemented):
            return self._analytic_integrate(limits=limits, norm_range=norm_range)
        return self._fallback_analytic_integrate(limits=limits, norm_range=norm_range)

    def _fallback_analytic_integrate(self, limits, norm_range):
        try:
            return self._analytic_integral.integrate(x=None, limits=limits, axes=limits.axes,
                                                     norm_range=norm_range, model=self, params=self.params)
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            raise AnalyticIntegralNotImplemented

    @_BaseModel_register_check_support(True)
    def _numeric_integrate(self, limits, norm_range):
        raise SpecificFunctionNotImplemented

    def numeric_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        """Numerical integration over the model.

        Args:
            limits: the limits to integrate over
            norm_range: the limits to normalize over

        Returns:
            The integral value
        """
        norm_range = self._check_input_norm_range(norm_range)
        limits = self._check_input_limits(limits=limits)

        return self._single_hook_numeric_integrate(limits=limits, norm_range=norm_range)

    def _single_hook_numeric_integrate(self, limits, norm_range):
        return self._hook_numeric_integrate(limits=limits, norm_range=norm_range)

    def _hook_numeric_integrate(self, limits, norm_range):
        return self._norm_numeric_integrate(limits=limits, norm_range=norm_range)

    def _norm_numeric_integrate(self, limits, norm_range):
        try:
            integral = self._limits_numeric_integrate(limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:
            assert not norm_range.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_numeric_integrate(limits=limits, norm_range=False)
            normalization = self._limits_numeric_integrate(limits=norm_range, norm_range=False)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_numeric_integrate(self, limits, norm_range):
        try:
            integral = self._call_numeric_integrate(limits=limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_numeric_integrate(limits=sub_limits, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_numeric_integrate(self, limits, norm_range):
        with suppress(FunctionNotImplemented):
            return self._numeric_integrate(limits=limits, norm_range=norm_range)
        return self._fallback_numeric_integrate(limits=limits, norm_range=norm_range)

    def _fallback_numeric_integrate(self, limits, norm_range):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm_range=norm_range)

    @_BaseModel_register_check_support(True)
    def _partial_integrate(self, x, limits, norm_range):
        raise SpecificFunctionNotImplemented

    @z.function(wraps='model')
    def partial_integrate(self, x: ztyping.XTypeInput, limits: ztyping.LimitsType,
                          norm_range: ztyping.LimitsType = None) -> ztyping.XTypeReturn:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: the limits to integrate over. Can contain only some axes
            norm_range: the limits to normalize over. Has to have all axes

        Returns:
            The value of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)
        limits = self._check_input_limits(limits=limits)
        with self._convert_sort_x(x, partial=True) as x:
            return self._single_hook_partial_integrate(x=x, limits=limits, norm_range=norm_range)

    def _single_hook_partial_integrate(self, x, limits, norm_range):
        return self._hook_partial_integrate(x=x, limits=limits, norm_range=norm_range)

    def _hook_partial_integrate(self, x, limits, norm_range):
        integral = self._norm_partial_integrate(x=x, limits=limits, norm_range=norm_range)
        return integral

    def _norm_partial_integrate(self, x, limits, norm_range):
        try:
            integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:
            assert not norm_range.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=False)
            normalization = self._hook_integrate(limits=norm_range, norm_range=False)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_integrate(self, x, limits, norm_range):
        try:
            integral = self._call_partial_integrate(x=x, limits=limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limit in limits:
                integrals.append(self._call_partial_integrate(x=x, limits=sub_limit, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_partial_integrate(self, x, limits, norm_range):

        with suppress(FunctionNotImplemented):
            return self._partial_integrate(x=x, limits=limits, norm_range=norm_range)
        with suppress(AnalyticIntegralNotImplemented):
            return self._hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)
        try:
            return self._fallback_partial_integrate(x=x, limits=limits, norm_range=norm_range)
        except FunctionNotImplemented:
            raise AnalyticIntegralNotImplemented

    def _fallback_partial_integrate(self, x, limits: ZfitSpace, norm_range: ZfitSpace):
        max_axes = self._analytic_integral.get_max_axes(limits=limits, axes=limits.axes)
        if max_axes:
            sublimits = limits.get_subspace(axes=max_axes)

            def part_int(x):  # change to partial integrate max axes?
                """Temporary partial integration function."""
                return self._hook_partial_analytic_integrate(x=x, limits=sublimits, norm_range=norm_range)

            axes = list(set(limits.axes) - set(max_axes))
            limits = limits.get_subspace(axes=axes)
        else:
            part_int = self._func_to_integrate

        assert limits.axes, "Internal Error! Axes should not be empty, maybe cleanup."
        integral_vals = self._auto_numeric_integrate(func=part_int, limits=limits, x=x, norm_range=norm_range)

        return integral_vals

    @_BaseModel_register_check_support(True)
    def _partial_analytic_integrate(self, x, limits, norm_range):
        raise SpecificFunctionNotImplemented

    @z.function(wraps='model')
    def partial_analytic_integrate(self, x: ztyping.XTypeInput, limits: ztyping.LimitsType,
                                   norm_range: ztyping.LimitsType = None) -> ztyping.XTypeReturn:
        """Do analytical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: the limits to integrate over. Can contain only some axes
            norm_range: the limits to normalize over. Has to have all axes

        Returns:
            The value of the partially integrated function evaluated at `x`.

        Raises:
            AnalyticIntegralNotImplementedError: if the *analytic* integral (over this limits) is not implemented
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)
        limits = self._check_input_limits(limits=limits)
        with self._convert_sort_x(x, partial=True) as x:
            return self._single_hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range):
        return self._hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

    def _hook_partial_analytic_integrate(self, x, limits, norm_range):
        return self._norm_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

    def _norm_partial_analytic_integrate(self, x, limits, norm_range):
        try:
            integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:
            assert not norm_range.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=False)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False)
            except AnalyticIntegralNotImplemented:
                raise NormRangeNotImplemented("Function does not support this (or even any) normalization range"
                                              " 'norm_range'. This usually means,that no analytic integral "
                                              "is available for this function. An analytical normalization has to "
                                              "be available and no attempt of numerical normalization was made.")
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_analytic_integrate(self, x, limits, norm_range):
        try:
            integral = self._call_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_partial_analytic_integrate(x=x, limits=sub_limits, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_partial_analytic_integrate(self, x, limits, norm_range):
        with suppress(FunctionNotImplemented, AnalyticIntegralNotImplemented):
            return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)
        return self._fallback_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

    def _fallback_partial_analytic_integrate(self, x, limits, norm_range):
        try:
            return self._analytic_integral.integrate(x=x, limits=limits, axes=limits.axes,
                                                     norm_range=norm_range, model=self, params=self.params)
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            raise AnalyticIntegralNotImplemented

    @_BaseModel_register_check_support(True)
    def _partial_numeric_integrate(self, x, limits, norm_range):
        raise SpecificFunctionNotImplemented

    @z.function(wraps='model')
    def partial_numeric_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType,
                                  norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        """Force numerical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: the limits to integrate over. Can contain only some axes
            norm_range: the limits to normalize over. Has to have all axes

        Returns:
            The value of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range)
        limits = self._check_input_limits(limits=limits)
        with self._convert_sort_x(x, partial=True) as x:
            return self._single_hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range):
        return self._hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)

    def _hook_partial_numeric_integrate(self, x, limits, norm_range):
        integral = self._norm_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)
        return integral

    def _norm_partial_numeric_integrate(self, x, limits, norm_range):
        try:
            integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)
        except NormRangeNotImplemented:
            assert not norm_range.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm_range=False)
            integral = unnormalized_integral / self._hook_numeric_integrate(limits=norm_range, norm_range=norm_range)
        return integral

    def _limits_partial_numeric_integrate(self, x, limits, norm_range):
        try:
            integral = self._call_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_partial_numeric_integrate(x=x, limits=sub_limits, norm_range=norm_range))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)
        return integral

    def _call_partial_numeric_integrate(self, x, limits, norm_range):
        with suppress(SpecificFunctionNotImplemented):
            return self._partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)
        return self._fallback_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range)

    def _fallback_partial_numeric_integrate(self, x, limits, norm_range=False):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm_range=norm_range, x=x)

    @no_norm_range
    @z.function(wraps='model')
    def _auto_numeric_integrate(self, func, limits, x=None, norm_range=False, **overwrite_options):
        integration_options = dict(func=func, limits=limits, n_axes=limits.n_obs, x=x, norm_range=norm_range,
                                   # auto from self
                                   dtype=self.dtype,
                                   mc_sampler=self.integration.mc_sampler,
                                   mc_options={
                                       "draws_per_dim": self.integration.draws_per_dim},
                                   **overwrite_options)
        return self._integration.auto_numeric_integrator(**integration_options)

    @supports()
    def _inverse_analytic_integrate(self, x):
        if not self._inverse_analytic_integral:
            raise AnalyticSamplingNotImplemented
        else:
            icdf = self._inverse_analytic_integral[0]
            params = inspect.signature(icdf).parameters
            if len(params) == 2:
                return icdf(x=x, params=self.params)
            elif len(params) == 3:
                return icdf(x=x, params=self.params, model=self)
            else:
                raise RuntimeError(f"icdf function does not have the right signature: {icdf}")

    def create_sampler(self, n: ztyping.nSamplingTypeIn = None, limits: ztyping.LimitsType = None,
                       fixed_params: Union[bool, List[ZfitParameter], Tuple[ZfitParameter]] = True) -> "Sampler":
        """Create a :py:class:`Sampler` that acts as `Data` but can be resampled, also with changed parameters and n.

            If `limits` is not specified, `space` is used (if the space contains limits).
            If `n` is None and the model is an extended pdf, 'extended' is used by default.


        Args:
            n: The number of samples to be generated. Can be a Tensor that will be
                or a valid string. Currently implemented:

                    - 'extended': samples `poisson(yield)` from each pdf that is extended.

            limits: From which space to sample.
            fixed_params: A list of `Parameters` that will be fixed during several `resample` calls.
                If True, all are fixed, if False, all are floating. If a :py:class:`~zfit.Parameter` is not fixed and
                its
                value gets updated (e.g. by a `Parameter.set_value()` call), this will be reflected in
                `resample`. If fixed, the Parameter will still have the same value as the `Sampler` has
                been created with when it resamples.

        Returns:
            :py:class:`~zfit.core.data.Sampler`

        Raises:
            NotExtendedPDFError: if 'extended' is chosen (implicitly by default or explicitly) as an
                option for `n` but the pdf itself is not extended.
            ValueError: if n is an invalid string option.
            InvalidArgumentError: if n is not specified and pdf is not extended.
        """

        limits = self._check_input_limits(limits=limits)

        if not limits.limits_are_set:
            limits = self.space  # TODO(Mayou36): clean up, better norm_range?
            if not limits.has_limits:
                raise ValueError("limits are False/None, have to be specified")

        if fixed_params is True:
            fixed_params = list(self.get_cache_deps(only_floating=False))
        elif fixed_params is False:
            fixed_params = []
        elif not isinstance(fixed_params, (list, tuple)):
            raise TypeError("`Fixed_params` has to be a list, tuple or a boolean.")

        def sample_func(n=n):
            return self._create_sampler_tensor(limits=limits, n=n)

        sample_data = Sampler.from_sample(sample_func=sample_func, n=n, obs=limits, fixed_params=fixed_params,
                                          dtype=self.dtype)

        return sample_data

    @z.function(wraps='model')
    def _create_sampler_tensor(self, limits, n):

        sample = self._single_hook_sample(n=n, limits=limits, x=None)
        return sample

    @_BaseModel_register_check_support(True)
    def _sample(self, n, limits: ZfitSpace):
        raise SpecificFunctionNotImplemented

    def sample(self, n: ztyping.nSamplingTypeIn = None,
               limits: ztyping.LimitsType = None,
               x: Optional[
                   ztyping.DataInputType] = None) -> SampleData:  # TODO: change poissonian top-level with multinomial
        """Sample `n` points within `limits` from the model.

        If `limits` is not specified, `space` is used (if the space contains limits).
        If `n` is None and the model is an extended pdf, 'extended' is used by default.

        Args:
            n: The number of samples to be generated. Can be a Tensor that will be
                or a valid string. Currently implemented:

                    - 'extended': samples `poisson(yield)` from each pdf that is extended.
            limits: In which region to sample in

        Returns:
            SampleData(n_obs, n_samples)

        Raises:
            NotExtendedPDFError: if 'extended' is (implicitly by default or explicitly) chosen as an
                option for `n` but the pdf itself is not extended.
            ValueError: if n is an invalid string option.
            InvalidArgumentError: if n is not specified and pdf is not extended.
        """
        if not isinstance(n, str):
            n = tf.convert_to_tensor(n)
            n = tf.cast(n, dtype=tf.int32)

        limits = self._check_input_limits(limits=limits)
        if not limits.limits_are_set:
            limits = self.space
            if not limits.has_limits:
                raise tf.errors.InvalidArgumentError("limits are False/None, have to be specified")
        limits = self._check_input_limits(limits=limits, none_is_error=True)

        @z.function(wraps='model_sampling')
        def run_tf(n, limits, x):
            sample = self._single_hook_sample(n=n, limits=limits, x=x)
            return sample

        with self._convert_sort_x(x, allow_none=True) as x:
            if x is not None:
                new_obs = limits * x.data_range
            else:
                new_obs = limits
            sample_data = SampleData.from_sample(sample=run_tf(n=n, limits=limits, x=x),
                                                 obs=new_obs)  # TODO: which limits?

        return sample_data

    def _single_hook_sample(self, n, limits, x=None):
        del x
        return self._hook_sample(n=n, limits=limits)

    def _hook_sample(self, limits, n):
        return self._norm_sample(n=n, limits=limits)

    def _norm_sample(self, n, limits):
        """Dummy function."""
        return self._limits_sample(n=n, limits=limits)

    def _limits_sample(self, n, limits):
        try:
            return self._call_sample(n=n, limits=limits)
        except MultipleLimitsNotImplemented as error:
            try:
                total_integral = self.analytic_integrate(limits, norm_range=False)
                sub_integrals = znp.concatenate([self.analytic_integrate(limit, norm_range=False) for limit in limits],
                                          axis=0)
            except AnalyticIntegralNotImplemented:
                raise MultipleLimitsNotImplemented("Cannot autohandle multiple limits as the analytic"
                                                   " integral is not available.") from error
            fracs = sub_integrals / total_integral
            n_samples = tf.unstack(z.random.counts_multinomial(n, probs=fracs), axis=0)

            samples = []
            for limit, n_sample in zip(limits, n_samples):
                sub_sample = self._call_sample(n=n_sample, limits=limit)
                if isinstance(sub_sample, ZfitData):
                    sub_sample = sub_sample.value()
                samples.append(sub_sample)
            sample = znp.concatenate(samples, axis=0)

        return sample

    def _call_sample(self, n, limits):
        with suppress(SpecificFunctionNotImplemented):
            return self._sample(n=n, limits=limits)
        with suppress(SpecificFunctionNotImplemented, AnalyticSamplingNotImplemented):
            return self._analytic_sample(n=n, limits=limits)
        return self._fallback_sample(n=n, limits=limits)

    def _analytic_sample(self, n, limits: ZfitSpace):  # TODO(Mayou36) implement multiple limits sampling
        if not self._inverse_analytic_integral:
            raise AnalyticSamplingNotImplemented  # TODO(Mayou36): create proper analytic sampling
        if limits.n_limits > 1:
            raise AnalyticSamplingNotImplemented
        try:
            lower_bound, upper_bound = limits.rect_limits_np
        except CannotConvertToNumpyError as err:
            raise WorkInProgressError("Currently, analytic sampling with Tensors not supported."
                                      " Needs implementation of analytic integrals with Tensors.") from err
        neg_infinities = (tuple((-float("inf"),) * limits.n_obs),)  # py34 change float("inf") to math.inf
        # to the cdf to get the limits for the inverse analytic integral
        try:
            lower_prob_lim = self._norm_analytic_integrate(limits=Space(limits=(neg_infinities,
                                                                                (lower_bound,)),
                                                                        axes=limits.axes),
                                                           norm_range=False)

            upper_prob_lim = self._norm_analytic_integrate(limits=Space(limits=(neg_infinities,
                                                                                (upper_bound,)),
                                                                        axes=limits.axes),
                                                           norm_range=False)
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            raise AnalyticSamplingNotImplemented(f"analytic sampling not possible because the analytic integral"
                                                 f" is not"" implemented for the boundaries: {limits}")
        prob_sample = z.random.uniform(shape=(n, limits.n_obs), minval=lower_prob_lim,
                                       maxval=upper_prob_lim)
        # with self._convert_sort_x(prob_sample) as x:
        x = prob_sample
        sample = self._inverse_analytic_integrate(x=x)
        return sample

    def _fallback_sample(self, n, limits):
        sample = zsample.accept_reject_sample(prob=self._func_to_sample_from, n=n, limits=limits,
                                              prob_max=None, dtype=self.dtype,
                                              sample_and_weights_factory=self._sample_and_weights)
        return sample

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
                " params=[{params}]".format(type_name=type(self),
                                            params=", ".join(sorted(str(p.name) for p in self.params.values())),
                                            ))

    def _check_input_x_function(self, func):
        # TODO: signature etc?
        if not callable(func):
            raise TypeError("Function {} is not callable.")
        return func

    def _get_dependencies(self) -> ztyping.DependentsType:
        return _extract_dependencies(self.get_params())

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


class SimpleModelSubclassMixin:
    """Subclass a model: implement the corresponding function and specify _PARAMS.

    In order to create a custom model, two things have to be implemented: the class attribute
    _PARAMS has to be a list containing the names of the parameters and the corresponding
    function (_unnormalized_pdf/_func) has to be overridden.

    Example:

    .. code:: python

        class MyPDF(zfit.pdf.ZPDF):
            _PARAMS = ['mu', 'sigma']

            def _unnormalized_pdf(self, x):
                mu = self.params['mu']
                sigma = self.params['sigma']
                x = z.unstack_x(x)
                return z.exp(-z.square((x - mu) / sigma))
    """

    def __init__(self, *args, **kwargs):
        try:
            params = OrderedDict((name, kwargs.pop(name)) for name in self._PARAMS)
        except KeyError:
            raise ValueError("The following parameters are not given (as keyword arguments): {}"
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
            raise TypeError(f"The following parameters are not strings in `_PARAMS`: ")
