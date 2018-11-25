import abc
import builtins
from collections import OrderedDict
import contextlib
from contextlib import suppress
import typing
import warnings
from typing import Type, Dict, List, Union

import tensorflow as tf
from tensorflow_probability.python import mcmc as mc

from zfit.util.container import convert_to_container
from .interfaces import ZfitModel, ZfitParameter
from . import integration as zintegrate, sample as zsample
from .baseobject import BaseObject, BaseNumeric
from .limits import no_norm_range, Range, convert_to_range, supports
from .parameter import convert_to_parameter
from ..settings import types as ztypes
from ..util import container as zcontainer, ztyping
from ..util.exception import BasePDFSubclassingError, NormRangeNotImplementedError, MultipleLimitsNotImplementedError
from zfit import ztf

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
        name = func.__name__
        _BaseModel_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BaseModel_register_check_support
        return func

    return register


class BaseModel(BaseNumeric, ZfitModel):  # __init_subclass__ backport
    """Base class for any generic model.

    # TODO instructions on how to use

    """
    _DEFAULTS_integration = zcontainer.DotDict()
    _DEFAULTS_integration.mc_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 4000
    _DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(self, dims, dtype: Type = None, name: str = "BaseModel",
                 parameters: Union[Dict[str, ZfitParameter], None] = None, **kwargs):
        """The base model to inherit from and overwrite `_unnormalized_pdf`.

        Args:
            dtype (Type): the dtype of the model
            name (str): the name of the model
            parameters (): the parameters the distribution depends on
        """
        super().__init__(name=name, dtype=dtype, parameters=parameters, **kwargs)

        self.dims = convert_to_container(dims, container=tuple, convert_none=False)

        self._integration = zcontainer.DotDict()
        self._integration.mc_sampler = self._DEFAULTS_integration.mc_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.auto_numeric_integrator = self._DEFAULTS_integration.auto_numeric_integrator

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

            # if we reach this points, somethings wrong
            raise BasePDFSubclassingError("Method {} has not been correctly wrapped with @supports "
                                          "OR been been wrapped but it should not be".format(method_name))

    @abc.abstractmethod
    def _func_to_integrate(self, x: ztyping.XType) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _func_to_sample_from(self, x: ztyping.XType) -> tf.Tensor:
        raise NotImplementedError

    def set_integration_options(self, mc_options: dict = None, numeric_options: dict = None,
                                general_options: dict = None, analytic_options: dict = None):
        mc_options = {} if mc_options is None else mc_options
        numeric_options = {} if numeric_options is None else numeric_options
        general_options = {} if general_options is None else general_options
        analytic_options = {} if analytic_options is None else analytic_options
        if analytic_options:
            raise NotImplementedError("analytic_options cannot be updated currently.")
        self._integration.update(mc_options)
        self._integration.update(numeric_options)
        self._integration.update(general_options)

    @abc.abstractmethod
    def gradient(self, x: ztyping.XType, params: ztyping.ParamsType = None):
        raise NotImplementedError

    def _check_input_norm_range(self, norm_range, dims, caller_name="",
                                none_is_error=False) -> typing.Union[Range, bool]:
        """Convert to :py:class:`Range`.

        Args:
            norm_range (None or Range compatible):
            dims (tuple(int)):
            caller_name (str): name of the calling function. Used for exception message.
            none_is_error (bool): if both `norm_range` and `self.norm_range` are None, the default
                value is `False` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two `None` will raise a Value error.

        Returns:
            Union[Range, False]:

        """
        if norm_range is None:
            if none_is_error:
                raise ValueError("Normalization range `norm_range` has to be specified, (but can be None as well)"
                                 "a default normalization range. Currently, both are None/False."
                                 "".format(name=caller_name))
            else:
                norm_range = False

        return convert_to_range(limits=norm_range, dims=dims)

    @property
    def n_dims(self):
        return self._automatic_n_dims

    @property
    @abc.abstractmethod
    def _n_dims(self) -> int:
        """The number of dimensions of this model. Overwrite when subclassing."""
        raise NotImplementedError

    @property
    def _automatic_n_dims(self):  # this is in case something has changed. TODO(mayou36): handle it
        return self._n_dims

    @property
    def dims(self) -> ztyping.DimsType:
        return self._dims

    @dims.setter
    def dims(self, value: ztyping.DimsType):  # TODO: what's the default return?
        self._dims = value

    def _check_convert_input_dims(self, dims):
        if dims is None:
            return dims

        dims = convert_to_container(dims, container=tuple)
        if len(dims) > self.n_dims:
            raise ValueError("`dims` has more dims than the instance has `n_dims`. This is not possible.")
        if self.dims is None:
            return dims
        else:
            try:
                dims = tuple(self.dims.index(dim) for dim in dims)
            except ValueError:
                missing_dims = set(dims) - set(self.dims)
                raise ValueError("The following dims are not specified in the pdf: {}".format(str(missing_dims)))
        return dims

    # Integrals
    @_BaseModel_register_check_support(True)
    def _integrate(self, limits, norm_range):
        raise NotImplementedError()

    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str):

        Returns:
            Tensor: the integral value
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)
        return self._hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        integral = self._norm_integrate(limits=limits, norm_range=norm_range, name=name)
        return integral

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
            for sub_limits in limits.subbounds():
                integrals.append(self._call_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._integrate(limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._norm_analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_integrate(limits=limits, norm_range=norm_range)

    def _fallback_integrate(self, limits, norm_range):
        dims = limits.dims
        max_dims = self._analytic_integral.get_max_dims(limits=limits, dims=dims)

        integral = None
        if max_dims and integral:  # TODO improve handling of available analytic integrals
            with suppress(NotImplementedError):
                def part_int(x):
                    """Temporary partial integration function."""
                    return self._norm_partial_analytic_integrate(x, limits=limits, norm_range=norm_range)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._norm_numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    @classmethod
    def register_analytic_integral(cls, func: typing.Callable, limits: ztyping.LimitsType = None,
                                   dims: ztyping.DimsType = None, priority: int = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral with the class.

        Args:
            func ():
            limits (): |limits_arg_descr|
            dims (tuple(int)):
            priority (int):
            supports_multiple_limits (bool):
            supports_norm_range (bool):

        Returns:

        """
        cls._analytic_integral.register(func=func, dims=dims, limits=limits, supports_norm_range=supports_norm_range,
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
    def _analytic_integrate(self, limits, norm_range):
        raise NotImplementedError

    def analytic_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                           name: str = "analytic_integrate") -> ztyping.XType:
        """Do analytical integration over function and raise Error if not possible.

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value
        Raises:
            NotImplementedError: If no analytical integral is available (for this limits).
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)
        return self._hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        integral = self._norm_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_analytic_integrate(self, limits, norm_range, name='_norm_analytic_integrate'):
        try:
            integral = self._limits_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:

            unnormalized_integral = self._limits_analytic_integrate(limits, norm_range=None,
                                                                    name=name)
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
            for sub_limits in limits.subbounds():
                integrals.append(self._call_analytic_integrate(limits=sub_limits, norm_range=norm_range,
                                                               name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_analytic_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_analytic_integrate(limits=limits, norm_range=norm_range)

    def _fallback_analytic_integrate(self, limits, norm_range):
        return self._analytic_integral.integrate(x=None, limits=limits, dims=limits.dims,
                                                 norm_range=norm_range, params=self.parameters)

    @_BaseModel_register_check_support(True)
    def _numeric_integrate(self, limits, norm_range):
        raise NotImplementedError

    def numeric_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                          name: str = "numeric_integrate") -> ztyping.XType:
        """Numerical integration over the model.

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value

        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)

        return self._hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        integral = self._norm_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_numeric_integrate(self, limits, norm_range, name='_norm_numeric_integrate'):
        try:
            integral = self._limits_numeric_integrate(limits, norm_range, name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_numeric_integrate(limits=limits, norm_range=False,
                                                                   name=name)
            normalization = self._limits_numeric_integrate(limits=norm_range, norm_range=False,
                                                           name=name + "_normalization")
            integral = unnormalized_integral / normalization
        return integral

    def _limits_numeric_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_numeric_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = tf.accumulate_n(integrals)

        return integral

    def _call_numeric_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            # TODO: anything?
            with suppress(NotImplementedError):
                return self._numeric_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_numeric_integrate(limits=limits, norm_range=norm_range)

    def _fallback_numeric_integrate(self, limits, norm_range):
        integral = self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm_range=norm_range)

        return integral

    @_BaseModel_register_check_support(True)
    def _partial_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType = None,
                          norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: FULL reasonable?
        limits = convert_to_range(limits, dims=dims)

        return self._hook_partial_integrate(x=x, limits=limits,
                                            norm_range=norm_range, name=name)

    def _hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        integral = self._norm_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=False, name=name)
            normalization = self._hook_integrate(limits=norm_range, norm_range=False)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_integrate(x=x, limits=limits,
                                                    norm_range=norm_range,
                                                    name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limit in limits.subbounds():
                integrals.append(self._call_partial_integrate(x=x, limits=sub_limit, norm_range=norm_range,
                                                              name=name))
            integral = ztf.reduce_sum(integrals, axis=0)

        return integral

    def _call_partial_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = ztf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_integrate(x=x, limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

            return self._fallback_partial_integrate(x=x, limits=limits,
                                                    norm_range=norm_range)

    def _fallback_partial_integrate(self, x, limits, norm_range):
        max_dims = self._analytic_integral.get_max_dims(limits=limits, dims=limits.dims)
        if max_dims:
            sublimits = limits.subspace(max_dims)

            def part_int(x):  # change to partial integrate max dims?
                """Temporary partial integration function."""
                return self._norm_partial_analytic_integrate(x=x, limits=sublimits, norm_range=norm_range)

            dims = list(set(limits.dims) - set(max_dims))
        else:
            part_int = self._func_to_integrate
            dims = limits.dims

        if norm_range is False:
            integral_vals = self._auto_numeric_integrate(func=part_int, limits=limits, dims=dims, x=x)
        else:
            raise NormRangeNotImplementedError
        return integral_vals

    @_BaseModel_register_check_support(True)
    def _partial_analytic_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_analytic_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType,
                                   norm_range: ztyping.LimitsType = None,
                                   name: str = "partial_analytic_integrate") -> ztyping.XType:
        """Do analytical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.

        Raises:
            NotImplementedError: if the *analytic* integral (over this limits) is not implemented
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.

        """
        norm_range = self._check_input_norm_range(norm_range=norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: full reasonable?
        limits = convert_to_range(limits, dims=dims)  # TODO: replace by limits.dims if dims is None?
        return self._hook_partial_analytic_integrate(x=x, limits=limits,
                                                     norm_range=norm_range, name=name)

    def _hook_partial_analytic_integrate(self, x, limits, norm_range,
                                         name='_hook_partial_analytic_integrate'):

        integral = self._norm_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_partial_analytic_integrate(self, x, limits, norm_range, name='_norm_partial_analytic_integrate'):
        try:
            integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_analytic_integrate(x=x, limits=limits,
                                                                            norm_range=False,
                                                                            name=name)
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

    def _limits_partial_analytic_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_analytic_integrate(x=x, limits=limits,
                                                             norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_partial_analytic_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                       name=name))
            integral = ztf.reduce_sum(integrals, axis=0)

        return integral

    def _call_partial_analytic_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = ztf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits,
                                                        norm_range=norm_range)
            return self._fallback_partial_analytic_integrate(x=x, limits=limits,
                                                             norm_range=norm_range)

    def _fallback_partial_analytic_integrate(self, x, limits, norm_range):
        return self._analytic_integral.integrate(x=x, limits=limits, dims=limits.dims,
                                                 norm_range=norm_range, params=self.parameters)

    @_BaseModel_register_check_support(True)
    def _partial_numeric_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_numeric_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType,
                                  norm_range: ztyping.LimitsType = None,
                                  name: str = "partial_numeric_integrate") -> ztyping.XType:
        """Force numerical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=dims)

        return self._hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _hook_partial_numeric_integrate(self, x, limits, norm_range,
                                        name='_hook_partial_numeric_integrate'):
        integral = self._norm_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        return integral

    def _norm_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_numeric_integrate(x=x, limits=limits,
                                                              norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_numeric_integrate(x=x, limits=limits,
                                                                           norm_range=None, name=name)
            integral = unnormalized_integral / self._hook_numeric_integrate(limits=norm_range, norm_range=norm_range)
        return integral

    def _limits_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_numeric_integrate(x=x, limits=limits,
                                                            norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_partial_numeric_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                      name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_partial_numeric_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = ztf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_numeric_integrate(x=x, limits=limits,
                                                       norm_range=norm_range)
            return self._fallback_partial_numeric_integrate(x=x, limits=limits,
                                                            norm_range=norm_range)

    @no_norm_range
    def _fallback_partial_numeric_integrate(self, x, limits, norm_range=False):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, x=x)

    def _auto_numeric_integrate(self, func, limits, x=None, norm_range=False, **overwrite_options):
        integration_options = dict(func=func, limits=limits, n_dims=limits.n_dims, x=x, norm_range=norm_range,
                                   # auto from self
                                   dtype=self.dtype,
                                   mc_sampler=self._integration.mc_sampler,
                                   mc_options={
                                       "draws_per_dim": self._integration.draws_per_dim},
                                   **overwrite_options)
        return self._integration.auto_numeric_integrator(**integration_options)

    @no_norm_range
    def _inverse_analytic_integrate(self, x):
        if self._inverse_analytic_integral is None:
            raise NotImplementedError
        else:
            return self._inverse_analytic_integral[0](x=x, params=self.parameters)

    @_BaseModel_register_check_support(True)
    def _sample(self, n, limits):
        raise NotImplementedError

    def sample(self, n: int, limits: ztyping.LimitsType, name: str = "sample") -> ztyping.XType:
        """Sample `n` points within `limits` from the model.

        Args:
            n (int): The number of samples to be generated
            limits (tuple, Range): In which region to sample in
            name (str):

        Returns:
            Tensor(n_dims, n_samples)
        """
        limits = convert_to_range(limits, dims=Range.FULL)
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
            n = ztf.convert_to_tensor(n, dtype=ztypes.int, name="n")

            with suppress(NotImplementedError):
                return self._sample(n=n, limits=limits)
            with suppress(NotImplementedError):
                return self._analytic_sample(n=n, limits=limits)
            return self._fallback_sample(n=n, limits=limits)

    def _analytic_sample(self, n, limits: Range):
        if len(limits) > 1:
            raise MultipleLimitsNotImplementedError()

        for lower_bound, upper_bound in zip(*limits.get_boundaries()):
            neg_infinities = (tuple((-float("inf"),) * limits.n_dims),)  # py34 change float("inf") to math.inf
            try:
                lower_prob_lim = self._norm_analytic_integrate(limits=Range.from_boundaries(lower=neg_infinities,
                                                                                            upper=lower_bound,
                                                                                            dims=limits.dims,
                                                                                            convert_none=True),
                                                               norm_range=False)

                upper_prob_lim = self._norm_analytic_integrate(limits=Range.from_boundaries(lower=neg_infinities,
                                                                                            upper=upper_bound,
                                                                                            dims=limits.dims,
                                                                                            convert_none=True),
                                                               norm_range=False)
            except NotImplementedError:
                raise NotImplementedError("analytic sampling not possible because the analytic integral is not"
                                          "implemented for the boundaries:".format(limits.get_boundaries()))
            prob_sample = ztf.random_uniform(shape=(n, limits.n_dims), minval=lower_prob_lim,
                                             maxval=upper_prob_lim)
            sample = self._inverse_analytic_integrate(x=prob_sample)
            return sample

    def _fallback_sample(self, n, limits):
        sample = zsample.accept_reject_sample(prob=self._func_to_sample_from, n=n, limits=limits,
                                              prob_max=None)  # None -> auto
        return sample

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with tf.name_scope(self.name):
            with tf.name_scope(name, values=(
                ([] if values is None else values))) as scope:
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
                # "'{self_name}'"
                " parameters=[{params}]"
                " dtype={dtype}>".format(
            type_name=type(self).__name__,
            # self_name=self.name,
            params=", ".join(sorted(str(p.name) for p in self.parameters.values())),
            dtype=self.dtype.name) +
                str(sum(" {k}={v}".format(k=str(k), v=str(v)) for k, v in self._get_additional_repr(sorted=True).items())))

    def _check_input_x_function(self, func):
        # TODO: signature etc?
        if not callable(func):
            raise TypeError("Function {} is not callable.")
        return func

    def _get_dependents(self) -> ztyping.DependentsType:
        return self._extract_dependents(self.get_parameters())

    def __add__(self, other):
        from . import operations
        return operations.add(self, other, dims=None)

    def __radd__(self, other):
        from . import operations
        return operations.add(other, self, dims=None)

    def __mul__(self, other):
        from . import operations
        return operations.multiply(self, other, dims=None)

    def __rmul__(self, other):
        from . import operations
        return operations.multiply(other, self, dims=None)


def model_dims_mixin(n):
    class NDimensional:
        @property
        def _n_dims(self):
            return n

    return NDimensional
