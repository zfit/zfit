"""Baseclass for a Model.

Handle integration and sampling
"""

#  Copyright (c) 2024 zfit

from __future__ import annotations

import abc
import builtins
import contextlib
import inspect
import math
import warnings
from collections.abc import Callable
from contextlib import suppress
from typing import Mapping, Optional

import tensorflow as tf
from dotmap import DotMap
from tensorflow_probability.python import mcmc as mc

import zfit.z.numpy as znp

from .. import z
from ..core.integration import Integration
from ..settings import ztypes
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.deprecation import deprecated_args, deprecated_norm_range
from ..util.exception import (
    AnalyticIntegralNotImplemented,
    AnalyticSamplingNotImplemented,
    BasePDFSubclassingError,
    BreakingAPIChangeError,
    CannotConvertToNumpyError,
    FunctionNotImplemented,
    MultipleLimitsNotImplemented,
    NormRangeNotImplemented,
    SpaceIncompatibleError,
    SpecificFunctionNotImplemented,
    SubclassingError,
    WorkInProgressError,
)
from . import integration as zintegrate
from . import sample as zsample
from .baseobject import BaseNumeric
from .data import Data, SamplerData, convert_to_data
from .dimension import BaseDimensional
from .interfaces import ZfitData, ZfitModel, ZfitParameter, ZfitSpace
from .sample import UniformSampleAndWeights
from .space import Space, convert_to_space, supports

_BaseModel_USER_IMPL_METHODS_TO_CHECK = {}


def _BaseModel_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the ``@supports`` decorator.

    Args:
        has_support: If True, flags that it **requires** the ``@supports`` decorator. If False,
            flags that the ``@supports`` decorator is **not allowed**.
    """
    if not isinstance(has_support, bool):
        msg = "Has to be boolean."
        raise TypeError(msg)

    def register(func):
        """Register a method to be checked to (if True) *has* ``support`` or (if False) has *no* ``support``.

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


class ParamArgsNotImplemented(Exception):
    pass


class BaseModel(BaseNumeric, GraphCachable, BaseDimensional, ZfitModel):
    """Base class for any generic model.

    # TODO instructions on how to use
    """

    DEFAULTS_integration = DotMap()
    DEFAULTS_integration.mc_sampler = lambda *args, **kwargs: mc.sample_halton_sequence(
        *args, randomized=False, **kwargs
    )
    DEFAULTS_integration.draws_per_dim = "auto"
    DEFAULTS_integration.draws_simpson = "auto"
    DEFAULTS_integration.max_draws = 1_200_000
    DEFAULTS_integration.tol = 3e-6
    DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        params: dict[str, ZfitParameter] | None = None,
        name: str = "BaseModel",
        dtype=ztypes.float,
        **kwargs,
    ):
        """The base model to inherit from and overwrite ``_unnormalized_pdf``.

        Args:
            dtype: the dtype of the model
            name: the name of the model
            params: A dictionary with the internal name of the parameter and
                the parameters itself the model depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, **kwargs)
        self._check_set_space(obs)

        self._integration = DotMap()
        self._integration.auto_numeric_integrator = self.DEFAULTS_integration.auto_numeric_integrator
        self.integration = Integration(
            mc_sampler=self.DEFAULTS_integration.mc_sampler,
            draws_per_dim=self.DEFAULTS_integration.draws_per_dim,
            max_draws=self.DEFAULTS_integration.max_draws,
            tol=self.DEFAULTS_integration.tol,
            draws_simpson=None,
        )
        self.update_integration_options(
            draws_per_dim=self.DEFAULTS_integration.draws_per_dim,
            draws_simpson=self.DEFAULTS_integration.draws_simpson,
        )

        self._sample_and_weights = UniformSampleAndWeights

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if subclass has decorator if required
        cls._subclass_check_support(
            methods_to_check=_BaseModel_USER_IMPL_METHODS_TO_CHECK,
            wrapper_not_overwritten=_BaseModel_register_check_support,
        )
        cls._analytic_integral = zintegrate.AnalyticIntegral()
        cls._inverse_analytic_integral = []
        cls._additional_repr = {}

    @classmethod
    def _subclass_check_support(cls, methods_to_check, wrapper_not_overwritten):
        for method_name, has_support in methods_to_check.items():
            method = getattr(cls, method_name)
            if hasattr(method, "__wrapped__") and method.__wrapped__ == wrapper_not_overwritten:
                continue  # not overwritten, fine

            # here means: overwritten
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == supports:
                    if has_support:
                        continue  # needs support, has been wrapped

                    msg = (
                        f"Method {method_name} has been wrapped with supports but"
                        f" is not allowed to. Has to handle all arguments."
                    )
                    raise BasePDFSubclassingError(msg)
                if has_support:
                    msg = (
                        f"Method {method_name} has been overwritten and *has to* be"
                        f" wrapped by `supports` decorator (don't forget () ) to call the decorator"
                        f" as it takes arguments"
                    )
                    raise BasePDFSubclassingError(msg)
                if not has_support:
                    continue  # no support, has not been wrapped with
            elif not has_support:
                continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            if method_name not in ["_pdf"]:
                msg = (
                    f"Method {method_name} has not been correctly wrapped with @supports "
                    "OR has been wrapped but it should not be"
                )
                raise BasePDFSubclassingError(msg)
            warnings.warn(
                "For the future, also decorate _pdf with @supports and specify what you support"
                " (such as 'norm=True' to keep the same behavior as before)",
                FutureWarning,
                stacklevel=2,
            )

    # since subclasses can be funcs of pdfs, we need to now what to sample/integrate from
    @abc.abstractmethod
    def _func_to_integrate(self, x: ztyping.XType, *, params=None) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _func_to_sample_from(self, x: ztyping.XType, *, params=None) -> Data:
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
    def _convert_sort_x(
        self, x: ztyping.XTypeInput, partial: bool = False, allow_none: bool = False, fallback_obs=None
    ) -> Data:
        del partial  # TODO: implement partial
        fallback_obs = self.obs if fallback_obs is None else fallback_obs
        if x is None:
            if not allow_none:
                msg = f"x {x} given to {self} must be non-empty (not None)."
                raise ValueError(msg)
            else:
                yield None
        else:
            x = convert_to_data(x, obs=fallback_obs)
            if x.obs is not None:
                x = x.with_obs(self.obs)
                # with x.sort_by_obs(obs=self.obs, allow_superset=True):
                yield x
            elif x.axes is not None:
                x = x.with_axes(self.space.axes)
                # with x.sort_by_axes(axes=self.axes):
                yield x
            else:
                msg = "Neither the `obs` nor the `axes` are specified in `Data`"
                raise AssertionError(msg)

    def update_integration_options(
        self,
        draws_per_dim=None,
        mc_sampler=None,
        tol=None,
        max_draws=None,
        draws_simpson=None,
    ):
        """Set the integration options.

        Args:
            max_draws (default ~1'000'000): Maximum number of draws when integrating . Typically 500'000 - 5'000'000.
            tol: Tolerance on the error of the integral. typically 1e-4 to 1e-8
            draws_per_dim: The draws for MC integration to do per iteration. Can be set to ``'auto``'.
            draws_simpson: Number of points in one dimensional Simpson integration. Can be set to ``'auto'``.
        """

        if draws_per_dim is not None:
            self.integration.draws_per_dim = draws_per_dim
        if mc_sampler is not None:
            self.integration.mc_sampler = mc_sampler
        if max_draws is not None:
            self.integration.max_draws = max_draws
        if tol is not None:
            if tol > 1 or tol < 0:
                msg = "tol has to be between 0 and 1 (larger does not make sense)"
                raise ValueError(msg)
            self.integration.tol = tol

        if draws_per_dim == "auto":
            logtolonly = max(int(abs(math.log10(self.integration.tol))), 0)
            logexp = max(int(abs(math.log(self.integration.tol))), 2)
            logtol = int((logexp) ** 0.6)
            high_draws = 2**logtol * 10**logtol
            draws = min({0: 10, 1: 15, 2: 150, 3: 300, 4: 600}.get(logtolonly, 1e30), high_draws)
            draws = int(min(draws, self.integration.max_draws))
        if draws_per_dim is not None:
            self.integration.draws_per_dim = draws

        if draws_simpson == "auto":
            logtol = abs(math.log10(self.integration.tol * 0.005))
            npoints = 10 + 100 * (logtol + 1) + 1.8**logtol
            draws_simpson = int(npoints)
        if draws_simpson is not None:
            self.integration.draws_simpson = draws_simpson

    def _check_input_norm(self, norm, none_is_error=False) -> ZfitSpace | None:
        """Convert to :py:class:`~zfit.Space`.

        Args:
            norm:
            none_is_error: if both ``norm_range`` and ``self.norm_range`` are None, the default
                value is ``False`` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two ``None`` will raise a Value error.

        Returns:
            Union[:py:class:`~zfit.Space`, False]:
        """
        if (norm is None or (isinstance(norm, ZfitSpace) and not norm.limits_are_set)) and none_is_error:
            msg = (
                "Normalization range `norm` has to be specified or"
                "a default normalization range has to be set. Currently, both are None"
            )
            raise ValueError(msg)

        return self._convert_sort_space(limits=norm)

    def _check_input_limits(self, limits, none_is_error=False):
        if (limits is None or (isinstance(limits, ZfitSpace) and not limits.has_limits)) and none_is_error:
            msg = "The `limits` have to be specified and not be None"
            raise ValueError(msg)

        return self._convert_sort_space(limits=limits)

    def _convert_sort_space(
        self,
        obs: ztyping.ObsTypeInput | ztyping.LimitsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        limits: ztyping.LimitsTypeInput = None,
    ) -> ZfitSpace | None:
        """Convert the inputs (using eventually ``obs``, ``axes``) to :py:class:`~zfit.ZfitSpace` and sort them
        according to own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        elif not set(obs).intersection(self.obs):
            msg = "The given space {obs} is not compatible with the obs of the pdfs{self.obs};" " they are disjoint."
            raise SpaceIncompatibleError(msg)
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:  # e.g. not the first call
            space = space.with_coords(self.space, allow_superset=True, allow_subset=True)
        return space

    # Integrals

    @_BaseModel_register_check_support(True)
    @deprecated_norm_range
    def _integrate(self, limits, norm, *, options=None, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        params: ztyping.ParamTypeInput = None,
        var=None,
    ) -> ztyping.XType:
        """Integrate the function over ``limits`` (normalized over ``norm_range`` if not False).

        If an analytic integration function is available, it is used, otherwise numerical methods will be invoked.
        If the integration is in more than one dimension and no analytical integration method is provided, this can
        be unstable and the PDF will warn that the accuracy target was not reached.

        Args:
            limits: |@doc:pdf.integrate.limits| Limits of the integration. |@docend:pdf.integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            options: |@doc:pdf.integrate.options| Options for the integration.
               Additional options for the integration. Currently supported options are:
               - type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen. |@docend:pdf.integrate.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The integral value as a scalar with shape ()
        """
        norm = self._check_input_norm(norm)
        limits = self._check_input_limits(limits=limits)
        if options is None:
            options = {}
        with self._convert_sort_x(var, allow_none=True) as varclean, self._check_set_input_params(params=params):
            integral = self._single_hook_integrate(limits=limits, norm=norm, x=varclean, options=options)
        return znp.reshape(integral, -1)

    @z.function(wraps="model")
    def _single_hook_integrate(self, limits, norm, x, *, options):
        del x  # TODO HACK: how and what to pass through?
        return self._hook_integrate(limits=limits, norm=norm, options=options)

    def _hook_integrate(self, limits, norm, *, options=None):
        return self._norm_integrate(limits=limits, norm=norm, options=options)

    def _norm_integrate(self, limits, norm, *, options=None):
        try:
            integral = self._limits_integrate(limits=limits, norm=norm, options=options)
        except NormRangeNotImplemented:
            unnormalized_integral = self._limits_integrate(limits=limits, norm=False, options=options)
            normalization = self._limits_integrate(limits=norm, norm=False, options=options)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_integrate(self, limits, norm, *, options=None):
        try:
            integral = self._call_integrate(limits=limits, norm=norm, options=options)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_integrate(limits=sub_limits, norm=norm, options=options))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)  # TODO: remove stack?
        return integral

    def _call_integrate(self, limits, norm, options):
        with suppress(FunctionNotImplemented):
            return self._integrate(limits, norm, options=options)
        with suppress(AnalyticIntegralNotImplemented):
            return self._hook_analytic_integrate(limits=limits, norm=norm)
        return self._fallback_integrate(limits=limits, norm=norm, options=options)

    def _fallback_integrate(self, limits, norm, options):
        max_axes = self._analytic_integral.get_max_axes(limits=limits)

        integral = None
        if max_axes and integral is None:  # TODO improve handling of available analytic integrals
            with suppress(AnalyticIntegralNotImplemented):

                def part_int(x):
                    """Temporary partial integration function."""
                    return self._hook_partial_analytic_integrate(x, limits=limits, norm=norm)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._hook_numeric_integrate(limits=limits, norm=norm, options=options)
        return integral

    @classmethod
    def register_analytic_integral(
        cls,
        func: Callable,
        limits: ztyping.LimitsType = None,
        priority: int | float = 50,
        *,
        supports_norm: bool | None = None,
        supports_norm_range: bool | None = None,
        supports_multiple_limits: bool | None = None,
    ) -> None:
        """Register an analytic integral with the class.

        Args:
            func: A function that calculates the (partial) integral over the axes ``limits``.
                The signature has to be the following:

                    * x (:py:class:`~zfit.core.interfaces.ZfitData`, None): the data for the remaining axes in a partial
                        integral. If it is not a partial integral, this will be None.
                    * limits (:py:class:`~zfit.ZfitSpace`): the limits to integrate over.
                    * norm_range (:py:class:`~zfit.ZfitSpace`, None): Normalization range of the integral.
                        If not ``supports_supports_norm_range``, this will be None.
                    * params (Dict[param_name, :py:class:`zfit.Parameters`]): The parameters of the model.
                    * model (:py:class:`~zfit.core.interfaces.ZfitModel`):The model that is being integrated.

            limits: |limits_arg_descr|
            priority: Priority of the function. If multiple functions cover the same space, the one with the
                highest priority will be used.
            supports_multiple_limits: If ``True``, the ``limits` given to the integration function can have
                multiple limits. If `False`, only simple limits will pass through and multiple limits will be
                auto-handled.
            supports_norm: If `True`, `norm` argument to the function may not be `None`.
                If `False`, `norm` will always be `None` and care is taken of the normalization automatically.
        """
        if supports_norm_range is not None:
            msg = "Use `supports_norm` instead of `supports_norm_range`."
            raise BreakingAPIChangeError(msg)
        if supports_norm is None:
            supports_norm = False
        if supports_multiple_limits is None:
            supports_multiple_limits = False
        cls._analytic_integral.register(
            func=func,
            limits=limits,
            supports_norm=supports_norm,
            priority=priority,
            supports_multiple_limits=supports_multiple_limits,
        )

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
    @deprecated_norm_range
    def _analytic_integrate(self, limits, norm):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def analytic_integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamsTypeInput | None = None,
    ) -> ztyping.XType:
        """Analytical integration over function and raise Error if not possible.

        Args:
            limits: |@doc:pdf.integrate.limits| Limits of the integration. |@docend:pdf.integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The integral value
        Raises:
            AnalyticIntegralNotImplementedError: If no analytical integral is available (for this limits).
            NormRangeNotImplementedError: if the *norm* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm is not available.
        """
        norm = self._check_input_norm(norm)
        limits = self._check_input_limits(limits=limits)
        with self._check_set_input_params(params=params):
            integral = self._single_hook_analytic_integrate(limits=limits, norm=norm)
        return znp.atleast_1d(integral)

    @z.function(wraps="model")
    def _single_hook_analytic_integrate(self, limits, norm):
        return self._hook_analytic_integrate(limits=limits, norm=norm)

    def _hook_analytic_integrate(self, limits, norm):
        return self._norm_analytic_integrate(limits=limits, norm=norm)

    def _norm_analytic_integrate(self, limits, norm):
        try:
            integral = self._limits_analytic_integrate(limits=limits, norm=norm)
        except NormRangeNotImplemented:
            unnormalized_integral = self._limits_analytic_integrate(limits, norm=False)
            try:
                normalization = self._limits_analytic_integrate(limits=norm, norm=False)
            except AnalyticIntegralNotImplemented:
                msg = (
                    "Function does not support this (or even any)"
                    "normalization range 'norm'."
                    " This usually means,that no analytic integral "
                    "is available for this function. Due to rule "
                    "safety, an analytical normalization has to "
                    "be available and no attempt of numerical "
                    "normalization was made."
                )
                raise NormRangeNotImplemented(msg) from None
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_analytic_integrate(self, limits, norm):
        try:
            integral = self._call_analytic_integrate(limits, norm=norm)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_analytic_integrate(limits=sub_limits, norm=norm))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)
        return integral

    def _call_analytic_integrate(self, limits, norm):
        with suppress(FunctionNotImplemented, AnalyticIntegralNotImplemented):
            return self._analytic_integrate(limits, norm)
        return self._fallback_analytic_integrate(limits=limits, norm=norm)

    def _fallback_analytic_integrate(self, limits, norm):
        try:
            return self._analytic_integral.integrate(
                x=None,
                limits=limits,
                axes=limits.axes,
                norm=norm,
                model=self,
                params={k: znp.array(v) for k, v in self.params.items()},
            )
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            raise AnalyticIntegralNotImplemented from None

    @property
    def has_analytic_integral(self):
        """Return whether the PDF has an analytic integral over its full dimension.

        This does not imply that all different integrals, i.e. over different ranges or just partial variable are
        available.

        Returns:
        """
        try:
            _ = self.analytic_integrate(self.space)  # what about extended?
        except AnalyticIntegralNotImplemented:
            return False
        except Exception as error:
            warnings.warn(
                f"Called analytic integral to test if available, but unknown error occured: {error}."
                f" This can be ignored, but may be reported as an issue (you're welcome to do so!)",
                stacklevel=1,
                category=UserWarning,
            )
            return False
        else:
            return True

    @_BaseModel_register_check_support(True)
    @deprecated_norm_range
    def _numeric_integrate(self, limits, norm, *, params: Mapping[str, ZfitParameter], options=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def numeric_integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        params: ztyping.ParamsTypeInput = None,
    ) -> ztyping.XType:
        """Numerical integration over the model.

        Args:
            limits: |@doc:pdf.integrate.limits| Limits of the integration. |@docend:pdf.integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            options: |@doc:pdf.integrate.options| Options for the integration.
               Additional options for the integration. Currently supported options are:
               - type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen. |@docend:pdf.integrate.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The integral value
        """
        norm = self._check_input_norm(norm)
        limits = self._check_input_limits(limits=limits)
        if options is None:
            options = {}
        with self._check_set_input_params(params=params):
            return self._single_hook_numeric_integrate(limits=limits, norm=norm, options=options)

    @z.function(wraps="model")
    def _single_hook_numeric_integrate(self, limits, norm, options):
        return self._hook_numeric_integrate(limits=limits, norm=norm, options=options)

    def _hook_numeric_integrate(self, limits, norm, options):
        return self._norm_numeric_integrate(limits=limits, norm=norm, options=options)

    def _norm_numeric_integrate(self, limits, norm, options):
        try:
            integral = self._limits_numeric_integrate(limits=limits, norm=norm, options=options)
        except NormRangeNotImplemented:
            assert not norm.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_numeric_integrate(limits=limits, norm=False, options=options)
            normalization = self._limits_numeric_integrate(limits=norm, norm=False, options=options)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_numeric_integrate(self, limits, norm, options):
        try:
            integral = self._call_numeric_integrate(limits=limits, norm=norm, options=options)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_numeric_integrate(limits=sub_limits, norm=norm, options=options))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_numeric_integrate(self, limits, norm, options):
        with suppress(FunctionNotImplemented):
            return self._numeric_integrate(
                limits, norm, options=options, params={k: znp.array(v) for k, v in self.params.items()}
            )
        return self._fallback_numeric_integrate(limits=limits, norm=norm, options=options)

    def _fallback_numeric_integrate(self, limits, norm, options):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, norm=norm, options=options)

    @_BaseModel_register_check_support(True)
    def _partial_integrate(self, x, limits, norm, *, params=None, options):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def partial_integrate(
        self,
        x: ztyping.XTypeInput,
        limits: ztyping.LimitsType,
        *,
        norm=None,
        options=None,
        params: ztyping.ParamsTypeInput = None,
    ) -> ztyping.XTypeReturn:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: |@doc:pdf.partial_integrate.limits| Limits of the integration that will be integrated out.
               Has to be a subset of the PDFs observables. |@docend:pdf.partial_integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            options: |@doc:pdf.integrate.options| Options for the integration.
               Additional options for the integration. Currently supported options are:
               - type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen. |@docend:pdf.integrate.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The value of the partially integrated function evaluated at `x`.
        """
        if options is None:
            options = {}
        norm = self._check_input_norm(norm=norm)
        limits = self._check_input_limits(limits=limits)
        fallback_obs = [obs for obs in self.obs if obs not in limits.obs]  # keep order
        with self._convert_sort_x(x, partial=True, fallback_obs=fallback_obs) as xclean, self._check_set_input_params(
            params=params
        ):
            return self._single_hook_partial_integrate(x=xclean, limits=limits, norm=norm, options=options)

    @z.function(wraps="model")
    def _single_hook_partial_integrate(self, x, limits, norm, *, options):
        return self._hook_partial_integrate(x=x, limits=limits, norm=norm, options=options)

    def _hook_partial_integrate(self, x, limits, norm, *, options):
        return self._norm_partial_integrate(x=x, limits=limits, norm=norm, options=options)

    def _norm_partial_integrate(self, x, limits, norm, *, options):
        try:
            integral = self._limits_partial_integrate(x=x, limits=limits, norm=norm, options=options)
        except NormRangeNotImplemented:
            assert not norm.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_integrate(x=x, limits=limits, norm=False, options=options)
            normalization = self._hook_integrate(limits=norm, norm=False, options=None)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_integrate(self, x, limits, norm, *, options):
        try:
            integral = self._call_partial_integrate(x=x, limits=limits, norm=norm, options=options)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limit in limits:
                integrals.append(self._call_partial_integrate(x=x, limits=sub_limit, norm=norm, options=options))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_partial_integrate(self, x, limits, norm, *, options):
        with suppress(FunctionNotImplemented):
            return self._partial_integrate(x=x, limits=limits, norm=norm, options=options)
        with suppress(AnalyticIntegralNotImplemented):
            return self._hook_partial_analytic_integrate(x=x, limits=limits, norm=norm)
        try:
            return self._fallback_partial_integrate(x=x, limits=limits, norm=norm, options=options)
        except FunctionNotImplemented:
            raise AnalyticIntegralNotImplemented from None

    def _fallback_partial_integrate(self, x, limits: ZfitSpace, norm: ZfitSpace, *, options):
        del options
        max_axes = self._analytic_integral.get_max_axes(limits=limits)
        if max_axes:
            sublimits = limits.get_subspace(axes=max_axes)

            def part_int(x):  # change to partial integrate max axes?
                """Temporary partial integration function."""
                return self._hook_partial_analytic_integrate(x=x, limits=sublimits, norm=norm)

            axes = list(set(limits.axes) - set(max_axes))
            limits = limits.get_subspace(axes=axes)
        else:
            part_int = self._func_to_integrate

        assert limits.axes, "Internal Error! Axes should not be empty, maybe cleanup."
        return self._auto_numeric_integrate(func=part_int, limits=limits, x=x, norm=norm)

    @_BaseModel_register_check_support(True)
    @deprecated_norm_range
    def _partial_analytic_integrate(self, x, limits, norm, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def partial_analytic_integrate(
        self,
        x: ztyping.XTypeInput,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XTypeReturn:
        """Do analytical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: |@doc:pdf.partial_integrate.limits| Limits of the integration that will be integrated out.
               Has to be a subset of the PDFs observables. |@docend:pdf.partial_integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The value of the partially integrated function evaluated at `x`.

        Raises:
            AnalyticIntegralNotImplementedError: if the *analytic* integral (over these limits) is not implemented
            NormRangeNotImplementedError: if the *norm* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm is not available.
        """
        norm = self._check_input_norm(norm=norm)
        limits = self._check_input_limits(limits=limits)
        fallback_obs = [obs for obs in self.obs if obs not in limits.obs]  # keep order
        with self._convert_sort_x(x, partial=True, fallback_obs=fallback_obs) as xclean, self._check_set_input_params(
            params=params
        ):
            return self._single_hook_partial_analytic_integrate(x=xclean, limits=limits, norm=norm)

    @z.function(wraps="model")
    def _single_hook_partial_analytic_integrate(self, x, limits, norm):
        return self._hook_partial_analytic_integrate(x=x, limits=limits, norm=norm)

    def _hook_partial_analytic_integrate(self, x, limits, norm):
        return self._norm_partial_analytic_integrate(x=x, limits=limits, norm=norm)

    def _norm_partial_analytic_integrate(self, x, limits, norm):
        try:
            integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm=norm)
        except NormRangeNotImplemented:
            assert not norm.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm=False)
            try:
                normalization = self._limits_analytic_integrate(limits=norm, norm=False)
            except AnalyticIntegralNotImplemented:
                msg = (
                    "Function does not support this (or even any) normalization range"
                    " 'norm'. This usually means,that no analytic integral "
                    "is available for this function. An analytical normalization has to "
                    "be available and no attempt of numerical normalization was made."
                )
                raise NormRangeNotImplemented(msg) from None
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_analytic_integrate(self, x, limits, norm):
        try:
            integral = self._call_partial_analytic_integrate(x=x, limits=limits, norm=norm)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_partial_analytic_integrate(x=x, limits=sub_limits, norm=norm))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)

        return integral

    def _call_partial_analytic_integrate(self, x, limits, norm):
        with suppress(FunctionNotImplemented, AnalyticIntegralNotImplemented):
            return self._partial_analytic_integrate(x=x, limits=limits, norm=norm)
        return self._fallback_partial_analytic_integrate(x=x, limits=limits, norm=norm)

    def _fallback_partial_analytic_integrate(self, x, limits, norm):
        try:
            return self._analytic_integral.integrate(
                x=x,
                limits=limits,
                axes=limits.axes,
                norm=norm,
                model=self,
                params={k: znp.array(v) for k, v in self.params.items()},
            )
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            raise AnalyticIntegralNotImplemented from None

    @_BaseModel_register_check_support(True)
    @deprecated_norm_range
    def _partial_numeric_integrate(self, x, limits, norm, *, norm_range):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def partial_numeric_integrate(
        self,
        x: ztyping.XType,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Force numerical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm` (if not False)

        Args:
            x: The value at which the partially integrated function will be evaluated
            limits: |@doc:pdf.partial_integrate.limits| Limits of the integration that will be integrated out.
               Has to be a subset of the PDFs observables. |@docend:pdf.partial_integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The value of the partially integrated function evaluated at `x`.
        """
        norm = self._check_input_norm(norm)
        limits = self._check_input_limits(limits=limits)
        fallback_obs = [obs for obs in self.obs if obs not in limits.obs]  # keep order
        with self._convert_sort_x(x, partial=True, fallback_obs=fallback_obs) as clean, self._check_set_input_params(
            params=params
        ):
            return self._single_hook_partial_numeric_integrate(x=clean, limits=limits, norm=norm)

    @z.function(wraps="model")
    def _single_hook_partial_numeric_integrate(self, x, limits, norm):
        return self._hook_partial_numeric_integrate(x=x, limits=limits, norm=norm)

    def _hook_partial_numeric_integrate(self, x, limits, norm):
        return self._norm_partial_numeric_integrate(x=x, limits=limits, norm=norm)

    def _norm_partial_numeric_integrate(self, x, limits, norm):
        try:
            integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm=norm)
        except NormRangeNotImplemented:
            assert not norm.limits_are_false, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_numeric_integrate(x=x, limits=limits, norm=False)
            integral = unnormalized_integral / self._hook_numeric_integrate(limits=norm, norm=norm)
        return integral

    def _limits_partial_numeric_integrate(self, x, limits, norm):
        try:
            integral = self._call_partial_numeric_integrate(x=x, limits=limits, norm=norm)
        except MultipleLimitsNotImplemented:
            integrals = []
            for sub_limits in limits:
                integrals.append(self._call_partial_numeric_integrate(x=x, limits=sub_limits, norm=norm))
            integral = z.reduce_sum(znp.stack(integrals), axis=0)
        return integral

    def _call_partial_numeric_integrate(self, x, limits, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._partial_numeric_integrate(x=x, limits=limits, norm=norm)
        return self._fallback_partial_numeric_integrate(x=x, limits=limits, norm=norm)

    def _fallback_partial_numeric_integrate(self, x, limits, norm=False):
        return self._auto_numeric_integrate(func=self._func_to_integrate, limits=limits, x=x, norm=norm)

    @supports(multiple_limits=True)
    @z.function(wraps="model")
    def _auto_numeric_integrate(self, func, limits, x=None, options=None, **overwrite_options):
        if options is None:
            options = {}
        norm_option = overwrite_options.pop("norm", None)
        assert (norm_option) in (
            None,
            False,
        ), f"norm should be None or False, should be caught, is {norm_option}"

        is_binned = options.get("type") == "bins"
        vectorizable = is_binned
        draws_per_dim = self.integration.draws_per_dim
        draws_simpson = self.integration.draws_simpson
        if is_binned:
            draws_per_dim = max(draws_per_dim // 30, 10)
            # draws_per_dim = 100
            draws_simpson = max(draws_simpson // 30, 10)
            # draws_simpson = 100

        integration_options = dict(
            func=func,
            limits=limits,
            n_axes=limits.n_obs,
            x=x,
            # auto from self
            vectorizable=vectorizable,
            dtype=self.dtype,
            mc_sampler=self.integration.mc_sampler,
            mc_options={
                "draws_per_dim": draws_per_dim,
                "max_draws": self.integration.max_draws,
            },
            tol=self.integration.tol,
            simpsons_options={"draws_simpson": draws_simpson},
            **overwrite_options,
        )
        return self._integration.auto_numeric_integrator(**integration_options)

    @supports()
    def _inverse_analytic_integrate(self, x):
        if not self._inverse_analytic_integral:
            raise AnalyticSamplingNotImplemented

        icdf = self._inverse_analytic_integral[0]
        params = inspect.signature(icdf).parameters
        if len(params) == 2:
            return icdf(x=x, params={k: znp.array(v) for k, v in self.params.items()})
        elif len(params) == 3:
            return icdf(x=x, params={k: znp.array(v) for k, v in self.params.items()}, model=self)
        else:
            msg = f"icdf function does not have the right signature: {icdf}"
            raise RuntimeError(msg)

    @deprecated_args(None, "Use `params` instead.", "fixed_params")
    def create_sampler(
        self,
        n: ztyping.nSamplingTypeIn = None,
        limits: ztyping.LimitsType = None,
        *,
        fixed_params: Optional[bool | list[ZfitParameter] | tuple[ZfitParameter]] = None,
        params: ztyping.ParamTypeInput = None,
    ) -> SamplerData:
        """Create a :py:class:`SamplerData` that acts as `Data` but can be resampled, also with changed parameters and
        n.

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
                `resample`. If fixed, the Parameter will still have the same value as the `SamplerData` has
                been created with when it resamples.
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            :py:class:`~zfit.core.data.SamplerData`

        Raises:
            NotExtendedPDFError: if 'extended' is chosen (implicitly by default or explicitly) as an
                option for `n` but the pdf itself is not extended.
            ValueError: if n is an invalid string option.
            InvalidArgumentError: if n is not specified and pdf is not extended.
        """
        # legacy start
        if fixed_params is not None:
            msg = (
                "`fixed_params` has been removed, the sampler will always sample from the parameters at the time of the creation/given to the creator"
                " _or_ by giving params to the `resample` method."
            )
            raise BreakingAPIChangeError(msg)

        # legacy end

        limits = self._check_input_limits(limits=limits)
        if isinstance(n, str):
            n = None
        # Do NOT convert to tensor here, it will be done in the sampler (could be stateful object)

        if not limits.limits_are_set:
            limits = self.space
            if not limits.has_limits:
                msg = "limits are False/None, have to be specified"
                raise ValueError(msg)

        params = self._check_convert_input_paramvalues(params=params)
        params = {
            p.name: params[p.name] if p.name in params else p.value()
            for p in self.get_params(floating=None, is_yield=None)
        }

        def sample_func(n, params, *, limits=limits):
            return self.sample(n=n, limits=limits, params=params).value()

        return SamplerData.from_sampler(
            sample_func=sample_func,
            n=n,
            obs=limits,
            params=params,
            dtype=self.dtype,
            guarantee_limits=True,
        )

    @z.function(wraps="sampler")
    def _create_sampler_tensor(self, limits, n):
        return self._single_hook_sample(n=n, limits=limits, x=None)

    @_BaseModel_register_check_support(True)
    def _sample(self, n, limits: ZfitSpace, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def sample(
        self,
        n: ztyping.nSamplingTypeIn = None,
        limits: ztyping.LimitsType = None,
        *,
        x: ztyping.DataInputType | None = None,
        params: ztyping.ParamTypeInput = None,
    ) -> Data:  # TODO: change poissonian top-level with multinomial
        """Sample `n` points within `limits` from the model.

        If `limits` is not specified, `space` is used (if the space contains limits).
        If `n` is None and the model is an extended pdf, 'extended' is used by default.

        Args:
            n: The number of samples to be generated. Can be a Tensor that will be
                or a valid string. Currently implemented:

                    - 'extended': samples `poisson(yield)` from each pdf that is extended.
            limits: In which region to sample in
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            Data(n_obs, n_samples): The observables are the `limits`

        Raises:
            NotExtendedPDFError: if 'extended' is (implicitly by default or explicitly) chosen as an
                option for `n` but the pdf itself is not extended.
            ValueError: if n is an invalid string option.
            InvalidArgumentError: if n is not specified and pdf is not extended.
        """
        if isinstance(n, str):
            n = None
        if n is not None:
            n = znp.asarray(n, dtype=tf.int32)

        limits = self._check_input_limits(limits=limits)
        if not limits.limits_are_set:
            limits = self.space
            if not limits.has_limits:
                msg = "limits are False/None, have to be specified"
                raise tf.errors.InvalidArgumentError(msg)
        limits = self._check_input_limits(limits=limits, none_is_error=True)

        def run_tf(n, limits, x):  # todo: add params
            return self._single_hook_sample(n=n, limits=limits, x=x)  # todo: make data a composite object

        with self._convert_sort_x(x, allow_none=True) as xclean, self._check_set_input_params(params=params):
            new_obs = limits * xclean.data_range if xclean is not None else limits
            tensor = run_tf(n=n, limits=limits, x=xclean)
        return Data.from_tensor(tensor=tensor, obs=new_obs)  # TODO: which limits?

    @z.function(wraps="sample")
    def _single_hook_sample(self, n, limits, x=None):
        del x
        return self._hook_sample(n=n, limits=limits)

    def _hook_sample(self, limits, n):
        return self._limits_sample(n=n, limits=limits)

    def _limits_sample(self, n, limits):
        try:
            return self._call_sample(n=n, limits=limits)
        except MultipleLimitsNotImplemented as error:
            try:
                total_integral = self.analytic_integrate(limits, norm=False)
                sub_integrals = znp.concatenate(
                    [self.analytic_integrate(limit, norm=False) for limit in limits],
                    axis=0,
                )
            except AnalyticIntegralNotImplemented:
                msg = "Cannot autohandle multiple limits as the analytic" " integral is not available."
                raise MultipleLimitsNotImplemented(msg) from error
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

    def _analytic_sample(self, n, limits: ZfitSpace):
        if not self._inverse_analytic_integral:
            raise AnalyticSamplingNotImplemented  # TODO(Mayou36): create proper analytic sampling
        if limits._depr_n_limits > 1:
            raise AnalyticSamplingNotImplemented
        try:
            lower_bound, upper_bound = limits.rect_limits_np
        except CannotConvertToNumpyError as err:
            msg = (
                "Currently, analytic sampling with Tensors not supported."
                " Needs implementation of analytic integrals with Tensors."
            )
            raise WorkInProgressError(msg) from err
        neg_infinities = (tuple((-math.inf,) * limits.n_obs),)
        # to the cdf to get the limits for the inverse analytic integral
        try:
            lower_prob_lim = self._norm_analytic_integrate(
                limits=Space(limits=(neg_infinities, (lower_bound,)), axes=limits.axes),
                norm=False,
            )

            upper_prob_lim = self._norm_analytic_integrate(
                limits=Space(limits=(neg_infinities, (upper_bound,)), axes=limits.axes),
                norm=False,
            )
        except (SpecificFunctionNotImplemented, AnalyticIntegralNotImplemented):
            msg = (
                "analytic sampling not possible because the analytic integral"
                " is not"
                " implemented for the boundaries: {limits}"
            )
            raise AnalyticSamplingNotImplemented(msg) from None
        x = z.random.uniform(shape=(n, limits.n_obs), minval=lower_prob_lim, maxval=upper_prob_lim)
        # with self._convert_sort_x(prob_sample) as x:
        return self._inverse_analytic_integrate(x=x)

    def _fallback_sample(self, n, limits):
        return zsample.accept_reject_sample(
            prob=self._func_to_sample_from,
            n=n,
            limits=limits,
            prob_max=None,
            dtype=self.dtype,
            sample_and_weights_factory=self._sample_and_weights,
        )

    @classmethod
    def _register_additional_repr(cls, **kwargs):
        """Register an additional attribute to add to the repr.

        Args:
            any keyword argument. The value has to be gettable from the instance (has to be an
            attribute or callable method of self.
        """
        if cls._additional_repr is None:
            cls._additional_repr = {}
        if overwritten_keys := set(kwargs).intersection(cls._additional_repr):
            warnings.warn(
                "The following keys have been overwritten while registering additional repr:"
                f"\n{[str(k) for k in overwritten_keys]}",
                RuntimeWarning,
                stacklevel=2,
            )
        cls._additional_repr = dict(cls._additional_repr, **kwargs)

    def _get_additional_repr(self, sorted=True):
        # nice name change
        sorted_ = sorted
        sorted = builtins.sorted
        # nice name change end

        additional_repr = {}
        for key, val in self._additional_repr.items():
            try:
                new_obj = getattr(self, val)
            except AttributeError as error:
                msg = (
                    f"The attribute {val} is not a valid attribute of this class {type(self)}."
                    "Cannot use it in __repr__. It was added using the"
                    "`register_additional_repr` function"
                )
                raise AttributeError(msg) from error
            else:
                if callable(new_obj):
                    new_obj = new_obj()
            additional_repr[key] = new_obj
        if sorted_:
            additional_repr = dict(sorted(additional_repr))
        return additional_repr

    def __repr__(self):  # TODO(mayou36):repr to baseobject with _repr
        return "<zfit.{type_name} " " params=[{params}]".format(
            type_name=type(self),
            params=", ".join(sorted(str(p.name) for p in self.params.values())),
        )

    def _check_input_x_function(self, func):
        # TODO: signature etc?
        if not callable(func):
            msg = "Function {} is not callable."
            raise TypeError(msg)
        return func

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
            params = {name: kwargs.pop(name) for name in self._PARAMS}
        except KeyError as error:
            msg = (
                f"The following parameters are not given (as keyword arguments): {[k for k in self._PARAMS if k not in kwargs]}"
                ""
            )
            raise ValueError(msg) from error
        super().__init__(params=params, *args, **kwargs)  # noqa: B026
        # super().__init__(params=params, *args, **kwargs)  # use if upper fails

    @classmethod
    def _check_simple_model_subclass(cls):
        try:
            params = cls._PARAMS
        except AttributeError as error:
            msg = (
                "Need to define `_PARAMS` in the definition of the subclass."
                "Example:"
                "class MyModel(ZPDF):"
                "    _PARAMS = ['mu', 'sigma']"
            )
            raise SubclassingError(msg) from error
        not_str = [param for param in params if not isinstance(param, str)]
        if not_str:
            msg = "The following parameters are not strings in `_PARAMS`: "
            raise TypeError(msg)
