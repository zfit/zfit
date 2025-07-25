#  Copyright (c) 2025 zfit

from __future__ import annotations

import typing
import warnings
from collections.abc import Iterable
from contextlib import suppress

import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z
from zfit._interfaces import ZfitParameter, ZfitPDF, ZfitSpace

from ..settings import run, ztypes
from ..util import ztyping
from ..util.cache import invalidate_graph
from ..util.deprecation import deprecated, deprecated_norm_range
from ..util.exception import (
    AlreadyExtendedPDFError,
    BreakingAPIChangeError,
    FunctionNotImplemented,
    NormNotImplemented,
    NotExtendedPDFError,
    SpecificFunctionNotImplemented,
)
from ..util.plotter import PDFPlotter
from ..util.ztyping import ExtendedInputType, NormInputType
from .basemodel import BaseModel
from .baseobject import extract_filter_params
from .parameter import Parameter, convert_to_parameter
from .sample import extended_sampling
from .space import Space, convert_to_space

if typing.TYPE_CHECKING:
    import zfit


_BasePDF_USER_IMPL_METHODS_TO_CHECK = {}


def _BasePDF_register_check_support(has_support: bool):
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
        _BasePDF_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BasePDF_register_check_support
        return func

    return register


class PDFMeta(type):
    def __call__(cls, *args, obs=None, **kwargs):
        if binned := (obs is not None and isinstance(obs, Space) and obs.binning is not None):
            binned_obs = obs
            obs = binned_obs.with_binning(None)
        if obs is not None:
            kwargs["obs"] = obs
        pdf = cls.__new__(cls)
        pdf.__init__(*args, **kwargs)
        if binned:
            pdf = pdf.to_binned(
                binned_obs,
                extended=kwargs.get("extended"),
                norm=kwargs.get("norm"),
                name=kwargs.get("name"),
                label=kwargs.get("label"),
            )

        return pdf


class BasePDF(ZfitPDF, BaseModel, metaclass=PDFMeta):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        params: dict[str, ZfitParameter] | None = None,
        *,
        dtype=ztypes.float,
        label=None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "BasePDF",
        **kwargs,
    ):
        self._yield = None
        self._plot = None

        super().__init__(obs=obs, dtype=dtype, name=name, params=params, **kwargs)
        self._label = label or self.name
        self._norm = self._check_init_norm(norm)
        if extended is not False and extended is not None:
            self._set_yield(extended)

        self._assert_params_unique()
        if self._plot is None:
            self._plot = PDFPlotter(self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclass_check_support(
            methods_to_check=_BasePDF_USER_IMPL_METHODS_TO_CHECK,
            wrapper_not_overwritten=_BasePDF_register_check_support,
        )

    def _check_init_norm(self, norm):
        if not isinstance(norm, ZfitSpace):
            norm = self.space if norm is None else self.space.with_limits(norm)

        elif set(norm.obs) != set(self.obs):
            msg = "The normalization space has to be the same as the observation space."
            raise ValueError(msg)
        else:
            norm = norm.with_coords(self.space)
        return norm

    def _check_input_norm(self, norm, none_is_error=False):
        if norm is None:
            norm = self.norm
        return super()._check_input_norm(norm=norm, none_is_error=none_is_error)

    def _check_input_params_tfp(self, *params):
        return tuple(convert_to_parameter(p) for p in params)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.pdf(x, norm=False)

    def _func_to_sample_from(self, x):
        return self.pdf(x, norm=False)

    @property
    def label(self) -> str:
        return self._label if self._label is not None else self.name

    @property
    @deprecated(None, "Use the `norm` attribute instead.")
    def norm_range(self) -> Space | None | bool:
        """Return the current normalization range. If None and the ``obs`` have limits, they are returned.

        Returns:
            The current normalization range.
        """
        return self.norm

    @property
    def norm(self) -> Space | None | bool:
        """Return the current normalization range. If None and the ``obs`` have limits, they are returned.

        Returns:
            The current normalization range.
        """
        norm = self._norm
        if norm is None:
            norm = self.space
        return norm

    @invalidate_graph
    def set_norm_range(self, _: ztyping.LimitsTypeInput):
        """Set the normalization range (temporarily if used with contextmanager).

        Args:
            norm:
        """
        msg = "Setting the norm range is not supported anymore. Use `norm` argument instead or create a new PDF."
        raise BreakingAPIChangeError(msg)

    @_BasePDF_register_check_support(True)
    def _normalization(self, norm, options, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def normalization(
        self,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        limits: ztyping.LimitsType = None,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Return the normalization of the function (usually the integral over ``norm``).

        Args:
            norm:  |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|
            options: |@doc:pdf.param.options| Options for the PDF.
               Additional options for the PDF. Currently supported options are:

               - type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen. |@docend:pdf.param.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The normalization value
        """
        if limits is not None:
            msg = "Use `norm` instead of `limits`."
            raise BreakingAPIChangeError(msg)
        if options is None:
            options = {}
        norm = self._check_input_norm(norm)
        with self._check_set_input_params(params=params):
            return self._single_hook_normalization(norm=norm, options=options)

    @z.function(wraps="model")
    def _single_hook_normalization(self, norm, options):  # TODO(Mayou36): add yield?
        return self._hook_normalization(norm=norm, options=options)

    def _hook_normalization(self, norm, options):
        return self._call_normalization(norm=norm, options=options)  # no _norm_* needed

    def _call_normalization(self, norm, options):
        # TODO: caching? alternative
        with suppress(FunctionNotImplemented):
            return self._normalization(norm=norm, options=options)
        return self._fallback_normalization(norm, options=options)

    def _fallback_normalization(self, norm, options):
        return self._hook_integrate(limits=norm, norm=False, options=options)

    def _unnormalized_pdf(self, x, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated(None, "Use `pdf(norm=False)` instead")
    def unnormalized_pdf(self, x: ztyping.XType) -> ztyping.XType:
        """PDF "unnormalized". Use ``functions`` for unnormalized pdfs. this is only for performance in special cases.

        Args:
            x: |@doc:pdf.param.x| Data to evaluate the method on. Should be ``ZfitData``
               or a mapping of *obs* to numpy-like arrays.
               If an array is given, the first dimension is interpreted as the events while
               the second is meant to be the dimensionality of a single event. |@docend:pdf.param.x|
        Returns:
            1-dimensional :py:class:`tf.Tensor` containing the unnormalized pdf.
        """
        with self._convert_sort_x(x) as xclean:
            return self._single_hook_unnormalized_pdf(xclean)

    def _single_hook_unnormalized_pdf(self, x):
        return self._call_unnormalized_pdf(x=x)

    def _call_unnormalized_pdf(self, x):
        # try:

        return self._unnormalized_pdf(x)

    @z.function(wraps="model")
    @deprecated_norm_range
    def ext_pdf(
        self,
        x: ztyping.XTypeInput,
        norm: ztyping.LimitsTypeInput = None,
        *,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Probability density function scaled by yield, normalized over ``norm``.

        Args:
          x: |@doc:pdf.param.x| Data to evaluate the method on. Should be ``ZfitData``
               or a mapping of *obs* to numpy-like arrays.
               If an array is given, the first dimension is interpreted as the events while
               the second is meant to be the dimensionality of a single event. |@docend:pdf.param.x|
          norm: |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|
          params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        norm = self._check_input_norm(norm, none_is_error=True)
        if not self.is_extended:
            msg = f"{self} is not extended, cannot call `ext_pdf`"
            raise NotExtendedPDFError(msg)
        with self._convert_sort_x(x) as xclean, self._check_set_input_params(params=params):
            return self._call_ext_pdf(xclean, norm)

    @z.function(wraps="model")
    def _call_ext_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_pdf(x, norm)

        # fallback
        return self.pdf(x=x, norm=norm) * self.get_yield()

    def _auto_ext_pdf(self, x, norm):
        try:
            probs = self._ext_pdf(x, norm)
        except NormNotImplemented:
            unnorm_probs = self._ext_pdf(x, False)
            normalization = self.normalization(norm)
            probs = unnorm_probs / normalization
        return probs

    @_BasePDF_register_check_support(True)
    def _ext_pdf(self, x, norm, *, norm_range=None, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented  # TODO: implement properly

    @deprecated_norm_range
    def ext_log_pdf(
        self,
        x: ztyping.XTypeInput,
        norm: ztyping.LimitsTypeInput = None,
        *,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Log of probability density function scaled by yield, normalized over ``norm``.

        Args:
          x: |@doc:pdf.param.x| Data to evaluate the method on. Should be ``ZfitData``
               or a mapping of *obs* to numpy-like arrays.
               If an array is given, the first dimension is interpreted as the events while
               the second is meant to be the dimensionality of a single event. |@docend:pdf.param.x|
          norm: |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|

          params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        norm = self._check_input_norm(norm, none_is_error=True)
        if not self.is_extended:
            msg = f"{self} is not extended, cannot call `ext_pdf`"
            raise NotExtendedPDFError(msg)
        with self._convert_sort_x(x) as xclean, self._check_set_input_params(params=params):
            return self._call_ext_log_pdf(xclean, norm)

    @z.function(wraps="model")
    def _call_ext_log_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_log_pdf(x, norm)

        # fallback
        return self.log_pdf(x=x, norm=norm) + znp.log(self.get_yield())

    def _auto_ext_log_pdf(self, x, norm):
        try:
            pdf = self._ext_log_pdf(x, norm)
        except NormNotImplemented:
            unnormed_pdf = self._ext_log_pdf(x, False)
            normalization = self.log_normalization(norm)
            pdf = unnormed_pdf - normalization
        return pdf

    @_BasePDF_register_check_support(True)
    def _ext_log_pdf(self, x, norm):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @_BasePDF_register_check_support(True)
    def _pdf(self, x, norm, *, norm_range=None, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def pdf(
        self,
        x: ztyping.XTypeInput,
        norm: ztyping.LimitsTypeInput = None,
        *,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Probability density function of ``x``, normalized over ``norm``.

        Args:
          x: |@doc:pdf.param.x| Data to evaluate the method on. Should be ``ZfitData``
               or a mapping of *obs* to numpy-like arrays.
               If an array is given, the first dimension is interpreted as the events while
               the second is meant to be the dimensionality of a single event. |@docend:pdf.param.x|
          norm: |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|
          params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|
        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        norm = self._check_input_norm(norm, none_is_error=True)
        with self._convert_sort_x(x) as xclean, self._check_set_input_params(params=params):
            value = self._single_hook_pdf(x=xclean, norm=norm)
        if run.numeric_checks:
            z.check_numerics(value, message="Check if pdf output contains any NaNs of Infs")
        return znp.atleast_1d(znp.asarray(z.to_real(value)))

    @z.function(wraps="model")
    def _single_hook_pdf(self, x, norm):
        return self._hook_pdf(x=x, norm=norm)

    def _hook_pdf(self, x, norm):
        return self._norm_pdf(x=x, norm=norm)

    def _norm_pdf(self, x, norm):
        try:
            return self._call_pdf(x=x, norm=norm)
        except NormNotImplemented:
            unnormed_pdf = self._call_pdf(x=x, norm=norm.with_limits(False))
            normalization = self.normalization(norm)
            return unnormed_pdf / normalization

    def _call_pdf(self, x, norm):
        with suppress(FunctionNotImplemented):
            return self._pdf(x, norm)
        with suppress(FunctionNotImplemented):
            return znp.exp(self._log_pdf(x, norm))
        if self.is_extended:
            with suppress(FunctionNotImplemented):
                return self._ext_pdf(x, norm) / self.get_yield()  # TODO: extend/refactor the calling

        return self._fallback_pdf(x, norm)

    def _fallback_pdf(self, x, norm):
        pdf = self._call_unnormalized_pdf(x)
        if norm.has_limits:
            pdf /= self._hook_normalization(norm=norm, options={})
        return pdf

    @_BasePDF_register_check_support(True)
    @deprecated_norm_range
    def _log_pdf(self, x, norm):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def log_pdf(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Log probability density function normalized over ``norm``.

        Args:
            x: |@doc:pdf.param.x| Data to evaluate the method on. Should be ``ZfitData``
               or a mapping of *obs* to numpy-like arrays.
               If an array is given, the first dimension is interpreted as the events while
               the second is meant to be the dimensionality of a single event. |@docend:pdf.param.x|
            norm: |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|
        Returns:
          A ``Tensor`` of type ``self.dtype``.
        """
        norm = self._check_input_norm(norm)
        with self._convert_sort_x(x) as xclean, self._check_set_input_params(params=params):
            return znp.asarray(z.to_real(self._single_hook_log_pdf(x=xclean, norm=norm)))

    @z.function(wraps="model")
    def _single_hook_log_pdf(self, x, norm):
        return self._hook_log_pdf(x=x, norm=norm)

    def _hook_log_pdf(self, x, norm):
        return self._norm_log_pdf(x=x, norm=norm)

    def _norm_log_pdf(self, x, norm):
        try:
            return self._call_log_pdf(x=x, norm=norm)
        except NormNotImplemented:
            unnormed_log_pdf = self._call_log_pdf(x=x, norm=False)
            normalization = self.log_normalization(norm)
            return unnormed_log_pdf - normalization

    def _call_log_pdf(self, x, norm):
        with suppress(FunctionNotImplemented):
            return self._log_pdf(x, norm)
        with suppress(FunctionNotImplemented):
            return znp.log(self._pdf(x, norm))
        return self._fallback_log_pdf(x, norm)

    def _fallback_log_pdf(self, x, norm):
        return znp.log(self._hook_pdf(x=x, norm=norm))

    @_BasePDF_register_check_support(True)
    @deprecated_norm_range
    def _log_normalization(self, norm, *, params=None, options):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def log_normalization(
        self,
        norm: ztyping.LimitsType,
        *,
        options=None,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Return the normalization of the function (usually the integral over ``norm``).

        Args:
            norm:  |@doc:pdf.param.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). Should be ``ZfitSpace`` to define the space
               to normalize over. |@docend:pdf.param.norm|
            options: Additional options for the normalization.
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The normalization value
        """
        if options is None:
            options = {}
        norm = self._check_input_norm(norm, none_is_error=True)
        with self._check_set_input_params(params=params):
            return self._single_hook_log_normalization(norm=norm, options=options)

    @z.function(wraps="model")
    def _single_hook_log_normalization(self, norm, options):  # TODO(Mayou36): add yield?
        return self._hook_normalization(norm=norm, options=options)

    def _hook_log_normalization(self, norm, options):
        return self._call_normalization(norm=norm, options=options)  # no _norm_* needed

    def _call_log_normalization(self, norm, options):
        # TODO: caching? alternative
        with suppress(FunctionNotImplemented):
            return self._normalization(norm=norm, options=options)
        return self._fallback_normalization(norm, options=options)

    def _fallback_log_normalization(self, norm, options):
        return znp.log(self._hook_normalization(norm=norm, options=options))

    @deprecated_norm_range
    def ext_integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        params: ztyping.ParamsTypeOpt = None,
    ) -> ztyping.XType:
        """Integrate the function over ``limits`` (normalized over ``norm`` if not False).

        Args:
            limits: |@doc:pdf.integrate.limits| Limits of the integration. |@docend:pdf.integrate.limits|
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            options: |@doc:pdf.integrate.options| Options for the integration.
               Additional options for the integration. Currently supported options are:

               * type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen.

               Other options *may* be available in the future. |@docend:pdf.integrate.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            The integral value as a scalar with shape ()
        """
        if options is None:
            options = {}
        norm = self._check_input_norm(norm)
        limits = self._check_input_limits(limits=limits)
        if not self.is_extended:
            msg = f"{self} is not extended, cannot call `ext_pdf`"
            raise NotExtendedPDFError(msg)
        with self._check_set_input_params(params=params):
            return self.integrate(limits=limits, norm=norm, options=options) * self.get_yield()

    def _apply_yield(self, value: float, norm: ztyping.LimitsType, log: bool) -> float | tf.Tensor:
        if self.is_extended and not norm.limits_are_false:
            if log:
                value += znp.log(self.get_yield())
            else:
                value *= self.get_yield()
        return value

    def _set_yield_inplace(self, *_, **__):
        """Make the model extended by setting a yield.

        This does not alter the general behavior of the PDF. If there is a
        ``norm`` given, the output of the above functions does not represent a normalized
        probability density function anymore but corresponds to a number probability.

        Args:
            value:
        """
        msg = "Setting the yield inplace is not supported anymore. Use the   `create_extended` method instead or create a new PDF."
        raise BreakingAPIChangeError(msg)

    def create_extended(
        self,
        yield_: ztyping.ParamTypeInput,
        name: str | None = None,
        *,
        name_addition: str | None = None,
    ) -> ZfitPDF:
        """Return an extended version of this pdf with yield ``yield_``. The parameters are shared.

        Args:
            yield_: |@doc:pdf.param.yield| Yield (expected number of events) of the PDF.
               This is the expected number of events.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.param.yield|
            name: New name of the PDF. If ``None``, the name of the PDF with a trailing "_ext" is used.

        Returns:
            :py:class:`~zfit.core.interfaces.ZfitPDF`: a new PDF that is extended
        """
        # TODO(Mayou36): fix copy
        if name_addition is not None:
            msg = "name_addition is not supported anymore, use `name` instead."
            raise BreakingAPIChangeError(msg)
        from zfit.models.functor import ProductPDF  # noqa: PLC0415

        name = f"{self.name}_ext" if name is None else name

        if isinstance(self, ProductPDF):
            warnings.warn(
                "As `copy` is not yet properly implemented, this may fails (for ProductPDF for example?). This"
                "will be fixed in the future.",
                category=UserWarning,
                stacklevel=2,
            )
        if self.is_extended:
            msg = "This PDF is already extended, cannot create an extended one."
            raise AlreadyExtendedPDFError(msg)
        try:
            new_pdf = self.copy(name=name)
        except Exception as error:
            msg = (
                f"PDF {self} could not be copied, therefore `create_extended` failed and a new "
                f"extended PDF cannot be created. As an alternative, you can use `set_yield`"
                f" to set the yield on the current PDF *inplace* (this won't return a new PDF but"
                f" instead modify the existing)."
            )
            raise RuntimeError(msg) from error
        new_pdf.set_yield(value=yield_)
        return new_pdf

    def to_positive(
        self,
        epsilon: float = 1e-100,
        name: str | None = None,
    ) -> zfit.pdf.PositivePDF:
        """Return a positive version of this PDF that ensures output values are at least epsilon.

        This method creates a PositivePDF functor that wraps the current PDF and ensures its output
        is always at least epsilon using znp.maximum. This is useful for PDFs
        that can produce negative values (e.g., KDE with negative weights) or numerical
        instabilities that lead to values very close to zero.

        Args:
            epsilon: The minimum positive value for the PDF output. Default is 1e-100.
            name: New name of the PDF. If ``None``, the name of the PDF with a trailing "_positive" is used.

        Returns:
            :py:class:`~zfit.core.interfaces.ZfitPDF`: a new PDF that is always positive
        """
        from zfit.models.postprocess import PositivePDF  # noqa: PLC0415

        name = f"{self.name}_positive" if name is None else name

        return PositivePDF(pdf=self, epsilon=epsilon, name=name)

    def to_cached(
        self,
        epsilon: float | None = None,
        name: str | None = None,
    ) -> zfit.pdf.CachedPDF:
        """Return a cached version of this PDF that stores the last calculated value for reuse.

        This method creates a CachedPDF functor that wraps the current PDF and caches
        the results of pdf() and integrate() calls. The cache is keyed by the input data
        and parameter values, and will return cached results when the same inputs are
        encountered again within the specified tolerance.

        This is particularly useful during optimization when the same PDF is evaluated
        multiple times with the same parameters and data, such as when using numerical
        gradients.

        Args:
            epsilon: Accuracy of absolute tolerance for comparing arguments (parameters, data)
                with cached values. If None, defaults to 1e-8.
            name: New name of the PDF. If ``None``, the name of the PDF with a trailing "_cached" is used.

        Returns:
            :py:class:`~zfit.core.interfaces.ZfitPDF`: a new PDF that caches evaluation results
        """
        from zfit.models.cache import CachedPDF  # noqa: PLC0415

        name = f"{self.name}_cached" if name is None else name

        return CachedPDF(
            pdf=self,
            epsilon=epsilon,
            name=name,
        )

    @deprecated(None, "Use `create_extended` instead or `extended=yield` when creating the PDF.")
    def set_yield(self, value):
        """Make the model extended **inplace** by setting a yield. If possible, prefer to use ``create_extended``.

        This does not alter the general behavior of the PDF. The ``pdf`` and ``integrate`` and similar methods will
        continue to return the same - normalized to 1 - values. However, not only can this parameter be accessed
        via ``get_yield``, the methods ``ext_pdf`` and ``ext_integral`` provide a version of ``pdf`` and ``integrate``
        respecetively that is multiplied by the yield.

        These can be useful for plotting and for binned likelihoods.

        Args:
            value: |@doc:pdf.param.yield| Yield (expected number of events) of the PDF.
               This is the expected number of events.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.param.yield|
        """
        self._set_yield(value=value)

    def _set_yield(self, value: ztyping.ParamTypeInput):
        if value is None:
            msg = "Cannot unset a yield (anymore)."
            raise BreakingAPIChangeError(msg)
        if self.is_extended:
            msg = f"Cannot extend {self}, is already extended."
            raise AlreadyExtendedPDFError(msg)
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

        # not ideal, should be in parametrized. But we don't have too many base classes, so this should work
        self._assert_params_unique()

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the model is extended or not.

        Returns:
            A boolean.
        """
        return self._yield is not None

    def _hook_sample(self, limits, n):
        if n is None and self.is_extended:
            n = "extended"
        if isinstance(n, str) and n == "extended":
            if not self.is_extended:
                msg = "Cannot use 'extended' as value for `n` on a non-extended pdf."
                raise NotExtendedPDFError(msg)
            samples = extended_sampling(pdfs=self, limits=limits)
        elif isinstance(n, str):
            msg = "`n` is a string and not 'extended'. Other options are currently not implemented."
            raise ValueError(msg)
        elif n is None:
            msg = "`n` (number of samples) cannot be `None` (only if pdf is extended)."
            raise ValueError(msg)
        else:
            samples = super()._hook_sample(limits=limits, n=n)
        return samples

    def get_yield(self) -> Parameter | None:
        """Return the yield (only for extended models).

        Returns:
            The yield of the current model or None
        """
        return self._yield

    @property
    def extended(self) -> Parameter | None:
        """Return the yield (only for extended models).

        Returns:
            The yield of the current model or None
        """
        return self.get_yield()

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        params = super()._get_params(
            floating, is_yield=is_yield, extract_independent=extract_independent, autograd=autograd
        )

        if is_yield is not False:
            if self.is_extended:
                yield_params = extract_filter_params(
                    self.get_yield(),
                    floating=floating,
                    extract_independent=extract_independent,
                )
                # we care if it supports autograd or not
                if autograd is False:
                    if "yield" not in self._autograd_params:  # it doesn't support autograd
                        yield_params.update(params)
                    else:
                        yield_params = params
                elif autograd is None:
                    yield_params.update(params)
                else:
                    msg = "autograd should either be None or False, internal error"
                    raise AssertionError(msg)
                params = yield_params
            elif is_yield is True:
                msg = "PDF is not extended but only yield parameters were requested."
                raise NotExtendedPDFError(msg)
        return params

    def _get_autograd_params(self):
        params = super()._get_autograd_params()
        if self.is_extended and "yield" in self._autograd_params:
            params.update(self.get_params(floating=None, is_yield=True, extract_independent=True))
        return params

    def create_projection_pdf(
        self,
        *,
        limits: ztyping.LimitsTypeInput = None,
        obs: ztyping.LimitsTypeInput = None,
        options=None,
        name: str | None = None,
        label: str | None = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
    ) -> ZfitPDF:
        """Create a PDF projection by integrating out some dimensions.

        The new projection pdf is still fully dependent on the pdf it was created with.

        Args:
            limits: Limits of the integration to project out. If not given, all observables that are not in `obs` are
                projected on using the default limits of the observables.
            obs: Observables to project on. If not given, all observables that are not in `limits` are projected on.
            options: |@doc:pdf.integrate.options| Options for the integration.
               Additional options for the integration. Currently supported options are:

               * type: one of (``bins``)
                 This hints that bins are integrated. A method that is vectorizable,
                 non-dynamic and therefore less suitable for complicated functions is chosen.

               Other options *may* be available in the future. |@docend:pdf.integrate.options|
            name: Name of the new PDF. If not given, it is created from the original name.
            label: Label of the new PDF. If not given, it is created from the original label.
            extended: If the new PDF should be extended. If not given, it is the same as the original PDF.
            norm: If the new PDF should be normalized. If not given, it is the same as the original PDF.

        Returns:
            A pdf without the dimensions from ``limits``.
        """
        from ..models.special import SimpleFunctorPDF  # noqa: PLC0415

        if limits is None:
            if obs is None:
                msg = "Either `limits` or `obs` have to be given."
                raise ValueError(msg)
            obs = convert_to_space(obs)
            limit_obs = [ob for ob in self.obs if ob not in obs.obs]
            if not limit_obs:
                msg = f"No observables to integrate out: `obs` contains all observables {obs}."
                raise ValueError(msg)
            limits = self.space.with_obs(limit_obs)
            if not obs.has_limits:
                obs = self.space.with_obs(obs.obs)
        else:
            limits = convert_to_space(limits)
            if not limits.has_limits:
                limits = self.space.with_obs(limits.obs)
            if obs is None:
                obs = self.space.with_obs([ob for ob in self.obs if ob not in limits.obs])
            else:
                obs = convert_to_space(obs)
                if not obs.has_limits:
                    obs = self.space.with_obs(obs.obs)
                if not set(obs.obs).isdisjoint(limits.obs):
                    msg = (
                        f"The `obs` to project on ({obs}) and the `limits` to integrate over ({limits}) "
                        "have to be disjoint."
                    )
                    raise ValueError(msg)

        def partial_integrate_wrapped(self_simple, x):
            del self_simple
            return self.partial_integrate(
                x, limits=limits, options=options, norm=False
            )  # todo: it should be fine not to normalize, right?

        if label is None:
            label = f"{self.label}_projon_{obs.obs[0]}"
        if extended is None:
            extended = self.is_extended
        if extended is True:
            extended = self.get_yield()
        return SimpleFunctorPDF(
            obs=obs,
            pdfs=(self,),
            func=partial_integrate_wrapped,
            name=name,
            label=label,
            extended=extended,
            norm=norm,
        )

    def copy(self, **override_parameters) -> BasePDF:
        """Creates a copy of the model.

        Note: the copy model may continue to depend on the original
        initialization arguments.

        Args:
          **override_parameters: String/value dictionary of initialization
            arguments to override with new value.

        Returns:
          A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters, i.e.,
            `dict(self.parameters, **override_parameters)`.
        """
        obs = self.norm

        # HACK(Mayou36): remove once copy is proper implemented
        from ..models.dist_tfp import WrapDistribution  # noqa: PLC0415
        from ..models.kde import GaussianKDE1DimV1  # noqa: PLC0415
        from ..models.polynomials import RecursivePolynomial  # noqa: PLC0415

        if type(self) is WrapDistribution:  # NOT isinstance! Because e.g. Gauss wraps that and takes different args
            parameters = {"distribution": self._distribution, "dist_params": self.dist_params}
        else:
            # HACK END

            parameters = dict(self.params)
            lam = parameters.pop("lambda", None)
            if lam is not None:
                parameters["lam"] = lam

        if type(self) is GaussianKDE1DimV1:
            msg = (
                "Cannot copy `GaussianKDE1DimV1` (yet). If you tried to make it extended, use "
                "`set_yield`"
                " instead and set it inplace."
            )
            raise RuntimeError(msg)
            parameters["data"] = self._original_data

        # HACK(Mayou36): copy the polynomial correct, replace 'c_0' with coeff0/coeff_0 or similar
        if isinstance(self, RecursivePolynomial):
            parameters["coeff0"] = parameters.pop("c_0", None)
            coeffs = []
            i_coeff = 1
            # collect coeffs and convert to 'coeff' list
            while True:
                coeff_name = f"c_{i_coeff}"
                try:
                    coeff = parameters.pop(coeff_name)
                except KeyError:
                    break
                else:
                    coeffs.append(coeff)
                i_coeff += 1
            parameters["coeffs"] = coeffs

        from zfit.models.functor import BaseFunctor, SumPDF  # noqa: PLC0415

        if isinstance(self, BaseFunctor):
            parameters = {}
            if isinstance(self, SumPDF):
                fracs = self.fracs
                if not self.is_extended:
                    fracs = fracs[:-1]
                parameters.update(fracs=fracs)
            parameters.update(pdfs=self.pdfs)

        parameters.update(obs=obs, name=self.name)
        parameters.update(**override_parameters)
        # if hasattr(self, "distribution"):
        #     parameters.update(distribution=self.distribution)
        yield_ = parameters.pop("yield", None)
        new_instance = type(self)(**parameters)
        if yield_ is not None:
            new_instance.set_yield(yield_)
        return new_instance

    @deprecated_norm_range
    def as_func(self, norm: ztyping.LimitsType = False):
        """Return a `Function` with the function `model(x, norm=norm)`.

        Args:
            norm: If not False or a `ZfitSpace`, this will be used to call the `pdf` function.
        """
        from .operations import convert_pdf_to_func  # prevent circular import  # noqa: PLC0415

        return convert_pdf_to_func(pdf=self, norm=norm)

    def __str__(self) -> str:
        """User-friendly string representation."""
        info = [f"{type(self).__name__}('{self.label}')"]

        # Add observables
        if hasattr(self, "obs") and self.obs:
            info.append(f"obs={self.obs}")

        # Add key parameters (limit to avoid clutter)
        if hasattr(self, "params") and self.params:
            param_strs = []
            for name, param in list(self.params.items())[:3]:  # Show first 3
                try:
                    if hasattr(param, "value"):
                        value = f"{param.value():.3g}" if run.executing_eagerly() else "symbolic"
                        param_strs.append(f"{name}={value}")
                except Exception:
                    param_strs.append(f"{name}=?")
            if len(self.params) > 3:
                param_strs.append("...")
            if param_strs:
                info.append(f"params=[{', '.join(param_strs)}]")

        # Add extended status
        if hasattr(self, "is_extended") and self.is_extended:
            try:
                yield_val = f"{self.get_yield().value():.3g}" if run.executing_eagerly() else "symbolic"
                info.append(f"extended=True(yield={yield_val})")
            except Exception:
                info.append("extended=True")

        return " ".join(info)

    def to_unbinned(self) -> ZfitPDF:
        """Convert to unbinned pdf, returns self if already unbinned."""
        return self

    def to_binned(
        self,
        space: ztyping.SpaceType,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        """Convert to binned pdf, returns self if already binned.

        Args:
            space: The space to bin the pdf in.
            extended: If the new PDF should be extended. If not given, it is the same as the original PDF.
            norm: If the new PDF should be normalized. If not given, it is the same as the original PDF.
            name: Name of the new PDF. If not given, it is created from the original name.
            label: Label of the new PDF. If not given, it is created from the original label.
        """
        from ..models.tobinned import BinnedFromUnbinnedPDF  # noqa: PLC0415

        return BinnedFromUnbinnedPDF(pdf=self, space=space, extended=extended, norm=norm, name=name, label=label)

    def to_truncated(
        self,
        limits: ZfitSpace | Iterable[ZfitSpace] | None = None,
        *,
        obs=None,
        extended=None,
        norm=None,
        name: str | None = None,
        label: str | None = None,
    ):
        """Convert the PDF to a truncated version with possibly different and multiple limits.

        The arguments are the same as for :py:class:`~zfit.pdf.TruncatedPDF`, the only difference being that
        if no limits are given, the limit of the PDF is used, thereby truncating the PDF to its original limits.

        Args:
            pdf: The PDF to be truncated.
            limits: The limits to truncate the PDF. Can be a single limit or multiple limits.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
               If None, the PDF will be extended if the original PDF is extended.
               If ``True`` and the original PDF is extended, the yield will be scaled to the
               fraction of the total integral that is within the limits.
               Therefore, the overall yield is comparable, i.e. the pdfs can be plotted
               "on top of each other".
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """

        from ..models.truncated import TruncatedPDF  # noqa: PLC0415

        if limits is None:
            limits = obs if obs is not None else self.space
        if obs is None:
            obs = self.space
        if name is None:
            name = self.name + "_truncated"
        if label is None:
            label = self.label + " _truncated"
        if norm is None:
            norm = self.norm
        return TruncatedPDF(pdf=self, obs=obs, limits=limits, extended=extended, norm=norm, name=name, label=label)
