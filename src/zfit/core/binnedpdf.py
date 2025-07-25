#  Copyright (c) 2025 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorflow.python.util.deprecation import deprecated_args

from ..util.plotter import PDFPlotter

if TYPE_CHECKING:
    import zfit

import operator
from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from uhi.typing.plottable import PlottableHistogram

import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit._data.binneddatav1 import BinnedData, BinnedSamplerData, move_axis_obs
from zfit._interfaces import (
    ZfitBinnedData,
    ZfitBinnedPDF,
    ZfitBinning,
    ZfitParameter,
    ZfitPDF,
    ZfitSpace,
    ZfitUnbinnedData,
)

from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_norm_range
from ..util.exception import (
    AlreadyExtendedPDFError,
    BasePDFSubclassingError,
    BreakingAPIChangeError,
    MultipleLimitsNotImplemented,
    NormNotImplemented,
    NotExtendedPDFError,
    SpaceIncompatibleError,
    SpecificFunctionNotImplemented,
    WorkInProgressError,
)
from .baseobject import BaseNumeric, extract_filter_params
from .binning import unbinned_to_binindex
from .data import Data
from .dimension import BaseDimensional
from .parameter import convert_to_parameter
from .space import convert_to_space, supports
from .tensorlike import OverloadableMixinValues

_BaseModel_USER_IMPL_METHODS_TO_CHECK = {}


def _BinnedPDF_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the ``@supports`` decorator.

    Args:
        has_support (bool): If True, flags that it **requires** the ``@supports`` decorator. If False,
            flags that the ``@supports`` decorator is **not allowed**.
    """
    if not isinstance(has_support, bool):
        msg = "Has to be boolean."
        raise TypeError(msg)

    def register(func):
        """Register a method to be checked to (if True) *has* ``support`` or (if False) has *no* ``support``.

        Args:
            func (function):
        Returns:
            function:
        """
        name = func.__name__
        _BaseModel_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BinnedPDF_register_check_support
        return func

    return register


class BaseBinnedPDF(
    BaseNumeric,
    GraphCachable,
    BaseDimensional,
    OverloadableMixinValues,
    ZfitBinnedPDF,
):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
        name: str | None = None,
        label: str | None = None,
        **kwargs,
    ):
        self._label = label
        self._plot = None
        super().__init__(dtype=znp.float64, name=name, **kwargs)

        self._space = self._check_convert_obs_init(obs)
        self._yield = None
        self._norm = self._check_convert_norm_init(norm)
        if extended is None:
            extended = False
        if extended is not False:
            self._set_yield(extended)

        if self._plot is None:  # todo: have a binned plotter
            self._plot = PDFPlotter(self)

    @property
    def label(self):
        return self._label

    def _check_convert_obs_init(self, obs):
        if not isinstance(obs, ZfitSpace) or not obs.is_binned:
            msg = f"`obs` have to be a Space with binning, not {obs}."
            raise ValueError(msg)
        return obs

    def _check_convert_norm_init(self, norm):
        if norm is not None and (not isinstance(norm, ZfitSpace) or not norm.has_limits):
            msg = f"`norm` has to be None or a Space with limits, not {norm}."
            raise ValueError(msg)
        return norm

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
                        f"Method {method_name} has been wrapped with supports "
                        "but is not allowed to. Has to handle all "
                        "arguments."
                    )
                    raise BasePDFSubclassingError(msg)
                if has_support:
                    msg = (
                        f"Method {method_name} has been overwritten and *has to* be "
                        "wrapped by `supports` decorator (don't forget () )"
                        "to call the decorator as it takes arguments"
                        ""
                    )
                    raise BasePDFSubclassingError(msg)
                if not has_support:
                    continue  # no support, has not been wrapped with
            elif not has_support:
                continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            msg = (
                f"Method {method_name} has not been correctly wrapped with @supports "
                "OR has been wrapped but it should not be"
            )
            raise BasePDFSubclassingError(msg)

    @property
    def axes(self):
        return self.space.binning

    def to_binneddata(self, **kwargs) -> zfit.data.BinnedData:
        """Create an Asimov dataset as ``BinnedData`` using either ``counts`` (for extended) or ``rel_counts``.

        Args:
            **kwargs (): arguments to ``counts`` or ``rel_counts``.

        Returns:
            BinnedData: Binned data representing the Asimov dataset of this PDF.
        """
        values = self.values(**kwargs)
        return BinnedData.from_tensor(space=self.space, values=values)

    def to_hist(self, **kwargs):
        """Create an Asimov histogram as ``Hist`` using either ``counts`` (for extended) or ``rel_counts``.

        Args:
            **kwargs (): arguments to ``counts`` or ``rel_counts``.

        Returns:
            ``hist.Hist``: Histogram representing the Asimov dataset of this PDF.
        """
        return self.to_binneddata(**kwargs).to_hist()

    @property
    def space(self):
        return self._space

    def _set_yield(self, value):
        if self.is_extended:
            msg = f"Cannot extend {self}, is already extended."
            raise AlreadyExtendedPDFError(msg)
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

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
                if autograd is None or (autograd is False and "yield" not in self._autograd_params):
                    params = yield_params | params

                assert autograd is not True, "autograd should either be None or False, internal error"
            elif is_yield is True:
                msg = "PDF is not extended but only yield parameters were requested."
                raise NotExtendedPDFError(msg)
        return params

    def _convert_input_binned_x(self, x, none_is_space=None):
        if x is None and none_is_space:
            return self.space
        if isinstance(x, PlottableHistogram) and not isinstance(x, ZfitBinnedData):
            x = BinnedData.from_hist(x)
        if not isinstance(x, ZfitBinnedData | ZfitSpace | ZfitUnbinnedData):
            # TODO: should we allow spaces? Or what?
            try:
                x = Data.from_tensor(obs=self.obs, tensor=x)
            except Exception as error:
                raise TypeError(
                    f"Data to {self} has to be Binned Data, not {x}. (It can also be unbinned Data)"
                    + " but conversion to it failed (see also above) with the following error:"
                    + f" {error})"
                ) from error
        return x

    def _check_convert_norm(self, norm, none_is_error=False):
        if norm is None:
            norm = self.norm
        if norm is None:
            if none_is_error:
                msg = "norm cannot be None for this function."
                raise ValueError(msg)
        elif norm is not False and not isinstance(norm, ZfitSpace):
            msg = f"`norm` needs to be a binned ZfitSpace, not {norm}."
            raise TypeError(msg)
        elif norm is not False and not norm.is_binned:
            norm = norm.with_binning(self.space.binning)
        return norm

    def _check_convert_limits(self, limits):
        if limits is None:
            limits = self.space
        if not isinstance(limits, ZfitSpace):
            limits = convert_to_space(obs=self.obs, limits=limits)
        if isinstance(limits, ZfitSpace) and not limits.is_binned:
            limits = limits.with_binning(self.space.binning)
        return limits

    @_BinnedPDF_register_check_support(True)
    def _pdf(self, x, norm):
        return self._call_rel_counts(x, norm=norm) / np.prod(self.space.binning.widths, axis=0)

    @deprecated_norm_range
    def pdf(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Probability density function, evaluated at ``x`` or in the bins of ``x``

        Args:
            x: |@doc:binnedpdf.pdf.x| Values to evaluate the PDF at.
               If this is a ``ZfitBinnedData``-like object, a histogram of *densities*
               will be returned. If x is a ``ZfitUnbinnedData``-like object, the densities will be
               evaluated at the points of ``x``. |@docend:binnedpdf.pdf.x|
            norm: |@doc:pdf.pdf.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). |@docend:pdf.pdf.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            ``Array-like``: probability density
        """

        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)

        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space
        with self._check_set_input_params(params=params):
            values = self._call_pdf(x, norm=norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    @z.function(wraps="model_binned")
    def _call_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_pdf(x, norm)
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_rel_counts(x, norm=norm) / np.prod(self.space.binning.widths, axis=0)
        return self._fallback_pdf(x, norm=norm)

    def _auto_pdf(self, x, norm):
        try:
            pdf = self._pdf(x, norm=norm)
        except NormNotImplemented:
            unnormed_pdf = self._pdf(x, norm=False)
            normalization = self.normalization(norm)
            pdf = unnormed_pdf / normalization
        return pdf

    def _fallback_pdf(self, x, norm):
        values = self._call_unnormalized_pdf(x)
        if norm is not False:
            values = values / self.normalization(norm)
        return values

    def _unnormalized_pdf(self, x):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def _call_unnormalized_pdf(self, x):
        return self._unnormalized_pdf(x)

    @_BinnedPDF_register_check_support(True)
    def _ext_pdf(self, x, norm):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_norm_range
    def ext_pdf(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Probability density function scaled by yield, evaluated at ``x`` or in the bins of ``x``

        Args:
            x: |@doc:binnedpdf.pdf.x| Values to evaluate the PDF at.
               If this is a ``ZfitBinnedData``-like object, a histogram of *densities*
               will be returned. If x is a ``ZfitUnbinnedData``-like object, the densities will be
               evaluated at the points of ``x``. |@docend:binnedpdf.pdf.x|
            norm: |@doc:pdf.pdf.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). |@docend:pdf.pdf.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            |@doc:binnedpdf.out.problike| If the input was unbinned, it returns an array
               of shape (nevents,). If the input was binned, the dimensions and ordering of
               the axes corresponds to the input axes. |@docend:binnedpdf.out.problike|
        """
        if not self.is_extended:
            raise NotExtendedPDFError
        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)
        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space

        with self._check_set_input_params(params=params):
            values = self._call_ext_pdf(x, norm=norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    @z.function(wraps="model_binned")
    def _call_ext_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_pdf(x, norm)
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_counts(x, norm=norm) / np.prod(self.space.binning.widths, axis=0)
        return self._fallback_ext_pdf(x, norm=norm)

    def _auto_ext_pdf(self, x, norm):
        try:
            pdf = self._ext_pdf(x, norm=norm)
        except NormNotImplemented:
            unnormed_pdf = self._ext_pdf(x, norm=False)
            normalization = self.normalization(norm)
            pdf = unnormed_pdf / normalization
        return pdf

    def _fallback_ext_pdf(self, x, norm):
        values = self._call_pdf(x, norm=norm)
        return values * self.get_yield()

    @_BinnedPDF_register_check_support(True)
    def _log_pdf(self, x, *, norm, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def log_pdf(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Log probability density function, evaluated at ``x`` or in the bins of ``x``

        Args:
            x: |@doc:binnedpdf.pdf.x| Values to evaluate the PDF at.
               If this is a ``ZfitBinnedData``-like object, a histogram of *densities*
               will be returned. If x is a ``ZfitUnbinnedData``-like object, the densities will be
               evaluated at the points of ``x``. |@docend:binnedpdf.pdf.x|
            norm: |@doc:pdf.pdf.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). |@docend:pdf.pdf.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            |@doc:binnedpdf.out.problike| If the input was unbinned, it returns an array
               of shape (nevents,). If the input was binned, the dimensions and ordering of
               the axes corresponds to the input axes. |@docend:binnedpdf.out.problike|
        """
        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)
        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space

        with self._check_set_input_params(params=params):
            values = self._call_log_pdf(x, norm=norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    @z.function(wraps="model_binned")
    def _call_log_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_log_pdf(x, norm)
        with suppress(SpecificFunctionNotImplemented):
            return znp.log(self._call_pdf(x, norm=norm))
        return self._fallback_log_pdf(x, norm=norm)

    def _auto_log_pdf(self, x, norm):
        try:
            pdf = self._log_pdf(x, norm=norm)
        except NormNotImplemented:
            unnormed_pdf = self._log_pdf(x, norm=False)
            log_normalization = self.log_normalization(norm)
            pdf = unnormed_pdf - log_normalization
        return pdf

    def _fallback_log_pdf(self, x, norm):
        values = self._call_pdf(x, norm=norm)
        return znp.log(values)

    @_BinnedPDF_register_check_support(True)
    def _ext_log_pdf(self, x, *, norm, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def ext_log_pdf(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Log probability density function scaled by yield, evaluated at ``x`` or in the bins of ``x``

        Args:
            x: |@doc:binnedpdf.pdf.x| Values to evaluate the PDF at.
               If this is a ``ZfitBinnedData``-like object, a histogram of *densities*
               will be returned. If x is a ``ZfitUnbinnedData``-like object, the densities will be
               evaluated at the points of ``x``. |@docend:binnedpdf.pdf.x|
            norm: |@doc:pdf.pdf.norm| Normalization of the function.
               By default, this is the ``norm`` of the PDF (which by default is the same as
               the space of the PDF). |@docend:pdf.pdf.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            |@doc:binnedpdf.out.problike| If the input was unbinned, it returns an array
               of shape (nevents,). If the input was binned, the dimensions and ordering of
               the axes corresponds to the input axes. |@docend:binnedpdf.out.problike|
        """
        if not self.is_extended:
            raise NotExtendedPDFError
        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)
        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space

        with self._check_set_input_params(params=params):
            values = self._call_ext_log_pdf(x, norm=norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    def _call_ext_log_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_log_pdf(x, norm)
        with suppress(SpecificFunctionNotImplemented):
            return znp.log(self._call_ext_pdf(x, norm=norm))
        return self._fallback_ext_log_pdf(x, norm=norm)

    def _auto_ext_log_pdf(self, x, norm):
        try:
            pdf = self._ext_log_pdf(x, norm=norm)
        except NormNotImplemented:
            unnormed_pdf = self._ext_log_pdf(x, norm=False)
            log_normalization = self.log_normalization(norm)
            pdf = unnormed_pdf - log_normalization
        return pdf

    def _fallback_ext_log_pdf(self, x, norm):
        values = self._call_ext_pdf(x, norm=norm)
        return znp.log(values)

    @_BinnedPDF_register_check_support(True)
    def _log_normalization(self, norm, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    @deprecated_args(None, "Use `norm` instead.", "limits")
    def log_normalization(self, norm, *, params=None, options=None, limits=None) -> ztyping.NumericalTypeReturn:
        """Normalization of the PDF. For a binned PDF, this is the sum over the counts or the integral over the density.

        Args:
            norm: |@doc:pdf.normalization.norm| Normalization of the function.
                By default, this is the ``norm`` of the PDF (which by default is the same as
                the space of the PDF). |@docend:pdf.normalization.norm|
            options:   |@doc:pdf.normalization.options| Additional options for the normalization.
                Currently supported options are:
                - type: one of (``bins``)
                  This hints that bins are integrated. A method that is vectorizable, non-dynamic and
                  therefore less suitable for complicated functions is chosen. |@docend:pdf.normalization.options|

            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
        """
        if limits is not None:
            norm = limits
        if options is None:
            options = {}
        norm = self._check_convert_norm(norm, none_is_error=True)
        with self._check_set_input_params(params=params):
            return self._call_log_normalization(norm, options=options)

    def _call_log_normalization(self, norm, *, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._log_normalization(norm, options=options)

        # fallback
        return znp.log(self._call_integrate(norm, norm=False, options=None))

    @_BinnedPDF_register_check_support(True)
    def _normalization(self, limits, *, options, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def normalization(self, norm=None, *, params=None, options=None, limits=None) -> ztyping.NumericalTypeReturn:
        """Normalization of the PDF. For a binned PDF, this is the sum over the counts or the integral over the density.

        Args:
            norm: |@doc:pdf.normalization.norm| Normalization of the function.
                By default, this is the ``norm`` of the PDF (which by default is the same as
                the space of the PDF). |@docend:pdf.normalization.norm|
            options:   |@doc:pdf.normalization.options| Additional options for the normalization.
                Currently supported options are:
                - type: one of (``bins``)
                  This hints that bins are integrated. A method that is vectorizable, non-dynamic and
                  therefore less suitable for complicated functions is chosen. |@docend:pdf.normalization.options|

            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
        """
        if limits is not None:
            msg = "Use `norm` instead of `limits`."
            raise BreakingAPIChangeError(msg)
        if options is None:
            options = {}
        norm = self._check_convert_norm(norm)
        with self._check_set_input_params(params=params):
            return self._call_normalization(norm, options=options)

    def _call_normalization(self, norm, *, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._normalization(norm, options=options)

        # fallback
        return self._call_integrate(norm, norm=False, options=None)

    @_BinnedPDF_register_check_support(True)
    def _integrate(self, limits, norm, options, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Integral of the PDF, the sum over all the bins normalized to the total integral.

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
            Scalar integration value.
        """
        if options is None:
            options = {}
        norm = self._check_convert_norm(norm)
        limits = self._check_convert_limits(limits)
        with self._check_set_input_params(params=params):
            return self._call_integrate(limits, norm, options)

    @z.function(wraps="model_binned")
    def _call_integrate(self, limits, norm, options=None):
        if options is None:
            options = {}
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_integrate(limits, norm, options)
        return self._fallback_integrate(limits, norm, options)

    def _fallback_integrate(self, limits, norm, options):
        del options  # not used
        bincounts = self._call_rel_counts(limits, norm=norm)  # TODO: fake data? not to integrate limits?
        edges = limits.binning.edges
        return binned_rect_integration(counts=bincounts, edges=edges, limits=limits)  # TODO: check integral, correct?

    def _auto_integrate(self, limits, norm, options):
        try:
            integral = self._integrate(limits=limits, norm=norm, options=options)
        except NormNotImplemented:
            unnormalized_integral = self._auto_integrate(limits=limits, norm=False, options=options)
            normalization = self.normalization(norm, options=options)
            integral = unnormalized_integral / normalization
        except MultipleLimitsNotImplemented:
            integrals = []  # TODO: map?
            for sub_limits in limits:
                integrals.append(self._auto_integrate(limits=sub_limits, norm=norm, options=options))
            integral = z.reduce_sum(integrals, axis=0)  # TODO: remove stack?
        return integral

    @deprecated_norm_range
    def ext_integrate(
        self,
        limits: ztyping.LimitsType,
        norm: ztyping.LimitsType = None,
        *,
        options=None,
        params: ztyping.ParamTypeInput = None,
    ) -> ztyping.XType:
        """Extended integral of the PDF, i.e. the expected counts or the integral scaled by the yield.

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
            Scalar integration value.
        """
        if not self.is_extended:
            raise NotExtendedPDFError
        if options is None:
            options = {}
        norm = self._check_convert_norm(norm)
        limits = self._check_convert_limits(limits)
        with self._check_set_input_params(params=params):
            return self._call_ext_integrate(limits, norm, options=options)

    @z.function(wraps="model_binned")
    def _call_ext_integrate(self, limits, norm, *, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_integrate(limits, norm, options=options)
        return self._fallback_ext_integrate(limits, norm, options=options)

    def _fallback_ext_integrate(self, limits, norm, *, options):  # TODO: rather use pdf?
        del options  # not used
        bincounts = self._call_counts(limits, norm=norm)  # TODO: fake data? not to integrate limits?
        edges = limits.binning.edges
        return binned_rect_integration(counts=bincounts, edges=edges, limits=limits)

    def _auto_ext_integrate(self, limits, norm, *, options):
        try:
            integral = self._ext_integrate(limits=limits, norm=norm, options=options)
        except NormNotImplemented:
            unnormalized_integral = self._auto_ext_integrate(limits=limits, norm=False, options=options)
            normalization = self.ext_normalization(norm, options=options)
            integral = unnormalized_integral / normalization
        except MultipleLimitsNotImplemented:
            integrals = []  # TODO: map?
            for sub_limits in limits:
                integrals.append(self._auto_integrate(limits=sub_limits, norm=norm, options=options))
            integral = z.reduce_sum(integrals, axis=0)  # TODO: remove stack?
        return integral

    @_BinnedPDF_register_check_support(True)
    def _ext_integrate(self, limits, norm, *, options, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def create_sampler(
        self,
        n: ztyping.nSamplingTypeIn = None,
        limits: ztyping.LimitsType = None,
        *,
        params: ztyping.ParamTypeInput = None,
        fixed_params: bool | list[ZfitParameter] | tuple[ZfitParameter] = True,  # todo: use params instead only?
    ) -> BinnedSamplerData:
        """Create a :py:class:`SamplerData` that acts as `Data` but can be resampled, also with changed parameters and
        n.

            If `limits` is not specified, `space` is used (if the space contains limits).
            If `n` is None and the model is an extended pdf, 'extended' is used by default.


        Args:
            n: The number of samples to be generated. Can be a Tensor that will be
                or a valid string. Currently implemented:

                    - 'extended': samples `poisson(yield)` from each pdf that is extended.

            limits: From which space to sample.
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|
            fixed_params: A list of `Parameters` that will be fixed during several `resample` calls.
                If True, all are fixed, if False, all are floating. If a :py:class:`~zfit.Parameter` is not fixed and
                its
                value gets updated (e.g. by a `Parameter.set_value()` call), this will be reflected in
                `resample`. If fixed, the Parameter will still have the same value as the `SamplerData` has
                been created with when it resamples.

        Returns:
            :py:class:`~zfit.core.data.BinnedSampler`

        Raises:
            NotExtendedPDFError: if 'extended' is chosen (implicitly by default or explicitly) as an
                option for `n` but the pdf itself is not extended.
            ValueError: if n is an invalid string option.
            InvalidArgumentError: if n is not specified and pdf is not extended.
        """

        if n is None:
            if self.is_extended:
                n = znp.random.poisson(self.get_yield(), size=1)
            else:
                msg = f"n cannot be None for sampling of {self} or needs to be extended."
                raise ValueError(msg)
        limits = self._check_convert_limits(limits)

        if fixed_params is not None:
            if params is not None:
                msg = "Cannot specify both `fixed_params` and `params`. Use `params` only."
                raise ValueError(msg)
            if fixed_params is False:
                msg = (
                    "`fixed_params` has been removed, the sampler will always sample from the parameters at the time of the creation/given to the creator"
                    " _or_ by giving params to the `resample` method."
                )
                raise BreakingAPIChangeError(msg)
            if fixed_params is True:
                fixed_params = None  # default behavior is to catch all anyway
            if isinstance(fixed_params, list | tuple):
                fixed_params = {param.name: znp.asarray(param) for param in fixed_params}
            params = fixed_params

        # legacy end
        params = self._check_convert_input_paramvalues(params=params)
        params = {
            p.name: params[pname] if (pname := p) in params or (pname := p.name) in params else p.value()
            for p in self.get_params(floating=None, is_yield=None)
        }

        def sample_and_variances_func(n, params, *, limits=limits):
            sample = self.sample(n=n, limits=limits, params=params)
            return sample.values(), sample.variances()

        return BinnedSamplerData.from_sampler(
            sample_and_variances_func=sample_and_variances_func,
            n=n,
            obs=limits,
            params=params,
        )

    @z.function(wraps="sampler")
    def _create_sampler_tensor(self, limits, n):
        return self._call_sample(n=n, limits=limits)

    def sample(
        self, n: int | None = None, limits: ztyping.LimitsType = None, *, params: ztyping.ParamTypeInput = None
    ) -> ZfitBinnedData:
        """Draw a random binned sample from the PDF.

        Args:
            n: |@doc:pdf.sample.n| Number of samples to draw.
               For an extended PDF, the argument is optional and will be the
               poisson-fluctuated expected number of events, i.e. the yield. |@docend:pdf.sample.n|
            limits: |@doc:pdf.sample.limits| Limits of the sampling.
               By default, this is the same as the default space of the PDF. |@docend:pdf.sample.limits|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            ``ZfitBinnedData``: Sampled dataset
        """
        if n is None:
            if self.is_extended:
                n = znp.random.poisson(self.get_yield(), size=1)
            else:
                msg = f"n cannot be None for sampling of {self} or needs to be extended."
                raise ValueError(msg)
        original_limits = limits
        limits = self._check_convert_limits(limits)
        with self._check_set_input_params(params=params):
            values = self._call_sample(n, limits)
        if not isinstance(values, ZfitBinnedData):
            values = BinnedData.from_tensor(space=limits, values=values, variances=None)
        if isinstance(original_limits, ZfitSpace):
            values = values.with_obs(original_limits)
        return values

    @z.function(wraps="sample")
    def _call_sample(self, n, limits):
        with suppress(SpecificFunctionNotImplemented):
            self._sample(n, limits)
        return self._fallback_sample(n, limits)

    def _fallback_sample(self, n, limits):
        if limits != self.space:
            msg = (
                "limits different from the default are not yet available."
                " Please open an issue if you need this:"
                " https://github.com/zfit/zfit/issues/new/choose"
            )
            raise WorkInProgressError(msg)
        probs = self.rel_counts(limits)
        return z.random.counts_multinomial(n, probs=probs, dtype=znp.float64)

    # ZfitMinimalHist implementation
    def values(self, *, var=None, params=None):
        """Histogram values that are either the counts or the normalized counts.

        If the PDF is extended, the counts are returned, otherwise the normalized counts are returned.

        Args:
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            ``ZfitBinnedData``: Histogram values
        """
        if var is not None:
            msg = "var argument for `values` is not supported in V1"
            raise RuntimeError(msg)
        if self.is_extended:
            return self.counts(var, params=params)
        else:
            return self.rel_counts(var, params=params)

    def update_integration_options(self, *_, **__):
        msg = "Integration options not available for BinnedPDF"
        raise RuntimeError(msg)

    def as_func(self, norm: ztyping.LimitsType = False):  # noqa: ARG002
        msg = "as_func not yet available for BinnedPDF"
        raise WorkInProgressError(msg)

    @property
    def is_extended(self) -> bool:
        return self._yield is not None

    def set_norm(self, norm):  # noqa: ARG002
        msg = "set_norm should not be used anymore. Create a new PDF with the desired normalization."
        raise RuntimeError(msg)

    def create_extended(self, yield_: ztyping.ParamTypeInput) -> ZfitPDF:  # noqa: ARG002
        msg = "create_extended not available for BinnedPDF. Use `extended` in the initialization instead."
        raise WorkInProgressError(msg)

    def get_yield(self) -> ZfitParameter | None:
        if not self.is_extended:
            raise NotExtendedPDFError
        return self._yield

    @classmethod
    def register_analytic_integral(
        cls,
        func: Callable,  # noqa: ARG003
        limits: ztyping.LimitsType = None,  # noqa: ARG003
        priority: int = 50,  # noqa: ARG003
        *,
        supports_norm: bool = False,  # noqa: ARG003
        supports_multiple_limits: bool = False,  # noqa: ARG003
    ):
        msg = "analytic integral not available for BinnedPDF"
        raise RuntimeError(msg)

    @deprecated_norm_range
    def partial_integrate(
        self,
        x: ztyping.XType,  # noqa: ARG002
        limits: ztyping.LimitsType,  # noqa: ARG002
        *,
        norm=None,  # noqa: ARG002
        options=None,  # noqa: ARG002
    ) -> ztyping.XType:
        msg = "partial_integrate not yet available for BinnedPDF"
        raise WorkInProgressError(msg)

    @classmethod
    def register_inverse_analytic_integral(cls, func: Callable):  # noqa: ARG003
        msg = "inverse analytic integral not available for BinnedPDF. It's a histogram, it's already 'analytic'"
        raise RuntimeError(msg)

    @_BinnedPDF_register_check_support(True)
    def _sample(self, n, limits, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def _copy(self, deep, name, overwrite_params):  # noqa: ARG002
        raise WorkInProgressError

    # factor out with unbinned pdf

    @property
    def norm(self):
        norm = self._norm
        if norm is None:
            norm = self.space
        return norm

    @property
    @deprecated(None, "use `norm` instead.")
    def norm_range(self):
        return self.norm

    # TODO: factor out with unbinned pdf

    def _convert_sort_space(
        self,
        obs: ztyping.ObsTypeInput | ztyping.LimitsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        limits: ztyping.LimitsTypeInput = None,
    ) -> ZfitSpace | None:
        """Convert the inputs (using eventually ``obs``, ``axes``) to :py:class:`~zfit.ZfitSpace` and sort them
        according to own ``obs``.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        elif not set(obs).intersection(self.obs):
            msg = "The given space {obs} is not compatible with the obs of the pdfs{self.obs}; they are disjoint."
            raise SpaceIncompatibleError(msg)
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:  # e.g. not the first call
            space = space.with_coords(self.space, allow_superset=True, allow_subset=True)
        return space

    def counts(
        self,
        x: ztyping.BinnedDataInputType = None,
        norm: ztyping.NormInputType = None,
        *,
        params: ztyping.ParamTypeInput = None,
    ) -> ZfitBinnedData:
        """Calculate the number of events in each bin.

        This is the integrals of the PDF in each bin.

        Args:
            x: |@doc:pdf.binned.counts.x| Data for the binned PDF.
               The returned counts correspond to the binned axis in ``x``. |@docend:pdf.binned.counts.x|
            norm: |@doc:pdf.binned.counts.norm| Normalization of the counts.
               This normalizes the counts so that the actual sum of all counts is
               equal to the yield. |@docend:pdf.binned.counts.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            ZfitBinnedData: A histogram with the number of events in each bin.
        """

        if not self.is_extended:
            raise NotExtendedPDFError
        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)
        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space

        with self._check_set_input_params(params=params):
            values = self._call_counts(x, norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    @z.function(wraps="model_binned")
    def _call_counts(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_counts(x, norm)
        return self._fallback_counts(x, norm)

    def ext_normalization(
        self, norm: ztyping.NormInputType, *, options=None, params: ztyping.ParamTypeInput = None
    ) -> ztyping.NumericalTypeReturn:
        """Calculate the normalization of the extended PDF, for a binned PDF this is the sum of the counts.

        Args:
            norm: |@doc:pdf.normalization.norm| Normalization of the function.
                By default, this is the ``norm`` of the PDF (which by default is the same as
                the space of the PDF). |@docend:pdf.normalization.norm|
            options:  |@doc:pdf.normalization.options| Additional options for the normalization.
                Currently supported options are:
                - type: one of (``bins``)
                    This hints that bins are integrated. A method that is vectorizable, non-dynamic and
                    therefore less suitable for complicated functions is chosen. |@docend:pdf.normalization.options|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            Scalar-like: The normalization of the extended PDF.
        """
        if not self.is_extended:
            raise NotExtendedPDFError
        if options is None:
            options = {}
        norm = self._check_convert_norm(norm, none_is_error=True)
        with self._check_set_input_params(params=params):
            return self._call_ext_normalization(norm, options=options)

    def _call_ext_normalization(self, norm, *, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._ext_normalization(norm, options=options)
        # fallback
        return self.normalization(norm, options=options) / self.get_yield()

    def _ext_normalization(self, norm, *, options):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def _auto_counts(self, x, norm):
        try:
            counts = self._counts(x, norm=norm)
        except NormNotImplemented:
            unnormed_counts = self._counts(x, norm=False)
            normalization = self.normalization(norm)
            counts = unnormed_counts / normalization
        return counts

    def _fallback_counts(self, x, norm):
        return self._auto_rel_counts(x, norm) * self.get_yield()

    @_BinnedPDF_register_check_support(True)
    def _counts(self, x, norm):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def rel_counts(
        self,
        x: ztyping.BinnedDataInputType = None,
        *,
        norm: ztyping.NormInputType = None,
        params: ztyping.ParamTypeInput = None,
    ) -> ZfitBinnedData:
        """Calculate the relative number of events in each bin.

        This is the integrals of the PDF in each bin divided by the integral of the PDF over the whole space.
        It is *not* equal to the density but rather a histogram scaled to 1.

        Args:
            x: |@doc:pdf.binned.counts.x| Data for the binned PDF.
               The returned counts correspond to the binned axis in ``x``. |@docend:pdf.binned.counts.x|
            norm: |@doc:pdf.binned.counts.norm| Normalization of the counts.
               This normalizes the counts so that the actual sum of all counts is
               equal to the yield. |@docend:pdf.binned.counts.norm|
            params: |@doc:model.args.params| Mapping of the parameter names to the actual
               values. The parameter names refer to the names of the parameters,
               typically :py:class:`~zfit.Parameter`, that
               the model was _initialized_ with, not the name of the models
               parametrization. |@docend:model.args.params|

        Returns:
            ZfitBinnedData: A histogram with the relative number of events in each bin.
        """
        # convert the input argument to a standardized form
        x = self._convert_input_binned_x(x, none_is_space=True)
        norm = self._check_convert_norm(norm, none_is_error=True)
        # sort it and remember the original sorting
        original_space = x if isinstance(x, ZfitSpace) else x.space
        x = x.with_obs(self.space.obs)  # we don't want to cut anything off -> only obs, not space

        # if it is unbinned, we get the binned version and gather the corresponding values
        is_unbinned = isinstance(x, ZfitUnbinnedData)
        binindices = None
        if is_unbinned:
            binindices = unbinned_to_binindex(x, self.space, flow=True)
            x = self.space

        with self._check_set_input_params(params=params):
            values = self._call_rel_counts(x, norm)

        if binindices is not None:  # because we have the flow, so we need to make it here with pads
            padded_values = znp.pad(
                values,
                znp.ones((z._get_ndims(values), 2), dtype=znp.float64),
                mode="constant",
            )  # for overflow
            ordered_values = tf.gather_nd(padded_values, indices=binindices)
        else:
            ordered_values = move_axis_obs(self.space, original_space, values)[0]  # only use values, not variance
        return znp.asarray(ordered_values)

    @z.function(wraps="model_binned")
    def _call_rel_counts(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_rel_counts(x, norm=norm)
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_counts(x, norm=norm) / self.get_yield()
        return self._fallback_rel_counts(x, norm)

    @_BinnedPDF_register_check_support(True)
    def _rel_counts(self, x, norm, *, params=None):  # noqa: ARG002
        raise SpecificFunctionNotImplemented

    def _auto_rel_counts(self, x, norm):
        try:
            rel_counts = self._rel_counts(x, norm=norm)
        except NormNotImplemented:
            unnormed_rel_counts = self._rel_counts(x, norm=False)
            normalization = self.normalization(norm)
            rel_counts = unnormed_rel_counts / normalization
        return rel_counts

    def _fallback_rel_counts(self, x, norm):
        density = self._call_pdf(x, norm)
        return density * np.prod(self.space.binning.widths, axis=0)

    def set_norm_range(self):
        msg = "set_norm_range is removed and should not be used anymore."
        raise RuntimeError(msg)

    def to_binned(self, space, *, extended=None, norm=None):
        """Convert the PDF to a binned PDF, returns self.

        For compatibility with unbinned PDFs.
        """
        if isinstance(space, ZfitBinning) and space != self.space.binning:
            msg = "The binning of the PDF and the binning of the space must be equal."
            raise ValueError(msg)
        if space != self.space:
            msg = f"Space must be the same as the PDF's space, as {self} is already a binned PDF."
            raise ValueError(msg)
        if extended is not None:
            msg = "extended is not implemented yet. Create an extended PDF manually."
            raise WorkInProgressError(msg)
        if norm is not None:
            msg = "norm is not implemented yet. Create a pdf with a different norm range manually."
            raise WorkInProgressError(msg)
        return self

    def to_unbinned(self):
        """Convert the PDF to an unbinned PDF."""
        from zfit.models.unbinnedpdf import UnbinnedFromBinnedPDF  # noqa: PLC0415

        return UnbinnedFromBinnedPDF(self, self.space.with_binning(None))


BaseBinnedPDFV1 = BaseBinnedPDF  # alias, backwards compatibility


def binned_rect_integration(
    *,
    limits: ZfitSpace,
    edges: Iterable[znp.array] | znp.array,
    counts: znp.array | None = None,
    density: znp.array | None = None,
    axis: Iterable[int] | int | None = None,
) -> znp.array:
    """Integrate a histogram over *limits*.

    This integrator does take into account that limits do not match the edges.

    Args:
        limits: Limits to integrate over. A possible binning is ignored.
        edges: The edges per axis. They should have the shape ``(1,..., 1, n, 1, ..., 1)``, where n is the *ith* axis.
            ``ZfitBinning`` provides this format on the ``edges` attribute.
        counts: Counts of the histogram. This is what most histograms have and is equal to the density multiplied by
            the binwidth.
            Exactly one of counts or density has to be provided.
        density: The density of a histogram is the bincount divided by the binwidth.
            Exactly one of counts or density has to be provided.
        axis: Which axes to integrate over. Defaults to all.

    Returns:
        Integral with shape corresponding to the non-integrated axes (or a scalar in case of all axes integrated).
    """
    edges = convert_to_container(edges)
    if not isinstance(limits, ZfitSpace):
        msg = f"limits has to be a ZfitSpace, not {limits}."
        raise TypeError(msg)
    if counts is not None:
        if density is not None:
            msg = "Either specify 'counts' or 'density' but not both."
            raise ValueError(msg)
        is_density = False
        values = counts
    elif density is not None:
        is_density = True
        values = density
    else:
        msg = "Need to specify either 'counts' or 'density', not None."
        raise ValueError(msg)
    ndims = z._get_ndims(values)
    # partial = axis is not None and len(axis) < ndims
    if axis is not None:
        axis = convert_to_container(axis)
        if len(axis) > ndims:
            msg = f"axis {axis} is larger than values has ndims {values.shape}."
            raise ValueError(msg)
    else:
        axis = list(range(ndims))

    scaled_edges, (lower_bins, upper_bins), unscaled_edges = cut_edges_and_bins(
        edges=edges, limits=limits, axis=axis, unscaled=True
    )

    values_cut = tf.slice(values, lower_bins, (upper_bins - lower_bins))  # since limits are inclusive

    rank = values.shape.rank
    binwidths = []
    if not is_density:
        binwidths_unscaled = []
    # calculate the binwidth in each dimension
    for i, edge in enumerate(scaled_edges):
        edge_lower_index = [0] * rank
        # int32 is needed! Otherwise the gradient will fail
        edge_lowest_index = znp.array(edge_lower_index, dtype=znp.int32)

        edge_lower_index[i] = 1
        edge_lower_index = znp.array(edge_lower_index, dtype=znp.int32)
        edge_upper_index = [1] * rank
        edge_highest_index = edge_upper_index.copy()
        len_edge = tf.shape(edge)[i]
        edge_highest_index[i] = len_edge
        edge_highest_index = znp.asarray(edge_highest_index, dtype=znp.int32)
        edge_upper_index[i] = len_edge - 1  # len n -> index max is n - 1

        edge_upper_index = znp.asarray(edge_upper_index, dtype=znp.int32)
        lower_edge = tf.slice(edge, edge_lowest_index, (edge_upper_index - edge_lowest_index))
        upper_edge = tf.slice(edge, edge_lower_index, (edge_highest_index - edge_lower_index))
        binwidths.append(upper_edge - lower_edge)

        if not is_density:
            # unscaled edges to get the ratio
            lower_edge_unscaled = tf.slice(
                unscaled_edges[i],
                edge_lowest_index,
                (edge_upper_index - edge_lowest_index),
            )
            upper_edge_unscaled = tf.slice(
                unscaled_edges[i],
                edge_lower_index,
                (edge_highest_index - edge_lower_index),
            )
            binwidths_unscaled.append(upper_edge_unscaled - lower_edge_unscaled)

    binareas = reduce(operator.mul, binwidths)  # needs to be python as znp or tf can't broadcast otherwise
    if not is_density:  # scale the counts by the fraction. This is mostly one.
        binareas_uncut = reduce(operator.mul, binwidths_unscaled)
        # binareas_uncut = znp.prod(binwidths_unscaled, axis=0)
        binareas /= binareas_uncut
    values_cut *= binareas
    return tf.reduce_sum(values_cut, axis=axis)


@z.function(wraps="tensor", keepalive=True)
def cut_edges_and_bins(
    edges: Iterable[znp.array], limits: ZfitSpace, axis=None, unscaled=None
) -> tuple[list[znp.array], tuple[znp.array, znp.array], list | None]:
    """Cut the *edges* according to *limits* and calculate the bins inside.

    The edges within limits are calculated and returned together with the corresponding bin indices. The indices
    mark the lowest and the highest index of the edges that are returned. Additionally, the unscaled edges are returned.

    If the limits are between two edges, this will be treated as the new edge. If the limits are outside the edges,
    all edges in this direction will be returned (but not extended to the limit). For example:

    [0, 0.5, 1., 1.5, 2.] and the limits (0.8, 3.) will return [0.8, 1., 1.5, 2.], ([1], [4])

    .. code-block::

        cut_edges_and_bins([[0., 0.5, 1., 1.5, 2.]], ([[0.8]], [[3]]))



    Args:
        edges: Iterable of tensor-like objects that describe the edges of a histogram. Every object should have rank n
            (where n is the length of *edges*) but only have the dimension i filled out. These are
            tensors that are ready to be broadcasted together.
        limits: The limits that will be used to confine the edges


    Returns:
        edges, (lower bins, upper bins), unscaled_edges:  The edges and the bins are returned.
            The upper bin number corresponds to
            the highest bin which was still (partially) inside the limits **plus one** (so it's the index of the
            edge that is right outside). The unscaled edges are like *edges* but the last edge is the edge
            that is lying not inside anymore, so the actual edge of the last bin number returend.
            This can be used to determine the fraction cut away.
    """
    if axis is not None:
        axis = convert_to_container(axis)
    if unscaled is None:
        unscaled = False
    cut_unscaled_edges = [] if unscaled else None
    cut_scaled_edges = []

    all_lower_bins = []
    all_upper_bins = []
    if isinstance(limits, ZfitSpace):  # todo: check shapes?
        lower, upper = limits.v0.limits
    else:
        lower, upper = limits
        lower = znp.asarray(lower)
        upper = znp.asarray(upper)
    lower_all = lower[0]
    upper_all = upper[0]
    rank = len(edges)
    current_axis = 0
    for i, edge in enumerate(edges):
        edge = znp.asarray(edge)
        edge = znp.reshape(edge, (-1,))
        if axis is None or i in axis:
            lower_i = lower_all[current_axis, None]
            edge_minimum = edge[0]
            # edge_minimum = tf.gather(edge, indices=0, axis=i)
            lower_i = znp.maximum(lower_i, edge_minimum)
            upper_i = upper_all[current_axis, None]
            edge_maximum = edge[-1]
            # edge_maximum = tf.gather(edge, indices=tf.shape(edge)[i] - 1, axis=i)
            upper_i = znp.minimum(upper_i, edge_maximum)
            # we get the bins that are just one too far. Then we update this whole bin tensor with the actual edge.
            # The bins index is the index below the value.
            lower_bin_float = tfp.stats.find_bins(lower_i, edge, extend_lower_interval=True, extend_upper_interval=True)
            lower_bin = znp.reshape(znp.asarray(lower_bin_float, dtype=znp.int32), [-1])
            # lower_bins = tf.tensor_scatter_nd_update(zero_bins, [[i]], lower_bin)
            # +1 below because the outer bin is searched, meaning the one that is higher than the value

            upper_bin_float = tfp.stats.find_bins(upper_i, edge, extend_lower_interval=True, extend_upper_interval=True)
            upper_bin = znp.reshape(znp.asarray(upper_bin_float, dtype=znp.int32), [-1]) + 1
            size = upper_bin - lower_bin
            new_edge = tf.slice(edge, lower_bin, size + 1)  # +1 because stop is exclusive
            new_edge = tf.tensor_scatter_nd_update(new_edge, [tf.constant([0]), size], [lower_i[0], upper_i[0]])

            if unscaled:
                new_edge_unscaled = tf.slice(edge, lower_bin, size + 1)  # +1 because stop is exclusive

            current_axis += 1
        else:
            lower_bin = [0]
            upper_bin = znp.asarray([edge.shape[0] - 1], dtype=znp.int32)
            new_edge = edge
            if unscaled:
                new_edge_unscaled = edge
        new_shape = [1] * rank
        new_shape[i] = -1
        new_edge = znp.reshape(new_edge, new_shape)
        all_lower_bins.append(lower_bin)
        all_upper_bins.append(upper_bin)
        cut_scaled_edges.append(new_edge)
        if unscaled:
            new_edge_unscaled = znp.reshape(new_edge_unscaled, new_shape)
            cut_unscaled_edges.append(new_edge_unscaled)

    # partial = axis is not None and len(axis) < rank
    #
    # if partial:
    #     scaled_edges_full = list(edges)
    #     for edge, ax in zip(cut_scaled_edges, axis):
    #         scaled_edges_full[ax] = edge
    #     scaled_edges = scaled_edges_full
    #     indices = tf.convert_to_tensor(axis)[:, None]
    #     lower_bins = tf.scatter_nd(indices, lower_bins, shape=(ndims,))
    #     upper_bins = tf.tensor_scatter_nd_update(tf.convert_to_tensor(values.shape),
    #                                              indices, upper_bins)
    # lower_bins_indices = tf.stack([lower_bins, dims], axis=-1)
    # upper_bins_indices = tf.stack([upper_bins, dims], axis=-1)
    # all_lower_bins = znp.asarray(znp.sum(all_lower_bins, axis=0), dtype=znp.int32)
    all_lower_bins = tf.concat(all_lower_bins, axis=0)
    all_upper_bins = tf.concat(all_upper_bins, axis=0)
    return cut_scaled_edges, (all_lower_bins, all_upper_bins), cut_unscaled_edges
