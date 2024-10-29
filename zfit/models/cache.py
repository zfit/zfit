#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import Callable, Literal

import pydantic.v1 as pydantic
import tensorflow as tf

import zfit.z.numpy as znp
from zfit.util.exception import AnalyticGradientNotAvailable

from ..core.serialmixin import SerializableMixin
from ..core.space import supports
from ..serialization import Serializer  # noqa: F401
from ..settings import ztypes
from ..util import ztyping
from .basefunctor import FunctorPDFRepr
from .functor import BaseFunctor


def get_value(cache: tf.Variable, flag: tf.Variable, func: Callable):
    @tf.custom_gradient
    def actual_func():
        def autoset_func():
            val = func()
            return cache.assign(val, read_value=True)

        def use_cache():
            return cache

        val = tf.cond(flag, use_cache, autoset_func)

        def grad_fn(dval, variables):  # noqa: ARG001
            msg = (
                "The analytic gradient is not implemented for caching PDF. Use the numerical gradient instead."
                "(either using zfit.run.set_autograd_mode(False) and/or by using the minimizer internal numerical gradient)"
            )
            raise AnalyticGradientNotAvailable(msg)

        return val, grad_fn

    return actual_func()


class CachedPDF(BaseFunctor, SerializableMixin):
    def __init__(
        self,
        pdf: ztyping.PDFInputType,
        *,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
        cache_tol=None,
        name: str | None = None,
        label: str | None = None,
    ):
        """Creates a PDF where ``pdf`` and ``integrate`` methods are cacheable.

        .. note::

           Analytic gradients are not available for the cached PDF. Use the numerical gradient instead,
           either by using a minimizers internal calculator (e.g. :py:class:`~zfit.minimize.Minuit(..., gradient=True)`) or by
           setting the autograd mode to False (e.g. :py:func:`~zfit.run.set_autograd_mode(False)`).
           An error will be raised if the analytic gradient is requested.

        The method stores the last calculated value of a function for a specific dataset and
        returns it when the input data and the parameters are the same. This can be useful when
        the pdf is called multiple times with the same data and parameters, for example in the
        minimization process when a numerical gradient is used.

        Args:
            pdf: pdf which methods to be cached
            cache_tol: accuracy of absolute tolerance comparing arguments (parameters, data) with cached values
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
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
        obs = pdf.space
        hs3_init = {
            "obs": obs,
            "extended": extended,
            "norm": norm,
            "cache_tol": cache_tol,
            "name": name,
        }
        name = name or pdf.name
        super().__init__(pdfs=pdf, obs=obs, name=name, extended=extended, norm=norm, label=label, autograd_params=[])
        params = list(pdf.get_params(floating=None, is_yield=None, extract_independent=True))

        param_cache = tf.Variable(
            znp.zeros(shape=tf.shape(tf.stack(params)), dtype=ztypes.float),
            trainable=False,
            validate_shape=False,
            dtype=tf.float64,
        )
        param_cache_int = tf.Variable(
            znp.zeros(shape=tf.shape(tf.stack(params)), dtype=ztypes.float),
            trainable=False,
            validate_shape=False,
            dtype=tf.float64,
        )

        self._cached_pdf_params = param_cache
        self._cached_pdf_params_for_integration = param_cache_int
        self._pdf_cache = None
        self._cached_x = None
        self._pdf_cache_valid = tf.Variable(initial_value=False, trainable=False)

        self._cached_integral_limits = None
        self._integral_cache = None
        self._integral_cache_valid = tf.Variable(initial_value=False, trainable=False)

        self._cache_tolerance = 1e-8 if cache_tol is None else cache_tol
        self.hs3.original_init.update(hs3_init)

    @supports(norm="space")
    def _pdf(self, x, norm):
        x = x.value()
        if self._pdf_cache is None:
            self._pdf_cache = tf.Variable(
                -999.0
                * znp.ones(
                    shape=tf.shape(x)[0],
                ),  # negative ones, to make sure these are unrealistic values
                trainable=False,
                validate_shape=False,
                dtype=ztypes.float,
            )

        if self._cached_x is None:
            self._cached_x = tf.Variable(
                x + 19.0,  # to make sure it's not the same
                trainable=False,
                validate_shape=False,
                dtype=ztypes.float,
            )
        x_same = tf.math.reduce_all(znp.abs(x - self._cached_x) < self._cache_tolerance)
        pdf_params = list(self.pdfs[0].get_params())
        if hasparams := len(pdf_params) > 0:
            stacked_pdf_params = tf.stack(pdf_params)
            params_same = tf.math.reduce_all(
                znp.abs(stacked_pdf_params - self._cached_pdf_params) < self._cache_tolerance
            )

            same_args = tf.math.logical_and(params_same, x_same)
        else:
            same_args = x_same
        assign1 = self._pdf_cache_valid.assign(same_args, read_value=False)

        def value_update_func():
            if hasparams:
                self._cached_pdf_params.assign(stacked_pdf_params, read_value=False)
            self._cached_x.assign(x, read_value=False)
            return self.pdfs[0].pdf(x, norm)

        with tf.control_dependencies([assign1]):
            return get_value(self._pdf_cache, self._pdf_cache_valid, value_update_func)

    @supports(norm="space")
    def _integrate(self, limits, norm, options=None):
        if self._cached_integral_limits is None:
            self._cached_integral_limits = tf.Variable(
                tf.stack(limits.v1.limits) + 19.0,  # to make sure it's not the same
                trainable=False,
                validate_shape=False,
                dtype=ztypes.float,
            )

        if self._integral_cache is None:
            self._integral_cache = tf.Variable(
                znp.zeros(shape=tf.shape([1])),
                trainable=False,
                validate_shape=False,
                dtype=ztypes.float,
            )

        stacked_integral_limits = tf.stack(limits.v1.limits)
        limits_same = tf.math.reduce_all(
            znp.abs(stacked_integral_limits - self._cached_integral_limits) < self._cache_tolerance
        )

        params = list(self.pdfs[0].get_params(floating=None))
        if hasparams := len(params) > 0:
            stacked_pdf_params = tf.stack(params)
            params_same = tf.math.reduce_all(
                znp.abs(stacked_pdf_params - self._cached_pdf_params_for_integration) < self._cache_tolerance
            )

            same_args = tf.math.logical_and(params_same, limits_same)
        else:
            same_args = limits_same
        assign1 = self._integral_cache_valid.assign(same_args, read_value=False)

        def value_update_func():
            if hasparams:
                self._cached_pdf_params_for_integration.assign(stacked_pdf_params, read_value=False)
            self._cached_integral_limits.assign(stacked_integral_limits, read_value=False)
            return self.pdfs[0].integrate(limits, norm, options=options)

        with tf.control_dependencies([assign1]):
            return get_value(self._integral_cache, self._integral_cache_valid, value_update_func)


class CachedPDFRepr(FunctorPDFRepr):
    _implementation = CachedPDF
    hs3_type: Literal["CachedPDF"] = pydantic.Field("CachedPDF", alias="type")

    def _to_orm(self, init) -> CachedPDF:
        init.pop("obs")
        init["pdf"] = init.pop("pdfs")[0]
        return super()._to_orm(init)
