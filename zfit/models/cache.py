from zfit import supports
from zfit.models.functor import BaseFunctor

import tensorflow as tf
import zfit.z.numpy as znp


def get_value(cache: tf.Variable, flag: tf.Variable, func, counter):
    def autoset_func():
        val = func()
        cache.assign(val, read_value=False)
        return cache

    def use_cache():
        counter.assign_add(1.)
        return cache

    val = tf.cond(flag, use_cache, autoset_func)
    return val


class CacheablePDF(BaseFunctor):
    def __init__(self, pdf, cache_tol=None, **kwargs):
        super().__init__(pdfs=pdf, obs=pdf.space, **kwargs)
        params = list(pdf.get_params())
        self.pdf_cache_counter = tf.Variable(0.)
        self.integrate_cache_counter = tf.Variable(0.)
        self._cached_pdf_params = tf.Variable(znp.zeros(shape=tf.shape(tf.stack(params))), trainable=False,
                                              validate_shape=False, dtype=tf.float64)
        self._cached_pdf_params_for_integration = tf.Variable(znp.zeros(shape=tf.shape(tf.stack(params))),
                                                              trainable=False,
                                                              validate_shape=False, dtype=tf.float64)
        self._pdf_cache = None
        self._cached_integral_limits = None
        self._integral_cache = None
        self._integral_cache_valid = tf.Variable(initial_value=False, trainable=False)

        self._cache_tolerance = 1e-8 if cache_tol is None else cache_tol

    @supports(norm="space")
    def _pdf(self, x, norm):
        if self._pdf_cache is None:
            self._pdf_cache = tf.Variable(znp.zeros(shape=tf.shape(x)[0]), trainable=False, validate_shape=False,
                                          dtype=tf.float64)
        pdf_params = list(self.pdfs[0].get_params())
        stacked_pdf_params = tf.stack(pdf_params)
        params_same = tf.math.reduce_all(
            tf.math.abs(stacked_pdf_params - self._cached_pdf_params) < self._cache_tolerance)
        self._cached_pdf_params.assign(stacked_pdf_params, read_value=False)
        pdf = get_value(self._pdf_cache, params_same, lambda: self.pdfs[0].pdf(x, norm), self.pdf_cache_counter)
        return pdf

    @supports(norm="space")
    def _integrate(self, limits, norm, options=None):
        if self._cached_integral_limits is None:
            self._cached_integral_limits = tf.Variable(znp.zeros(shape=tf.shape(tf.stack(limits.limits))),
                                                       trainable=False, validate_shape=False, dtype=tf.float64)

        if self._integral_cache is None:
            self._integral_cache = tf.Variable(znp.zeros(shape=tf.shape([1])), trainable=False, validate_shape=False,
                                               dtype=tf.float64)
        pdf_params = list(self.pdfs[0].get_params())
        stacked_pdf_params = tf.stack(pdf_params)
        params_same = tf.math.reduce_all(
            tf.math.abs(stacked_pdf_params - self._cached_pdf_params_for_integration) < self._cache_tolerance)

        stacked_integral_limits = tf.stack(limits.limits)
        limits_same = tf.math.reduce_all(
            tf.math.abs(stacked_integral_limits - self._cached_integral_limits) < self._cache_tolerance)

        self._integral_cache_valid.assign(tf.math.logical_and(params_same, limits_same), read_value=False)

        self._cached_pdf_params_for_integration.assign(stacked_pdf_params, read_value=False)
        self._cached_integral_limits.assign(stacked_integral_limits, read_value=False)

        integral = get_value(self._integral_cache, self._integral_cache_valid,
                             lambda: self.pdfs[0].integrate(limits, norm, options=None), self.integrate_cache_counter)
        return integral
