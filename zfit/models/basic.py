"""
Basic PDFs are provided here. Gauss, exponential... that can be used together with Functors to
build larger models.
"""

#  Copyright (c) 2019 zfit

import math as mt
from typing import Type, Any
import warnings

import numpy as np
import tensorflow as tf

from zfit import ztf
from ..util.exception import DueToLazynessNotImplementedError
from ..util.temporary import TemporarilySet
from ..settings import ztypes
from ..util import ztyping
from ..core.limits import Space, ANY_LOWER, ANY_UPPER
from ..core.basepdf import BasePDF

infinity = mt.inf


class CustomGaussOLD(BasePDF):

    def __init__(self, mu, sigma, obs, name="Gauss"):
        super().__init__(name=name, obs=obs, params=dict(mu=mu, sigma=sigma))

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        mu = self.params['mu']
        sigma = self.params['sigma']
        gauss = tf.exp(- 0.5 * tf.square((x - mu) / sigma))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params, model):
    return tf.sqrt(2 * ztf.pi) * params['sigma']


CustomGaussOLD.register_analytic_integral(func=_gauss_integral_from_inf_to_inf,
                                          limits=Space.from_axes(limits=(-infinity, infinity), axes=(0,)))


class Exponential(BasePDF):
    _N_OBS = 1

    def __init__(self, lambda_, obs: ztyping.ObsTypeInput, name: str = "Exponential",
                 **kwargs):
        """Exponential function exp(lambda * x).

        The function is normalized over a finite range and therefore a pdf. So the PDF is precisely
        defined as :math:`\\frac{ e^{\\lambda \\cdot x}}{ \\int_{lower}^{upper} e^{\\lambda \\cdot x} dx}`

        Args:
            lambda_ (:py:class:`~zfit.Parameter`): Accessed as parameter "lambda".
            obs (:py:class:`~zfit.Space`): The :py:class:`~zfit.Space` the pdf is defined in.
            name (str): Name of the pdf.
            dtype (DType):
        """
        params = {'lambda': lambda_}
        super().__init__(obs, name=name, params=params, **kwargs)
        self._numerics_data_shift = None

    def _unnormalized_pdf(self, x):
        lambda_ = self.params['lambda']
        x = x.unstack_x()
        return self._numerics_shifted_exp(x=x, lambda_=lambda_)  # Don't use exp! will overflow.

    def _numerics_shifted_exp(self, x, lambda_):  # needed due to overflow in exp otherwise, prevents by shift
        return ztf.exp(lambda_ * (x - self._numerics_data_shift))

    def _set_numerics_data_shift(self, limits):
        lower, upper = limits.limits
        lower_val = min([lim[0] for lim in lower])
        upper_val = max([lim[0] for lim in upper])

        value = (upper_val + lower_val) / 2

        if max(abs(lower_val - value), abs(upper_val - value)) > 710:
            warnings.warn(
                "Boundaries can be too wide for exponential (assuming lambda ~ 1), expect `inf` in exp(x) and `NaN`s."
                "(upper - lower) * lambda should be smaller than 1400 roughly",
                category=RuntimeWarning)

        def setter(value):
            self._numerics_data_shift = value

        def getter():
            return self._numerics_data_shift

        return TemporarilySet(value=value, getter=getter, setter=setter)

    # All hooks are needed to set the right shift when "entering" the pdf. The norm range is taken where both are
    # available. No special need needs to be taken for sampling (it samples from the correct region, the limits, and
    # uses the predictions by the `unnormalized_prob` -> that is shifted correctly
    def _single_hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_integrate(limits, norm_range, name)

    def _single_hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_analytic_integrate(limits, norm_range, name)

    def _single_hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_numeric_integrate(limits, norm_range, name)

    def _single_hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_partial_integrate(x, limits, norm_range, name)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range, name='_hook_partial_analytic_integrate'):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_partial_analytic_integrate(x, limits, norm_range, name)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range, name='_hook_partial_numeric_integrate'):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_partial_numeric_integrate(x, limits, norm_range, name)

    def _single_hook_normalization(self, limits, name):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_normalization(limits, name)

    def _single_hook_unnormalized_pdf(self, x, component_norm_range, name):
        if component_norm_range.limits is not None:
            with self._set_numerics_data_shift(limits=component_norm_range):
                return super()._single_hook_unnormalized_pdf(x, component_norm_range, name)
        else:
            return super()._single_hook_unnormalized_pdf(x, component_norm_range, name)

    def _single_hook_pdf(self, x, norm_range, name):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_pdf(x, norm_range, name)

    def _single_hook_log_pdf(self, x, norm_range, name):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_log_pdf(x, norm_range, name)

    def _single_hook_sample(self, n, limits, name):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_sample(n, limits, name)

    # def _log_pdf(self, x, norm_range: Space):
    #     lambda_ = self.params['lambda']
    #     x = ztf.unstack_x(x)
    #     func = x * lambda_
    #     if norm_range.n_limits > 1:
    #         raise DueToLazynessNotImplementedError(
    #             "Not implemented, it's more of a hack. I Should implement log_pdf and "
    #             "norm probarly")
    #     (lower,), (upper,) = norm_range.limits
    #     lower = lower[0]
    #     upper = upper[0]
    #
    #     assert False, "WIP, add log integral"


def _exp_integral_from_any_to_any(limits, params, model):
    lambda_ = params['lambda']

    def raw_integral(x):
        return model._numerics_shifted_exp(x=x, lambda_=lambda_) / lambda_  # needed due to overflow in exp otherwise

    (lower,), (upper,) = limits.limits
    if lower[0] == - upper[0] == np.inf:
        raise NotImplementedError
    lower_int = raw_integral(x=ztf.constant(lower))
    upper_int = raw_integral(x=ztf.constant(upper))
    return (upper_int - lower_int)[0]


# Exponential.register_inverse_analytic_integral()


limits = Space.from_axes(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Exponential.register_analytic_integral(func=_exp_integral_from_any_to_any, limits=limits)
