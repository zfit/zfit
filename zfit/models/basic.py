"""Basic PDFs are provided here.

Gauss, exponential... that can be used together with Functors to build larger models.
"""

#  Copyright (c) 2021 zfit
import contextlib

import numpy as np
import tensorflow as tf

import zfit.z.math
import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.space import ANY_LOWER, ANY_UPPER, Space
from ..util import ztyping
from ..util.exception import (AnalyticIntegralNotImplemented,
                              BreakingAPIChangeError)
from ..util.warnings import warn_advanced_feature


class Exponential(BasePDF):
    _N_OBS = 1

    def __init__(self, lam=None, obs: ztyping.ObsTypeInput = None, name: str = "Exponential", lambda_=None):
        """Exponential function exp(lambda * x).

        The function is normalized over a finite range and therefore a pdf. So the PDF is precisely
        defined as :math:`\\frac{ e^{\\lambda \\cdot x}}{ \\int_{lower}^{upper} e^{\\lambda \\cdot x} dx}`

        Args:
            lam: Accessed as parameter "lambda".
            obs: The :py:class:`~zfit.Space` the pdf is defined in.
            name: Name of the pdf.
            dtype:
        """
        if lambda_ is not None:
            if lam is None:
                lam = lambda_
            else:
                raise BreakingAPIChangeError("The 'lambda' parameter has been renamed from 'lambda_' to 'lam'.")
        params = {'lambda': lam}
        super().__init__(obs, name=name, params=params)

        self._calc_numerics_data_shift = lambda: z.constant(0.)

        if not self.space.has_limits:
            warn_advanced_feature("Exponential pdf relies on a shift of the input towards 0 to keep the numerical "
                                  f"stability high. The space {self.space} does not have limits set and no shift"
                                  f" will occure. To set it manually, set _numerics_data_shift to the expected"
                                  f" average values given to this function _in case you want things to be set_."
                                  f"If this sounds unfamiliar, regard this as an error and use a normalization range.",
                                  identifier='exp_shift')
        self._set_numerics_data_shift(self.space)

    def _unnormalized_pdf(self, x):
        lambda_ = self.params['lambda']
        x = x.unstack_x()
        probs = z.exp(lambda_ * (self._shift_x(x)))
        tf.debugging.assert_all_finite(probs, f"Exponential PDF {self} has non valid values. This is likely caused"
                                              f" by numerical problems: if the exponential is too steep, this will"
                                              f" yield NaNs or infs. Make sure that your lambda is small enough and/or"
                                              f" the initial space is in the same"
                                              f" region as your data (and norm_range, if explicitly set differently)."
                                              f" If this issue still persists, please oben an issue on Github:"
                                              f" https://github.com/zfit/zfit")
        return probs  # Don't use exp! will overflow.

    def _shift_x(self, x):
        return x - self._calc_numerics_data_shift()

    @contextlib.contextmanager
    def _set_numerics_data_shift(self, limits):
        if limits:
            def calc_numerics_data_shift():
                lower, upper = [], []
                for limit in limits:
                    low, up = limit.rect_limits
                    lower.append(z.convert_to_tensor(low[:, 0]))
                    upper.append(z.convert_to_tensor(up[:, 0]))
                lower = z.convert_to_tensor(lower)
                upper = z.convert_to_tensor(upper)
                lower_val = znp.min(lower, axis=0)
                upper_val = znp.max(upper, axis=0)

                return (upper_val + lower_val) / 2

            old_value = self._calc_numerics_data_shift

            self._calc_numerics_data_shift = calc_numerics_data_shift
            yield
            self._calc_numerics_data_shift = old_value
        else:
            yield

    # All hooks are needed to set the right shift when "entering" the pdf. The norm range is taken where both are
    # available. No special need needs to be taken for sampling (it samples from the correct region, the limits, and
    # uses the predictions by the `unnormalized_prob` -> that is shifted correctly
    def _single_hook_integrate(self, limits, norm_range, x):
        with self._set_numerics_data_shift(norm_range):
            return super()._single_hook_integrate(limits, norm_range, x=x)

    def _single_hook_analytic_integrate(self, limits, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_analytic_integrate(limits, norm_range)

    def _single_hook_numeric_integrate(self, limits, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_numeric_integrate(limits, norm_range)

    def _single_hook_partial_integrate(self, x, limits, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_partial_integrate(x, limits, norm_range)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_partial_analytic_integrate(x, limits, norm_range)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_partial_numeric_integrate(x, limits, norm_range)

    # def _single_hook_normalization(self, limits):
    #     with self._set_numerics_data_shift(limits=limits):
    #         return super()._single_hook_normalization(limits)

    #
    # # TODO: remove component_norm_range? But needed for integral?
    # def _single_hook_unnormalized_pdf(self, x, name):
    #     if component_norm_range.limits_are_false:
    #         component_norm_range = self.space
    #     if component_norm_range.limits_are_set:
    #         with self._set_numerics_data_shift(limits=component_norm_range):
    #             return super()._single_hook_unnormalized_pdf(x, name)
    #     else:
    #         return super()._single_hook_unnormalized_pdf(x, name)
    #
    def _single_hook_pdf(self, x, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_pdf(x, norm_range)

    #
    def _single_hook_log_pdf(self, x, norm_range):
        with self._set_numerics_data_shift(limits=norm_range):
            return super()._single_hook_log_pdf(x, norm_range)

    def _single_hook_sample(self, n, limits, x=None):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_sample(n, limits, x)


def _exp_integral_from_any_to_any(limits, params, model):
    lambda_ = params['lambda']
    lower, upper = limits.rect_limits_np
    if any(np.isinf([lower, upper])):
        raise AnalyticIntegralNotImplemented

    integral = _exp_integral_func_shifting(lambd=lambda_, lower=lower, upper=upper, model=model)
    return integral[0]


def _exp_integral_func_shifting(lambd, lower, upper, model):
    def raw_integral(x):
        return z.exp(lambd * (model._shift_x(x))) / lambd  # needed due to overflow in exp otherwise

    lower_int = raw_integral(x=z.constant(lower))
    upper_int = raw_integral(x=z.constant(upper))
    integral = (upper_int - lower_int)
    return integral


def exp_icdf(x, params, model):
    lambd = params['lambda']
    x = z.unstack_x(x)
    x = model._shift_x(x)
    return zfit.z.math.log(lambd * x) / lambd


# Exponential.register_inverse_analytic_integral(exp_icdf)  # TODO: register icdf for exponential
# TODO: cleanup, make cdf registrable _and_ inverse integral, but real

limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Exponential.register_analytic_integral(func=_exp_integral_from_any_to_any, limits=limits)
