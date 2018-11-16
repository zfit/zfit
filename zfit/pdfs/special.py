"""
Special PDFs are provided in this module. One example is a normal function `Function` that allows to
simply define a non-normalizable function.
"""

import functools
from types import MethodType

import tensorflow as tf

from zfit.core.basepdf import BasePDF
from zfit.core.limits import no_norm_range
from zfit.util.exception import NormRangeNotImplementedError


class SimplePDF(BasePDF):
    def __init__(self, func, name="SimplePDF", **parameters):
        super().__init__(name=name, **parameters)
        self._unnormalized_prob_func = self._check_input_x_function(func)

    def _unnormalized_prob(self, x, norm_range=False):
        return self._unnormalized_prob_func(x **self.get_parameters(only_floating=False))


def raise_error_if_norm_range(func):
    func = no_norm_range(func)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NormRangeNotImplementedError:  # TODO: silently remove norm_range? Or loudly fail?
            raise tf.errors.InvalidArgumentError("Norm_range given to Function: cannot be normalized.")

    return wrapped

# class BaseFunction(BasePDF):  # TODO: what about yield? Illegal too? And other way? (change normalization value?)
#     def __init__(self, name="BaseFunction", **kwargs):
#         super().__init__(name=name, **kwargs)
#
#         self.set_norm_range(False)
#
#
# # decorate every function to prevent any norm_range given.
# BaseFunction._norm_analytic_integrate = MethodType(raise_error_if_norm_range(super()._norm_analytic_integrate),
#                                                    BaseFunction)
# BaseFunction._norm_integrate = MethodType(raise_error_if_norm_range(super()._norm_integrate),
#                                           BaseFunction)
# BaseFunction._norm_log_prob = MethodType(raise_error_if_norm_range(super()._norm_log_prob),
#                                          BaseFunction)
# BaseFunction._norm_numeric_integrate = MethodType(raise_error_if_norm_range(super()._norm_numeric_integrate),
#                                                   BaseFunction)
# BaseFunction._norm_partial_analytic_integrate = MethodType(
#     raise_error_if_norm_range(super()._norm_partial_analytic_integrate),
#     BaseFunction)
# BaseFunction._norm_partial_integrate = MethodType(raise_error_if_norm_range(super()._norm_partial_integrate),
#                                                   BaseFunction)
# BaseFunction._norm_partial_numeric_integrate = MethodType(
#     raise_error_if_norm_range(super()._norm_partial_numeric_integrate),
#     BaseFunction)
# BaseFunction._norm_prob = MethodType(raise_error_if_norm_range(super()._norm_prob),
#                                      BaseFunction)
# BaseFunction._norm_sample = MethodType(raise_error_if_norm_range(super()._norm_sample),
#                                        BaseFunction)
