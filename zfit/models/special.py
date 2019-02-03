"""
Special PDFs are provided in this module. One example is a normal function `Function` that allows to
simply define a non-normalizable function.
"""

import functools
from types import MethodType

import tensorflow as tf

from zfit.core.basemodel import SimpleModelSubclassMixin
from zfit.core.basepdf import BasePDF
from zfit.util import ztyping

from ..core.basepdf import BasePDF
from ..core.limits import no_norm_range
from ..util.exception import NormRangeNotImplementedError


class SimplePDF(BasePDF):
    def __init__(self, func, name="SimplePDF", n_dims=1, **params):
        super().__init__(name=name, **params)
        self._unnormalized_prob_func = self._check_input_x_function(func)
        self._user_n_dims = n_dims

    def _unnormalized_pdf(self, x):
        return self._unnormalized_prob_func(x)

    @property
    def _n_dims(self):
        return self._user_n_dims


def raise_error_if_norm_range(func):
    func = no_norm_range(func)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NormRangeNotImplementedError:  # TODO: silently remove norm_range? Or loudly fail?
            raise tf.errors.InvalidArgumentError("Norm_range given to Function: cannot be normalized.")

    return wrapped


class ZPDF(SimpleModelSubclassMixin, BasePDF):
    def __init__(self, obs: ztyping.ObsTypeInput, name: str = "ZPDF", **params):
        super().__init__(obs=obs, name=name, **params)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._check_simple_model_subclass()
