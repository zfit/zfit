"""Special PDFs are provided in this module.

One example is a normal function `Function` that allows to simply define a non-normalizable function.
"""

#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools

import tensorflow as tf

from ..core.basemodel import SimpleModelSubclassMixin
from ..core.basepdf import BasePDF
from ..core.space import supports
from ..util import ztyping
from ..util.exception import NormRangeNotImplemented
from .functor import BaseFunctor


class SimplePDF(BasePDF):
    def __init__(self, obs, func, name="SimplePDF", label=None, norm=None, extended=None, **params):
        super().__init__(name=name, params=params, obs=obs, norm=norm, extended=extended, label=label)
        self._unnormalized_prob_func = self._check_input_x_function(func)

    def _unnormalized_pdf(self, x):
        try:
            return self._unnormalized_prob_func(x)
        except TypeError:
            return self._unnormalized_prob_func(self, x)

    def copy(self, **override_parameters) -> BasePDF:
        override_parameters.update(func=self._unnormalized_prob_func)
        return super().copy(**override_parameters)


class SimpleFunctorPDF(BaseFunctor, SimplePDF):
    def __init__(self, obs, pdfs, func, name="SimpleFunctorPDF", label=None, norm=None, extended=None, **params):
        super().__init__(obs=obs, pdfs=pdfs, func=func, name=name, label=label, norm=norm, extended=extended, **params)


def raise_error_if_norm_range(func):
    func = supports(norm=False)(func)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NormRangeNotImplemented:  # TODO: silently remove norm? Or loudly fail?
            msg = "Norm_range given to Function: cannot be normalized."
            raise tf.errors.InvalidArgumentError(msg) from None

    return wrapped


class ZPDF(SimpleModelSubclassMixin, BasePDF):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        *,
        name: str = "ZPDF",
        label: str | None = None,
        norm=None,
        extended=None,
        **params,
    ):
        super().__init__(obs=obs, name=name, norm=norm, extended=extended, label=label, **params)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._check_simple_model_subclass()
