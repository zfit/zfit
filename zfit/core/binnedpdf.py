#  Copyright (c) 2020 zfit
from contextlib import suppress
from typing import Union, Callable

import numpy as np
import tensorflow as tf

from .baseobject import BaseNumeric
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedPDF, ZfitSpace, ZfitParameter
from .. import convert_to_parameter
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.exception import SpecificFunctionNotImplementedError, WorkInProgressError, NotExtendedPDFError, \
    AlreadyExtendedPDFError


class BaseBinnedPDF(BaseNumeric, GraphCachable, BaseDimensional, ZfitBinnedPDF):

    def __init__(self, obs, **kwargs):
        super().__init__(**kwargs)

        self._space = obs
        self._yield = None
        self._norm_range = None
        self._normalization_value = None

    @property
    def space(self):
        return self._space

    def set_yield(self, value):
        if self.is_extended:
            raise AlreadyExtendedPDFError(f"Cannot extend {self}, is already extended.")
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

    def _get_dependencies(self) -> ztyping.DependentsType:
        return super(BaseBinnedPDF, self)._get_dependencies()

    def _pdf(self, x, norm_range):
        raise SpecificFunctionNotImplementedError

    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        return self._call_pdf(x, norm_range=norm_range)

    def _call_pdf(self, x, norm_range):
        with suppress(SpecificFunctionNotImplementedError):
            return self._pdf(x, norm_range)
        return self._fallback_pdf(x, norm_range=norm_range)

    def _fallback_pdf(self, x, norm_range):
        values = self._call_unnormalized_pdf(x)
        if norm_range is not False:
            values = values / self.normalization(norm_range)
        return values

    def _unnormalized_pdf(self, x):
        raise SpecificFunctionNotImplementedError

    def _call_unnormalized_pdf(self, x):
        return self._unnormalized_pdf(x)

    def _ext_pdf(self, x, norm_range):
        raise SpecificFunctionNotImplementedError

    def ext_pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        if not self.is_extended:
            raise NotExtendedPDFError
        return self._call_ext_pdf(x, norm_range=norm_range)

    def _call_ext_pdf(self, x, norm_range):
        with suppress(SpecificFunctionNotImplementedError):
            return self._ext_pdf(x, norm_range)
        return self._fallback_ext_pdf(x, norm_range=norm_range)

    def _fallback_ext_pdf(self, x, norm_range):
        values = self._call_pdf(x, norm_range=norm_range)
        return values * self.get_yield()

    def normalization(self, limits: ztyping.LimitsType) -> ztyping.NumericalTypeReturn:
        return self.integrate(limits)

    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        # TODO HACK
        def binned_rect_integration(bincount: tf.Tensor, edges: np.ndarray, limits: ZfitSpace) -> tf.Tensor:

            # HACK
            limits.inside = lambda x: x.astype(bool)

            print("HACK ACTIVE in integration.py, binned_integration, limits not working")

            # HACK END
            # bincount_cut = bincount[limits.inside(edges)]
            bincount_cut = bincount
            binwidths = [(edge[1:] - edge[:-1]) for edge in edges]

            def outer_tensordot_recursive(tensors):
                """Outer product of the tensors."""
                if len(tensors) > 1:
                    return tf.tensordot(tensors[0], outer_tensordot_recursive(tensors[1:]), axes=0)
                else:
                    return tensors[0]

            areas = outer_tensordot_recursive(binwidths)

            bincount_cut *= areas
            integral = tf.reduce_sum(bincount_cut, axis=limits.axes)
            return integral

        bincounts = self.pdf(limits, norm_range=False)
        edges = limits.binning.get_edges()
        return binned_rect_integration(bincount=bincounts, edges=edges, limits=limits)

    def update_integration_options(self, *args, **kwargs):
        raise WorkInProgressError

    def as_func(self, norm_range: ztyping.LimitsType = False):
        raise WorkInProgressError

    @property
    def is_extended(self) -> bool:
        return self._yield is not None

    def set_norm_range(self, norm_range):
        return self._norm_range

    def create_extended(self, yield_: ztyping.ParamTypeInput) -> "ZfitPDF":
        raise WorkInProgressError

    def get_yield(self) -> Union[ZfitParameter, None]:
        # TODO: catch
        return self._yield

    @classmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None, priority: int = 50, *,
                                   supports_norm_range: bool = False, supports_multiple_limits: bool = False):
        raise WorkInProgressError

    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType,
                          norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        raise WorkInProgressError

    @classmethod
    def register_inverse_analytic_integral(cls, func: Callable):
        raise WorkInProgressError

    def _sample(self, n, limits):
        raise SpecificFunctionNotImplementedError

    def sample(self, n: int, limits: ztyping.LimitsType = None) -> ztyping.XType:
        return self._call_sample(n, limits)

    def _call_sample(self, n, limits):
        with suppress(SpecificFunctionNotImplementedError):
            self._sample(n, limits)
        return self._fallback_sample(n, limits)

    def _fallback_sample(self, n, limits):
        raise WorkInProgressError

    def _copy(self, deep, name, overwrite_params):
        raise WorkInProgressError
