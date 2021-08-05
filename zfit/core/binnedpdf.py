from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Callable
#  Copyright (c) 2021 zfit
from contextlib import suppress
from typing import Union, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from zfit import z
from zfit.core.binneddata import BinnedData
from .baseobject import BaseNumeric
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedPDF, ZfitParameter, ZfitSpace, ZfitMinimalHist
from .tensorlike import OverloadableMixinValues, register_tensor_conversion
from .. import convert_to_parameter, convert_to_space
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.exception import (AlreadyExtendedPDFError, NotExtendedPDFError,
                              SpaceIncompatibleError,
                              SpecificFunctionNotImplemented,
                              WorkInProgressError)


class BaseBinnedPDF(
    BaseNumeric,
    GraphCachable,
    BaseDimensional,
    OverloadableMixinValues,
    ZfitMinimalHist,
    ZfitBinnedPDF):

    def __init__(self, obs, extended=None, norm=None, **kwargs):
        super().__init__(dtype=znp.float64, **kwargs)

        self._space = obs
        self._yield = None
        self._norm = norm
        if extended is not None:
            self._set_yield(extended)

    @property
    def axes(self):
        return self.space.binning

    @property
    def space(self):
        return self._space

    def _set_yield(self, value):
        if self.is_extended:
            raise AlreadyExtendedPDFError(f"Cannot extend {self}, is already extended.")
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

    def _get_dependencies(self) -> ztyping.DependentsType:
        return super()._get_dependencies()

    def _pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    def pdf(self, x: ztyping.XType, norm: ztyping.LimitsType = None) -> ztyping.XType:
        return self._call_pdf(x, norm=norm)

    def _call_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._pdf(x, norm)
        return self._fallback_pdf(x, norm=norm)

    def _fallback_pdf(self, x, norm):
        values = self._call_unnormalized_pdf(x)
        if norm is not False:
            values = values / self.normalization(norm)
        return values

    def _unnormalized_pdf(self, x):
        raise SpecificFunctionNotImplemented

    def _call_unnormalized_pdf(self, x):
        return self._unnormalized_pdf(x)

    def _ext_pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    def ext_pdf(self, x: ztyping.XType, norm: ztyping.LimitsType = None) -> ztyping.XType:
        if not self.is_extended:
            raise NotExtendedPDFError
        return self._call_ext_pdf(x, norm=norm)

    def _call_ext_pdf(self, x, norm):
        with suppress(SpecificFunctionNotImplemented):
            return self._ext_pdf(x, norm)
        return self._fallback_ext_pdf(x, norm=norm)

    def _fallback_ext_pdf(self, x, norm):
        values = self._call_pdf(x, norm=norm)
        return values * self.get_yield()

    def normalization(self, limits: ztyping.LimitsType) -> ztyping.NumericalTypeReturn:
        with suppress(SpecificFunctionNotImplemented):
            return self._normalization(limits)
        return self.integrate(limits)

    def _normalization(self, limits):
        raise SpecificFunctionNotImplemented

    def integrate(self,
                  limits: ztyping.LimitsType,
                  norm: ztyping.LimitsType = None) -> ztyping.XType:
        # TODO HACK

        bincounts = self.pdf(limits, norm=False)
        edges = limits.binning.get_edges()
        return binned_rect_integration(values=bincounts, edges=edges, limits=limits)

    def ext_integrate(self,
                      limits: ztyping.LimitsType,
                      norm: ztyping.LimitsType = None) -> ztyping.XType:
        # TODO HACK

        bincounts = self.ext_pdf(limits, norm=False)
        edges = limits.binning.edges
        return binned_rect_integration(values=bincounts, edges=edges, limits=limits)

    def sample(self, n: int, limits: ztyping.LimitsType = None) -> ztyping.XType:
        if limits is None:
            limits = self.space
        return self._call_sample(n, limits)

    def _call_sample(self, n, limits):
        with suppress(SpecificFunctionNotImplemented):
            self._sample(n, limits)
        return self._fallback_sample(n, limits)

    def _fallback_sample(self, n, limits):
        if limits != self.space:
            raise WorkInProgressError("limits different from the default are not yet available."
                                      " Please open an issue if you need this:"
                                      " https://github.com/zfit/zfit/issues/new/choose")
        probs = self.ext_pdf(None) / self.get_yield()
        values = z.random.counts_multinomial(n, probs=probs)
        data = BinnedData.from_tensor(space=limits, values=values, variances=None)
        return data

    # ZfitMinimalHist implementation
    def values(self):
        if self.is_extended:
            return self.ext_pdf(None)
        else:
            return self.pdf(None)

    # WIPWIPWIP

    def update_integration_options(self, *args, **kwargs):
        raise WorkInProgressError

    def as_func(self, norm_range: ztyping.LimitsType = False):
        raise WorkInProgressError

    @property
    def is_extended(self) -> bool:
        return self._yield is not None

    def set_norm_range(self, norm_range):
        return self._norm

    def create_extended(self, yield_: ztyping.ParamTypeInput) -> ZfitPDF:
        raise WorkInProgressError

    def get_yield(self) -> ZfitParameter | None:
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
        raise SpecificFunctionNotImplemented

    def _copy(self, deep, name, overwrite_params):
        raise WorkInProgressError

    # factor out with unbinned pdf
    @property
    def norm_range(self):
        return self._norm

    # TODO: factor out with unbinned pdf
    def convert_sort_space(self, obs: ztyping.ObsTypeInput | ztyping.LimitsTypeInput = None,
                           axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> ZfitSpace | None:
        """Convert the inputs (using eventually `obs`, `axes`) to :py:class:`~zfit.ZfitSpace` and sort them according to
        own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        elif not set(obs).intersection(self.obs):
            raise SpaceIncompatibleError("The given space {obs} is not compatible with the obs of the pdfs{self.obs};"
                                         " they are disjoint.")
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:  # e.g. not the first call
            space = space.with_coords(self.space, allow_superset=True, allow_subset=True)
        return space


register_tensor_conversion(BaseBinnedPDF)


class BinnedFromUnbinned(BaseBinnedPDF):

    def __init__(self, pdf, space, extended=None, norm=None):
        super().__init__(obs=space, extended=extended, norm=norm, params={})
        self.pdfs = pdf

    def _get_params(self, floating: bool | None = True, is_yield: bool | None = None,
                    extract_independent: bool | None = True) -> set[ZfitParameter]:
        return self.pdfs[0].get_params(floating=floating, is_yield=is_yield, extract_independent=extract_independent)


# TODO: extend with partial integration. Axis parameter?
def binned_rect_integration(values: znp.array, edges: Iterable[znp.array], limits: ZfitSpace) -> tf.Tensor:
    scaled_edges, (lower_bins, upper_bins) = cut_edges_and_bins(edges=edges, limits=limits)
    values_cut = tf.slice(values, lower_bins, (upper_bins - lower_bins))
    rank = values.shape.rank
    binwidths = []
    for i, edge in enumerate(scaled_edges):
        edge_lower_index = [0] * rank
        edge_lowest_index = znp.asarray(edge_lower_index.copy(), dtype=znp.int64)

        edge_lower_index[i] = 1
        edge_lower_index = znp.asarray(edge_lower_index, dtype=znp.int64)
        edge_upper_index = [1] * rank
        edge_highest_index = edge_upper_index.copy()
        max_index = tf.shape(edge)[i]
        edge_highest_index[i] = max_index
        edge_highest_index = znp.asarray(edge_highest_index, dtype=znp.int64)
        edge_upper_index[i] = max_index - 1  # we want to skip the last (as we skip also the first)
        edge_upper_index = znp.asarray(edge_upper_index, dtype=znp.int64)
        lower_edge = tf.slice(edge, edge_lowest_index, (edge_upper_index - edge_lowest_index))
        upper_edge = tf.slice(edge, edge_lower_index, (edge_highest_index - edge_lower_index))
        binwidths.append(upper_edge - lower_edge)
    # binwidths = [(edge[1:] - edge[:-1]) for edge in scaled_edges]
    binareas = np.prod(binwidths, axis=0)  # needs to be np as znp or tf can't broadcast otherwise

    integral = tf.reduce_sum(values_cut * binareas, axis=limits.axes)
    return integral


@z.function(wraps='tensor')
def cut_edges_and_bins(edges: Iterable[znp.array], limits: ZfitSpace) -> tuple[
    list[znp.array], tuple[znp.array, znp.array]]:
    """Cut the *edges* according to *limits* and calculate the bins inside.

    The edges within limits are calculated and returned together with the corresponding bin indices. The indices
    mark the lowest and the highest index of the edges that are returned.

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
        edges, (lower bins, upper bins): The edges and the bins are returned. The upper bin number corresponds to
            the highest bin which was still (partially) inside the limits **plus one** (so it's the index of the
            edge that is right outside).
    """
    cut_scaled_edges = []
    all_lower_bins = []
    all_upper_bins = []
    if isinstance(limits, ZfitSpace):
        lower, upper = limits.limits
    else:
        lower, upper = limits
        lower = znp.asarray(lower)
        upper = znp.asarray(upper)
    lower_all = lower[0]
    upper_all = upper[0]
    rank = len(edges)
    zero_bins = znp.zeros([rank], dtype=znp.int64)
    for i, edge in enumerate(edges):
        edge = znp.asarray(edge)
        lower_i = lower_all[i, None]
        edge_minimum = tf.gather(edge, indices=0, axis=i)
        lower_i = znp.maximum(lower_i, edge_minimum)
        upper_i = upper_all[i, None]
        edge_maximum = tf.gather(edge, indices=tf.shape(edge)[i] - 1, axis=i)
        upper_i = znp.minimum(upper_i, edge_maximum)
        # we get the bins that are just one too far. Then we update this whole bin tensor with the actual edge.
        # The bins index is the index below the value.
        flat_edge = tf.reshape(edge, [-1])
        lower_bin_float = tfp.stats.find_bins(lower_i, flat_edge,
                                              extend_lower_interval=True,
                                              extend_upper_interval=True)
        lower_bin = tf.reshape(tf.cast(lower_bin_float, dtype=znp.int64), [-1])
        lower_bins = tf.tensor_scatter_nd_update(zero_bins, [[i]], lower_bin)
        # +1 below because the outer bin is searched, meaning the one that is higher than the value

        upper_bin_float = tfp.stats.find_bins(upper_i, flat_edge,
                                              extend_lower_interval=True,
                                              extend_upper_interval=True)
        upper_bin = tf.reshape(tf.cast(upper_bin_float, dtype=znp.int64), [-1])
        upper_bin_p1 = upper_bin + 1
        upper_bins = tf.tensor_scatter_nd_update(zero_bins, [[i]], upper_bin_p1)
        size = upper_bins - lower_bins
        new_edge = tf.slice(edge, lower_bins, size + 1)  # +1 because stop is exclusive
        new_edge = tf.tensor_scatter_nd_update(new_edge, [zero_bins, size],
                                               [tf.reshape(lower_i, [-1])[0], tf.reshape(upper_i, [-1])[0]])
        all_lower_bins.append(lower_bins)
        all_upper_bins.append(upper_bins)
        cut_scaled_edges.append(new_edge)
    # lower_bins_indices = tf.stack([lower_bins, dims], axis=-1)
    # upper_bins_indices = tf.stack([upper_bins, dims], axis=-1)
    all_lower_bins = znp.sum(all_lower_bins, axis=0)
    all_upper_bins = znp.sum(all_upper_bins, axis=0)
    return cut_scaled_edges, (all_lower_bins, all_upper_bins)


class BaseBinnedPDFV1(BaseNumeric, GraphCachable, BaseDimensional, ZfitBinnedPDF):

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
        return super()._get_dependencies()

    def _pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        return self._call_pdf(x, norm_range=norm_range)

    def _call_pdf(self, x, norm_range):
        with suppress(SpecificFunctionNotImplemented):
            return self._pdf(x, norm_range)
        return self._fallback_pdf(x, norm_range=norm_range)

    def _fallback_pdf(self, x, norm_range):
        values = self._call_unnormalized_pdf(x)
        if norm_range is not False:
            values = values / self.normalization(norm_range)
        return values

    def _unnormalized_pdf(self, x):
        raise SpecificFunctionNotImplemented

    def _call_unnormalized_pdf(self, x):
        return self._unnormalized_pdf(x)

    def _ext_pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    def ext_pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        if not self.is_extended:
            raise NotExtendedPDFError
        return self._call_ext_pdf(x, norm_range=norm_range)

    def _call_ext_pdf(self, x, norm_range):
        with suppress(SpecificFunctionNotImplemented):
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

    def create_extended(self, yield_: ztyping.ParamTypeInput) -> ZfitPDF:
        raise WorkInProgressError

    def get_yield(self) -> ZfitParameter | None:
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
        raise SpecificFunctionNotImplemented

    def sample(self, n: int, limits: ztyping.LimitsType = None) -> ztyping.XType:
        return self._call_sample(n, limits)

    def _call_sample(self, n, limits):
        with suppress(SpecificFunctionNotImplemented):
            self._sample(n, limits)
        return self._fallback_sample(n, limits)

    def _fallback_sample(self, n, limits):
        raise WorkInProgressError

    def _copy(self, deep, name, overwrite_params):
        raise WorkInProgressError

    # factor out with unbinned pdf
    @property
    def norm_range(self):
        return self._norm_range

    # TODO: factor out with unbinned pdf
    def convert_sort_space(self, obs: ztyping.ObsTypeInput | ztyping.LimitsTypeInput = None,
                           axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> ZfitSpace | None:
        """Convert the inputs (using eventually `obs`, `axes`) to :py:class:`~zfit.ZfitSpace` and sort them according to
        own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        elif not set(obs).intersection(self.obs):
            raise SpaceIncompatibleError("The given space {obs} is not compatible with the obs of the pdfs{self.obs};"
                                         " they are disjoint.")
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:  # e.g. not the first call
            space = space.with_coords(self.space, allow_superset=True, allow_subset=True)
        return space
