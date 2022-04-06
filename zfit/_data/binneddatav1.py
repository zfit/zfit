#  Copyright (c) 2022 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

import boost_histogram as bh
import hist
import tensorflow as tf

from zfit._variables.axis import histaxes_to_binning, binning_to_histaxes
from zfit.core.interfaces import ZfitBinnedData, ZfitSpace, ZfitData
from zfit.z import numpy as znp
from ..util import ztyping
from ..util.exception import ShapeIncompatibleError


# @tfp.experimental.auto_composite_tensor()
class BinnedHolder(
    # tfp.experimental.AutoCompositeTensor
):
    def __init__(self, space, values, variances):
        self._check_init_values(space, values, variances)
        self.space = space
        self.values = values
        self.variances = variances

    def _check_init_values(self, space, values, variances):
        value_shape = tf.shape(values)
        edges_shape = znp.array(
            [tf.shape(znp.reshape(edge, (-1,)))[0] for edge in space.binning.edges]
        )
        values_rank = value_shape.shape[0]
        if variances is not None:
            variances_shape = tf.shape(variances)
            variances_rank = variances_shape.shape[0]
            if values_rank != variances_rank:
                raise ShapeIncompatibleError(
                    f"Values {values} and variances {variances} differ in rank: {values_rank} vs {variances_rank}"
                )
            tf.assert_equal(
                variances_shape,
                value_shape,
                message=f"Variances and values do not have the same shape:"
                f" {variances_shape} vs {value_shape}",
            )
        binning_rank = len(space.binning.edges)
        if binning_rank != values_rank:
            raise ShapeIncompatibleError(
                f"Values and binning  differ in rank: {values_rank} vs {binning_rank}"
            )
        tf.assert_equal(
            edges_shape - 1,
            value_shape,
            message=f"Edges (minus one) and values do not have the same shape:"
            f" {edges_shape} vs {value_shape}",
        )

    def with_obs(self, obs):
        space = self.space.with_obs(obs)
        values = move_axis_obs(self.space, space, self.values)
        variances = self.variances
        if variances is not None:
            variances = move_axis_obs(self.space, space, self.variances)
        return type(self)(space=space, values=values, variances=variances)


def move_axis_obs(original, target, values):
    new_axes = [original.obs.index(ob) for ob in target.obs]
    values = znp.moveaxis(values, tuple(range(target.n_obs)), new_axes)
    return values


flow = False  # TODO: track the flow or not?


# @tfp.experimental.auto_composite_tensor()
class BinnedData(
    ZfitBinnedData,
    # tfp.experimental.AutoCompositeTensor, OverloadableMixinValues, ZfitBinnedData
):
    def __init__(self, *, holder):
        self.holder: BinnedHolder = holder

    @classmethod  # TODO: add overflow bins if needed
    def from_tensor(
        cls, space: ZfitSpace, values: znp.array, variances: znp.array | None = None
    ) -> BinnedData:
        """Create a binned dataset defined in *space* where values are considered to be the counts.

        Args:
            space: The space of the data. Variables need to match the values dimensions. The space has to be binned
                and carry the information about the edges.
            values: Actual counts of the histogram.
            variances: Uncertainties of the histogram values. If `True`, the uncertainties are taken to be poissonian
                distributed.
        """
        values = znp.asarray(values, znp.float64)
        if variances is True:
            variances = znp.sqrt(values)
        elif variances is not None:
            variances = znp.asarray(variances)
        return cls(holder=BinnedHolder(space=space, values=values, variances=variances))

    @classmethod
    def from_unbinned(cls, space: ZfitSpace, data: ZfitData):
        from zfit.core.binning import unbinned_to_binned

        return unbinned_to_binned(data, space)

    @classmethod
    def from_hist(cls, hist: hist.NamedHist) -> BinnedData:
        """Create a binned dataset from a `hist` histogram.

        Args:
            hist: A NamedHist. The axes will be used as the binning in zfit.
        """
        from zfit import Space

        space = Space(binning=histaxes_to_binning(hist.axes))
        values = znp.asarray(hist.values(flow=flow))
        variances = hist.variances(flow=flow)
        if variances is not None:
            variances = znp.asarray(variances)
        holder = BinnedHolder(space=space, values=values, variances=variances)
        return cls(holder=holder)

    def with_obs(self, obs: ztyping.ObsTypeInput) -> BinnedData:
        """Return a subset of the data in the ordering of *obs*.

        Args:
            obs: Which obs to return
        """
        return type(self)(holder=self.holder.with_obs(obs))

    @property
    def kind(self):
        return "COUNT"

    @property
    def n_obs(self) -> int:
        return self.rank

    @property
    def rank(self) -> int:
        return self.space.n_obs

    @property
    def obs(self):
        return self.space.obs

    def to_hist(self) -> hist.NamedHist:
        """Convert the binned data to a :py:class:`~hist.NamedHist`.

        While a binned data object can be used inside zfit (PDFs,...), it lacks many convenience features that the `hist
        library <https://hist.readthedocs.io/>`_ offers, such as plots.
        """
        binning = binning_to_histaxes(self.holder.space.binning)
        h = hist.Hist(*binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values()  # TODO: flow?
        h.view(flow=flow).variance = self.variances()  # TODO: flow?
        return h

    def _to_boost_histogram_(self):
        binning = binning_to_histaxes(self.holder.space.binning)
        h = bh.Histogram(*binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values()  # TODO: flow?
        h.view(flow=flow).variance = self.variances()  # TODO: flow?
        return h

    @property
    def space(self):
        return self.holder.space

    @property
    def axes(self):
        return self.binning

    @property
    def binning(self):
        return self.space.binning

    def values(self) -> znp.array:  # , flow=False
        """Values of the histogram as an ndim array.

        Compared to `hist`, zfit does not make a difference between a view and a copy; tensors are immutable.
        This distinction is made in the traced function by the compilation backend.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        vals = self.holder.values
        # if not flow:  # TODO: flow?
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)
        return vals

    def variances(self) -> None | znp.array:  # , flow=False
        """Variances, if available, of the histogram as an ndim array.

        Compared to `hist`, zfit does not make a difference between a view and a copy; tensors are immutable.
        This distinction is made in the traced function by the compilation backend.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        vals = self.holder.variances
        # if not flow:  # TODO: flow?
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)
        return vals

    def counts(self):
        """Effective counts of the histogram as an ndim array.

        Compared to `hist`, zfit does not make a difference between a view and a copy; tensors are immutable.
        This distinction is made in the traced function by the compilation backend.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        return self.values()

    # dummy
    @property
    def data_range(self):
        return self.space

    @property
    def nevents(self):
        return znp.sum(self.values())

    @property
    def _approx_nevents(self):
        return znp.sum(self.values())

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def to_unbinned(self):
        meshed_center = znp.meshgrid(*self.axes.centers, indexing="ij")
        flat_centers = [znp.reshape(center, (-1,)) for center in meshed_center]
        centers = znp.stack(flat_centers, axis=-1)
        flat_weights = znp.reshape(self.values(), (-1,))  # TODO: flow?
        space = self.space.copy(binning=None)
        from zfit import Data

        return Data.from_tensor(obs=space, tensor=centers, weights=flat_weights)

    def __str__(self):
        import zfit

        if zfit.run.executing_eagerly():
            return self.to_hist().__str__()
        else:
            return f"Binned data, {self.obs} (non-eager)"

    def _repr_html_(self):
        import zfit

        if zfit.run.executing_eagerly():
            return self.to_hist()._repr_html_()
        else:
            return f"Binned data, {self.obs} (non-eager)"


# tensorlike.register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)
