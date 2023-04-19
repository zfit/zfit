#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..core.parameter import set_values

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
        """Create a binned data object from a :py:class:`~zfit.core.data.BinnedHolder`.

        Prefer to use the constructors ``from_*`` of :py:class:`~zfit.core.data.BinnedData`
        like :py:meth:`~zfit.core.data.BinnedData.from_hist`, :py:meth:`~zfit.core.data.BinnedData.from_tensor`
        or :py:meth:`~zfit.core.data.BinnedData.from_unbinned`.

        Args:
            holder:
        """
        self.holder: BinnedHolder = holder
        self.name = "BinnedData"  # TODO: improve naming

    @classmethod  # TODO: add overflow bins if needed
    def from_tensor(
        cls, space: ZfitSpace, values: znp.array, variances: znp.array | None = None
    ) -> BinnedData:
        """Create a binned dataset defined in *space* where values are considered to be the counts.

        Args:
            space: |@doc:binneddata.param.space| Binned space of the data.
               The space is used to define the binning and the limits of the data. |@docend:binneddata.param.space|
            values: |@doc:binneddata.param.values| Corresponds to the counts of the histogram.
               Follows the definition of the
               `Unified Histogram Interface (UHI) <https://uhi.readthedocs.io/en/latest/plotting.html#plotting>`_. |@docend:binneddata.param.values|
            variances: |@doc:binneddata.param.variances| Corresponds to the uncertainties of the histogram.
               If ``True``, the uncertainties are created assuming that ``values``
               have been drawn from a Poisson distribution. Follows the definition of the
               `Unified Histogram Interface (UHI) <https://uhi.readthedocs.io/en/latest/plotting.html#plotting>`_. |@docend:binneddata.param.variances|
        """
        values = znp.asarray(values, znp.float64)
        if variances is True:
            variances = znp.sqrt(values)
        elif variances is not None:
            variances = znp.asarray(variances)
        return cls(holder=BinnedHolder(space=space, values=values, variances=variances))

    @classmethod
    def from_unbinned(cls, space: ZfitSpace, data: ZfitData):
        """Convert an unbinned dataset to a binned dataset.

        Args:
            space: |@doc:binneddata.param.space| Binned space of the data.
               The space is used to define the binning and the limits of the data. |@docend:binneddata.param.space|
            data: Unbinned data to be converted to binned data

        Returns:
            ZfitBinnedData: The binned data
        """
        from zfit.core.binning import unbinned_to_binned

        return unbinned_to_binned(data, space)

    @classmethod
    def from_hist(cls, h: hist.NamedHist) -> BinnedData:
        """Create a binned dataset from a ``hist`` histogram.

        A histogram (following the UHI definition) with named axes.

        Args:
            h: A NamedHist. The axes will be used as the binning in zfit.
        """
        from zfit import Space

        space = Space(binning=histaxes_to_binning(h.axes))
        values = znp.asarray(h.values(flow=flow))
        variances = h.variances(flow=flow)
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

    def to_hist(self) -> hist.Hist:
        """Convert the binned data to a :py:class:`~hist.NamedHist`.

        While a binned data object can be used inside zfit (PDFs,...), it lacks many convenience features that the
        `hist library <https://hist.readthedocs.io/>`_
        offers, such as plots.
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

        Compared to ``hist``, zfit does not make a difference between a view and a copy; tensors are immutable.
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

        Compared to ``hist``, zfit does not make a difference between a view and a copy; tensors are immutable.
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
        """Effective counts of the histogram as a ndim array.

        Compared to ``hist``, zfit does not make a difference between a view and a copy; tensors are immutable.
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
    def n_events(self):  # LEGACY, what should be the name?
        return self.nevents

    @property
    def _approx_nevents(self):
        return znp.sum(self.values())

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def to_unbinned(self):
        """Use the bincenters as unbinned data with values as counts.

        Returns:
            ``ZfitData``: Unbinned data
        """
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


class SampleHolder(BinnedHolder):
    def with_obs(self, obs):
        assert False, "INTERNAL ERROR, should never be used directly"


class BinnedSampler(BinnedData):
    _cache_counting = 0

    def __init__(
        self,
        dataset: SampleHolder,
        sample_func: Callable,
        sample_holder: tf.Variable,
        n: ztyping.NumericalScalarType | Callable,
        fixed_params: dict[zfit.Parameter, ztyping.NumericalScalarType] = None,
    ):
        super().__init__(holder=dataset)
        if fixed_params is None:
            fixed_params = {}
        if isinstance(fixed_params, (list, tuple)):
            fixed_params = {param: param.numpy() for param in fixed_params}

        self._initial_resampled = False

        self.fixed_params = fixed_params
        self.sample_holder = sample_holder
        self.sample_func = sample_func
        self.n = n
        self._n_holder = n
        self.resample()  # to be used for precompilations etc

    @property
    def n_samples(self):
        return self._n_holder

    @property
    def _approx_nevents(self):
        nevents = super()._approx_nevents
        if nevents is None:
            nevents = self.n
        return nevents

    @property
    def hashint(self) -> int | None:
        return None  # since the variable can be changed but this may stays static... and using 128 bits we can't have
        # a tf.Variable that keeps the int

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(
        cls,
        sample_func: Callable,
        n: ztyping.NumericalScalarType,
        obs: ztyping.ObsTypeInput,
        fixed_params=None,
        dtype=None,
    ):
        from ..core.space import convert_to_space

        obs = convert_to_space(obs)

        if fixed_params is None:
            fixed_params = []
        if dtype is None:
            from .. import ztypes

            dtype = ztypes.float
        # from tensorflow.python.ops.variables import VariableV1
        sample_holder = tf.Variable(
            initial_value=sample_func(n),
            dtype=dtype,
            trainable=False,
            # validate_shape=False,
            shape=(None,) * obs.n_obs,
            name=f"sample_hist_holder_{cls.get_cache_counting()}",
        )
        dataset = SampleHolder(space=obs, values=sample_holder, variances=None)

        return cls(
            dataset=dataset,
            sample_holder=sample_holder,
            sample_func=sample_func,
            fixed_params=fixed_params,
            n=n,
        )

    def resample(self, param_values: Mapping = None, n: int | tf.Tensor = None):
        """Update the sample by newly sampling. This affects any object that used this data already.

        All params that are not in the attribute ``fixed_params`` will use their current value for
        the creation of the new sample. The value can also be overwritten for one sampling by providing
        a mapping with ``param_values`` from ``Parameter`` to the temporary ``value``.

        Args:
            param_values: a mapping from :py:class:`~zfit.Parameter` to a `value`. For the current sampling,
                `Parameter` will use the `value`.
            n: the number of samples to produce. If the `Sampler` was created with
                anything else then a numerical or tf.Tensor, this can't be used.
        """
        if n is None:
            n = self.n

        temp_param_values = self.fixed_params.copy()
        if param_values is not None:
            temp_param_values.update(param_values)

        with set_values(
            list(temp_param_values.keys()), list(temp_param_values.values())
        ):
            new_sample = self.sample_func(n)
            self.sample_holder.assign(new_sample, read_value=False)
            self._initial_resampled = True

    def with_obs(self, obs: ztyping.ObsTypeInput) -> BinnedSampler:
        """Create a new :py:class:`~zfit.core.data.BinnedSampler` with the same sample but different ordered
        observables.

        Args:
            obs: The new observables
        """
        from ..core.space import convert_to_space

        obs = convert_to_space(obs)
        if obs.obs == self.obs:
            return self

        def new_sample_func(n):
            sample = self.sample_func(n)
            values = move_axis_obs(self.space, obs, sample)
            return values

        return BinnedSampler.from_sample(
            sample_func=new_sample_func,
            n=self.n,
            obs=obs,
            fixed_params=self.fixed_params,
        )

    def values(self) -> znp.array:
        return znp.asarray(super().values())

    def __str__(self) -> str:
        return f"<BinnedSampler: {self.name} obs={self.obs}>"
