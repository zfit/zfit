#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
import xxhash
from tensorflow.python.util.deprecation import deprecated
from zfit_interface.typing import TensorLike

from ..core.baseobject import convert_param_values

if TYPE_CHECKING:
    pass

import boost_histogram as bh
import hist
import tensorflow as tf

from zfit.z import numpy as znp

from .._variables.axis import binning_to_histaxes, histaxes_to_binning
from ..core.interfaces import ZfitBinnedData, ZfitData, ZfitSpace
from ..util import ztyping
from ..util.exception import BreakingAPIChangeError, ShapeIncompatibleError


# @tfp.experimental.auto_composite_tensor()
class BinnedHolder:
    def __init__(self, space, values, variances):
        self._check_init_values(space, values, variances)
        self.space = space
        self.values = values
        self.variances = variances

    def _check_init_values(self, space, values, variances):
        value_shape = tf.shape(values)
        edges_shape = znp.array([tf.shape(znp.reshape(edge, (-1,)))[0] for edge in space.binning.edges])
        values_rank = value_shape.shape[0]
        if variances is not None:
            variances_shape = tf.shape(variances)
            variances_rank = variances_shape.shape[0]
            if values_rank != variances_rank:
                msg = f"Values {values} and variances {variances} differ in rank: {values_rank} vs {variances_rank}"
                raise ShapeIncompatibleError(msg)
            tf.assert_equal(
                variances_shape,
                value_shape,
                message=f"Variances and values do not have the same shape:" f" {variances_shape} vs {value_shape}",
            )
        if (binning_rank := len(space.binning.edges)) != values_rank:
            msg = f"Values and binning  differ in rank: {values_rank} vs {binning_rank}"
            raise ShapeIncompatibleError(msg)
        tf.assert_equal(
            edges_shape - 1,
            value_shape,
            message=f"Edges (minus one) and values do not have the same shape:" f" {edges_shape} vs {value_shape}",
        )

    @classmethod
    def from_hist(cls, h: hist.NamedHist):
        """Create a binned data object from a ``hist`` histogram.

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
        return cls(space=space, values=values, variances=variances)

    def with_obs(self, obs: ztyping.ObsTypeInput):
        """Return a new binned data object with updated observables order.

        Args:
            obs: The observables in the new order.
        """
        space = self.space.with_obs(obs)
        values, variances = move_axis_obs(self.space, space, values=self.values, variances=self.variances)
        return type(self)(space=space, values=values, variances=variances)

    def with_variances(self, variances: znp.array | None):
        """Return a new binned data object with updated variances.

        Args:
            variances: The new variances

        Returns:
            BinnedHolder: A new binned data object with updated variances.
        """
        if variances is not None:
            variances = znp.asarray(variances)
            from zfit import run

            if run.numeric_checks:
                tf.debugging.assert_equal(
                    tf.shape(variances),
                    tf.shape(self.values),
                    message=f"Variances {variances} and values {self.values} do not have the same shape",
                )
        return type(self)(space=self.space, values=self.values, variances=variances)


def move_axis_obs(original, target, values, variances=None):
    new_axes = [original.obs.index(ob) for ob in target.obs]
    newobs = tuple(range(target.n_obs))
    values = znp.moveaxis(values, newobs, new_axes)
    if variances is not None:
        variances = znp.moveaxis(variances, newobs, new_axes)
    return values, variances


flow = False  # TODO: track the flow or not?


# @tfp.experimental.auto_composite_tensor()
class BinnedData(
    ZfitBinnedData,
    # tfp.experimental.AutoCompositeTensor, OverloadableMixinValues, ZfitBinnedData
):
    USE_HASH = False

    def __init__(self, *, holder, use_hash=None, name: Optional[str] = None, label: Optional[str] = None):
        """Create a binned data object from a :py:class:`~zfit.core.data.BinnedHolder`.

        Prefer to use the constructors ``from_*`` of :py:class:`~zfit.core.data.BinnedData`
        like :py:meth:`~zfit.core.data.BinnedData.from_hist`, :py:meth:`~zfit.core.data.BinnedData.from_tensor`
        or :py:meth:`~zfit.core.data.BinnedData.from_unbinned`.

        Args:
            holder:
        """
        if use_hash is None:
            use_hash = self.USE_HASH
        self._use_hash = use_hash
        self._hashint = None
        self.holder: BinnedHolder = holder
        self.name = name or "BinnedData"
        self.label = label or self.name
        self._update_hash()

    def with_variances(self, variances: znp.array) -> BinnedData:
        """Return a new binned data object with updated variances.

        Args:
            variances: The new variances
        """
        return type(self)(holder=self.holder.with_variances(variances), name=self.name, label=self.label)

    def enable_hashing(self):
        """Enable hashing for this data object if it was disabled.

        A hash allows some objects to be cached and reused. If a hash is enabled, the data object will be hashed and the
        hash _can_ be used for caching. This can speedup various objects, however, it maybe doesn't have an effect at
        all. For example, if an object was already called before with the data object, the hash will probably not be
        used, as the object is already compiled.
        """
        from zfit import run

        run.assert_executing_eagerly()
        self._use_hash = True
        self._update_hash()

    @property
    def _using_hash(self):
        from zfit import run

        return self._use_hash and run.hashing_data()

    @classmethod  # TODO: add overflow bins if needed
    def from_tensor(
        cls,
        space: ZfitSpace,
        values: znp.array,
        variances: znp.array | None = None,
        name: str | None = None,
        label: str | None = None,
        use_hash: bool | None = None,
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
        return cls(
            holder=BinnedHolder(space=space, values=values, variances=variances),
            name=name,
            label=label,
            use_hash=use_hash,
        )

    @classmethod
    def from_unbinned(
        cls,
        space: ZfitSpace,
        data: ZfitData,
        *,
        use_hash: bool | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> BinnedData:
        """Convert an unbinned dataset to a binned dataset.

        Args:
            space: |@doc:binneddata.param.space| Binned space of the data.
               The space is used to define the binning and the limits of the data. |@docend:binneddata.param.space|
            data: Unbinned data to be converted to binned data

        Returns:
            ZfitBinnedData: The binned data
        """
        from zfit.core.binning import unbinned_to_binned

        return unbinned_to_binned(
            data,
            space,
            binned_class=cls,
            initkwargs={
                "name": name or data.name,
                "label": label or data.label,
                "use_hash": use_hash if use_hash is not None else data._use_hash,
            },
        )

    @classmethod
    def from_hist(cls, h: hist.NamedHist) -> BinnedData:
        """Create a binned dataset from a ``hist`` histogram.

        A histogram (following the UHI definition) with named axes.

        Args:
            h: A NamedHist. The axes will be used as the binning in zfit.
        """
        holder = BinnedHolder.from_hist(h)
        return cls(holder=holder)

    def with_obs(self, obs: ztyping.ObsTypeInput) -> BinnedData:
        """Return a subset of the data in the ordering of *obs*.

        Args:
            obs: Which obs to return
        """
        return BinnedData(holder=self.holder.with_obs(obs), name=self.name, label=self.label)
        # no subclass, as this allows the sampler to be the same still and not reinitiated

    def _update_hash(self):
        from zfit import run

        if not run.executing_eagerly() or not self._using_hash:
            self._hashint = None
        else:
            hashval = xxhash.xxh128(np.asarray(self.values()))
            if (variances := self.variances()) is not None:
                hashval.update(np.asarray(variances))
            if hasattr(self, "_hashint"):
                self._hashint = hashval.intdigest() % (64**2)
            else:  # if the dataset is not yet initialized; this is allowed
                self._hashint = None

    @property
    def hashint(self) -> int | None:
        return self._hashint

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
        return self.holder.values
        # if not flow:  # TODO: flow?
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)

    def variances(self) -> None | znp.array:  # , flow=False
        """Variances, if available, of the histogram as an ndim array.

        Compared to ``hist``, zfit does not make a difference between a view and a copy; tensors are immutable.
        This distinction is made in the traced function by the compilation backend.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        return self.holder.variances
        # if not flow:  # TODO: flow?
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)

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
        return f"Binned data {self.axes} (compiled, no preview)"

    def _repr_html_(self):
        import zfit

        if zfit.run.executing_eagerly():
            return self.to_hist()._repr_html_()
        return f"Binned data {self.axes} (compiled, no preview)"


# tensorlike.register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)


class BinnedSamplerData(BinnedData):
    _cache_counting = 0

    def __init__(
        self,
        dataset: BinnedHolder,
        *,
        sample_and_variances_func: Callable,
        sample_holder: tf.Variable = None,
        variances_holder: tf.Variable = None,
        n: ztyping.NumericalScalarType | Callable = None,
        params: ztyping.ParamValuesMap = None,
        name: str | None = None,
        label: str | None = None,
    ):
        """The ``BinnedSampler`` is a binned data object that can be resampled, i.e. modified in-place.

        Use `from_sampler` to create a `BinnedSampler`.

        Args:
            dataset: The data holder that contains the sample and the variances.
            sample_and_variances_func: A function that samples the data and returns the sample and the variances.
            sample_holder: The tensor that holds the sample.
            variances_holder: The tensor that holds the variances.
            n: The number of samples to produce. If the `SamplerData` was created with
                anything else then a numerical or tf.Tensor, this can't be used.
            params: A mapping from `Parameter` to a fixed value that should be used for the sampling.
            name: The name of the data object.
            label: The label of the data object.
        """
        super().__init__(holder=dataset, name=name, label=label, use_hash=True)
        params = convert_param_values(params)

        self._initial_resampled = False

        self.params = params
        self._sample_holder = sample_holder
        self._variances_holder = variances_holder
        self._sample_and_variances_func = sample_and_variances_func
        self.n = n
        self._n_holder = n

        # we need to use a hash because it could change -> for loss etc to know when data changes
        self._hashint_holder = tf.Variable(
            initial_value=0,
            dtype=tf.int64,
            trainable=False,
            shape=(),
        )
        self.update_data(dataset.values, variances=dataset.variances)

    @property
    @deprecated(None, "Use `params` instead.")
    def fixed_params(self):
        return self.params

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
        return self._hashint_holder.value()

    def _update_hash(self):
        super()._update_hash()
        if hasattr(self, "_hashint_holder"):  # initialization
            self._hashint_holder.assign(self._hashint % (64**2))

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(
        cls,
        sample_func: Callable,  # noqa: ARG003
        n: ztyping.NumericalScalarType,  # noqa: ARG003
        obs: ztyping.ObsTypeInput,  # noqa: ARG003
        fixed_params=None,  # noqa: ARG003
    ):
        msg = " Use `from_sampler` (with `r` at the end instead."
        raise BreakingAPIChangeError(msg)

    @classmethod
    def from_sampler(
        cls,
        *,
        sample_func: Callable | None = None,
        sample_and_variances_func: Callable | None = None,
        n: ztyping.NumericalScalarType,
        obs: ztyping.AxesTypeInput,
        params: ztyping.ParamValuesMap = None,
        fixed_params=None,
        name: str | None = None,
        label: str | None = None,
    ):
        """Create a binned sampler from a sample function.

        This is a binned data object that can be modified in-place by updating/resampling the sample.

        Args:
            sample_func: A function that samples the data.
            sample_and_variances_func: A function that samples the data and returns the sample and the variances.
            n: The number of samples to produce.
            obs: The observables of the data.
            params: A mapping from :py:class:~`zfit.Parameter` or string (the name) to a fixed value that should be used for the sampling.
            name: The name of the data object.
            label: The label of the data object.
        """
        if fixed_params is not None:
            msg = "Use `params` instead of `fixed_params`."
            raise BreakingAPIChangeError(msg)
        if int(sample_func is not None) + int(sample_and_variances_func is not None) != 1:
            msg = "Exactly one of `sample`, `sample_func` or `sample_and_variances_func` must be provided."
            raise ValueError(msg)

        if sample_func is not None:

            def sample_and_variances_func(n, params, *, sample_func=sample_func):
                sample = sample_func(n, params=params)
                return sample, None

            del sample_func

        from ..core.space import convert_to_space

        obs = convert_to_space(obs)

        from .. import ztypes

        dtype = ztypes.float

        params = convert_param_values(params)

        initval, initvar = sample_and_variances_func(n, params=params)  # todo: preprocess, cut data?
        sample_holder = tf.Variable(
            initial_value=initval,
            dtype=dtype,
            trainable=False,
            # validate_shape=False,
            shape=(None,) * obs.n_obs,
            name=f"sample_hist_holder_{cls.get_cache_counting()}",
        )
        if initvar is not None:
            variances_holder = tf.Variable(
                initial_value=initvar,
                dtype=dtype,
                trainable=False,
                # validate_shape=False,
                shape=(None,) * obs.n_obs,
                name=f"variances_hist_holder_{cls.get_cache_counting()}",
            )
        else:
            variances_holder = None
        dataset = BinnedHolder(space=obs, values=sample_holder, variances=variances_holder)

        return cls(
            dataset=dataset,
            sample_holder=sample_holder,
            sample_and_variances_func=sample_and_variances_func,
            variances_holder=variances_holder,
            name=name,
            label=label,
            params=params,
            n=n,
        )

    def resample(
        self,
        params: ztyping.ParamValuesMap = None,
        *,
        n: int | tf.Tensor = None,
        param_values: ztyping.ParamValuesMap = None,
    ):
        """Update the sample by new sampling *inplace*; This affects any object that used this data already.

        All params that are not in the attribute ``params`` will use their current value for
        the creation of the new sample. The value can also be overwritten for one sampling by providing
        a mapping with ``param_values`` from ``Parameter`` to the temporary ``value``.

        Args:
            params: a mapping from :py:class:`~zfit.Parameter` to a `value` that should be used for the sampling.
                Any parameter that is not in this mapping will use the value in `params`.
            n: the number of samples to produce. If the `SamplerData` was created with
                anything else then a numerical or tf.Tensor, this can't be used.
        """
        if n is None:
            n = self.n

        if param_values is not None:
            if params is not None:
                msg = "Cannot specify both `fixed_params` and `params`."
                raise ValueError(msg)
            params = param_values
        temp_param_values = self.params.copy()
        if params is not None:
            params = convert_param_values(params)
            temp_param_values.update(params)

        new_sample, new_variances = self._sample_and_variances_func(n, params=temp_param_values)
        self.update_data(new_sample, new_variances)

    def update_data(self, sample: TensorLike, variances: TensorLike | None = None):
        """Update the data, and optionally the variances, of the sampler in-place.

        This change will be reflected in any object that used this data already.

        Args:
            sample: The new sample.
            variances: The new variances. Can only be provided if the sampler was initialized with variances and *must* be
                provided if the sampler was initialized with variances.
        """
        sample = znp.asarray(sample)
        if variances is not None:
            variances = znp.asarray(variances)
        self._sample_holder.assign(sample, read_value=False)
        if variances is not None:
            if self._variances_holder is None:
                msg = "Variances were not initialized, cannot update them."
                raise ValueError(msg)
            self._variances_holder.assign(variances, read_value=False)
        elif self._variances_holder is not None:
            msg = "Variances were initialized, cannot remove them."
            raise ValueError(msg)
        self._initial_resampled = True
        self._update_hash()

    def values(self) -> znp.array:
        """Values/counts of the histogram as an ndim array.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        return znp.asarray(super().values())  # otherwise, shape is not correct -> use handler if variable is needed

    def variances(self) -> znp.array:
        """Variances of the histogram as an ndim array or `None` if no variances are available.

        Returns:
            Tensor of shape (nbins0, nbins1, ...) with nbins the number of bins in each observable.
        """
        if (variances := super().variances()) is not None:
            variances = znp.asarray(variances)
        return variances

    def __repr__(self) -> str:
        return f"<BinnedSampler: {self.name} obs={self.obs}>"
