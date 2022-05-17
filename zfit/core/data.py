#  Copyright (c) 2022 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from tensorflow.python.util.deprecation import deprecated_args, deprecated

if TYPE_CHECKING:
    import zfit

from collections.abc import Mapping
from collections.abc import Callable

from collections import OrderedDict
from contextlib import ExitStack

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot

# from ..settings import types as ztypes
import zfit
import zfit.z.numpy as znp

from .. import z
from ..settings import ztypes
from ..util import ztyping
from ..util.cache import GraphCachable, invalidate_graph
from ..util.container import convert_to_container
from ..util.exception import (
    ObsIncompatibleError,
    ShapeIncompatibleError,
    WorkInProgressError,
)
from ..util.temporary import TemporarilySet
from .baseobject import BaseObject
from .coordinates import convert_to_obs_str
from .dimension import BaseDimensional
from .interfaces import ZfitSpace, ZfitUnbinnedData
from .tensorlike import register_tensor_conversion, OverloadableMixin
from .space import Space, convert_to_space


# TODO: make cut only once, then remember
class Data(
    GraphCachable, ZfitUnbinnedData, BaseDimensional, BaseObject, OverloadableMixin
):
    BATCH_SIZE = 1000000  # 1 mio

    def __init__(
        self,
        dataset: tf.data.Dataset | LightDataset,
        obs: ztyping.ObsTypeInput = None,
        name: str = None,
        weights=None,
        iterator_feed_dict: dict = None,
        dtype: tf.DType = None,
    ):
        """Create a data holder from a `dataset` used to feed into `models`.

        Args:
            dataset: A dataset storing the actual values
            obs: Observables where the data is defined in
            name: Name of the `Data`
            iterator_feed_dict:
            dtype: |dtype_arg_descr|
        """
        if name is None:
            name = "Data"
        if dtype is None:
            dtype = ztypes.float
        super().__init__(name=name)
        # if iterator_feed_dict is None:
        #     iterator_feed_dict = {}
        self._permutation_indices_data = None
        self._next_batch = None
        self._dtype = dtype
        self._nevents = None
        self._weights = None

        self._data_range = None
        self._set_space(obs)
        self._original_space = self.space
        self._data_range = (
            self.space
        )  # TODO proper data cuts: currently set so that the cuts in all dims are applied
        self.batch_size = self.BATCH_SIZE
        self.dataset = dataset.batch(self.batch_size)
        self._name = name
        self.iterator_feed_dict = iterator_feed_dict
        self.iterator = None
        self.set_weights(weights=weights)

    @property
    def nevents(self):
        nevents = self._nevents
        if nevents is None:
            nevents = self._get_nevents()
        return nevents

    # TODO: which naming? nevents or n_events

    @property
    def _approx_nevents(self):
        return self.nevents

    @property
    def n_events(self):
        return self.nevents

    @property
    def has_weights(self):
        return self._weights is not None

    @property
    def dtype(self):
        return self._dtype

    def _set_space(self, obs: Space):
        obs = convert_to_space(obs)
        self._check_n_obs(space=obs)
        obs = obs.with_autofill_axes(overwrite=True)
        self._space = obs

    @property
    def data_range(self):
        data_range = self._data_range
        if data_range is None:
            data_range = self.space
        return data_range

    @invalidate_graph
    def set_data_range(self, data_range):
        data_range = self._check_input_data_range(data_range=data_range)

        def setter(value):
            self._data_range = value

        def getter():
            return self._data_range

        return TemporarilySet(value=data_range, setter=setter, getter=getter)

    @property
    def weights(self):
        # TODO: refactor below more general, when to apply a cut?
        if self.data_range.has_limits and self.has_weights:
            raw_values = self._value_internal(obs=self.data_range.obs, filter=False)
            is_inside = self.data_range.inside(raw_values)
            weights = self._weights[is_inside]
        else:
            weights = self._weights
        return weights

    @deprecated(None, "Do not set the weights on a data set, create a new one instead.")
    @invalidate_graph
    def set_weights(self, weights: ztyping.WeightsInputType):
        """Set (temporarily) the weights of the dataset.

        Args:
            weights:
        """
        if weights is not None:
            weights = z.convert_to_tensor(weights)
            weights = z.to_real(weights)
            if weights.shape.ndims != 1:
                raise ShapeIncompatibleError("Weights have to be 1-Dim objects.")

        def setter(value):
            self._weights = value

        def getter():
            return self.weights

        return TemporarilySet(value=weights, getter=getter, setter=setter)

    @property
    def space(self) -> ZfitSpace:
        return self._space

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        obs: ztyping.ObsTypeInput = None,
        weights: ztyping.WeightsInputType | str = None,
        name: str = None,
        dtype: tf.DType = None,
    ):
        """Create a `Data` from a pandas DataFrame. If `obs` is `None`, columns are used as obs.

        Args:
            df: pandas DataFrame that contains the data. If `obs` is `None`, columns are used as obs. Can be
                a superset of obs.
            obs: obs to use for the data. obs have to be the columns in the data frame.
                If `None`, columns are used as obs.
            weights: Weights of the data. Has to be 1-D and match the shape
                of the data (nevents) or a string that is a column in the dataframe.
            name:
        """
        if obs is None:
            obs = list(df.columns)

        obs = convert_to_space(obs)
        not_in_df = set(obs.obs) - set(df.columns)
        if not_in_df:
            raise ValueError(
                f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            )
        if isinstance(weights, str):
            if weights not in df.columns:
                raise ValueError(
                    f"Weights {weights} is a string and not in dataframe with columns {df.columns}"
                )
            weights = df[weights]
        array = df[list(obs.obs)].values

        return cls.from_numpy(
            obs=obs, array=array, weights=weights, name=name, dtype=dtype
        )

    @classmethod
    @deprecated_args(None, "Use obs instead.", "branches")
    @deprecated_args(
        None,
        "Use obs_alias instead and make sure to invert the logic! I.e. it's a mapping from"
        " the observable name to the actual branch name.",
        "branches_alias",
    )
    def from_root(
        cls,
        path: str,
        treepath: str,
        obs: ZfitSpace = None,
        *,
        weights: ztyping.WeightsStrInputType = None,
        obs_alias: Mapping[str, str] = None,
        name: str = None,
        dtype: tf.DType = None,
        root_dir_options=None,
        # deprecated
        branches: list[str] = None,
        branches_alias: dict = None,
    ) -> Data:
        """Create a `Data` from a ROOT file. Arguments are passed to `uproot`.

        The arguments are passed to uproot directly.

        Args:
            path: Path to the root file.
            treepath: Name of the tree in the root file.
            obs: Observables of the data. This will also be the columns of the data if not *obs_alias* is given.
            weights: Weights of the data. Has to be 1-D and match the shape
                of the data (nevents). Can be a column of the ROOT file by using a string corresponding to a
                column.
            obs_alias: A mapping from the `obs` (as keys) to the actual `branches` (as values) in the root file.
                This allows to have different `observable` names, independent of the branch name in the file.
            name:
            root_dir_options:

        Returns:
            `zfit.Data`: A `Data` object containing the unbinned data.
        """
        # begin deprecated legacy arguments
        if branches:
            obs = branches
            del branches
        if branches_alias is not None:
            if obs_alias is not None:
                raise ValueError("Cannot use both `branches_alias` and `obs_alias`.")
            obs_alias = {obs: branch for branch, obs in branches_alias.items()}
            del branches_alias

        # end legacy
        if root_dir_options is None:
            root_dir_options = {}
        if obs_alias is None and obs is None:
            raise ValueError("Either branches or branches_alias has to be specified.")
        if obs_alias is None:
            obs_alias = {}
        if obs is None:
            obs = list(obs_alias.values())

        obs = convert_to_space(obs)

        branches = [obs_alias.get(branch, branch) for branch in obs.obs]

        weights_are_branch = isinstance(weights, str)

        def uproot_loader():
            with uproot.open(path, **root_dir_options)[treepath] as root_tree:
                if weights_are_branch:
                    branches_with_weights = branches + [weights]
                else:
                    branches_with_weights = branches
                branches_with_weights = tuple(branches_with_weights)
                data = root_tree.arrays(expressions=branches_with_weights, library="pd")
            data_np = data[branches].values
            if weights_are_branch:
                weights_np = data[weights]
            else:
                weights_np = None
            return data_np, weights_np

        data, weights_np = uproot_loader()
        if not weights_are_branch:
            weights_np = weights
        dataset = LightDataset.from_tensor(data)

        return Data(
            dataset=dataset, obs=obs, weights=weights_np, name=name, dtype=dtype
        )

    @classmethod
    def from_numpy(
        cls,
        obs: ztyping.ObsTypeInput,
        array: np.ndarray,
        weights: ztyping.WeightsInputType = None,
        name: str = None,
        dtype: tf.DType = None,
    ):
        """Create `Data` from a `np.array`.

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            array: Numpy array containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name:

        Returns:
            `zfit.Data`: A `Data` object containing the unbinned data.
        """

        if not isinstance(array, (np.ndarray)) and not (
            tf.is_tensor(array) and hasattr(array, "numpy")
        ):
            raise TypeError(
                f"`array` has to be a `np.ndarray`. Is currently {type(array)}"
            )
        if dtype is None:
            dtype = ztypes.float
        tensor = tf.cast(array, dtype=dtype)
        return cls.from_tensor(
            obs=obs, tensor=tensor, weights=weights, name=name, dtype=dtype
        )

    @classmethod
    def from_tensor(
        cls,
        obs: ztyping.ObsTypeInput,
        tensor: tf.Tensor,
        weights: ztyping.WeightsInputType = None,
        name: str = None,
        dtype: tf.DType = None,
    ) -> Data:
        """Create a `Data` from a `tf.Tensor`. `Value` simply returns the tensor (in the right order).

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            tensor: Tensor containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: Name of the data.

        Returns:
            `zfit.Data`: A `Data` object containing the unbinned data.
        """
        # dataset = LightDataset.from_tensor(tensor=tensor)
        if dtype is None:
            dtype = ztypes.float
        tensor = tf.cast(tensor, dtype=dtype)
        if len(tensor.shape) == 0:
            tensor = znp.expand_dims(tensor, -1)
        if len(tensor.shape) == 1:
            tensor = znp.expand_dims(tensor, -1)
        # dataset = tf.data.Dataset.from_tensor_slices(tensor)
        dataset = LightDataset.from_tensor(tensor)

        return Data(dataset=dataset, obs=obs, name=name, weights=weights, dtype=dtype)

    def with_obs(self, obs):
        values = self.value(obs)
        return type(self).from_tensor(
            obs=self.space, tensor=values, weights=self.weights, name=self.name
        )

    def to_pandas(self, obs: ztyping.ObsTypeInput = None):
        """Create a `pd.DataFrame` from `obs` as columns and return it.

        Args:
            obs: The observables to use as columns. If `None`, all observables are used.

        Returns:
        """
        values = self.value(obs=obs)
        if obs is None:
            obs = self.obs
        obs_str = convert_to_obs_str(obs)
        values = values.numpy()
        df = pd.DataFrame(data=values, columns=obs_str)
        return df

    def unstack_x(self, obs: ztyping.ObsTypeInput = None, always_list: bool = False):
        """Return the unstacked data: a list of tensors or a single Tensor.

        Args:
            obs: which observables to return
            always_list: If True, always return a list (also if length 1)

        Returns:
            List(tf.Tensor)
        """
        return z.unstack_x(self.value(obs=obs))

    def value(self, obs: ztyping.ObsTypeInput = None):
        return znp.asarray(self._value_internal(obs=obs))
        # TODO: proper iterations
        # value_iter = self._value_internal(obs=obs)
        # value = next(value_iter)
        # try:
        #     next(value_iter)
        # except StopIteration:  # it's ok, we're not batched
        #     return value
        # else:
        #     raise DataIsBatchedError(
        #         f"Data {self} is batched, cannot return only the value. Iterate through it (WIP, make"
        #         f"an issue on Github if this feature is needed now)")

    def numpy(self):
        return self.value().numpy()

    def _cut_data(self, value, obs=None):
        if self.data_range.has_limits:
            data_range = self.data_range.with_obs(obs=obs)
            value = data_range.filter(value)

        return value

    def _value_internal(self, obs: ztyping.ObsTypeInput = None, filter: bool = True):
        if obs is not None:
            obs = convert_to_obs_str(obs)
        # for raw_value in self.dataset:
        # value = self._check_convert_value(raw_value)
        value = self.dataset.value()
        if filter:
            value = self._cut_data(value, obs=self._original_space.obs)
        value_sorted = self._sort_value(value=value, obs=obs)
        return value_sorted

    def _check_convert_value(self, value):
        # TODO(Mayou36): add conversion to right dimension? (n_events, n_obs)? # check if 1-D?
        if len(value.shape.as_list()) == 0:
            value = znp.expand_dims(value, -1)
        if len(value.shape.as_list()) == 1:
            value = znp.expand_dims(value, -1)

        # cast data to right type
        if value.dtype != self.dtype:
            value = tf.cast(value, dtype=self.dtype)
        return value

    def _sort_value(self, value, obs: tuple[str]):
        obs = convert_to_container(value=obs, container=tuple)
        # TODO CURRENT: deactivated below!
        perm_indices = (
            self.space.axes
            if self.space.axes != tuple(range(value.shape[-1]))
            else False
        )

        # permutate = perm_indices is not None
        if obs:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError(
                    "The observable(s) {} are not contained in the dataset. "
                    "Only the following are: {}".format(
                        frozenset(obs) - frozenset(self.obs), self.obs
                    )
                )
            perm_indices = self.space.get_reorder_indices(obs=obs)
            # values = list(values[self.obs.index(o)] for o in obs if o in self.obs)
        if perm_indices:
            value = z.unstack_x(value, always_list=True)
            value = [value[i] for i in perm_indices]
            value = z.stack_x(value)

        return value

    # TODO(Mayou36): use Space to permute data?
    # TODO(Mayou36): raise error is not obs <= self.obs?
    @invalidate_graph
    def sort_by_axes(self, axes: ztyping.AxesTypeInput, allow_superset: bool = True):
        if not allow_superset:
            if not frozenset(axes) <= frozenset(self.axes):
                raise ValueError(
                    "The observable(s) {} are not contained in the dataset. "
                    "Only the following are: {}".format(
                        frozenset(axes) - frozenset(self.axes), self.axes
                    )
                )
        space = self.space.with_axes(axes=axes, allow_subset=True)

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    @invalidate_graph
    def sort_by_obs(self, obs: ztyping.ObsTypeInput, allow_superset: bool = False):
        if not allow_superset:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError(
                    "The observable(s) {} are not contained in the dataset. "
                    "Only the following are: {}".format(
                        frozenset(obs) - frozenset(self.obs), self.obs
                    )
                )

        space = self.space.with_obs(
            obs=obs, allow_subset=True, allow_superset=allow_superset
        )

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    def _check_input_data_range(self, data_range):
        data_range = self._convert_sort_space(limits=data_range)
        if not frozenset(self.data_range.obs) == frozenset(data_range.obs):
            raise ObsIncompatibleError(
                f"Data range has to cover the full observable space {self.data_range.obs}, not "
                f"only {data_range.obs}"
            )
        return data_range

    # TODO(Mayou36): refactor with pdf or other range things?
    def _convert_sort_space(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        limits: ztyping.LimitsTypeInput = None,
    ) -> Space | None:
        """Convert the inputs (using eventually `obs`, `axes`) to
        :py:class:`~zfit.Space` and sort them according to own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:
            space = space.with_coords(self.space, allow_subset=True)
        return space

    def _get_nevents(self):
        return tf.shape(input=self.value())[0]

    def __str__(self) -> str:
        return f"<zfit.Data: {self.name} obs={self.obs}>"

    def to_binned(self, space):
        from zfit._data.binneddatav1 import BinnedData

        return BinnedData.from_unbinned(space=space, data=self)


class SampleData(Data):
    _cache_counting = 0

    def __init__(
        self,
        dataset: tf.data.Dataset | LightDataset,
        obs: ztyping.ObsTypeInput = None,
        weights=None,
        name: str = None,
        dtype: tf.DType = ztypes.float,
    ):
        super().__init__(
            dataset,
            obs,
            name=name,
            weights=weights,
            iterator_feed_dict=None,
            dtype=dtype,
        )

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(
        cls,
        sample: tf.Tensor,
        obs: ztyping.ObsTypeInput,
        name: str = None,
        weights=None,
    ):
        dataset = LightDataset.from_tensor(sample)
        return SampleData(dataset=dataset, obs=obs, name=name, weights=weights)


class Sampler(Data):
    _cache_counting = 0

    def __init__(
        self,
        dataset: LightDataset,
        sample_func: Callable,
        sample_holder: tf.Variable,
        n: ztyping.NumericalScalarType | Callable,
        weights=None,
        fixed_params: dict[zfit.Parameter, ztyping.NumericalScalarType] = None,
        obs: ztyping.ObsTypeInput = None,
        name: str = None,
        dtype: tf.DType = ztypes.float,
    ):

        super().__init__(
            dataset=dataset,
            obs=obs,
            name=name,
            weights=weights,
            iterator_feed_dict=None,
            dtype=dtype,
        )
        if fixed_params is None:
            fixed_params = OrderedDict()
        if isinstance(fixed_params, (list, tuple)):
            fixed_params = OrderedDict(
                (param, param.numpy()) for param in fixed_params
            )  # TODO: numpy -> read_value?

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

    def _value_internal(self, obs: ztyping.ObsTypeInput = None, filter: bool = True):
        if not self._initial_resampled:
            raise RuntimeError(
                "No data generated yet. Use `resample()` to generate samples or directly use `model.sample()`"
                "for single-time sampling."
            )
        return super()._value_internal(obs=obs, filter=filter)

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
        name: str = None,
        weights=None,
        dtype=None,
    ):
        obs = convert_to_space(obs)

        if fixed_params is None:
            fixed_params = []
        if dtype is None:
            dtype = ztypes.float
        # from tensorflow.python.ops.variables import VariableV1
        sample_holder = tf.Variable(
            initial_value=sample_func(),
            dtype=dtype,
            trainable=False,  # HACK: sample_func
            # validate_shape=False,
            shape=(None, obs.n_obs),
            name=f"sample_data_holder_{cls.get_cache_counting()}",
        )
        dataset = LightDataset.from_tensor(sample_holder)

        return Sampler(
            dataset=dataset,
            sample_holder=sample_holder,
            sample_func=sample_func,
            fixed_params=fixed_params,
            n=n,
            obs=obs,
            name=name,
            weights=weights,
        )

    def resample(self, param_values: Mapping = None, n: int | tf.Tensor = None):
        """Update the sample by newly sampling. This affects any object that used this data already.

        All params that are not in the attribute `fixed_params` will use their current value for
        the creation of the new sample. The value can also be overwritten for one sampling by providing
        a mapping with `param_values` from `Parameter` to the temporary `value`.

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

        with ExitStack() as stack:

            _ = [
                stack.enter_context(param.set_value(val))
                for param, val in temp_param_values.items()
            ]

            # if not (n and self._initial_resampled):  # we want to load and make sure that it's initialized
            #     # means it's handled inside the function
            #     # TODO(Mayou36): check logic; what if new_samples loaded? get's overwritten by initializer
            #     # fixed with self.n, needs cleanup
            #     if not (isinstance(self.n_samples, str) or self.n_samples is None):
            #         self.sess.run(self.n_samples.initializer)
            # if n:
            #     if not isinstance(self.n_samples, tf.Variable):
            #         raise RuntimeError("Cannot set a new `n` if not a Tensor-like object was given")
            # self.n_samples.assign(n)

            new_sample = self.sample_func(n)
            # self.sample_holder.assign(new_sample)
            self.sample_holder.assign(new_sample, read_value=False)
            self._initial_resampled = True

    def __str__(self) -> str:
        return f"<Sampler: {self.name} obs={self.obs}>"


register_tensor_conversion(Data, name="Data", overload_operators=True)


class LightDataset:
    def __init__(self, tensor):
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            tensor = z.convert_to_tensor(tensor)
        self.tensor = tensor

    def batch(self, batch_size):  # ad-hoc just empty
        return self

    def __iter__(self):
        yield self.value()

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor=tensor)

    def value(self):
        return self.tensor


def sum_samples(
    sample1: ZfitUnbinnedData,
    sample2: ZfitUnbinnedData,
    obs: ZfitSpace,
    shuffle: bool = False,
):
    samples = [sample1, sample2]
    if obs is None:
        raise WorkInProgressError
    sample2 = sample2.value(obs=obs)
    if shuffle:
        sample2 = tf.random.shuffle(sample2)
    sample1 = sample1.value(obs=obs)
    tensor = sample1 + sample2
    if any([s.weights is not None for s in samples]):
        raise WorkInProgressError("Cannot combine weights currently")
    weights = None

    return SampleData.from_sample(sample=tensor, obs=obs, weights=weights)
