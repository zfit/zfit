#  Copyright (c) 2023 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, List

import pydantic
from pydantic import Field

from ..serialization import SpaceRepr

from typing import Literal

import xxhash
from tensorflow.python.util.deprecation import deprecated_args, deprecated

from .parameter import set_values
from .serialmixin import ZfitSerializable, SerializableMixin
from ..serialization.serializer import BaseRepr, to_orm_init

if TYPE_CHECKING:
    import zfit

from collections.abc import Mapping
from collections.abc import Callable

from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot

import zfit
import zfit.z.numpy as znp

from .. import z
from ..settings import ztypes, run
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
from .space import Space, convert_to_space


# TODO: make cut only once, then remember
class Data(
    ZfitUnbinnedData,
    BaseDimensional,
    BaseObject,
    GraphCachable,
    SerializableMixin,
    ZfitSerializable,
):
    BATCH_SIZE = 1000000  # 1 mio

    def __init__(
        self,
        dataset: tf.data.Dataset | LightDataset,
        obs: ztyping.ObsTypeInput = None,
        name: str = None,
        weights=None,
        dtype: tf.DType = None,
        use_hash: bool = None,
    ):
        """Create a data holder from a ``dataset`` used to feed into ``models``.

        Args:
            dataset: A dataset storing the actual values
            obs: Observables where the data is defined in
            name: Name of the ``Data``
            weights: Weights of the data
            dtype: |dtype_arg_descr|
            use_hash: Whether to use a hash for caching
        """
        if use_hash is None:
            use_hash = run.hashing_data()
        self._use_hash = use_hash
        if name is None:
            name = "Data"
        if dtype is None:
            dtype = ztypes.float
        super().__init__(name=name)

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
        self.dataset = dataset.batch(100_000_000)
        self._name = name

        self._set_weights(weights=weights)
        self._hashint = None
        self._update_hash()

    @property
    def nevents(self):
        nevents = self._nevents
        if nevents is None:
            nevents = self._get_nevents()
        return nevents

    @property
    def hashint(self) -> int | None:
        return self._hashint

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
        self._update_hash()

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
            self._update_hash()

        def getter():
            return self._data_range

        return TemporarilySet(value=data_range, setter=setter, getter=getter)

    @property
    def weights(self):
        """Get the weights of the data."""
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

        # weights = self._set_weights(weights)

        def setter(value):
            self._set_weights(value)

        def getter():
            return self.weights

        return TemporarilySet(value=weights, getter=getter, setter=setter)

    def _set_weights(self, weights):
        if weights is not None:
            weights = z.convert_to_tensor(weights)
            weights = z.to_real(weights)
            if weights.shape.ndims != 1:
                if weights.shape.ndims == 2 and weights.shape[1] == 1:
                    weights = znp.reshape(weights, (-1,))
                else:
                    raise ShapeIncompatibleError("Weights have to be 1-Dim objects.")
        self._weights = weights
        self._update_hash()
        return weights

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
        use_hash: bool = None,
    ):
        """Create a ``Data`` from a pandas DataFrame. If ``obs`` is ``None``, columns are used as obs.

        Args:
            df: pandas DataFrame that contains the data. If ``obs`` is ``None``, columns are used as obs. Can be
                a superset of obs.
            obs: obs to use for the data. obs have to be the columns in the data frame.
                If ``None``, columns are used as obs.
            weights: Weights of the data. Has to be 1-D and match the shape
                of the data (nevents) or a string that is a column in the dataframe. By default, looks for a column ``""``, i.e.
                 an empty string.
            name:
            dtype: dtype of the data
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.
        """
        weights_requested = weights is not None
        if weights is None:
            weights = ""
        if obs is None:
            obs = list(df.columns)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        obs = convert_to_space(obs)
        not_in_df = set(obs.obs) - set(df.columns)
        if not_in_df:
            raise ValueError(
                f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            )
        space = obs
        if isinstance(weights, str):  # it's in the df
            if weights not in df.columns:
                if weights_requested:
                    raise ValueError(
                        f"Weights {weights} is a string and not in dataframe with columns {df.columns}"
                    )
                weights = None
            else:
                obs = [o for o in space.obs if o != weights]
                weights = df[weights]
                space = space.with_obs(obs=obs)

        not_in_df = set(space.obs) - set(df.columns)
        if not_in_df:
            raise ValueError(
                f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            )

        array = df[list(space.obs)].values

        return Data.from_numpy(  # *not* class, if subclass, keep constructor
            obs=space,
            array=array,
            weights=weights,
            name=name,
            dtype=dtype,
            use_hash=use_hash,
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
        use_hash: bool = None,
        # deprecated
        branches: list[str] = None,
        branches_alias: dict = None,
    ) -> Data:
        """Create a ``Data`` from a ROOT file. Arguments are passed to ``uproot``.

        The arguments are passed to uproot directly.

        Args:
            path: Path to the root file.
            treepath: Name of the tree in the root file.
            obs: Observables of the data. This will also be the columns of the data if not *obs_alias* is given.
            weights: Weights of the data. Has to be 1-D and match the shape
                of the data (nevents). Can be a column of the ROOT file by using a string corresponding to a
                column.
            obs_alias: A mapping from the ``obs`` (as keys) to the actual ``branches`` (as values) in the root file.
                This allows to have different ``observable`` names, independent of the branch name in the file.
            name:
            root_dir_options:

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
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

        return Data(  # *not* class, if subclass, keep constructor
            dataset=dataset,
            obs=obs,
            weights=weights_np,
            name=name,
            dtype=dtype,
            use_hash=use_hash,
        )

    @classmethod
    def from_numpy(
        cls,
        obs: ztyping.ObsTypeInput,
        array: np.ndarray,
        weights: ztyping.WeightsInputType = None,
        name: str = None,
        dtype: tf.DType = None,
        use_hash=None,
    ):
        """Create ``Data`` from a ``np.array``.

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            array: Numpy array containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: Name of the data.
            dtype: dtype of the data.
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """

        if not isinstance(array, (np.ndarray)) and not (
            tf.is_tensor(array) and hasattr(array, "numpy")
        ):
            raise TypeError(
                f"`array` has to be a `np.ndarray`. Is currently {type(array)}"
            )
        if dtype is None:
            dtype = ztypes.float
        array = znp.asarray(array)
        tensor = tf.cast(array, dtype=dtype)
        return Data.from_tensor(  # *not* class, if subclass, keep constructor
            obs=obs,
            tensor=tensor,
            weights=weights,
            name=name,
            dtype=dtype,
            use_hash=use_hash,
        )

    @classmethod
    def from_tensor(
        cls,
        obs: ztyping.ObsTypeInput,
        tensor: tf.Tensor,
        weights: ztyping.WeightsInputType = None,
        name: str = None,
        dtype: tf.DType = None,
        use_hash=None,
    ) -> Data:
        """Create a ``Data`` from a ``tf.Tensor``. ``Value`` simply returns the tensor (in the right order).

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            tensor: Tensor containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: Name of the data.

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """
        if dtype is None:
            dtype = ztypes.float
        tensor = tf.cast(tensor, dtype=dtype)
        if len(tensor.shape) == 0:
            tensor = znp.expand_dims(tensor, -1)
        if len(tensor.shape) == 1:
            tensor = znp.expand_dims(tensor, -1)
        dataset = LightDataset.from_tensor(tensor)

        return Data(  # *not* class, if subclass, keep constructor
            dataset=dataset,
            obs=obs,
            name=name,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
        )

    def _update_hash(self):
        if not run.executing_eagerly() or not self._use_hash:
            self._hashint = None
        else:
            try:
                hashval = xxhash.xxh128(np.asarray(self.value()))
                if self.has_weights:
                    hashval.update(np.asarray(self.weights))
                self._hashint = hashval.intdigest()
            except (
                AttributeError
            ):  # if the dataset is not yet initialized; this is allowed
                self._hashint = None

    def with_obs(self, obs):
        """Create a new ``Data`` with a subset of the data using the *obs*.

        Args:
            obs: Observables to return. Has to be a subset of the original observables.

        Returns:
            ``zfit.Data``: A new ``Data`` object containing the subset of the data.
        """
        values = self.value(obs)
        return type(self).from_tensor(
            obs=self.space, tensor=values, weights=self.weights, name=self.name
        )

    def to_pandas(
        self, obs: ztyping.ObsTypeInput = None, weightsname: str | None = None
    ):
        """Create a ``pd.DataFrame`` from ``obs`` as columns and return it.

        Args:
            obs: The observables to use as columns. If ``None``, all observables are used.
            weightsname: The name of the weights column if the data has weights. If ``None``, defaults to ``""``, an empty string.

        Returns:
            ``pd.DataFrame``: A ``pd.DataFrame`` containing the data and the weights (if present).
        """
        values = self.value(obs=obs)
        if obs is None:
            obs = self.obs
        obs_str = list(convert_to_obs_str(obs))
        values = values.numpy()
        if self.has_weights:
            weights = self.weights.numpy()
            if weightsname is None:
                weightsname = ""
            values = np.concatenate((values, weights[:, None]), axis=1)
            obs_str = obs_str + [weightsname]
        df = pd.DataFrame(data=values, columns=obs_str)
        return df

    def unstack_x(self, obs: ztyping.ObsTypeInput = None, always_list=None):
        """Return the unstacked data: a list of tensors or a single Tensor.

        Args:
            obs: which observables to return

        Returns:
            List(tf.Tensor)
        """
        return z.unstack_x(self.value(obs=obs), always_list=always_list)

    def value(self, obs: ztyping.ObsTypeInput = None):
        """Return the data as a numpy-like object in ``obs`` order.

        Args:
            obs: Observables to return. If ``None``, all observables are returned. Can be a subset of the original
                observables. If a string is given, a 1-D array is returned with shape (nevents,). If a list of strings
                or a ``zfit.Space`` is given, a 2-D array is returned with shape (nevents, nobs).

        Returns:
        """
        out = znp.asarray(self._value_internal(obs=obs))
        if isinstance(obs, str):
            out = znp.squeeze(out, axis=-1)
        return out

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

        if obs:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError(
                    "The observable(s) {} are not contained in the dataset. "
                    "Only the following are: {}".format(
                        frozenset(obs) - frozenset(self.obs), self.obs
                    )
                )
            perm_indices = self.space.get_reorder_indices(obs=obs)
        if perm_indices:
            value = z.unstack_x(value, always_list=True)
            value = [value[i] for i in perm_indices]
            value = z.stack_x(value)

        return value

    # TODO(Mayou36): use Space to permute data?
    # TODO(Mayou36): raise error is not obs <= self.obs?
    @invalidate_graph
    def sort_by_axes(self, axes: ztyping.AxesTypeInput, allow_superset: bool = True):
        if not allow_superset and not frozenset(axes) <= frozenset(self.axes):
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
        if not allow_superset and not frozenset(obs) <= frozenset(self.obs):
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
        if frozenset(self.data_range.obs) != frozenset(data_range.obs):
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
        """Convert the inputs (using eventually ``obs``, ``axes``) to
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

    def __getitem__(self, item):
        try:
            value = getitem_obs(self, item)
        except Exception as error:
            raise RuntimeError(
                f"Failed to retrieve {item} from data {self}. This can be changed behavior (since zfit 0.11): data can"
                f" no longer be accessed numpy-like but instead the 'obs' can be used, i.e. strings or spaces. This"
                f" resembles more closely the behavior of a pandas DataFrame."
            ) from error
        return value


# TODO(serialization): add to serializer
class DataRepr(BaseRepr):
    _implementation = Data
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["Data"] = Field("Data", alias="type")

    data: np.ndarray
    space: Union[SpaceRepr, List[SpaceRepr]]
    name: Optional[str] = None
    weights: Optional[np.ndarray] = None

    @pydantic.root_validator(pre=True)
    def extract_data(cls, values):
        if cls.orm_mode(values):
            values = dict(values)
            values["data"] = values["value"]()
        return values

    @pydantic.validator("space", pre=True)
    def flatten_spaces(cls, v):
        if cls.orm_mode(v):
            v = [v.get_subspace(o) for o in v.obs]
        return v

    @pydantic.validator("data", pre=True)
    def convert_data(cls, v):
        v = np.asarray(v)
        return v

    @pydantic.validator("weights", pre=True)
    def convert_weights(cls, v):
        if v is not None:
            v = np.asarray(v)
        return v

    @to_orm_init
    def _to_orm(self, init):
        dataset = LightDataset(znp.asarray(init.pop("data")))
        init["dataset"] = dataset
        init["obs"] = init.pop("space")

        spaces = init["obs"]
        space = spaces[0]
        for sp in spaces[1:]:
            space *= sp
        init["obs"] = space
        out = super()._to_orm(init)
        return out


def getitem_obs(self, item):
    if not isinstance(item, str):
        item = convert_to_obs_str(item)
    return self.value(item)


class SampleData(Data):
    _cache_counting = 0

    def __init__(
        self,
        dataset: tf.data.Dataset | LightDataset,
        obs: ztyping.ObsTypeInput = None,
        weights=None,
        name: str = None,
        dtype: tf.DType = ztypes.float,
        use_hash: bool = None,
    ):
        super().__init__(
            dataset,
            obs,
            name=name,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
        )

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(  # TODO(deprecate and remove? use normal data?
        cls,
        sample: tf.Tensor,
        obs: ztyping.ObsTypeInput,
        name: str = None,
        weights=None,
        use_hash: bool = None,
    ):
        return Data.from_tensor(
            tensor=sample, obs=obs, name=name, weights=weights, use_hash=use_hash
        )


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
        use_hash: bool = None,
    ):
        super().__init__(
            dataset=dataset,
            obs=obs,
            name=name,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
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
        name: str = None,
        weights=None,
        dtype=None,
        use_hash: bool = None,
    ):
        obs = convert_to_space(obs)

        if fixed_params is None:
            fixed_params = []
        if dtype is None:
            dtype = ztypes.float
        sample_holder = tf.Variable(
            initial_value=sample_func(),
            dtype=dtype,
            trainable=False,
            shape=(None, obs.n_obs),
            name=f"sample_data_holder_{cls.get_cache_counting()}",
        )
        dataset = LightDataset.from_tensor(sample_holder)

        return cls(
            dataset=dataset,
            sample_holder=sample_holder,
            sample_func=sample_func,
            fixed_params=fixed_params,
            n=n,
            obs=obs,
            name=name,
            weights=weights,
            use_hash=use_hash,
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
        self._update_hash()

    def __str__(self) -> str:
        return f"<Sampler: {self.name} obs={self.obs}>"


# register_tensor_conversion(Data, name="Data", overload_operators=True)


class LightDataset:
    def __init__(self, tensor):
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            tensor = z.convert_to_tensor(tensor)
        self.tensor = tensor

    def batch(self, _):  # ad-hoc just empty, mimicking tf.data.Dataset interface
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
        sample2 = z.random.shuffle(sample2)
    sample1 = sample1.value(obs=obs)
    tensor = sample1 + sample2
    if any(s.weights is not None for s in samples):
        raise WorkInProgressError("Cannot combine weights currently")
    weights = None

    return SampleData.from_sample(sample=tensor, obs=obs, weights=weights)
