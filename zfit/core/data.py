#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

import pydantic
import xxhash
from pydantic import Field
from tensorflow.python.types.core import TensorLike
from tensorflow.python.util.deprecation import deprecated

from ..serialization import SpaceRepr
from ..serialization.serializer import BaseRepr, to_orm_init
from .parameter import set_values
from .serialmixin import SerializableMixin, ZfitSerializable

if TYPE_CHECKING:
    import zfit

from collections import OrderedDict
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot

import zfit
import zfit.z.numpy as znp

from .. import z
from ..settings import run, ztypes
from ..util import ztyping
from ..util.cache import GraphCachable, invalidate_graph
from ..util.container import convert_to_container
from ..util.exception import (
    BreakingAPIChangeError,
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


def convert_to_data(data, obs=None):
    if isinstance(data, ZfitUnbinnedData):
        return data
    elif isinstance(data, LightDataset):
        return Data(dataset=data, obs=obs)
    elif isinstance(data, pd.DataFrame):
        return Data.from_pandas(df=data, obs=obs)
    elif isinstance(data, Mapping):
        return Data.from_mapping(mapping=data, obs=obs)

    if obs is None:
        msg = f"If data is not a Data-like object, obs has to be specified. Data is {data} and obs is {obs}."
        raise ValueError(msg)
    if isinstance(data, (int, float)):
        data = znp.array([data])
    if isinstance(data, Iterable):
        data = znp.array(data)
    if isinstance(data, np.ndarray):
        return Data.from_numpy(obs=obs, array=data)
    if isinstance(data, (tf.Tensor, znp.ndarray, tf.Variable)):
        return Data.from_tensor(obs=obs, tensor=data)

    msg = f"Cannot convert {data} to a Data object."
    raise TypeError(msg)


class Data(
    ZfitUnbinnedData,
    BaseDimensional,
    BaseObject,
    GraphCachable,
    SerializableMixin,
    ZfitSerializable,
):
    USE_HASH = False
    BATCH_SIZE = 1000000  # 1 mio

    def __init__(
        self,
        dataset: LightDataset,
        obs: ztyping.ObsTypeInput = None,
        name: str | None = None,
        weights: TensorLike = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create a data holder from a ``dataset`` used to feed into ``models``.

        Args:
            label:
            dataset: A dataset storing the actual values
            obs: Observables where the data is defined in
            name: Name of the ``Data``
            weights: Weights of the data
            dtype: |dtype_arg_descr|
            use_hash: Whether to use a hash for caching
        """
        if use_hash is None:
            use_hash = self.USE_HASH
        self._use_hash = use_hash

        if dtype is None:
            dtype = ztypes.float
        super().__init__(name=name)

        self._permutation_indices_data = None
        self._next_batch = None
        self._dtype = dtype
        self._nevents = None
        self._weights = None
        self._label = label if label is not None else name

        self._data_range = None
        self._set_space(obs)
        self._original_space = self.space
        self._data_range = self.space

        if not guarantee_limits:
            tensormap = dataset._tensormap if (ismap := dataset._tensor is None) else dataset.value()
            value, weights = check_cut_data_weights(limits=self.space, data=tensormap, weights=weights)
            if ismap:
                dataset = LightDataset(tensormap=value, ndims=self.space.n_obs)
            else:
                dataset = LightDataset.from_tensor(value, ndims=self.space.n_obs)

        self._name = name
        self._hashint = None

        self.dataset = dataset
        self._set_weights(weights=weights)
        # check that dimensions are compatible

        self._update_hash()

    @property
    def _using_hash(self):
        return self._use_hash and run.hashing_data()

    @property
    def label(self):
        return self._label

    @property
    def nevents(self):
        nevents = self._nevents
        if nevents is None:
            nevents = self._get_nevents()
        return nevents

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

    @property
    @deprecated(None, "Use `space` instead.")
    def data_range(self):
        data_range = self._data_range
        if data_range is None:
            data_range = self.space
        return data_range

    @invalidate_graph
    @deprecated(
        None,
        "Do not change the range, preferably use pandas or similar, or use `with_obs` instead.",
    )
    def set_data_range(self, data_range):
        data_range = self._check_input_data_range(data_range=data_range)

        def setter(value):
            self._data_range = value
            self._update_hash()

        def getter():
            return self._data_range

        return TemporarilySet(value=data_range, setter=setter, getter=getter)

    def _copy(self, deep, name, overwrite_params):
        """Copy the object, overwrite params with overwrite_params."""
        del deep  # no meaning...
        newpar = {
            "obs": self.space,
            "weights": self.weights,
            "name": name,
            "dataset": self.dataset,
            "label": self.label,
            "dtype": self.dtype,
            "use_hash": self._use_hash,
            **overwrite_params,
        }
        newpar["guarantee_limits"] = (
            "obs" not in overwrite_params
            and "dataset" not in overwrite_params
            and overwrite_params.get("guarantee_limits") is not False
        )
        if "tensor" in overwrite_params:
            msg = "do not give tensor in copy, instead give a LightDataset."
            raise BreakingAPIChangeError(msg)

        return Data(**newpar)

    @property
    def weights(self):
        """Get the weights of the data."""
        return self._weights

    def with_weights(self, weights: ztyping.WeightsInputType):
        """Create a new ``Data`` with a different set of weights.

        Args:
            weights: The new weights to use. Has to be 1-D and match the shape of the data (nevents).

        Returns:
            ``zfit.Data``: A new ``Data`` object containing the new weights.
        """
        run.assert_executing_eagerly()

        if weights is not None:
            weights = znp.asarray(weights)
            if weights.shape.ndims != 1:
                msg = "Weights have to be 1-Dim objects."
                raise ValueError(msg)
            if weights.shape[0] != self.nevents:
                msg = "Weights have to have the same length as the data."
                raise ValueError(msg)
        return self.copy(weights=weights, guarantee_limits=True)

    @deprecated(None, "Use `with_weights` instead.")
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
        if weights is not None and not isinstance(
            weights, tf.Variable
        ):  # tf.Variable means it's changeable and we trust it
            weights = znp.asarray(weights, dtype=self.dtype)
            weights = znp.atleast_1d(weights)
            if weights.shape.ndims > 1:
                msg = f"Weights have to be 1-Dim objects, is currently {weights} with shape {weights.shape}."
                raise ShapeIncompatibleError(msg)
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
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
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
            name: |@doc:data.init.name| Name of the data.
               Maybe has implications on the serialization and deserialization of the data.
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|
        """
        weights_requested = weights is not None
        if dtype is None:
            dtype = ztypes.float
        if weights is None:
            weights = ""
        if obs is None:
            obs = list(df.columns)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        obs = convert_to_space(obs)
        if not_in_df := set(obs.obs) - set(df.columns):
            msg = f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            raise ValueError(msg)
        space = obs
        if isinstance(weights, str):  # it's in the df
            if weights not in df.columns:
                if weights_requested:
                    msg = f"Weights {weights} is a string and not in dataframe with columns {df.columns}"
                    raise ValueError(msg)
                weights = None
            else:
                obs = [o for o in space.obs if o != weights]
                weights = df[weights].to_numpy(dtype=np.float64)
                space = space.with_obs(obs=obs)

        not_in_df = set(space.obs) - set(df.columns)
        if not_in_df:
            msg = f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            raise ValueError(msg)

        mapping = df[list(space.obs)].to_dict(orient="series")  # pandas indexes with lists, not tuples
        return Data.from_mapping(
            mapping=mapping,
            obs=space,
            weights=weights,
            name=name,
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, np.ndarray],
        obs: ztyping.ObsTypeInput = None,
        *,
        weights: TensorLike | None = None,
        label: str | None = None,
        name: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ) -> Data:
        """Create a ``Data`` from a mapping of observables to arrays.

        Args:
            mapping: A mapping from the observables to the data.
            obs: Observables of the data. They will be matched to the data in the same order.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: |@doc:data.init.name| Name of the data.
               Maybe has implications on the serialization and deserialization of the data.
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """
        if obs is None:
            obs = tuple(mapping.keys())
        obs = convert_to_space(obs)
        if missing_obs := set(obs.obs) - set(mapping.keys()):
            msg = f"Not all observables (missing: {missing_obs}) requested ({obs}) are in the mapping: {mapping}."
            raise ValueError(msg)
        tensormap = {i: znp.asarray(mapping[obs], dtype=dtype) for i, obs in enumerate(obs.obs)}
        weights = znp.asarray(weights, dtype=dtype) if weights is not None else None
        dataset = LightDataset(tensormap=tensormap, ndims=obs.n_obs)
        return Data(  # *not* class, if subclass, keep constructor
            dataset=dataset,
            obs=obs,
            weights=weights,
            name=name,
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_root(
        cls,
        path: str,
        treepath: str,
        obs: ZfitSpace = None,
        *,
        weights: ztyping.WeightsStrInputType = None,
        obs_alias: Mapping[str, str] | None = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        root_dir_options=None,
        use_hash: bool | None = None,
        # deprecated
        branches: list[str] | None = None,
        branches_alias: dict | None = None,
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
            name: |@doc:data.init.name| Name of the data.
               Maybe has implications on the serialization and deserialization of the data.
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            root_dir_options: Options passed to uproot.
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """
        # begin deprecated legacy arguments
        if branches:
            msg = "Use `obs` instead of `branches`."
            raise BreakingAPIChangeError(msg)
        if branches_alias is not None:
            msg = "Use `obs_alias` instead of `branches_alias`."
            raise BreakingAPIChangeError(msg)
        # end legacy
        if root_dir_options is None:
            root_dir_options = {}
        if obs_alias is None and obs is None:
            msg = "Either branches or branches_alias has to be specified."
            raise ValueError(msg)
        if obs_alias is None:
            obs_alias = {}
        if obs is None:
            obs = list(obs_alias.values())

        obs = convert_to_space(obs)

        branches = [obs_alias.get(branch, branch) for branch in obs.obs]

        weights_are_branch = isinstance(weights, str)

        def uproot_loader():
            with uproot.open(path, **root_dir_options)[treepath] as root_tree:
                branches_with_weights = [*branches, weights] if weights_are_branch else branches
                branches_with_weights = tuple(branches_with_weights)
                data = root_tree.arrays(expressions=branches_with_weights, library="pd")
            data_np = data[branches].to_numpy()
            weights_np = data[weights].to_numpy() if weights_are_branch else None
            return data_np, weights_np

        data, weights_np = uproot_loader()
        if not weights_are_branch:
            weights_np = weights
        dataset = LightDataset.from_tensor(data, ndims=obs.n_obs)

        return Data(
            dataset=dataset, obs=obs, name=name, weights=weights_np, dtype=dtype, use_hash=use_hash, label=label
        )

    @classmethod
    def from_numpy(
        cls,
        obs: ztyping.ObsTypeInput,
        array: np.ndarray,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        guarantee_limits: bool = False,
    ):
        """Create ``Data`` from a ``np.array``.

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            array: Numpy array containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: |@doc:data.init.name| Name of the data.
               Maybe has implications on the serialization and deserialization of the data.
               For a human-readable name or description, use the label. |@docend:data.init.name|
            dtype: dtype of the data.
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """

        if not isinstance(array, (np.ndarray)) and not (tf.is_tensor(array) and hasattr(array, "numpy")):
            msg = f"`array` has to be a `np.ndarray`. Is currently {type(array)}"
            raise TypeError(msg)
        if dtype is None:
            dtype = ztypes.float
        tensor = znp.asarray(array, dtype=dtype)
        return Data.from_tensor(  # *not* class, if subclass, keep constructor
            obs=obs,
            tensor=tensor,
            weights=weights,
            name=name,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_tensor(
        cls,
        obs: ztyping.ObsTypeInput,
        tensor: tf.Tensor,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        *,
        guarantee_limits: bool = False,
    ) -> Data:
        """Create a ``Data`` from a ``tf.Tensor``. ``Value`` simply returns the tensor (in the right order).

        Args:
            obs: Observables of the data. They will be matched to the data in the same order.
            tensor: Tensor containing the data.
            weights: Weights of the data. Has to be 1-D and match the shape of the data (nevents).
            name: |@doc:data.init.name| Name of the data.
               Maybe has implications on the serialization and deserialization of the data.
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: If ``True``, a hash of the data is created and is used to identify it in caching.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|

        Returns:
            ``zfit.Data``: A ``Data`` object containing the unbinned data.
        """
        if dtype is None:
            dtype = ztypes.float
        tensor = znp.asarray(tensor, dtype=dtype)
        tensor = znp.atleast_1d(tensor)
        if len(tensor.shape) == 1:
            tensor = znp.expand_dims(tensor, -1)
        space = convert_to_space(obs)
        dataset = LightDataset.from_tensor(tensor, ndims=space.n_obs)

        return Data(
            dataset=dataset,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    def _update_hash(self):
        if not run.executing_eagerly() or not self._use_hash:
            self._hashint = None
        else:
            hashval = self.dataset.calc_hash()
            if self.has_weights:
                hashval.update(np.asarray(self.weights))
            if hasattr(self, "_hashint"):
                self._hashint = hashval.intdigest() % (64**2)

            else:  # if the dataset is not yet initialized; this is allowed
                self._hashint = None

    def with_obs(self, obs, *, guarantee_limits: bool = False):
        """Create a new ``Data`` with a subset of the data using the *obs*.

        Args:
            obs: Observables to return. Has to be a subset of the original observables.
            guarantee_limits: If ``True``, the limits are guaranteed to be correct and will not be checked.


        Returns:
            ``zfit.Data``: A new ``Data`` object containing the subset of the data.
        """
        if not isinstance(obs, ZfitSpace):
            if not isinstance(obs, (list, tuple)):
                obs = [obs]
            if isinstance(obs[0], str):
                obs = self.space.with_obs(obs)
            elif isinstance(obs[0], int):
                obs = self.space.with_axes(obs)
            guarantee_limits = True
        elif obs == self.space.with_obs(obs):
            guarantee_limits = True
        indices = self._get_permutation_indices(obs=obs)
        dataset = self.dataset.with_indices(indices)
        weights = self.weights
        return self.copy(obs=obs, dataset=dataset, weights=weights, guarantee_limits=guarantee_limits)

    def to_pandas(self, obs: ztyping.ObsTypeInput = None, weightsname: str | None = None):
        """Create a ``pd.DataFrame`` from ``obs`` as columns and return it.

        Args:
            obs: The observables to use as columns. If ``None``, all observables are used.
            weightsname: The name of the weights column if the data has weights. If ``None``, defaults to ``""``, an empty string.

        Returns:
            ``pd.DataFrame``: A ``pd.DataFrame`` containing the data and the weights (if present).
        """
        if obs is None:
            obs = self.obs
        obs_str = list(convert_to_obs_str(obs))
        data = {ob: self.value(obs=ob) for ob in obs_str}
        if self.has_weights:
            weights = self.weights.numpy()
            if weightsname is None:
                weightsname = ""
            data.update({weightsname: weights})
        return pd.DataFrame.from_dict(data)

    def unstack_x(self, obs: ztyping.ObsTypeInput = None, always_list=None):
        """Return the unstacked data: a list of tensors or a single Tensor.

        Args:
            obs: Observables to return. If ``None``, all observables are returned. Can be a subset of the original
            always_list: If ``True``, always return a list, even if only one observable is requested.

        Returns:
            List(tf.Tensor)
        """
        if always_list is None:
            always_list = False
        nolist = (not always_list) and isinstance(obs, str)
        if obs is None:
            obs_str = self.obs
            if len(obs_str) == 1:
                nolist = True  # legacy behavior
        else:
            obs_str = convert_to_obs_str(obs)

            if missingobs := set(obs_str) - set(self.obs):
                msg = f"Observables {missingobs} not in data. Available observables: {self.obs}"
                raise ObsIncompatibleError(msg)
        values = [self.value(obs=ob) for ob in obs_str]
        if nolist:
            return values[0]
        return values

    def value(self, obs: ztyping.ObsTypeInput = None, axis: int | None = None):
        """Return the data as a numpy-like object in ``obs`` order.

        Args:
            obs: Observables to return. If ``None``, all observables are returned. Can be a subset of the original
                observables. If a string is given, a 1-D array is returned with shape (nevents,). If a list of strings
                or a ``zfit.Space`` is given, a 2-D array is returned with shape (nevents, nobs).
            axis: If given, the axis to return instead of the full data. If ``obs`` is a string, this has to be ``None``.

        Returns:
        """
        if axis is not None:
            if obs is not None:
                msg = "Cannot specify both `obs` and `axis`."
                raise ValueError(msg)
            indices = convert_to_container(axis, container=tuple)
            if not all(isinstance(ax, int) for ax in indices):
                msg = "All axes have to be integers."
                raise ValueError(msg)
            if not set(indices).issubset(set(self.space.axes)):
                msg = "All axes have to be in the space."
                raise ValueError(msg)
        else:
            indices = self.space.with_obs(obs=obs).axes
        out = self.dataset.value(indices)
        if isinstance(obs, str) or axis is not None:
            out = znp.squeeze(out, axis=-1)
        return out

    def numpy(self):
        return self.to_numpy()

    def to_numpy(self):
        """Return the data as a numpy array.

        Pandas DataFrame equivalent method
        Returns:
            np.ndarray: The data as a numpy array.
        """
        return self.value().numpy()

    def _value_internal(self, obs: ztyping.ObsTypeInput = None):
        if obs is not None:
            obs = convert_to_obs_str(obs)
        perm_indices = self._get_permutation_indices(obs)
        return self.dataset.value(perm_indices)

    def _get_permutation_indices(self, obs):
        obs = convert_to_obs_str(obs)
        perm_indices = self.space.axes  # if self.space.axes != no_change_indices else False
        if obs:
            if not frozenset(obs) <= frozenset(self.obs):
                msg = (
                    f"The observable(s) {frozenset(obs) - frozenset(self.obs)} are not contained in the dataset. "
                    f"Only the following are: {self.obs}"
                )
                raise ValueError(msg)
            perm_indices = self.space.get_reorder_indices(obs=obs)

        return perm_indices

    @invalidate_graph
    @deprecated(None, "Use `with_obs` instead.")
    def sort_by_axes(self, axes: ztyping.AxesTypeInput, allow_superset: bool = True):
        if not allow_superset and not frozenset(axes) <= frozenset(self.axes):
            msg = (
                f"The observable(s) {frozenset(axes) - frozenset(self.axes)} are not contained in the dataset. "
                f"Only the following are: {self.axes}"
            )
            raise ValueError(msg)
        space = self.space.with_axes(axes=axes, allow_subset=True)

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    @invalidate_graph
    @deprecated(None, "Use `with_obs` instead.")
    def sort_by_obs(self, obs: ztyping.ObsTypeInput, allow_superset: bool = False):
        if not allow_superset and not frozenset(obs) <= frozenset(self.obs):
            msg = (
                f"The observable(s) {frozenset(obs) - frozenset(self.obs)} are not contained in the dataset. "
                f"Only the following are: {self.obs}"
            )
            raise ValueError(msg)

        space = self.space.with_obs(obs=obs, allow_subset=True, allow_superset=allow_superset)

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    def _check_input_data_range(self, data_range):
        data_range = self._convert_sort_space(limits=data_range)
        if frozenset(self.data_range.obs) != frozenset(data_range.obs):
            msg = (
                f"Data range has to cover the full observable space {self.data_range.obs}, not "
                f"only {data_range.obs}"
            )
            raise ObsIncompatibleError(msg)
        return data_range

    def _convert_sort_space(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        limits: ztyping.LimitsTypeInput = None,
    ) -> Space | None:
        """Convert the inputs (using eventually ``obs``, ``axes``) to :py:class:`~zfit.Space` and sort them according to
        own `obs`.

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
        return self.dataset.nevents

    def __str__(self) -> str:
        return f"<zfit.Data: {self.label} obs={self.obs}>"

    def to_binned(self, space):
        from zfit._data.binneddatav1 import BinnedData

        return BinnedData.from_unbinned(space=space, data=self)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.value(axis=item)
        try:
            value = getitem_obs(self, item)
        except Exception as errorobs:
            msg = (
                f"Failed to retrieve {item} from data {self}. This can be changed behavior (since zfit 0.11): data can"
                f" no longer be accessed numpy-like but instead the 'obs' can be used, i.e. strings or spaces. This"
                f" resembles more closely the behavior of a pandas DataFrame."
            )
            raise RuntimeError(msg) from errorobs
        return value


class DataRepr(BaseRepr):
    _implementation = Data
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["Data"] = Field("Data", alias="type")

    data: np.ndarray
    space: Union[SpaceRepr, list[SpaceRepr]]
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
        return np.asarray(v)

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
        return super()._to_orm(init)


def getitem_obs(self, item):
    if not isinstance(item, str):
        item = convert_to_obs_str(item)
    return self.value(item)


def check_cut_datamap_weights(limits, data, weights, guarantee_limits):
    inside = None
    datanew = {}
    for ax in limits.axes:
        limit = limits.with_axes(ax)
        arr = data[ax]
        arr = znp.atleast_1d(arr)
        if not guarantee_limits and limit.has_limits:
            inside = limit.inside(arr) if inside is None else inside & limit.inside(arr)
        datanew[ax] = arr

    if inside is not None and not (run.executing_eagerly() and np.all(inside)):
        for ax, arr in datanew.items():
            datanew[ax] = arr[inside]
        if weights is not None:
            weights = weights[inside]
    return datanew, weights


def check_cut_data_weights(
    limits: ZfitSpace,
    data: TensorLike | Mapping[str, TensorLike],
    weights: TensorLike | None = None,
    guarantee_limits: bool = False,
):
    """Check and cut the data and weights according to the limits.

    Args:
        limits: Limits to cut the data to.
        data: Data to cut.
        weights: Weights to cut.
        guarantee_limits: If True, the limits are guaranteed to be correct and the data is not checked.

    Returns:
    """
    if weights is not None:
        weights = znp.atleast_1d(weights)
        if weights.shape.ndims != 1:
            msg = f"Weights have to be 1-D, not {weights.shape}."
            raise ValueError(msg)
        datashape = next(iter(data.values())).shape[0] if isinstance(data, Mapping) else data.shape[0]
        if run.executing_eagerly() and weights.shape[0] != datashape:
            msg = f"Weights have to have the same length as the data, not {weights.shape[0]} != {datashape}."
            raise ValueError(msg)

    if isinstance(data, Mapping):
        return check_cut_datamap_weights(limits=limits, data=data, weights=weights, guarantee_limits=guarantee_limits)
    else:
        data = znp.atleast_1d(data)
        if len(data.shape) == 1 and limits.n_obs == 1:
            data = data[:, None]
        if data.shape.ndims != 2:
            msg = f"Data has to be 2-D, i.e. (nevents, nobs)., not {data.shape}, with data={data}."
            raise ValueError(msg)

    if limits.has_limits and not guarantee_limits:
        inside = limits.inside(data)
        data = data[inside]
        if weights is not None:
            weights = weights[inside]
    return data, weights


class SamplerData(Data):
    _cache_counting = 0

    def __init__(
        self,
        dataset: LightDataset,
        sample_and_weights_func: Callable,
        sample_holder: tf.Variable,
        n: ztyping.NumericalScalarType | Callable,
        weights=None,
        weights_holder: tf.Variable | None = None,
        fixed_params: dict[zfit.Parameter, ztyping.NumericalScalarType] | None = None,
        obs: ztyping.ObsTypeInput = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = ztypes.float,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )
        if fixed_params is None:
            fixed_params = OrderedDict()
        if isinstance(fixed_params, (list, tuple)):
            fixed_params = OrderedDict((param, param.numpy()) for param in fixed_params)  # TODO: numpy -> read_value?

        self._initial_resampled = False

        self.fixed_params = fixed_params
        self._sample_holder = sample_holder
        self._weights_holder = weights_holder
        self._weights = self._weights_holder
        self._sample_and_weights_func = sample_and_weights_func
        if isinstance(n, tf.Variable):
            msg = "Using a tf.Variable as `n` is not supported anymore. Use a numerical value or a callable instead."
            raise BreakingAPIChangeError(msg)
        self.n = n
        self._n_holder = n
        self._hashint_holder = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.update_data(dataset.value(), weights=weights)  # to be used for precompilations etc
        self._sampler_guarantee_limits = guarantee_limits

    @property
    def n_samples(self):
        return self._n_holder

    @property
    def _approx_nevents(self):
        nevents = super()._approx_nevents
        if nevents is None:
            nevents = self.n
        return nevents

    def _update_hash(self):
        if not run.executing_eagerly() or not self._using_hash:
            self._hashint = None
            return
        super()._update_hash()
        if hasattr(self, "_hashint_holder"):
            self._hashint_holder.assign(self._hashint % (64**2))

    def _value_internal(self, obs: ztyping.ObsTypeInput = None):
        if hasattr(self, "_initial_resampled") and not self._initial_resampled:  # if not initialized, we can't sample
            msg = (
                "No data generated yet. Use `resample()` to generate samples or directly use `model.sample()`"
                "for single-time sampling."
            )
            raise RuntimeError(msg)
        return super()._value_internal(obs=obs)

    @property
    def hashint(self) -> int | None:
        if run.executing_eagerly():
            return (
                self._hashint
            )  # since the variable can be changed but this may stays static... and using 128 bits we can't have
        else:
            return self._hashint_holder.value()
        # a tf.Variable that keeps the int

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    @deprecated(None, "Use `from_sampler` instead (with an 'r' at the end).")
    def from_sample(
        cls,
        sample_func: Callable,
        n: ztyping.NumericalScalarType,
        obs: ztyping.ObsTypeInput,
        fixed_params=None,
        name: str | None = None,
        weights=None,
        dtype=None,
        use_hash: bool | None = None,
    ):
        return cls.from_sampler(
            sample_func=sample_func,
            n=n,
            obs=obs,
            fixed_params=fixed_params,
            name=name,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
        )

    @classmethod
    def from_sampler(
        cls,
        *,
        sample_func: Optional[Callable] = None,
        sample_and_weights_func: Optional[Callable] = None,
        n: ztyping.NumericalScalarType,
        obs: ztyping.ObsTypeInput,
        fixed_params=None,
        name: str | None = None,
        label: str | None = None,
        dtype=None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create a `SamplerData` from a sampler function.

        This is a more flexible way to create a `SamplerData`. Instead of providing a fixed sample, a sampler function
        is provided that will be called to sample the data. If the data is used in the loss, the sampler function will
        updated the value in the compiled version.

        .. note::

            If any method of the `SamplerData` is used to create a new data object, such as `with_obs`, the resulting
            data will be a `Data` object and not a `SamplerData` object; the data will be fixed and not resampled.

        Args:
            sample_func: A callable that takes as argument `n` and returns a sample of the data. The sample has to have the same number of
                observables as the `obs` of the `SamplerData`. If `None`, `sample_and_weights_func` has to be given.
            sample_and_weights_func: A callable that takes as argument `n` and returns a tuple of the sample and the weights of the data.
                The sample has to have the same number of observables as the `obs` of the `SamplerData`. If `None`, `sample_func` has to be given.

            n: The number of samples to produce initially. This is used to have a first sample that can be used for compilation.
            obs: The observables of the data.
            fixed_params: A mapping from `Parameter` to a fixed value. If a fixed value is given, the parameter will
                not be sampled but instead use the fixed value.
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: The dtype of the data.
            use_hash: If `True`, a hash of the data is created and is used to identify it in caching.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|
        """
        if sample_func is None and sample_and_weights_func is None:
            msg = "Either `sample_func` or `sample_and_weights_func` has to be given."
            raise ValueError(msg)
        if sample_func is not None and sample_and_weights_func is not None:
            msg = "Only one of `sample_func` or `sample_and_weights_func` can be given."
            raise ValueError(msg)
        if sample_func is not None:
            if not callable(sample_func):
                msg = (
                    "sample_func has to be a callable. If you want to use a fixed sample, use `sample_func=lambda x=sample: x`, "
                    "this will use the sample as a fixed sample when using `resample`."
                )
                raise TypeError(msg)

            def sample_and_weights_func(n):
                return sample_func(n), None
        elif not callable(sample_and_weights_func):
            msg = "sample_and_weights_func has to be a callable."
            raise TypeError(msg)

        obs = convert_to_space(obs)

        if fixed_params is None:
            fixed_params = []
        if dtype is None:
            dtype = ztypes.float

        init_val, init_weights = sample_and_weights_func(n)

        init_val, init_weights = check_cut_data_weights(
            limits=obs, data=init_val, weights=init_weights, guarantee_limits=guarantee_limits
        )
        sample_holder = tf.Variable(
            initial_value=init_val,
            dtype=dtype,
            trainable=False,
            shape=(None, obs.n_obs),
            name=f"sample_data_holder_{cls.get_cache_counting()}",
        )
        dataset = LightDataset.from_tensor(sample_holder)

        weights = init_weights
        weights_holder = None
        if weights is not None:
            weights_holder = tf.Variable(
                initial_value=weights,
                dtype=dtype,
                trainable=False,
                shape=(None,),
                name=f"weights_data_holder_{cls.get_cache_counting()}",
            )

        return cls(
            dataset=dataset,
            sample_holder=sample_holder,
            weights_holder=weights_holder,
            sample_and_weights_func=sample_and_weights_func,
            fixed_params=fixed_params,
            n=n,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            use_hash=use_hash,
            guarantee_limits=True,
            dtype=dtype,
        )

    def update_data(self, sample: TensorLike, weights: TensorLike | None = None, guarantee_limits: bool = False):
        """Load a new sample into the dataset, presumably similar to the previous one.

        Args:
            sample: The new sample to load. Has to have the same number of observables as the `obs` of the `SamplerData` but
                can have a different number of events.
            weights: The weights of the new sample. If `None`, the weights are not changed. If the `SamplerData` was
                initialized with weights, this has to be given. If the `SamplerData` was initialized without weights,
                this cannot be given.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits. |@docend:data.init.guarantee_limits|
        """
        sample = znp.asarray(sample, dtype=self.dtype)

        if sample.shape.rank == 1:
            sample = sample[:, None]
        elif sample.shape.rank != 2:
            msg = f"Sample has to have 1 or 2 dimensions, got {sample.shape.rank}."
            raise ValueError(msg)
        if sample.shape[-1] != self.space.n_obs:
            msg = (
                f"Sample has to have the same number of observables as the `obs` of the `SamplerData`. "
                f"Got {sample.shape[-1]} observables, expected {self.space.n_obs}."
            )
            raise ValueError(msg)
        if not guarantee_limits:
            sample, weights = check_cut_data_weights(limits=self.space, data=sample, weights=weights)
        self._sample_holder.assign(sample, read_value=False)
        if weights is not None:
            if self._weights_holder is None:
                msg = "Cannot set weights if no weights were given at initialization."
                raise ValueError(msg)
            weights = znp.asarray(weights, dtype=ztypes.float)
            self._weights_holder.assign(weights, read_value=False)
        elif self._weights_holder is not None:
            msg = "No weights given but weights_holder was initialized."
            raise ValueError(msg)

        self._n_holder = tf.shape(sample)[0]
        self._initial_resampled = True
        self._update_hash()

    def resample(self, param_values: Mapping | None = None, n: int | tf.Tensor = None):
        """Update the sample by newly sampling. This affects any object that used this data already internally.

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

        with set_values(list(temp_param_values.keys()), list(temp_param_values.values())):
            new_sample, new_weight = self._sample_and_weights_func(n)
            new_sample.set_shape((n, self.space.n_obs))
            if new_weight is not None:
                new_weight.set_shape((n,))
            self.update_data(sample=new_sample, weights=new_weight, guarantee_limits=self._sampler_guarantee_limits)

    def __str__(self) -> str:
        return f"<SamplerData: {self.label} obs={self.obs} size={int(self.nevents)} weighted={self.has_weights}>"

    @classmethod
    def get_repr(cls):  # acts as data object once serialized
        return DataRepr


# register_tensor_conversion(Data, name="Data", overload_operators=True)


class LightDataset:
    def __init__(self, tensor=None, tensormap=None, ndims=None):
        """A light-weight dataset that can be used for sampling and is aware of the mapping of the tensor with axes.

        Args:
            tensor: The tensor that contains the data. Has to be 2-D.
            tensormap: A mapping from the axes of the tensor to the actual axes in the data. If `None`, the tensor is
                assumed to be the data.
            ndims: The number of dimensions of the data. If `None`, it is inferred from the tensor or the tensormap.
        """
        if tensor is None and isinstance(tensormap, Mapping):
            tensormap = tensormap.copy()
        elif tensormap is None:  # the actual preprocessing, otherwise we pass it through
            if not isinstance(tensor, tf.Variable):
                tensor = znp.asarray(tensor)

            if tensor.shape.rank != 2:
                msg = "Only 2D tensors are allowed."
                raise ValueError(msg)
            if ndims is not None and tensor.shape[1] != ndims:
                msg = f"Second dimension has to be {ndims} but is {tensor.shape[1]}"
                raise ValueError(msg)
            ndims = tensor.shape[1]
            tensormap = {i: i for i in range(ndims)}

        if ndims is None:
            ndims = len(tensormap)
        self._tensor = tensor
        self._tensormap = tensormap
        self._ndims = ndims
        self._nevents = None

    def batch(self, _):  # ad-hoc just empty, mimicking tf.data.Dataset interface
        return self

    @property
    def nevents(self):
        return (
            tf.shape(self._tensor)[0] if self._tensor is not None else tf.shape(next(iter(self._tensormap.values())))[0]
        )

    @property
    def ndims(self):
        if (ndims := self._ndims) is None:
            ndims = len(self._tensormap)
        return ndims

    def __iter__(self):
        yield self.value()

    @classmethod
    def from_tensor(cls, tensor, ndims=None):
        del ndims  # not used
        return cls(tensor=tensor, ndims=None)

    def with_indices(self, indices: int | tuple[int] | list[int]):
        """Return a new `LightDataset` with the indices reshuffled.

        Args:
            indices: The indices to reshuffle the data. Can be a single index, a list or a tuple of indices.
        """
        if isinstance(indices, int):
            indices = (indices,)
        if not isinstance(indices, (list, tuple)):
            msg = f"Indices have to be an int, list or tuple, not {indices}"
            raise TypeError(msg)

        tensor, tensormap = self._get_tensor_and_tensormap()

        newmap = {}
        for i, idx in enumerate(indices):  # these are either indices that we reshuffle or a mapping to the new array
            newmap[i] = tensormap[idx]
        return LightDataset(tensor=tensor, tensormap=newmap)

    def _get_tensor_and_tensormap(self, forcemap=False):
        """Get the tensor and the tensor map, if needed, convert the tensor to the tensormap.

        Args:
            forcemap: Force the conversion of the tensor to the tensormap.

        Returns:
        """
        tensormap = self._tensormap
        if (tensor := self._tensor) is not None:
            if isvar := isinstance(tensor, tf.Variable):
                tensor = znp.asarray(tensor.value())  # to make sure the variable changes won't be reflected
            if forcemap:
                tensormap = {i: tensor[:, tensormap[i]] for i in range(self.ndims)}
                tensor = None
                if not isvar:  # we don't want to destroy the variable
                    self._tensormap = tensormap
                    self._tensor = None
        # do NOT update self, it could be a variable that we don't want to touch
        return tensor, tensormap

    def value(self, index: int | tuple[int] | list[int] | None = None):
        """Return the data as a tensor or a subset of the data as a tensor.

        Args:
            index: The axes to return. If `None`, the full tensor is returned. If an integer, a single axis is returned.

        Returns:
        """
        forcemap = False
        trivial_index = tuple(range(self.ndims))
        if index is None:
            index = trivial_index
        else:  # convert tensor to tensormap, if needed
            if not isinstance(index, (int, tuple, list)):
                msg = f"Index has to be an integer or a tuple/list of integers, not {index}"
                raise TypeError(msg)
            forcemap = len(set(index)) < self.ndims  # we will need a subset

        tensor, tensormap = self._get_tensor_and_tensormap(forcemap=forcemap)
        if tensor is None:
            # tensormap is filled, we can now return the values, either a single one or a stacked tensor
            if isinstance(index, int):
                return tensormap[index]
            return znp.stack([tensormap[i] for i in index], axis=-1)
        else:
            if isint := isinstance(index, int):
                index = (index,)
            newindex = [tensormap[i] for i in index]
            if newindex != trivial_index:
                tensor = tf.gather(tensor, newindex, axis=-1)
            if isint:
                tensor = znp.squeeze(tensor, axis=-1)
            return tensor

    def calc_hash(self):
        """Calculate a hash of the data."""
        tensor, tensormap = self._get_tensor_and_tensormap(forcemap=False)
        hashval = xxhash.xxh128()
        for dim in range(self.ndims):
            index_or_array = tensormap[dim]
            if tensor is not None:
                index_or_array = tensor[:, index_or_array]
            hashval.update(index_or_array)

        return hashval

    def __hash__(self):
        return self.calc_hash().intdigest()

    def __eq__(self, other):
        if not isinstance(other, LightDataset):
            return False
        return self.calc_hash() == other.calc_hash()


def sum_samples(
    sample1: ZfitUnbinnedData,
    sample2: ZfitUnbinnedData,
    obs: ztyping.ObsTypeInput = None,
    weights: ztyping.WeightsInputType = None,
    shuffle: bool = False,
):
    """Add the events of two samples together.

    Args:
        sample1: The first sample to add.
        sample2: The second sample to add.
        obs: The observables of the data. The sum will be done in this order and on this subset of observables.
        weights:  The new weights, as the sum cannot be done with the weights. If `False`, the weights are dropped.
        shuffle: If `True`, the second sample will be shuffled before adding it to the first sample.

    Returns:
    """
    samples = [sample1, sample2]
    if obs is None:
        obs = sample1.obs
        obs = convert_to_space(obs)
        obs2 = sample2.obs
        obs2 = convert_to_space(obs2)
        if obs != obs2:
            msg = "Observables of both samples have to be the same _or_ the observables have to be given as `obs` and must not be `None`."
            raise ValueError(msg)

    sample2 = sample2.value(obs=obs)
    if shuffle:
        sample2 = z.random.shuffle(sample2)
    sample1 = sample1.value(obs=obs)
    tensor = sample1 + sample2
    if any(s.weights is not None for s in samples) and weights is not False:
        msg = "Cannot combine weights currently. Either specify `weights=False` to drop them or give the weights explicitly."
        raise WorkInProgressError(msg)
    if weights is False:
        weights = None

    return Data.from_tensor(tensor=tensor, obs=obs, weights=weights)


class Sampler(SamplerData):
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        msg = "The class `Sampler` has been renamed to `SamplerData`."
        raise BreakingAPIChangeError(msg)
