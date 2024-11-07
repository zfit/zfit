#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

import pydantic.v1 as pydantic
import xxhash
from pydantic.v1 import Field
from tensorflow.python.types.core import TensorLike
from tensorflow.python.util.deprecation import deprecated, deprecated_args

from ..exception import OutsideLimitsError
from ..serialization import SpaceRepr
from ..serialization.serializer import BaseRepr, to_orm_init
from .serialmixin import SerializableMixin, ZfitSerializable

if TYPE_CHECKING:
    import zfit

from collections import Counter
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
from .baseobject import BaseObject, convert_param_values
from .coordinates import convert_to_obs_str
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData, ZfitSpace, ZfitUnbinnedData
from .space import Space, convert_to_space


def convert_to_data(data, obs=None, *, check_limits=False):
    if isinstance(data, ZfitUnbinnedData):
        return data
    elif isinstance(data, LightDataset):
        return Data(data=data, obs=obs)

    if check_limits:
        if not isinstance(obs, ZfitSpace):
            msg = "If check_limits is True, obs has to be a ZfitSpace."
            raise ValueError(msg)
        data_nocut = convert_to_data(data, obs=obs.obs, check_limits=False)
        not_inside = ~obs.inside(data_nocut.value())
        if np.any(not_inside):
            msg = f"Data {data} is not inside the limits {obs}."
            raise OutsideLimitsError(msg)
    if isinstance(data, pd.DataFrame):
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


class DataMeta(type):
    def __call__(cls, data, obs=None, *args, **kwargs):
        """Construct an instance of a class whose metaclass is Meta."""
        assert isinstance(cls, DataMeta)
        if binned := (obs is not None and isinstance(obs, ZfitSpace) and obs.is_binned):
            binned_obs = obs
            obs = obs.with_binning(False)

        if isinstance(data, LightDataset):
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(data, obs=obs, **kwargs)
        elif isinstance(data, pd.DataFrame):
            obj = cls.from_pandas(data, obs=obs, **kwargs)
        elif isinstance(data, Mapping):
            obj = cls.from_mapping(data, obs=obs, **kwargs)
        elif tf.is_tensor(data):
            obj = cls.from_tensor(tensor=data, obs=obs, **kwargs)
        elif isinstance(data, np.ndarray):
            obj = cls.from_numpy(array=data, obs=obs, **kwargs)
        else:
            try:
                obj = cls.from_numpy(array=data, obs=obs, **kwargs)
            except Exception as error:
                msg = f"Cannot convert {data} to a Data object. Use an explicit constructor (`from_pandas`, `from_mapping`, `from_tensor`, `from_numpy` etc)."
                raise TypeError(msg) from error
        if binned:
            obj = obj.to_binned(binned_obs)

        return obj


class Data(
    ZfitUnbinnedData,
    BaseDimensional,
    BaseObject,
    GraphCachable,
    SerializableMixin,
    ZfitSerializable,
    metaclass=DataMeta,
):
    USE_HASH = False
    BATCH_SIZE = 1_000_000

    def __init__(
        self,
        data: LightDataset | pd.DataFrame | Mapping[str, np.ndarray] | tf.Tensor | np.ndarray | zfit.Data,
        *,
        obs: ztyping.ObsTypeInput = None,
        weights: TensorLike = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create data, a thin wrapper around an array-like structure that supports weights and limits.

        Instead of creating a `Data` object directly, the `from_*` constructors, such as `from_pandas`, `from_mapping`,
        `from_tensor`, and `from_numpy`, can be used for a more fine-grained control of some arguments and
        for more extensive documentation on the allowed arguments.

        The data is unbinned, i.e. it is a collection of events. The data can be weighted and is defined in a
        space, which is a set of observables, whose limits are enforced.

        Args:
            data: A dataset storing the actual values. A variety of data-types are possible, as long as they are
               array-like.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|

                Some data-types, such as `pd.DataFrame`, already have
                observables defined implicitly. If `obs` is `None`, the observables are inferred from the data.
                If the ``obs`` is binned, the unbinned data will be binned according to the binning of the ``obs``
                and a :py:class:`~zfit.data.BinnedData` will be returned.

            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits| For example, if the data is a `pd.DataFrame` and the limits
                of ``obs`` have already been enforced through a ``query`` on the DataFrame, the limits are guaranteed
                to be correct and the data will not be checked again.
                Possible speedup, should not have any effect on the result.
            dtype: |dtype_arg_descr|
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ShapeIncompatibleError: If the shape of the data is incompatible with the observables.
            ValueError: If the data is not a recognized type.
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
        self._label = label if label is not None else (name if name is not None else "Data")

        self._data_range = None
        self._set_space(obs)
        self._original_space = self.space
        self._data_range = self.space

        if not guarantee_limits:
            tensormap = data._tensormap if (ismap := data._tensor is None) else data.value()
            value, weights = check_cut_data_weights(limits=self.space, data=tensormap, weights=weights)
            if ismap:
                data = LightDataset(tensormap=value, ndims=self.space.n_obs)
            else:
                data = LightDataset.from_tensor(value, ndims=self.space.n_obs)

        self._name = name
        self._hashint = None

        self.dataset = data
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

    def _set_space(self, obs: Space, autofill=True):
        obs = convert_to_space(obs)
        self._check_n_obs(space=obs)
        if autofill:
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
            "data": self.dataset,
            "label": self.label,
            "dtype": self.dtype,
            "use_hash": self._use_hash,
            **overwrite_params,
        }
        newpar["guarantee_limits"] = (
            "obs" not in overwrite_params
            and "data" not in overwrite_params
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

    def with_weights(self, weights: ztyping.WeightsInputType) -> Data:
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
        *,
        weights: ztyping.WeightsInputType | str = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a pandas DataFrame. If ``obs`` is ``None``, columns are used as obs.

        Args:
            df: pandas DataFrame that contains the data. If ``obs`` is ``None``, columns are used as obs. Can be
                a superset of obs.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
                If ``None``, columns are used as obs.

            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits| For example, if the data is a `pd.DataFrame` and the limits
                of ``obs`` have already been enforced through a ``query`` on the DataFrame, the limits are guaranteed
                to be correct and the data will not be checked again.
                Possible speedup, should not have any effect on the result.
            dtype: |dtype_arg_descr|
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ValueError: If the observables are not in the dataframe.
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
        mapping: Mapping[str, ztyping.ArrayLike],
        obs: ztyping.ObsTypeInput = None,
        *,
        weights: TensorLike | None = None,
        label: str | None = None,
        name: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool | None = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a mapping of observables to arrays.

        Args:
            mapping: A mapping from the observables to the data, with the observables as keys and the data as values.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
                They will be matched to the data in the same order. Can be omitted, in which case the keys of the
                mapping are used as observables.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
                Can also be a string that is a column in the dataframe. By default, look for a column ``""``, i.e.,
                an empty string.
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ValueError: If the observables are not in the mapping.
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
            data=dataset,
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
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
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

        return Data(data=dataset, obs=obs, name=name, weights=weights_np, dtype=dtype, use_hash=use_hash, label=label)

    @classmethod
    def from_numpy(
        cls,
        obs: ztyping.ObsTypeInput,
        array: np.ndarray,
        *,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create ``Data`` from a ``np.array``.

        Args:
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
            array: Numpy array containing the data. Has to be of shape (nevents, nobs) or,
                if only one observable, (nevents) is also possible.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            TypeError: If the array is not a numpy array.
        """
        # todo: should we switch orders
        # # legacy, switch input arguments
        # if isinstance(obs, np.ndarray) or isinstance(array, (str, ZfitSpace)) or (isinstance(array, (list, tuple)) and isinstance(array[0], str)):
        #     warn_once("The order of the arguments `obs` and `array` has been swapped, array goes first (as any other `from_` constructor.", identifier="data_from_numpy")
        #     obs, array = array, obs
        # # legacy end
        if isinstance(array, (float, int)):
            array = np.array([array])
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
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_tensor(
        cls,
        obs: ztyping.ObsTypeInput,
        tensor: tf.Tensor,
        *,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a ``tf.Tensor``

        Args:
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
            tensor: Tensor containing the data. Has to be of shape (nevents, nobs) or, if only one observable,
                (nevents) is also possible.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            TypeError: If the tensor is not a tensorflow tensor.
        """
        # todo: should we switch orders
        # # legacy start
        # if isinstance(obs, (np.ndarray, tf.Tensor)) or tf.is_tensor(obs) or isinstance(tensor, (str, ZfitSpace)) or (isinstance(tensor, (list, tuple)) and isinstance(tensor[0], str)):
        #     warn_once("The order of the arguments `obs` and `array` has been swapped, array goes first (as any other `from_` constructor.", identifier="data_from_numpy")
        #     obs, tensor = tensor, obs
        # # legacy end
        if dtype is None:
            dtype = ztypes.float
        tensor = znp.asarray(tensor, dtype=dtype)
        tensor = znp.atleast_1d(tensor)
        if len(tensor.shape) == 1:
            tensor = znp.expand_dims(tensor, -1)
        space = convert_to_space(obs)
        dataset = LightDataset.from_tensor(tensor, ndims=space.n_obs)

        return Data(
            data=dataset,
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

    def with_obs(self, obs: ztyping.ObsTypeInput, *, guarantee_limits: bool = False) -> Data:
        """Create a new ``Data`` with a subset of the data using the *obs*.

        Args:
            obs: Observables to return. Has to be a subset of the original observables.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|
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
        if obs.is_binned:
            msg = "obs is binned, no implicit conversion to binned data allowed. Use `to_binned` instead."
            raise ValueError(msg)
        indices = self._get_permutation_indices(obs=obs)
        dataset = self.dataset.with_indices(indices)
        weights = self.weights
        return self.copy(obs=obs, data=dataset, weights=weights, guarantee_limits=guarantee_limits)

    def to_pandas(self, obs: ztyping.ObsTypeInput = None, weightsname: str | None = None) -> pd.DataFrame:
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
            weights = self.weights
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

    def value(self, obs: ztyping.ObsTypeInput = None, axis: int | None = None) -> tf.Tensor:
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

    def numpy(self) -> np.ndarray:
        return self.to_numpy()

    @property
    def shape(self):
        return self.dataset.nevents

    def to_numpy(self) -> np.ndarray:
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

    def sort_by_axes(self, *_, **__):
        msg = "Use `with_axes` instead."
        raise BreakingAPIChangeError(msg)

    def sort_by_obs(self, *_, **__):
        msg = "Use `with_obs` instead."
        raise BreakingAPIChangeError(msg)

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

    def to_binned(
        self,
        space: ztyping.SpaceType,
        *,
        name: str | None = None,
        label: str | None = None,
        use_hash: bool | None = None,
    ) -> ZfitBinnedData:
        """Bins the data using ``space`` and returns a ``BinnedData`` object.

        Args:
            space: The space to bin the data in.
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            ``zfit.BinnedData``: A new ``BinnedData`` object containing the binned data.
        """
        from zfit._data.binneddatav1 import BinnedData

        return BinnedData.from_unbinned(
            space=space,
            data=self,
            name=name or self.name,
            label=label or self.label,
            use_hash=use_hash or self._use_hash,
        )

    def __len__(self):
        return self.nevents

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

    def __str__(self) -> str:
        return f"zfit.Data: {self.label} obs={self.obs} array={self.value()}"

    def __repr__(self) -> str:
        nevents = self.nevents
        try:
            nevents = int(round(float(nevents), ndigits=2))
        except Exception:
            nevents = None
        return f"<zfit.Data: {self.label} obs={self.obs} shape={(nevents, self.n_obs)}>"


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
        init["data"] = dataset
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
        data: LightDataset,
        *,
        sample_and_weights_func: Callable,
        sample_holder: tf.Variable,
        n: ztyping.NumericalScalarType | Callable,
        weights=None,
        weights_holder: tf.Variable | None = None,
        params: dict[zfit.Parameter, ztyping.NumericalScalarType] | None = None,
        obs: ztyping.ObsTypeInput = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = ztypes.float,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create a `SamplerData` object.

        Use constructor `from_sampler` instead.
        """
        if use_hash is not None and not use_hash:
            msg = "use_hash is required for SamplerData."
            raise ValueError(msg)
        use_hash = True
        super().__init__(
            data=data,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )
        params = convert_param_values(params)

        self._initial_resampled = False

        self.params = params
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
        self.update_data(data.value(), weights=weights)  # to be used for precompilations etc
        self._sampler_guarantee_limits = guarantee_limits

    # legacy
    @property
    @deprecated(None, "Use `params` instead.")
    def fixed_params(self):
        return self.params

    # legacy end
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
    @deprecated_args(None, "Use `params` instead.", "fixed_params")
    def from_sampler(
        cls,
        *,
        sample_func: Optional[Callable] = None,
        sample_and_weights_func: Optional[Callable] = None,
        n: ztyping.NumericalScalarType,
        obs: ztyping.ObsTypeInput,
        params: ztyping.ParamValuesMap = None,
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
            obs: Observables of the data. If the space has limits, the data will be cut to the limits.
            params: A mapping from `Parameter` or a string to a numerical value. This is used as the default values for the
                parameters in the `sample_func` or `sample_and_weights_func` and needs to fully specify the parameters.
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: The dtype of the data.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|
        """
        # legacy start
        if fixed_params is not None:
            msg = "Use `params` instead of `fixed_params`."
            raise BreakingAPIChangeError(msg)
        # legacy end
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

            def sample_and_weights_func(n, params):
                return sample_func(n, params), None
        elif not callable(sample_and_weights_func):
            msg = "sample_and_weights_func has to be a callable."
            raise TypeError(msg)

        obs = convert_to_space(obs)

        if dtype is None:
            dtype = ztypes.float

        params = convert_param_values(params)
        init_val, init_weights = sample_and_weights_func(n, params)

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
        dataset = LightDataset.from_tensor(sample_holder, ndims=obs.n_obs)

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
            data=dataset,
            sample_holder=sample_holder,
            weights_holder=weights_holder,
            sample_and_weights_func=sample_and_weights_func,
            params=params,
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
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|
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

    @deprecated_args(None, "Use `params` instead.", "param_values")
    def resample(
        self,
        params: ztyping.ParamValuesMap = None,
        *,
        n: TensorLike = None,
        param_values: ztyping.ParamValuesMap = None,
    ):
        """Update the sample by newly sampling *inplace*, affecting objects that have it internally, like likelihoods.

        The values of the parameters used to sample the PDF are taken from the creation of the Sampler and won't change
        by setting their values, i.e. using `param.set_values`. Parameter values (some or all) can be overridden
        by providing the ``params`` argument.

        Args:
            params: a mapping from :py:class:`~zfit.Parameter` or string to a `value` so that the sampler will use
                this value for the sampling. If not given, the `params` will be used.
            n: the number of samples to produce. If the `Sampler` was created with
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

        new_sample, new_weight = self._sample_and_weights_func(n, params=temp_param_values)
        new_sample.set_shape((n, self.space.n_obs))
        if new_weight is not None:
            new_weight.set_shape((n,))
        self.update_data(sample=new_sample, weights=new_weight, guarantee_limits=self._sampler_guarantee_limits)

    def __str__(self) -> str:
        return f"<SamplerData: {self.label} obs={self.obs} size={int(self.nevents)} weighted={self.has_weights} array={self.value()}>"

    @classmethod
    def get_repr(cls):  # acts as data object once serialized
        return DataRepr


def concat(
    datasets: Iterable[Data],
    *,
    obs: ztyping.ObsTypeInput = None,
    axis: int | str | None = None,
    name: str | None = None,
    label: str | None = None,
    use_hash: bool | None = None,
) -> Data:
    """Concatenate multiple `Data` objects into a single one.

    Args:
        datasets: The `Data` objects to concatenate.
        obs: The observables to use. If ``None``, the observables of the first ``Data`` object are used. They have the same
            function as on a single ``Data`` object.
        axis: The axis along which to concatenate the data. If `None`, the data is concatenated along the first axis.
            Possible options are `0/index` or `1/obs`. If `obs`, the data is concatenated along the observable axis.
        name: The name of the new `Data` object. |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
        label: The label of the new `Data` object. |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
        use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|


    Returns:
        A new `Data` object containing the concatenated data.

    Raises:
        tf.errors.InvalidArgumentError: If the number of events in the datasets is not equal.
        ObsIncompatibleError: If the observables are not unique or not the same in all datasets for merging along the observable axis.
    """
    # todo: only works for obs, not yet for axes, but needed?
    if axis is None or axis in (0, "index"):
        axis = 0
    elif axis in (1, "obs", "columns"):
        axis = 1
    else:
        msg = f"Invalid axis {axis}. Valid options are 0/index or 1/obs."
        raise ValueError(msg)

    datasets = convert_to_container(datasets, container=tuple)
    if len(datasets) == 0:
        msg = "No `Data` objects given to concatenate."
        raise ValueError(msg)

    if axis == 0:
        return concat_data_index(datasets=datasets, obs=obs, name=name, label=label, use_hash=use_hash)
    else:
        return concat_data_obs(datasets=datasets, obs=obs, name=name, label=label, use_hash=use_hash)


def concat_data_obs(datasets, obs, name, label, use_hash):
    # check if there are overlapping observables
    all_obs = [ob for data in datasets for ob in data.obs]
    obscounter = Counter(all_obs)
    if any(count > 1 for count in obscounter.values()):
        msg = "Observables have to be unique in the concatenated data."
        raise ObsIncompatibleError(msg)
    space = None
    if obs is not None:
        space = convert_to_space(obs)
        if set(space.obs) != (set_allobs := set(all_obs)):
            msg = f"The given observables ({space.obs}) have to be the same as the observables in the data ({set_allobs})."
            raise ObsIncompatibleError(msg)

    weights_new = []
    new_spaces = None
    nevents = []
    data_new = {} if (use_tensormap := all(data.dataset._tensor is None for data in datasets)) else []

    for data in datasets:
        if use_tensormap:
            value = {ob: data.value(ob) for ob in data.obs}
            data_new.update(value)
            nevents.extend([tf.shape(val) for val in value.values()])
        else:
            value = data.value()
            nevents.append(tf.shape(value)[0])
        data_new.append(value)
        if new_spaces is None:
            new_spaces = data.space
        else:
            new_spaces *= data.space

        if data.has_weights:
            weights_new.append(data.weights)

    tf.debugging.assert_equal(
        tf.reduce_all(tf.equal(nevents, nevents[0])),
        True,
        message=f"Number of events in the datasets {datasets} have to be equal.",
    )
    newweights = znp.prod(weights_new, axis=0) if weights_new else None
    if use_tensormap:
        Data.from_mapping(data_new, obs=space, weights=newweights, name=name, label=label, use_hash=use_hash)
    else:
        newval = znp.concatenate(data_new, axis=-1)
        data = Data.from_tensor(
            tensor=newval,
            obs=new_spaces,
            weights=newweights,
            name=name,
            label=label,
            use_hash=use_hash,
            guarantee_limits=True,
        )
        if space is not None:
            data = data.with_obs(space)
    return data


def concat_data_index(datasets, obs, name, label, use_hash):
    if obs is None:
        space = datasets[0].space
        obs = space.obs
    else:
        if not isinstance((space := obs), ZfitSpace):
            space = datasets[0].space.with_obs(obs)
        obs = space.obs

    if no_obs := [data for data in datasets if data.space.obs is None]:
        msg = f"Data objects {no_obs} have no observables."
        raise ValueError(msg)
    if not all(set(obs) == set(data.obs) for data in datasets):
        msg = "All `Data` objects have to have the same observables."
        raise ValueError(msg)
    weighted = any(data.has_weights for data in datasets)

    if obs is None:
        all_space_equal = all(data.space.with_obs(obs) == space for data in datasets)
        if not all_space_equal:
            msg = "All `Data` objects have to have the same space, i.e. the same limits."
            raise ValueError(msg)

    newval = []
    if weighted:
        newweights = []
    for data in datasets:
        values = data.value(obs=space.obs)
        newval.append(values)
        if weighted:
            weights = tf.ones(tf.shape(values)[0:1]) if not data.has_weights else data.weights
            newweights.append(weights)
    newval = znp.concatenate(newval, axis=0)
    newweights = znp.concatenate(newweights, axis=0) if weighted else None

    return Data.from_tensor(
        tensor=newval, obs=space, weights=newweights, name=name, label=label, use_hash=use_hash, guarantee_limits=True
    )


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
            for _key, value in tensormap.items():
                if value.dtype not in (ztypes.float, znp.float32, znp.float64, znp.int32, znp.int64):
                    msg = f"Value of tensormap has to be a float, not {value.dtype}."
                    raise TypeError(msg)
                if value.dtype != ztypes.float:
                    value = znp.array(value, dtype=ztypes.float)
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
    def from_tensor(cls, tensor, ndims):
        if run.executing_eagerly():
            if tensor.shape[1] != ndims:
                msg = f"Second dimension of {tensor} has to be {ndims} but is {tensor.shape[1]}"
                raise ShapeIncompatibleError(msg)
        elif run.numeric_checks:
            tf.debugging.assert_equal(tf.shape(tensor)[1], ndims)
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
