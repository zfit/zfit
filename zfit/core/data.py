#  Copyright (c) 2020 zfit

import warnings
from collections import OrderedDict
from contextlib import ExitStack
from types import MethodType
from typing import List, Tuple, Union, Dict, Mapping, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# from ..settings import types as ztypes
import zfit
from zfit import z
from zfit.core.interfaces import ZfitSpace
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitData
from .limits import Space, convert_to_space, convert_to_obs_str
from .sample import EventSpace
from ..settings import ztypes
from ..util import ztyping
from ..util.cache import Cachable, invalidates_cache
from ..util.container import convert_to_container
from ..util.exception import LogicalUndefinedOperationError, ShapeIncompatibleError, \
    ObsIncompatibleError
from ..util.temporary import TemporarilySet


class Data(Cachable, ZfitData, BaseDimensional, BaseObject):

    def __init__(self, dataset: Union[tf.data.Dataset, "LightDataset"], obs: ztyping.ObsTypeInput = None,
                 name: str = None, weights=None, iterator_feed_dict: Dict = None,
                 dtype: tf.DType = None):
        """Create a data holder from a `dataset` used to feed into `models`.

        Args:
            dataset (): A dataset storing the actual values
            obs (): Observables where the data is defined in
            name (): Name of the `Data`
            iterator_feed_dict ():
            dtype ():
        """
        if name is None:
            name = "Data"
        if dtype is None:
            dtype = ztypes.float
        super().__init__(name=name)
        if iterator_feed_dict is None:
            iterator_feed_dict = {}
        self._permutation_indices_data = None
        self._next_batch = None
        self._dtype = dtype
        self._nevents = None
        self._weights = None

        self._data_range = None
        self._set_space(obs)
        self._original_obs = self.space.obs
        self._data_range = self.space  # TODO proper data cuts: currently set so that the cuts in all dims are applied
        self.dataset = dataset
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

    @invalidates_cache
    def set_data_range(self, data_range):
        # warnings.warn("Setting the data_range may currently has an unexpected behavior and does not affect the range."
        #               "If you set it once in the beginning, it's ok. Otherwise, it's currently unsafe.")
        data_range = self._check_input_data_range(data_range=data_range)

        def setter(value):
            self._data_range = value

        def getter():
            return self._data_range

        return TemporarilySet(value=data_range, setter=setter, getter=getter)

    @property
    def weights(self):
        return self._weights

    @invalidates_cache
    def set_weights(self, weights: ztyping.WeightsInputType):
        """Set (temporarily) the weights of the dataset.

        Args:
            weights (`tf.Tensor`, np.ndarray, None):


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
    def space(self) -> "ZfitSpace":
        space = self._space
        # if space.limits is None:
        #     if self._data_range is not None:
        #         space = self._data_range
        return space

    @property
    def iterator(self):
        if self._iterator is None:
            self.initialize()
        return self._iterator

    @iterator.setter
    def iterator(self, value):
        self._iterator = value

    # constructors
    @classmethod
    def from_root_iter(cls, path, treepath, branches=None, entrysteps=None, name=None, **kwargs):
        # branches = convert_to_container(branches)
        warnings.warn(
            "Using the iterator is hardcore and will most probably fail! Don't use it (yet) if you don't fully "
            "understand what happens.")

        def uproot_generator():
            for data in uproot.iterate(path=path, treepath=treepath,
                                       branches=branches, entrysteps=entrysteps, **kwargs):
                data = np.array([data[branch] for branch in branches])
                yield data

        dataset = tf.data.Dataset.from_generator(uproot_generator, output_types=ztypes.float)
        dataset.prefetch(2)
        return Data(dataset=dataset, name=name)

    # @classmethod
    # def from_root(cls, path, treepath, branches=None, branches_alias=None, name=None, root_dir_options=None):
    #     if branches_alias is None:
    #         branches_alias = {}
    #
    #     branches = convert_to_container(branches)
    #     if root_dir_options is None:
    #         root_dir_options = {}
    #
    #     def uproot_generator():
    #         root_tree = uproot.open(path, **root_dir_options)[treepath]
    #         data = root_tree.arrays(branches)
    #         data = np.array([data[branch] for branch in branches])
    #         yield data
    #
    #     dataset = tf.data.Dataset.from_generator(uproot_generator, output_types=ztypes.float)
    #
    #     dataset = dataset.repeat()
    #     obs = [branches_alias.get(branch, branch) for branch in branches]
    #     return Data(dataset=dataset, obs=obs, name=name)

    @classmethod
    def from_root(cls, path: str, treepath: str, branches: List[str] = None, branches_alias: Dict = None,
                  weights: ztyping.WeightsStrInputType = None,
                  name: str = None,
                  dtype: tf.DType = None,
                  root_dir_options=None) -> "Data":
        """Create a `Data` from a ROOT file. Arguments are passed to `uproot`.

        Args:
            path (str):
            treepath (str):
            branches (List[str]]):
            branches_alias (dict): A mapping from the `branches` (as keys) to the actual `observables` (as values).
                This allows to have different `observable` names, independent of the branch name in the file.
            weights (tf.Tensor, None, np.ndarray, str]): Weights of the data. Has to be 1-D and match the shape
                of the data (nevents). Can be a column of the ROOT file by using a string corresponding to a
                column.
            name (str):
            root_dir_options ():

        Returns:
            `zfit.Data`:
        """
        if branches_alias is None and branches is None:
            raise ValueError("Either branches or branches_alias has to be specified.")

        if branches_alias is None:
            branches_alias = {}
        if branches is None:
            branches = list(branches_alias.values())

        weights_are_branch = isinstance(weights, str)

        branches = convert_to_container(branches)
        if root_dir_options is None:
            root_dir_options = {}

        def uproot_loader():
            root_tree = uproot.open(path, **root_dir_options)[treepath]
            if weights_are_branch:
                branches_with_weights = branches + [weights]
            else:
                branches_with_weights = branches
            data = root_tree.arrays(branches_with_weights, namedecode="utf-8")
            data_np = np.array([data[branch] for branch in branches])
            if weights_are_branch:
                weights_np = data[weights]
            else:
                weights_np = None
            return data_np.transpose(), weights_np

        data, weights_np = uproot_loader()
        if not weights_are_branch:
            weights_np = weights
        shape = data.shape
        dataset = LightDataset.from_tensor(data)

        # dataset = dataset.repeat()
        obs = [branches_alias.get(branch, branch) for branch in branches]
        return Data(dataset=dataset, obs=obs, weights=weights_np, name=name, dtype=dtype)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, obs: ztyping.ObsTypeInput = None, weights: ztyping.WeightsInputType = None,
                    name: str = None, dtype: tf.DType = None):
        """Create a `Data` from a pandas DataFrame. If `obs` is `None`, columns are used as obs.

        Args:
            df (`pandas.DataFrame`):
            weights (tf.Tensor, None, np.ndarray, str]): Weights of the data. Has to be 1-D and match the shape
                of the data (nevents).
            obs (`zfit.Space`):
            name (str):
        """
        if obs is None:
            obs = list(df.columns)
        array = df.values
        return cls.from_numpy(obs=obs, array=array, weights=weights, name=name, dtype=dtype)

    @classmethod
    def from_numpy(cls, obs: ztyping.ObsTypeInput, array: np.ndarray, weights: ztyping.WeightsInputType = None,
                   name: str = None, dtype: tf.DType = None):
        """Create `Data` from a `np.array`.

        Args:
            obs ():
            array (numpy.ndarray):
            name (str):

        Returns:
            zfit.Data:
        """

        if not isinstance(array, (np.ndarray)) and not (tf.is_tensor(array) and hasattr(array, 'numpy')):
            raise TypeError("`array` has to be a `np.ndarray`. Is currently {}".format(type(array)))
        if dtype is None:
            dtype = ztypes.float
        tensor = tf.cast(array, dtype=dtype)
        return cls.from_tensor(obs=obs, tensor=tensor, weights=weights, name=name, dtype=dtype)
        # np_placeholder = tf.compat.v1.placeholder(dtype=array.dtype, shape=array.shape)
        # iterator_feed_dict = {np_placeholder: array}
        # dataset = tf.data.Dataset.from_tensors(np_placeholder)
        #
        # dataset = dataset.batch(len(array))
        # dataset = dataset.repeat()
        # return Data(dataset=dataset, obs=obs, name=name, weights=weights, dtype=dtype,
        #             iterator_feed_dict=iterator_feed_dict)

    @classmethod
    def from_tensor(cls, obs: ztyping.ObsTypeInput, tensor: tf.Tensor, weights: ztyping.WeightsInputType = None,
                    name: str = None, dtype: tf.DType = None) -> "Data":
        """Create a `Data` from a `tf.Tensor`. `Value` simply returns the tensor (in the right order).

        Args:
            obs (Union[str, List[str]):
            tensor (`tf.Tensor`):
            name (str):

        Returns:
            zfit.core.Data:
        """
        dataset = LightDataset.from_tensor(tensor=tensor)
        return Data(dataset=dataset, obs=obs, name=name, weights=weights, dtype=dtype)

    def to_pandas(self, obs: ztyping.ObsTypeInput = None):
        """Create a `pd.DataFrame` from `obs` as columns and return it.

        Args:
            obs (): The observables to use as columns. If `None`, all observables are used.

        Returns:

        """
        values = self.value(obs=obs)
        if obs is None:
            obs = self.obs
        obs_str = convert_to_obs_str(obs)
        values = values.numpy()
        df = pd.DataFrame(data=values, columns=obs_str)
        return df

    def initialize(self):
        iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        # TODO(Mayou36): TF2 convert correct
        self.sess.run(iterator.initializer, self.iterator_feed_dict)
        self.iterator = iterator

    def get_iteration(self):
        if isinstance(self.dataset, LightDataset):
            return self.dataset.value()
        if self._next_batch is None:
            self._next_batch = self.iterator.get_next()
        return self._next_batch

    def unstack_x(self, obs: ztyping.ObsTypeInput = None, always_list: bool = False):
        """Return the unstacked data: a list of tensors or a single Tensor.

        Args:
            obs (): which observables to return
            always_list (bool): If True, always return a list (also if length 1)

        Returns:
            List(tf.Tensor)
        """
        return z.unstack_x(self._value_internal(obs=obs))

    def value(self, obs: ztyping.ObsTypeInput = None):
        return self._value_internal(obs=obs)

    def numpy(self):
        return self.value().numpy()

    def _cut_data(self, value, obs=None):
        if self.data_range.limits is not None:
            data_range = self.data_range.with_obs(obs=obs)

            inside_limits = []
            # value = tf.transpose(value)
            for lower, upper in data_range.iter_limits():
                if isinstance(data_range, EventSpace):  # TODO(Mayou36): remove EventSpace hack once more general
                    upper = tf.cast(tf.transpose(upper), dtype=self.dtype)
                    lower = tf.cast(tf.transpose(lower), dtype=self.dtype)

                below_upper = tf.reduce_all(input_tensor=tf.less_equal(value, upper), axis=1)  # if all obs inside
                above_lower = tf.reduce_all(input_tensor=tf.greater_equal(value, lower), axis=1)
                inside_limits.append(tf.logical_and(above_lower, below_upper))
            inside_any_limit = tf.reduce_any(input_tensor=inside_limits, axis=0)  # has to be inside one limit

            value = tf.boolean_mask(tensor=value, mask=inside_any_limit)
            # value = tf.transpose(value)

        return value

    def _value_internal(self, obs: ztyping.ObsTypeInput = None):
        if obs is not None:
            obs = convert_to_obs_str(obs)
        raw_value = self._value()
        value = self._cut_data(raw_value, obs=self._original_obs)
        value_sorted = self._sort_value(value=value, obs=obs)
        return value_sorted

    def _value(self):
        values = self.get_iteration()
        # TODO(Mayou36): add conversion to right dimension? (n_events, n_obs)? # check if 1-D?
        if len(values.shape.as_list()) == 0:
            values = tf.expand_dims(values, -1)
        if len(values.shape.as_list()) == 1:
            values = tf.expand_dims(values, -1)

        # cast data to right type
        if not values.dtype == self.dtype:
            values = tf.cast(values, dtype=self.dtype)
        return values

    def _sort_value(self, value, obs: Tuple[str]):
        obs = convert_to_container(value=obs, container=tuple)
        perm_indices = self.space.axes if self.space.axes != tuple(range(value.shape[-1])) else False

        # permutate = perm_indices is not None
        if obs:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError("The observable(s) {} are not contained in the dataset. "
                                 "Only the following are: {}".format(frozenset(obs) - frozenset(self.obs),
                                                                     self.obs))
            perm_indices = self.space.get_axes(obs=obs)
            # values = list(values[self.obs.index(o)] for o in obs if o in self.obs)
        if perm_indices:
            value = z.unstack_x(value)
            value = list(value[i] for i in perm_indices)
            value = z.stack_x(value)

        return value

    # TODO(Mayou36): use Space to permute data?
    # TODO(Mayou36): raise error is not obs <= self.obs?
    @invalidates_cache
    def sort_by_axes(self, axes: ztyping.AxesTypeInput, allow_superset: bool = False):
        if not allow_superset:
            if not frozenset(axes) <= frozenset(self.axes):
                raise ValueError("The observable(s) {} are not contained in the dataset. "
                                 "Only the following are: {}".format(frozenset(axes) - frozenset(self.axes),
                                                                     self.axes))
        space = self.space.with_axes(axes=axes)

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    @invalidates_cache
    def sort_by_obs(self, obs: ztyping.ObsTypeInput, allow_superset: bool = False):
        if not allow_superset:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError("The observable(s) {} are not contained in the dataset. "
                                 "Only the following are: {}".format(frozenset(obs) - frozenset(self.obs),
                                                                     self.obs))

        space = self.space.with_obs(obs=obs)

        def setter(value):
            self._space = value

        def getter():
            return self.space

        return TemporarilySet(value=space, setter=setter, getter=getter)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype is not None:
            if dtype != self.dtype:
                # return ValueError("From Mayou36", self.dtype)
                return NotImplemented
        if as_ref:
            # return "NEVER READ THIS"
            raise LogicalUndefinedOperationError("There is no ref for the `Data`")
        else:
            return self.value()

    def _AsTensor(self):
        return self.value()

    @staticmethod
    def _OverloadAllOperators():  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            Data._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        setattr(Data, "__getitem__", array_ops._SliceHelperVar)

    @staticmethod
    def _OverloadOperator(operator):  # pylint: disable=invalid-name
        """Defer an operator overload to `ops.Tensor`.
        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
        Args:
          operator: string. The operator name.
        """

        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args):
            # pylint: disable=protected-access
            value = a._AsTensor()
            return tensor_oper(value, *args)

        # Propagate __doc__ to wrapper
        try:
            _run_op.__doc__ = tensor_oper.__doc__
        except AttributeError:
            pass

        setattr(Data, operator, _run_op)

    def _check_input_data_range(self, data_range):
        data_range = self.convert_sort_space(limits=data_range)
        if not frozenset(self.data_range.obs) == frozenset(data_range.obs):
            raise ObsIncompatibleError(f"Data range has to cover the full observable space {self.data_range.obs}, not "
                                       f"only {data_range.obs}")
        return data_range

    # TODO(Mayou36): refactor with pdf or other range things?
    def convert_sort_space(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> Union[Space, None]:
        """Convert the inputs (using eventually `obs`, `axes`) to :py:class:`~zfit.Space` and sort them according to
        own `obs`.

        Args:
            obs ():
            axes ():
            limits ():

        Returns:

        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        self_space = self._space
        if self_space is not None:
            space = space.with_obs_axes(self_space.get_obs_axes(), ordered=True, allow_subset=True)
        return space

    def _get_nevents(self):
        nevents = tf.shape(input=self.value())[0]
        return nevents


class SampleData(Data):
    _cache_counting = 0

    def __init__(self, dataset: Union[tf.data.Dataset, "LightDataset"], sample_holder: tf.Tensor,
                 obs: ztyping.ObsTypeInput = None, weights=None, name: str = None,
                 dtype: tf.DType = ztypes.float):
        super().__init__(dataset, obs, name=name, weights=weights, iterator_feed_dict=None, dtype=dtype)

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(cls, sample: tf.Tensor, obs: ztyping.ObsTypeInput, name: str = None,
                    weights=None):
        dataset = LightDataset.from_tensor(sample)
        return SampleData(dataset=dataset, sample_holder=sample, obs=obs, name=name, weights=weights)


class Sampler(Data):
    _cache_counting = 0

    def __init__(self, dataset: "LightDataset", sample_func: Callable, sample_holder: tf.Variable,
                 n: Union[ztyping.NumericalScalarType, Callable], weights=None,
                 fixed_params: Dict["zfit.Parameter", ztyping.NumericalScalarType] = None,
                 obs: ztyping.ObsTypeInput = None, name: str = None,
                 dtype: tf.DType = ztypes.float):

        super().__init__(dataset=dataset, obs=obs, name=name, weights=weights, iterator_feed_dict=None, dtype=dtype)
        if fixed_params is None:
            fixed_params = OrderedDict()
        if isinstance(fixed_params, (list, tuple)):
            fixed_params = OrderedDict((param, param.numpy()) for param in fixed_params)

        self._initial_resampled = False

        self.fixed_params = fixed_params
        self.sample_holder = sample_holder
        self.sample_func = sample_func
        self.n = n
        self._n_holder = n

    @property
    def n_samples(self):
        return self._n_holder

    def _value_internal(self, obs: ztyping.ObsTypeInput = None):
        if not self._initial_resampled:
            raise RuntimeError(
                "No data generated yet. Use `resample()` to generate samples or directly use `model.sample()`"
                "for single-time sampling.")
        return super()._value_internal(obs)

    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting

    @classmethod
    def from_sample(cls, sample_func: Callable, n: ztyping.NumericalScalarType, obs: ztyping.ObsTypeInput,
                    fixed_params=None, name: str = None, weights=None, dtype=None):
        obs = convert_to_space(obs)

        if fixed_params is None:
            fixed_params = []
        if dtype is None:
            dtype = ztypes.float
        # from tensorflow.python.ops.variables import VariableV1
        sample_holder = tf.Variable(initial_value=sample_func(), dtype=dtype, trainable=False,  # HACK: sample_func
                                    # validate_shape=False,
                                    shape=(None, obs.n_obs),
                                    name="sample_data_holder_{}".format(cls.get_cache_counting()))
        dataset = LightDataset.from_tensor(sample_holder)

        return Sampler(dataset=dataset, sample_holder=sample_holder, sample_func=sample_func, fixed_params=fixed_params,
                       n=n, obs=obs, name=name, weights=weights)

    def resample(self, param_values: Mapping = None, n: Union[int, tf.Tensor] = None):
        """Update the sample by newly sampling. This affects any object that used this data already.

        All params that are not in the attribute `fixed_params` will use their current value for
        the creation of the new sample. The value can also be overwritten for one sampling by providing
        a mapping with `param_values` from `Parameter` to the temporary `value`.

        Args:
            param_values (Dict): a mapping from :py:class:`~zfit.Parameter` to a `value`. For the current sampling,
                `Parameter` will use the `value`.
            n (int, tf.Tensor): the number of samples to produce. If the `Sampler` was created with
                anything else then a numerical or tf.Tensor, this can't be used.
        """
        if n is None:
            n = self.n

        temp_param_values = self.fixed_params.copy()
        if param_values is not None:
            temp_param_values.update(param_values)

        with ExitStack() as stack:

            _ = [stack.enter_context(param.set_value(val)) for param, val in temp_param_values.items()]

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


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(Data, _dense_var_to_tensor)
fetch_function = lambda data: ([data.value()],
                               lambda val: val[0])
feed_function = lambda data, feed_val: [(data.value(), feed_val)]
feed_function_for_partial_run = lambda data: [data.value()]

from tensorflow.python.client.session import register_session_run_conversion_functions

# ops.register_dense_tensor_like_type()

register_session_run_conversion_functions(tensor_type=Data, fetch_function=fetch_function,
                                          feed_function=feed_function,
                                          feed_function_for_partial_run=feed_function_for_partial_run)

Data._OverloadAllOperators()


class LightDataset:

    def __init__(self, tensor):
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            tensor = z.convert_to_tensor(tensor)
        self.tensor = tensor

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor=tensor)

    def value(self):
        return self.tensor
