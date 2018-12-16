from collections import OrderedDict
from typing import List
import warnings

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import uproot
import numpy as np

# from ..settings import types as ztypes
import zfit
from zfit.core.baseobject import BaseObject
from zfit.core.interfaces import ZfitData
from zfit.core.limits import NamedSpace, convert_to_space
from zfit.settings import types as ztypes
from zfit.util.exception import LogicalUndefinedOperationError, NoSessionSpecifiedError
from zfit.util.temporary import TemporarilySet


class Data(ZfitData, BaseObject):

    def __init__(self, dataset, obs=None, name=None, iterator_feed_dict=None, dtype=ztypes.float):

        if name is None:
            name = "Data"
        super().__init__(name=name)
        if iterator_feed_dict is None:
            iterator_feed_dict = {}
        self._data_range = None
        self._permutation_indices_data = None
        self._next_batch = None
        self._dtype = dtype

        self._set_space(obs)
        self.dataset = dataset
        self._name = name
        self.iterator_feed_dict = iterator_feed_dict
        self.iterator = None

    @property
    def obs(self):
        return self._space.obs

    @property
    def dtype(self):
        return self._dtype

    def _set_space(self, obs: NamedSpace):
        obs = convert_to_space(obs)
        obs = obs.with_autofill_axes()
        self._space = obs

    @property
    def data_range(self):
        data_range = self._data_range
        if data_range is None:
            data_range = self._space
        return data_range

    @property
    def iterator(self):
        if self._iterator is None:
            self.initialize()
        return self._iterator

    @iterator.setter
    def iterator(self, value):
        self._iterator = value

    # constructor

    @classmethod
    def from_root_iter(cls, path, treepath, branches=None, entrysteps=None, name=None, **kwargs):
        # branches = convert_to_container(branches)
        warnings.warn("Using the iterator is hardcore! Don't do it if you don't fully understand what happens.")

        def uproot_generator():
            for data in uproot.iterate(path=path, treepath=treepath,
                                       branches=branches, entrysteps=entrysteps, **kwargs):
                data = np.array([data[branch] for branch in branches])
                yield data

        dataset = tf.data.Dataset.from_generator(uproot_generator, output_types=ztypes.float)
        dataset.prefetch(2)
        return Data(dataset=dataset, name=name)

    @classmethod
    def from_root(cls, path, treepath, branches=None, name=None, root_dir_options=None, **kwargs):
        # branches = convert_to_container(branches)
        if root_dir_options is None:
            root_dir_options = {}

        def uproot_generator():
            root_tree = uproot.open(path, **root_dir_options)[treepath]
            data = root_tree.arrays(branches)
            data = np.array([data[branch] for branch in branches])
            yield data

        dataset = tf.data.Dataset.from_generator(uproot_generator, output_types=ztypes.float)
        # dataset.prefetch(2)

        # dataset = dataset.batch(int(5))
        dataset = dataset.repeat()
        return Data(dataset=dataset, obs=branches, name=name)

    @classmethod
    def from_numpy(cls, obs, array, name=None):
        if not isinstance(array, np.ndarray):
            raise TypeError("`array` has to be a `np..ndarray`. Is currently {}".format(type(array)))
        np_placeholder = tf.placeholder(dtype=array.dtype, shape=array.shape)
        iterator_feed_dict = {np_placeholder: array}
        dataset = tf.data.Dataset.from_tensors(np_placeholder)

        # dataset = dataset.batch(len(array))
        dataset = dataset.repeat()
        return Data(dataset=dataset, obs=obs, name=name, iterator_feed_dict=iterator_feed_dict)

    @classmethod
    def from_tensors(cls, obs, tensors, name=None):
        dataset = tf.data.Dataset.from_tensors(tensors=tensors)
        dataset = dataset.repeat()
        return Data(dataset=dataset, obs=obs, name=name)

    def initialize(self, sess=None):
        iterator = self.dataset.make_initializable_iterator()
        if sess is None:
            sess = zfit.sess
            if sess is None:
                raise NoSessionSpecifiedError()
        sess.run(iterator.initializer, self.iterator_feed_dict)
        self.iterator = iterator

    def get_iteration(self):
        if self._next_batch is None:
            self._next_batch = self.iterator.get_next()
        return self._next_batch

    def value(self, obs: List[str] = None):
        values = self.get_iteration()
        perm_indices = self._permutation_indices_data

        # permutate = perm_indices is not None
        if perm_indices or obs:
            values = tf.unstack(values)
            if perm_indices:
                values = list(values[i] for i in perm_indices)
            if obs:
                if not frozenset(obs) <= frozenset(self.obs):
                    raise ValueError("The observable(s) {} are not contained in the dataset. "
                                     "Only the following are: {}".format(frozenset(obs) - frozenset(self.obs),
                                                                         self.obs))
                values = list(values[self.obs.index(o)] for o in obs if o in self.obs)
            values = tf.stack(values)

        return values

    # TODO(Mayou36): use Space to permute data?
    # TODO(Mayou36): raise error is not obs <= self.obs?
    def sort_by_obs(self, obs, allow_not_subset=False):
        if not allow_not_subset:
            if not frozenset(obs) <= frozenset(self.obs):
                raise ValueError("The observable(s) {} are not contained in the dataset. "
                                 "Only the following are: {}".format(frozenset(obs) - frozenset(self.obs),
                                                                     self.obs))
        permutation_indices = tuple(self.obs.index(o) for o in obs if o in self.obs)
        if self._permutation_indices_data is None:
            permutation_indices_data = permutation_indices
        else:
            permutation_indices_data = tuple(self._permutation_indices_data[i] for i in permutation_indices)
        obs_axes_items = tuple(self._space.get_axes(as_dict=True, autofill=True).items())
        obs_axes = OrderedDict(obs_axes_items[i] for i in permutation_indices)

        space = self._space.with_obs_axes(obs_axes=obs_axes, ordered=True)
        value = space, permutation_indices_data

        def setter(value):
            space, permutation_indices_data = value
            self._permutation_indices_data = permutation_indices_data
            self._space = space

        def getter():
            return self._space, self._permutation_indices_data

        return TemporarilySet(value=value, setter=setter, getter=getter)

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
        for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
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

        tensor_oper = getattr(ops.Tensor, operator)

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

if __name__ == '__main__':

    from skhep_testdata import data_path

    # path_root = "/data/uni/b2k1ee/classification_new/2012/"
    # big_root = 'Bu2KpipiEE-MC-12125000-2012-MagAll-StrippingBu2LLK.root'
    # small_root = 'small.root'
    #
    # # path_root += big_root
    # path_root += small_root
    path_root = data_path("uproot-Zmumu.root")

    branches = [b'pt1', b'pt2']  # b needed currently -> uproot

    data = Data.from_root(path=path_root, treepath='events', branches=branches)
    import time

    with tf.Session() as sess:
        data.initialize(sess=sess)
        x = data.value()

        for i in range(1):
            print(i)
            try:
                func = tf.log(x) * tf.sin(x) * tf.reduce_mean(x ** 2 - tf.cos(x) ** 2) / tf.reduce_sum(x)
                start = time.time()
                result_grad = sess.run(tf.gradients(func, x))
                result = sess.run(func)
                end = time.time()
                print("time needed", (end - start))
            except tf.errors.OutOfRangeError:
                print("finished at i = ", i)
                break
            print(np.shape(result))
            print(result[:, :10])
            print(result_grad)

    # directory = open_tree[]
    # directory = directory['DecayTree']
    # directory = directory['B_P']
