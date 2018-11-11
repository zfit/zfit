import warnings

import tensorflow as tf
import uproot
import numpy as np

# from ..settings import types as ztypes
from zfit.settings import types as ztypes

class Data:

    def __init__(self, dataset, columns=None, name=None, iterator_feed_dict=None):

        if name is None:
            name = "Data"
        if iterator_feed_dict is None:
            iterator_feed_dict = {}
        self.dataset = dataset
        self.name = name
        self.iterator_feed_dict = iterator_feed_dict
        self.iterator = None

    @property
    def iterator(self):
        if self._iterator is None:
            raise RuntimeError("Dataset not initialized. Will be done automatically in the future.")
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
        return Data(dataset=dataset, columns=branches, name=name)

    @classmethod
    def from_numpy(cls, array, name=None):
        if not isinstance(array, np.array):
            raise TypeError("`array` has to be a `np.array`. Is currently {}".format(type(array)))
        np_placeholder = tf.placeholder(dtype=array.dtype, shape=array.shape)
        iterator_feed_dict = {np_placeholder: array}
        dataset = tf.data.Dataset.from_tensors(np_placeholder)

        # dataset = dataset.batch(len(array))
        dataset = dataset.repeat()
        return Data(dataset=dataset, name=name, iterator_feed_dict=iterator_feed_dict)

    def initialize(self, sess=None):
        iterator = self.dataset.make_initializable_iterator()
        sess.run(iterator.initializer, self.iterator_feed_dict)
        self.iterator = iterator

    def get_iteration(self):
        next_batch = self.iterator.get_next()
        return next_batch

    def values(self):
        return self.get_iteration()


if __name__ == '__main__':

    path_root = "/data/uni/b2k1ee/classification_new/2012/"
    big_root = 'Bu2KpipiEE-MC-12125000-2012-MagAll-StrippingBu2LLK.root'
    small_root = 'small.root'

    # path_root += big_root
    path_root += small_root

    branches = [b'B_PT', b"B_M"]

    # def uproot_generator():
    #     for data in uproot.iterate(path=path_root, treepath='DecayTree',
    #                                branches=branches, entrysteps=3000):
    #         data = np.array([data[branch] for branch in branches])
    #
    #         yield data
    #
    #
    # data = tf.data.Dataset.from_generator(uproot_generator,
    #                                       output_types=tf.float64)
    #
    # data = data.batch(batch_size=4)
    # data = data.prefetch(100)
    #
    # iterator = data.make_one_shot_iterator()
    #
    # x = iterator.get_next()
    data = Data.from_root(path=path_root, treepath='DecayTree', branches=branches)
    import time

    with tf.Session() as sess:
        data.initialize(sess=sess)
        x = data.values()

        for i in range(1):
            print(i)
            try:
                func = tf.log(x) * tf.sin(x) * tf.reduce_mean(x**2 - tf.cos(x)**2) / tf.reduce_sum(x)
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
