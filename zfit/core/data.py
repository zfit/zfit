import tensorflow as tf
import uproot
import numpy as np

from ..settings import types as ztypes


class Data:

    def __init__(self, dataset, name="Data"):
        self.dataset = dataset
        self.name = name

    # constructor
    @classmethod
    def from_root(cls, path, treepath, branches=None, entrysteps=20000, name=None, **kwargs):
        def uproot_generator():
            for data in uproot.iterate(path=path, treepath=treepath,
                                       branches=branches, entrysteps=entrysteps, **kwargs):
                data = np.array([data[branch] for branch in branches])
                yield data

        dataset = tf.data.Dataset.from_generator(uproot_generator, output_types=ztypes.float)
        dataset.prefetch(2)
        return Data(dataset=dataset, name=name)

    def get_iteration(self):
        iterator = self.dataset.make_one_shot_iterator()
        next = iterator.get_next()
        return next

if __name__ == '__main__':

    path_root = "/data/uni/b2k1ee/classification_new/2012/"
    big_root = 'Bu2KpipiEE-MC-12125000-2012-MagAll-StrippingBu2LLK.root'
    small_root = 'small.root'

    # path_root += big_root
    path_root += small_root

    branches = [b'B_PT', b"B_M"]


    def uproot_generator():
        for data in uproot.iterate(path=path_root, treepath='DecayTree',
                                   branches=branches, entrysteps=3000):
            data = np.array([data[branch] for branch in branches])

            yield data


    data = tf.data.Dataset.from_generator(uproot_generator,
                                          output_types=tf.float64)

    # data = data.batch(batch_size=1)
    data = data.prefetch(100)

    iterator = data.make_one_shot_iterator()

    next = iterator.get_next()

    with tf.Session() as sess:
        for i in range(500):
            print(i)
            try:
                result = sess.run(next)
            except tf.errors.OutOfRangeError:
                print("finished at i = ", i)
                break
            print(np.shape(result))

    # directory = open_tree[]
    # directory = directory['DecayTree']
    # directory = directory['B_P']
