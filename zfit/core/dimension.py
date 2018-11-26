import functools

import numpy as np
import tensorflow as tf

from zfit import ztf

@functools.lru_cache(maxsize=500)
def get_same_dims(dims):
    deps = [set() for _ in range(len(dims))]
    for i, dim in enumerate(dims):
        for j, other_dim in enumerate(dims[i + 1:]):
            if not set(dim).isdisjoint(other_dim):
                deps[i].add(i)
                deps[i].add(j + i + 1)
                deps[j + i + 1].add(i)
    return deps


def unstack_x_dims(x, dims):
    if len(np.shape(dims[0])) == ():
        dims = (dims,)
    x_unstacked = ztf.unstack_x(x)

    x_per_dims = [tf.stack([x_unstacked[d] for d in dim]) for dim in dims]
    return x_per_dims
