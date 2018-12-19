import functools

import tensorflow as tf

from zfit.settings import types as ztypes


def log(x, name=None):
    return tf.log(x=x, name=name)


def exp(x, name=None):
    return tf.exp(x=x, name=name)


@functools.wraps(tf.convert_to_tensor)
def convert_to_tensor(value, dtype=ztypes.float, name=None, preferred_dtype=ztypes.float):
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, preferred_dtype=preferred_dtype)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=ztypes.float, seed=None, name=None):
    return tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)


def random_uniform(shape, minval=0, maxval=None, dtype=ztypes.float, seed=None, name=None):
    return tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed, name=name)

#
# @functools.wraps(tf.reduce_sum)
# def reduce_sum(*args, **kwargs):
#     return tf.reduce_sum(*args, **kwargs)
reduce_sum = tf.reduce_sum
