#  Copyright (c) 2019 zfit

import tensorflow.compat.v1 as tf

tf.enable_resource_variables()  # forward compat
tf.enable_v2_tensorshape()  # forward compat
tf.disable_eager_execution()

from zfit.settings import ztypes


def _auto_upcast(tensor: tf.Tensor):
    if isinstance(tensor, tf.Tensor):
        new_dtype = ztypes[tensor.dtype]
        if new_dtype != tensor.dtype:
            tensor = tf.cast(x=tensor, dtype=new_dtype)
    return tensor

# def _wrap_auto_upcast(func):
