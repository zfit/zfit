import tensorflow as tf

from zfit.settings import ztypes


def _auto_upcast(tensor: tf.Tensor):
    if isinstance(tensor, tf.Tensor):
        new_dtype = ztypes[tensor.dtype]
        if new_dtype != tensor.dtype:
            tensor = tf.cast(x=tensor, dtype=new_dtype)
    return tensor

# def _wrap_auto_upcast(func):
