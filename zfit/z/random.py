#  Copyright (c) 2020 zfit
from functools import wraps
from typing import Union, Iterable, Any

import tensorflow as tf
import tensorflow_probability as tfp

from .zextension import function as function

__all__ = ["counts_multinomial"]


from ..settings import ztypes


def counts_multinomial(total_count: Union[int, tf.Tensor], probs: Iterable[Union[float, tf.Tensor]] = None,
                       logits: Iterable[Union[float, tf.Tensor]] = None, dtype=tf.int64) -> tf.Tensor:
    """Get the number of counts for different classes with given probs/logits.

    Args:
        total_count: The total number of draws.
        probs: Length k (number of classes) object where the k-1th entry contains the probability to
            get a single draw from the class k. Have to be from [0, 1] and sum up to 1.
        logits: Same as probs but from [-inf, inf] (will be transformet to [0, 1])

    Returns:
        Shape (k,) tensor containing the number of draws.
    """
    from .. import z

    total_count = tf.convert_to_tensor(total_count)
    probs = z.convert_to_tensor(probs) if probs is not None else probs
    logits = tf.convert_to_tensor(logits) if logits is not None else logits

    if probs is not None:
        probs = tf.cast(probs, dtype=tf.float64)
        float_dtype = probs.dtype
    elif logits is not None:
        logits = tf.cast(logits, tf.float64)
        float_dtype = logits.dtype
    else:
        raise ValueError("Exactly one of `probs` or`logits` have to be specified")
    total_count = tf.cast(total_count, dtype=float_dtype)

    # needed since otherwise shape of sample will be (1, n_probs)
    # total_count = tf.broadcast_to(total_count, shape=probs_logits_shape)

    @function
    def wrapped_func(dtype, logits, probs, total_count):

        dist = tfp.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits)
        counts = dist.sample()
        counts = tf.cast(counts, dtype=dtype)
        return counts

    return wrapped_func(dtype, logits, probs, total_count)


@wraps(tf.random.normal)
def normal(shape, mean=0.0, stddev=1.0, dtype=ztypes.float, seed=None, name=None):
    return tf.random.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)


@wraps(tf.random.uniform)
def uniform(shape, minval=0, maxval=None, dtype=ztypes.float, seed=None, name=None):
    return tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed, name=name)


@wraps(tf.random.poisson)
def poisson(lam: Any, shape: Any, dtype: tf.DType = ztypes.float, seed: Any = None, name: Any = None):
    return tf.random.poisson(lam=lam, shape=shape, dtype=dtype, seed=seed, name=name)
