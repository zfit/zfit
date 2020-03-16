#  Copyright (c) 2020 zfit

from typing import Union, Iterable

import tensorflow as tf
import tensorflow_probability as tfp

from .zextension import tf_function as function

__all__ = ["counts_multinomial"]


def counts_multinomial(total_count: Union[int, tf.Tensor], probs: Iterable[Union[float, tf.Tensor]] = None,
                       logits: Iterable[Union[float, tf.Tensor]] = None, dtype=tf.int64) -> tf.Tensor:
    """Get the number of counts for different classes with given probs/logits.

    Args:
        total_count (int): The total number of draws.
        probs: Length k (number of classes) object where the k-1th entry contains the probability to
            get a single draw from the class k. Have to be from [0, 1] and sum up to 1.
        logits: Same as probs but from [-inf, inf] (will be transformet to [0, 1])

    Returns:
        :py:class:`tf.Tensor`: shape (k,) tensor containing the number of draws.
    """
    total_count = tf.convert_to_tensor(total_count)
    probs = tf.convert_to_tensor(probs) if probs is not None else probs
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
