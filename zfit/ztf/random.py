#  Copyright (c) 2019 zfit

from typing import Union, Iterable, Sized

import tensorflow_probability as tfp
import tensorflow as tf

from .wrapping_tf import convert_to_tensor
from ..util.container import convert_to_container

__all__ = ["counts_multinomial"]


def counts_multinomial(total_count: Union[int, tf.Tensor], probs: Iterable[Union[float, tf.Tensor]] = None,
                       logits: Iterable[Union[float, tf.Tensor]] = None, dtype=tf.int32) -> tf.Tensor:
    """Get the number of counts for different classes with given probs/logits.

    Args:
        total_count (int): The total number of draws.
        probs: Length k (number of classes) object where the k-1th entry contains the probability to
            get a single draw from the class k. Have to be from [0, 1] and sum up to 1.
        logits: Same as probs but from [-inf, inf] (will be transformet to [0, 1])

    Returns:
        :py:class.`tf.Tensor`: shape (k,) tensor containing the number of draws.
    """
    control_deps = []
    if probs is not None:
        if not isinstance(probs, (tf.Tensor, tf.Variable)):
            probs = convert_to_container(probs)
            if len(probs) < 2:
                raise ValueError("`probs` has to have length 2 at least.")
            probs = tf.convert_to_tensor(probs)
        probs = tf.cast(probs, tf.float32)
        control_deps.append(probs)
        # probs_logits_shape = tf.shape(probs)
    elif logits is not None:

        if not isinstance(logits, (tf.Tensor, tf.Variable)):
            logits = convert_to_container(logits)
            if len(logits) < 2:
                raise ValueError("`logits` has to have length 2 at least.")
            logits = tf.convert_to_tensor(logits, dtype=None)
        logits = tf.cast(logits, tf.float32)
        control_deps.append(logits)
        # probs_logits_shape = tf.shape(logits)
    else:
        raise ValueError("Exactly one of `probs` or`logits` have to be specified")
    if not isinstance(total_count, tf.Variable):
        total_count = convert_to_tensor(total_count, dtype=None)
    total_count = tf.cast(total_count, dtype=tf.float32)
    control_deps.append(total_count)
    # needed since otherwise shape of sample will be (1, n_probs)
    # total_count = tf.broadcast_to(total_count, shape=probs_logits_shape)

    with tf.control_dependencies(control_deps):
        dist = tfp.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits)
        counts = dist.sample()
    counts = tf.cast(counts, dtype=dtype)
    with tf.control_dependencies([counts]):
        return counts
