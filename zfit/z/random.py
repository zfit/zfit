#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from .zextension import function as function

__all__ = ["counts_multinomial", "sample_with_replacement"]

from ..settings import ztypes
from ..z import numpy as znp

generator = None


def get_prng():
    """Get the global random number generator.

    Returns:
        zfit random number generator
    """
    global generator
    if generator is None:  # initialization
        generator = tf.random.Generator.from_non_deterministic_state()
    return generator


def sample_with_replacement(
    a: tf.Tensor, axis: int, sample_shape: tuple[int]
) -> tf.Tensor:
    """Sample from ``a`` with replacement to return a Tensor with ``sample_shape``.

    Args:
        a (): Tensor to sample from
        axis (): int axis to sample along
        sample_shape (): Shape of the new samples along the axis

    Returns:
        tf.Tensor of shape a.shape[:axis] + samples_shape + a.shape[axis + 1:]


    Examples:

    .. code:: pycon

        >>> a = tf.random.uniform(shape=(10, 20, 30), dtype=tf.float32)
        >>> random_choice(a, axis=0)
        <tf.Tensor 'GatherV2:0' shape=(1, 20, 30) dtype=float32>
        >>> random_choice(a, axis=1, samples_shape=(2, 3))
        <tf.Tensor 'GatherV2_2:0' shape=(10, 2, 3, 30) dtype=float32
        >>> random_choice(a, axis=0, samples_shape=(100,))
        <tf.Tensor 'GatherV2_3:0' shape=(100, 20, 30) dtype=float32>
    """

    dim = tf.shape(a)[axis]
    choice_indices = get_prng().uniform(
        sample_shape, minval=0, maxval=dim, dtype=tf.int32
    )
    samples = tf.gather(a, choice_indices, axis=axis)
    return samples


def counts_multinomial(
    total_count: int | tf.Tensor,
    probs: Iterable[float | tf.Tensor] = None,
    logits: Iterable[float | tf.Tensor] = None,
    dtype=tf.int32,
) -> tf.Tensor:
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

    # @function

    return _wrapped_multinomial_func(dtype, logits, probs, total_count)


@function(wraps="tensor")
def _wrapped_multinomial_func(dtype, logits, probs, total_count):
    if probs is not None:
        shape = tf.shape(probs)
        probs = znp.reshape(probs, [-1])
    else:
        shape = tf.shape(logits)
        logits = znp.reshape(logits, [-1])
    dist = tfp.distributions.Multinomial(
        total_count=total_count, probs=probs, logits=logits
    )
    counts_flat = dist.sample()
    counts_flat = tf.cast(counts_flat, dtype=dtype)
    counts = znp.reshape(counts_flat, shape)
    return counts


@wraps(tf.random.normal)
def normal(shape, mean=0.0, stddev=1.0, dtype=ztypes.float, name=None):
    return get_prng().normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, name=name
    )


@wraps(tf.random.uniform)
def uniform(shape, minval=0, maxval=None, dtype=ztypes.float, name=None):
    return get_prng().uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=dtype, name=name
    )


@wraps(tf.random.poisson)
def poisson(
    lam: Any,
    shape: Any,
    seed: Any = None,
    dtype: tf.DType = ztypes.float,
    name: Any = None,
):
    if seed is None:
        seed = get_prng().make_seeds(1)[:, 0]
    return tf.random.stateless_poisson(
        lam=lam, seed=seed, shape=shape, dtype=dtype, name=name
    )


def shuffle(value, seed=None, name=None):
    if seed is None:
        seed = get_prng().make_seeds(1)[:, 0]
    return tf.random.experimental.stateless_shuffle(value, seed=seed, name=name)
