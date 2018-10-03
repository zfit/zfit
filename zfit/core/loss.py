from __future__ import print_function, division, absolute_import

import tensorflow as tf


def unbinned_nll(probs=None, weights=None, log_probs=None):
    """Return unbinned negative log likelihood graph for a PDF

    Args:
        probs (graph):
        weights (Tensor): Weights of the `probs`

    Returns:
        graph: the unbinned nll

    Raises:
        ValueError: if both `probs` and `log_probs` are specified.
    """
    if probs and log_probs:
        raise ValueError("Cannot specify 'probs' and 'log_probs'")
    if probs:
        log_probs = tf.log(probs)
    if weights:
        log_probs *= weights
    nll = -tf.reduce_sum(log_probs)
    return nll
