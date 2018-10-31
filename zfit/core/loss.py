
import tensorflow as tf


def unbinned_nll(probs=None, weights=None, log_probs=None):
    """Return unbinned negative log likelihood graph for a PDF

    Args:
        probs (Tensor): The probabilities
        weights (Tensor): Weights of the `probs`
        log_probs (Tensor): The logarithmic probabilites

    Returns:
        graph: the unbinned nll

    Raises:
        ValueError: if both `probs` and `log_probs` are specified.
    """
    if probs is not None and log_probs is not None:
        raise ValueError("Cannot specify 'probs' and 'log_probs'")
    if probs is not None:
        log_probs = tf.log(probs)
    if weights is not None:
        log_probs += tf.log(weights)
    nll = -tf.reduce_sum(log_probs)
    return nll
