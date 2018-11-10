import tensorflow as tf
from typing import Optional


def unbinned_nll(probs: tf.Tensor = None, weights: Optional[tf.Tensor] = None, log_probs: Optional[tf.Tensor] = None,
                 constraints: Optional[dict] = None) -> tf.Tensor:
    """Return unbinned negative log likelihood graph for a PDF

    Args:
        probs (Tensor): The probabilities
        weights (Tensor): Weights of the `probs`
        log_probs (Tensor): The logarithmic probabilites
        constraints (dict): A dictionary containing the constraints for certain parameters. The key
            is the parameter while the value is a pdf with at least a `prob(x)` method.

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
    if constraints:
        constraints_log_prob = tf.reduce_sum([tf.log(dist.prob(param)) for param, dist in constraints.items()])
        nll -= constraints_log_prob
    return nll

#
# def extended_unbinned_NLL(pdfs, integrals, n_obs, nsignals,
#                           param_gauss=None, param_gauss_mean=None, param_gauss_sigma=None,
#                           log_multi_gauss=None):
#     """
#     Return unbinned negative log likelihood graph for a PDF
#     pdfs       : concatenated array of several PDFs (different regions/channels)
#     integrals  : array of precalculated integrals of the corresponding pdfs
#     n_obs       : array of observed num. of events, used in the extended fit and in the
#     normalization of the pdf
#                  (needed since when I concatenate the pdfs I loose the information on how many
#                  data points are fitted with the pdf)
#     nsignals   : array of fitted number of events resulted from the extended fit (function of the
#     fit parameters, prop to BR)
#     param_gauss : list of parameter to be gaussian constrained (CKM pars, etc.)
#     param_gauss_mean : mean of parameter to be gaussian constrained
#     param_gauss_sigma : sigma parameter to be gaussian constrained
#     log_multi_gauss : log of the multi-gaussian to be included in the Likelihood (FF & alphas)
#     """
#     # tf.add_n(log(pdf(x))) - tf.add_n(Nev*Norm)
#     nll = - (tf.reduce_sum(tf.log(pdfs)) - tf.reduce_sum(
#         tf.cast(n_obs, tf.float64) * tf.log(integrals)))
#
#     # Extended fit to number of events
#     nll += - tf.reduce_sum(-nsignals + tf.cast(n_obs, tf.float64) * tf.log(nsignals))
#
#     # gaussian constraints on parameters (CKM) # tf.add_n( (par-mean)^2/(2*sigma^2) )
#     if param_gauss is not None:
#         nll += tf.reduce_sum(
#             tf.square(param_gauss - param_gauss_mean) / (2. * tf.square(param_gauss_sigma)))
#
#     # multivariate gaussian constraints on param that have correlations (alphas, FF)
#     if log_multi_gauss is not None:
#         nll += - log_multi_gauss
#
#     return nll
