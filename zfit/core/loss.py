import abc

import pep487
import tensorflow as tf
from typing import Optional

import zfit
from .limits import convert_to_range, Range
import zfit.util.checks


def unbinned_nll(pdf, data, fit_range, constraints: Optional[dict] = None) -> tf.Tensor:
    """Return unbinned negative log likelihood graph for a PDF

    Args:
        fit_range ():
        pdf (Tensor): The probabilities
        data (Tensor): Weights of the `probs`
        constraints (dict): A dictionary containing the constraints for certain parameters. The key
            is the parameter while the value is a pdf with at least a `prob(x)` method.

    Returns:
        graph: the unbinned nll

    Raises:
        ValueError: if both `probs` and `log_probs` are specified.
    """
    if constraints is None:
        constraints = {}
    if zfit.util.checks.isiterable(pdf):
        if not len(pdf) == len(data) == len(fit_range):  # TODO: pay attention to fit_range, what if just one range?
            raise ValueError("The number of given pdfs, data and fit_ranges do not match.")
        else:
            if not isinstance(fit_range[0], Range):
                raise ValueError("If several pdfs are given, the ranges in `fit_range` have to be a "
                                 "`Range` and not just tuples (because of disambiguity).")
            nlls = [unbinned_nll(pdf=p, data=d, fit_range=r, constraints=constraints)
                    for p, d, r in zip(pdf, data, fit_range)]
            nll_finished = tf.reduce_sum(nlls)
    else:  # TODO: complicated limits?
        fit_range = convert_to_range(fit_range, dims=Range.FULL)
        limits = fit_range.get_boundaries()
        assert len(limits[0]) == 1, "multiple limits not (yet) supported in nll."
        (lower,), (upper,) = limits

        in_limits = tf.logical_and(lower <= data, data <= upper)
        data = tf.boolean_mask(tensor=data, mask=in_limits)
        log_probs = tf.log(pdf.prob(data, norm_range=fit_range))
        nll = -tf.reduce_sum(log_probs)
        if constraints:
            constraints_log_prob = tf.reduce_sum([tf.log(dist.prob(param)) for param, dist in constraints.items()])
            nll -= constraints_log_prob
        nll_finished = nll
    return nll_finished



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

class LossInterface(pep487.ABC):

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_constraint(self, constraint):
        raise NotImplementedError

    @abc.abstractmethod
    def errordef(self, func):
        raise NotImplementedError

    @property
    # @abc.abstractmethod
    def pdf(self):
        raise NotImplementedError

    @property
    # @abc.abstractmethod
    def data(self):
        raise NotImplementedError

    @property
    # @abc.abstractmethod
    def fit_range(self):
        raise NotImplementedError


class BaseLoss(LossInterface):

    def __init__(self, pdf, data, fit_range, constraints=None):
        # constraints = {} if constraints == {} else constraints # protect mutable argument
        self._simultaneous = None
        if constraints is None:
            constraints = {}
        pdf, data, fit_range = self._input_check(pdf, data, fit_range)
        self._pdf = pdf
        self._data = data
        self._fit_range = fit_range
        self._constraints = constraints

    def __init_subclass__(cls, **kwargs):
        cls._name = "UnnamedSubBaseLoss"

    def _input_check(self, pdf, data, fit_range):
        # TODO
        return pdf, data, fit_range

    def add_constraint(self, constraint):
        if not isinstance(constraint, dict):
            raise TypeError("`constraint` has to be a dict, is currently {}".format(type(constraint)))
        overwritting_keys = set(constraint).intersection(self._constraints)
        if overwritting_keys:
            raise ValueError("Cannot change existing constraints but only add (currently). Constrain for "
                             "parameter(s) {} already there.".format(overwritting_keys))
        self._constraints.update(constraint)

    @property
    def name(self):
        return self._name

    @property
    def pdf(self):
        return self._pdf

    @property
    def data(self):
        return self._data

    @property
    def fit_range(self):
        return self._fit_range

    @property
    def constraints(self):
        return self._constraints

    @abc.abstractmethod
    def _eval(self):
        raise NotImplementedError

    def eval(self):
        try:
            return self._eval(pdf=self.pdf, data=self.data, fit_range=self.fit_range, constraints=self.constraints)
        except NotImplementedError:
            raise NotImplementedError("_eval not defined!")


def errordef_nll(sigma):
    return sigma

class UnbinnedNLL(BaseLoss):

    _name = "unbinned_nll"

    def _eval(self, pdf, data, fit_range, constraints):
        return unbinned_nll(pdf=pdf, data=data, fit_range=fit_range, constraints=constraints)

    def errordef(self, sigma):
        return errordef_nll(sigma)




