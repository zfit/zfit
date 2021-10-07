#  Copyright (c) 2021 zfit
from typing import Iterable, Optional, Union, Set

import numpy as np
import tensorflow as tf

from .. import z
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitPDF, ZfitData
from ..core.loss import BaseLoss
from ..util import ztyping
from ..util.checks import NONE
from ..util.warnings import warn_advanced_feature
from ..z import numpy as znp


@z.function(wraps='tensor')
def _spd_transform(values, probs, variances):
    # Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    scale = values * tf.math.reciprocal_no_nan(variances)
    return values * scale, probs * scale


@z.function(wraps='tensor')
def poisson_loss_calc(probs, values, log_offset=None, variances=None):
    if variances is not None:
        values, probs = _spd_transform(values, probs, variances=variances)
    values += znp.asarray(1e-307, dtype=znp.float64)
    probs += znp.asarray(1e-307, dtype=znp.float64)
    poisson_term = tf.nn.log_poisson_loss(values,  # TODO: correct offset
                                          znp.log(
                                              probs), compute_full_loss=False)  # TODO: optimization?
    if log_offset is not None:
        poisson_term += log_offset
    return poisson_term


class BaseBinned(BaseLoss):
    def create_new(self,
                   model: Optional[Union[ZfitPDF, Iterable[ZfitPDF]]] = NONE,
                   data: Optional[Union[ZfitData, Iterable[ZfitData]]] = NONE,
                   constraints=NONE,
                   options=NONE):
        if model is NONE:
            model = self.model
        if data is NONE:
            data = self.data
        if constraints is NONE:
            constraints = self.constraints
            if constraints is not None:
                constraints = constraints.copy()
        if options is NONE:
            options = self._options
            if isinstance(options, dict):
                options = options.copy()
        return type(self)(model=model, data=data, constraints=constraints, options=options)


class ExtendedBinnedNLL(BaseBinned):

    def __init__(self, model: ztyping.ModelsInputType, data: ztyping.DataInputType,
                 constraints: ztyping.ConstraintsTypeInput = None, options=None):
        self._errordef = 0.5
        super().__init__(model=model, data=data, constraints=constraints, fit_range=None, options=options)

    @z.function(wraps='loss')
    def _loss_func(self, model: Iterable[ZfitBinnedPDF], data: Iterable[ZfitBinnedData],
                   fit_range, constraints, log_offset):
        poisson_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            probs = mod.counts(dat)
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            poisson_terms.append(poisson_term)  # TODO: change None
        nll = znp.sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll

    @property
    def is_extended(self):
        return True

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:

        return super()._get_params(floating, is_yield, extract_independent)


class BinnedNLL(BaseBinned):

    def __init__(self, model: ztyping.ModelsInputType, data: ztyping.DataInputType,
                 constraints: ztyping.ConstraintsTypeInput = None, options=None):
        self._errordef = 0.5
        super().__init__(model=model, data=data, constraints=constraints, fit_range=None, options=options)

    @z.function(wraps='loss')
    def _loss_func(self, model: Iterable[ZfitBinnedPDF], data: Iterable[ZfitBinnedData],
                   fit_range, constraints, log_offset):
        poisson_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            probs = mod.rel_counts(dat)
            probs *= znp.sum(values)
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            poisson_terms.append(poisson_term)
        nll = znp.sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll

    @property
    def is_extended(self):
        return False

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent)


@z.function(wraps='tensor')
def chi2_loss_calc(probs, values, variances, log_offset=None, ignore_empty=None):
    if ignore_empty is None:
        ignore_empty = True
    chi2_term = tf.math.squared_difference(probs, values)
    if ignore_empty:
        one_over_var = tf.math.reciprocal_no_nan(variances)
    else:
        one_over_var = tf.math.reciprocal(variances)
    chi2_term *= one_over_var
    chi2_term = znp.sum(chi2_term)
    if log_offset is not None:
        chi2_term += log_offset
    return chi2_term


def _check_small_counts_chi2(data, ignore_empty):
    for dat in data:
        variances = dat.variances()
        smaller_than_six = dat.values() < 6
        if variances is None:
            raise ValueError(f"variances cannot be None for Chi2: {dat}")
        elif np.any(variances <= 0) and not ignore_empty:
            raise ValueError(f"Variances of {dat} contains zeros or negative numbers, cannot calculate chi2."
                             f" {variances}")
        elif np.any(smaller_than_six):
            warn_advanced_feature(f"Some values in {dat} are < 6, the chi2 assumption of gaussian distributed"
                                  f" uncertainties most likely won't hold anymore. Use Chi2 for large samples."
                                  f"For smaller samples, consider using (Extended)BinnedNLL (or an unbinned fit).",
                                  identifier='chi2_counts_small')


class BinnedChi2(BaseBinned):
    def __init__(self, model: ztyping.ModelsInputType, data: ztyping.DataInputType,
                 constraints: ztyping.ConstraintsTypeInput = None, options=None):
        self._errordef = 1.
        super().__init__(model=model, data=data, constraints=constraints, fit_range=None, options=options)

    def _precompile(self):
        super()._precompile()
        ignore_empty = self._options.get('ignore_empty', True)
        data = self.data
        _check_small_counts_chi2(data, ignore_empty)

    @z.function(wraps='loss')
    def _loss_func(self, model: Iterable[ZfitBinnedPDF], data: Iterable[ZfitBinnedData],
                   fit_range, constraints, log_offset):
        del fit_range
        ignore_empty = self._options.get('ignore_empty', True)
        chi2_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            if variances is None:
                raise ValueError(f"variances cannot be None for Chi2: {dat}")
            probs = mod.rel_counts(dat)
            probs *= znp.sum(values)
            chi2_term = chi2_loss_calc(probs, values, variances, log_offset, ignore_empty=ignore_empty)
            chi2_terms.append(chi2_term)
        chi2_term = znp.sum(chi2_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            chi2_term += constraints

        return chi2_term

    @property
    def is_extended(self):
        return False

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent)


class ExtendedBinnedChi2(BaseBinned):
    def __init__(self, model: ztyping.BinnedPDFInputType, data: ztyping.BinnedDataInputType,
                 constraints: ztyping.ConstraintsTypeInput = None, options=None):
        self._errordef = 1.
        super().__init__(model=model, data=data, constraints=constraints, fit_range=None, options=options)

    def _precompile(self):
        super()._precompile()
        ignore_empty = self._options.get('ignore_empty', True)
        data = self.data
        _check_small_counts_chi2(data, ignore_empty)

    @z.function(wraps='loss')
    def _loss_func(self, model: Iterable[ZfitBinnedPDF], data: Iterable[ZfitBinnedData],
                   fit_range, constraints, log_offset):
        del fit_range
        ignore_empty = self._options.get('ignore_empty', True)
        chi2_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            if variances is None:
                raise ValueError(f"variances cannot be None for Chi2: {dat}")
            probs = mod.counts(dat)
            chi2_term = chi2_loss_calc(probs, values, variances, log_offset, ignore_empty=ignore_empty)
            chi2_terms.append(chi2_term)
        chi2_term = znp.sum(chi2_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            chi2_term += constraints

        return chi2_term

    @property
    def is_extended(self):
        return True
