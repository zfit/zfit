#  Copyright (c) 2021 zfit
from typing import Iterable, Optional, Union

import tensorflow as tf

from .. import z
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitPDF, ZfitData
from ..core.loss import BaseLoss
from ..util import ztyping
from ..util.checks import NONE
from ..z import numpy as znp


def _spd_transform(values, probs, variances):
    # Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    scale = values * tf.math.reciprocal_no_nan(variances)
    return values * scale, probs * scale


@z.function(wraps='tensor')
def poisson_loss_calc(probs, values, log_offset, variances=None):
    if variances is not None:
        values, probs = _spd_transform(values, probs, variances=variances)
    poisson_term = tf.nn.log_poisson_loss(values,  # TODO: correct offset
                                          znp.log(
                                              probs)) + log_offset
    return poisson_term


class BaseBinnedNLL(BaseLoss):
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


class ExtendedBinnedNLL(BaseBinnedNLL):

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
        nll = tf.reduce_sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll


class BinnedNLL(BaseBinnedNLL):

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
            poisson_terms.append(poisson_term)  # TODO: change None
        nll = znp.sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll
