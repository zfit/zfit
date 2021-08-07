#  Copyright (c) 2021 zfit
from typing import Iterable, Optional, Union

import tensorflow as tf

from .. import z
from ..z import numpy as znp
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitPDF, ZfitData
from ..core.loss import BaseLoss
from ..util import ztyping
from ..util.checks import NONE


def _spd_transform(values, probs, variances):
    # Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    scale = values * tf.math.reciprocal_no_nan(variances)
    return values * scale, probs * scale

class ExtendedBinnedNLL(BaseLoss):

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
            probs = mod.values()
            if variances is not None:
                values, probs = _spd_transform(values, probs, variances=variances)

            poisson_terms.append(tf.nn.log_poisson_loss(values,  # TODO: correct offset
                                                        znp.log(
                                                            probs)) + log_offset)  # TODO: change None
        nll = tf.reduce_sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll

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
