#  Copyright (c) 2021 zfit
from typing import Iterable

from .. import z
from ..core.interfaces import ZfitBinnedPDF, ZfitBinnedData
from ..core.loss import BaseLoss
from ..util import ztyping

import tensorflow as tf


class ExtendedBinnedNLL(BaseLoss):

    def __init__(self, model: ztyping.ModelsInputType, data: ztyping.DataInputType,
                 constraints: ztyping.ConstraintsTypeInput = None):
        self._errordef = 0.5
        super().__init__(model=model, data=data, constraints=constraints, fit_range=None)


    @z.function(wraps='loss')
    def _loss_func(self, model: Iterable[ZfitBinnedPDF], data: Iterable[ZfitBinnedData],
                   fit_range, constraints):
        poisson_terms = []
        for mod, dat in zip(model, data):
            poisson_terms.append(tf.nn.log_poisson_loss(dat.get_counts(obs=mod.obs),
                                                        tf.math.log(mod.ext_pdf(None))))  # TODO: change None
        nll = tf.reduce_sum(poisson_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints

        return nll
