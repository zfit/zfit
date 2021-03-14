#  Copyright (c) 2021 zfit
from typing import Optional, Iterable

import tensorflow as tf

from .baseminimizer import BaseStepMinimizer, minimize_supports
from ..core.interfaces import ZfitIndependentParameter, ZfitLoss
from ..util.exception import OperationNotAllowedError


class WrapOptimizer(BaseStepMinimizer):
    # Todo: Write documentation for api.
    def __init__(self,
                 optimizer,
                 tol=None,
                 criterion=None,
                 strategy=None,
                 verbosity=None,
                 name=None,
                 **kwargs):
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError("optimizer {} has to be from class Optimizer".format(str(optimizer)))
        super().__init__(tol=tol, criterion=criterion, strategy=strategy, verbosity=verbosity, name=name,
                         minimizer_options=None, **kwargs)
        self._optimizer_tf = optimizer

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init):
        try:
            return super()._minimize(loss, params, init)
        except ValueError as error:
            if 'No gradients provided for any variable' in error.args[0]:
                raise OperationNotAllowedError("Cannot use TF optimizer with"
                                               " a numerical gradient (non-TF function)") from None
            else:
                raise

    def _step(self,
              loss: ZfitLoss,
              params: Iterable[ZfitIndependentParameter],
              init: Optional["zfit.result.FitResult"]
              ) -> tf.Tensor:
        self._optimizer_tf.minimize(loss=loss.value, var_list=params)
        return loss.value()
