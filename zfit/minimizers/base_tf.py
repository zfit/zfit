#  Copyright (c) 2020 zfit
from contextlib import ExitStack

import tensorflow as tf

from .baseminimizer import BaseMinimizer


class WrapOptimizer(BaseMinimizer):
    # Todo: Write documentation for api.
    def __init__(self, optimizer, tolerance=None, verbosity=None, name=None, **kwargs):
        if tolerance is None:
            tolerance = 1e-5
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError("optimizer {} has to be from class Optimizer".format(str(optimizer)))
        super().__init__(tolerance=tolerance, verbosity=verbosity, name=name, minimizer_options=None, **kwargs)
        self._optimizer_tf = optimizer

    def _step_tf(self, loss, params):
        minimization_step = self._optimizer_tf.minimize(loss=loss, var_list=params)

        return minimization_step
