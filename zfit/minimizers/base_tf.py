#  Copyright (c) 2019 zfit
from contextlib import ExitStack

import tensorflow as tf

from .baseminimizer import BaseMinimizer


class WrapOptimizer(BaseMinimizer):
    def __init__(self, optimizer, tolerance=None, verbosity=None, name=None, **kwargs):
        if tolerance is None:
            tolerance = 1e-8
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError("optimizer {} has to be from class Optimizer".format(str(optimizer)))
        super().__init__(tolerance=tolerance, verbosity=verbosity, name=name, minimizer_options=None, **kwargs)
        self._optimizer_tf = optimizer

    def _step_tf(self, loss, params):
        # loss = loss.value()
        # var_list = self.get_params()
        var_list = params
        with ExitStack() as stack:
            _ = [stack.enter_context(param._hack_set_tf_name()) for param in params]
            minimization_step = self._optimizer_tf.minimize(loss=loss, var_list=var_list)

        return minimization_step
