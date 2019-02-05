import tensorflow as tf

from .baseminimizer import BaseMinimizer


class WrapOptimizer(BaseMinimizer):
    def __init__(self, optimizer, tolerance=None, verbosity=None, name=None, **kwargs):
        if tolerance is None:
            tolerance = 1e-8
        if not isinstance(optimizer, tf.train.Optimizer):
            raise TypeError("optimizer {} has to be from class Optimizer".format(str(optimizer)))
        super().__init__(tolerance=tolerance, verbosity=verbosity, name=name, minimizer_options=None, **kwargs)
        self._optimizer_tf = optimizer

    def _step_tf(self, loss, params):
        loss = loss.value()
        # var_list = self.get_params()
        var_list = params
        minimization_step = self._optimizer_tf.minimize(loss=loss, var_list=var_list)

        # auto-initialize variables from optimizer
        all_params = list(self._optimizer_tf.variables())
        is_initialized = [tf.is_variable_initialized(p) for p in all_params]
        is_initialized = self.sess.run(is_initialized)
        inits = [p.initializer for p, is_init in zip(all_params, is_initialized) if not is_init]
        if inits:
            self.sess.run(inits)

        return minimization_step
