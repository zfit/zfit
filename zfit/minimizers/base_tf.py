import tensorflow as tf

from zfit.core.minimizer import BaseMinimizer


class HelperAdapterTFOptimizer(object):
    """Adapter for tf.Optimizer to convert the step-by-step minimization to full minimization"""

    def __init__(self, *args, **kwargs):  # self check
        # assert issubclass(self.__class__, tf.train.Optimizer)  # assumption
        super(HelperAdapterTFOptimizer, self).__init__(*args, **kwargs)


    def _step_tf(self):
        """One step of the minimization. Equals to `tf.train.Optimizer.minimize`

        Args:
            loss (graph): The loss function to minimize
            var_list (list(tf.Variable...)): A list of tf.Variables that will be optimized.


        """
        init = tf.global_variables_initializer()
        self.sess.run(init)
        init = tf.initialize_all_variables()
        self.sess.run(init)
        loss = self.loss
        var_list = self.get_parameters()
        minimum = super(HelperAdapterTFOptimizer, self).minimize(loss=loss,
                                                                 var_list=var_list)

        return minimum


class AdapterTFOptimizer(BaseMinimizer, HelperAdapterTFOptimizer):
    def _hook_minimize(self):
        optimizer_vars = list(self.variables())
        # HACK
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #HACK END
        all_params = self.get_parameters(only_floating=False) + optimizer_vars
        is_initialized = [tf.is_variable_initialized(p) for p in all_params]
        is_initialized = self.sess.run(is_initialized)
        inits = [p.initializer for p, is_init in zip(all_params, is_initialized) if not is_init]
        if inits:
            self.sess.run(inits)
        return super()._hook_minimize()
