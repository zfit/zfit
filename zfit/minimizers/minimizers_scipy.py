import tensorflow as tf

from zfit.core.minimizer import MinimizerInterface, BaseMinimizer


class ScipyMinimizer(BaseMinimizer):

    def __init__(self, loss, params=None, tolerance=None, name="ScipyMinimizer", **kwargs):
        super().__init__(loss=loss, params=params, tolerance=tolerance, name=name)
        self._scipy_init_kwargs = kwargs

    def _minimize(self, params):
        loss = self.loss.value()
        # var_list = self.get_parameters()
        var_list = params
        # params_name = self._extract_parameter_names(var_list)
        var_to_bounds = {p.name: (p.lower_limit, p.upper_limit) for p in var_list}
        minimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=loss, var_list=var_list,
                                                           var_to_bounds=var_to_bounds,
                                                           **self._scipy_init_kwargs)
        self._scipy_minimizer = minimizer
        result = minimizer.minimize(session=self.sess)
        result_values = result  # TODO change to receive scipy result, change external optimizer interface
        # self._assign_parameters(params=var_list, values=list(result_values))

        # TODO: make better
        edm = -999  # TODO: get from scipy result or how?
        fmin = self.sess.run(loss)  # TODO: get from scipy result
        self.get_state(copy=False)._set_new_state(params=var_list, edm=edm,
                                                  fmin=fmin, status={})
        return "NOTHING! Allow parameter copy or similar. TODO"
        # return self.get_state()
