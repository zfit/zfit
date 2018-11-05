import iminuit
import numpy as np
import tensorflow as tf

from zfit.core.minimizer import BaseMinimizer


class MinuitMinimizer(BaseMinimizer):

    # def __init__(self):

    def _minimize(self):
        loss = self.loss
        # params = [p for p in tf.trainable_variables() if p.floating]
        params = self.get_parameters()
        gradients = tf.gradients(loss, params)
        updated_params = [param.update_op for param in params]
        placeholders = [param.placeholder for param in params]
        assign_params = [param.assign for param in params]

        def func(values):

            # updated_params = []
            # feed_dict = collections.OrderedDict()
            # for param, val in zip(params, values):

            feed_dict = {p: v for p, v in zip(placeholders, values)}
            self.sess.run(updated_params, feed_dict=feed_dict)
            # with tf.control_dependencies(updated_params):
            # loss_new = tf.identity(loss)
            loss_new = loss
            loss_evaluated = self.sess.run(loss_new)
                # print("++++++++++++++++++++++++++++++")
            # print(loss_evaluated)
                # print(sess.run(params))
            # print(values)
            return loss_evaluated

        def grad_func(values):
            updated_params = []
            # for param, val in zip(params, values):
            #     updated_params.append(tf.assign(param, ztf.to_float(val)))
            # gradients1 = tf.identity(gradients)
            gradients1 = gradients
            gradients_values = self.sess.run(gradients1)
            # print(gradients_values)
            return gradients_values

        # HACK
        # func = grad_func
        # HACK END
        minimizer = iminuit.Minuit(fcn=func, use_array_call=True,
                                   grad=grad_func,
                                   forced_parameters="abc",
                                   error_a=0.1,
                                   error_b=0.1,
                                   error_c=0.1,
                                   # limit_a=(-1, 5), a=2.5,
                                   # limit_b=(-1, 8), b=6,
                                   # limit_c=(-3, 12), c=6,
                                   )
        result = minimizer.migrad(ncall=10000, nsplit=1, precision=1e-8)
        params = [p_dict['value'] for p_dict in result[1]]
        self.sess.run([assign(p) for assign, p in zip(assign_params, params)])
        return params


class MinuitTFMinimizer(tf.contrib.opt.ExternalOptimizerInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                  equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                  packed_bounds, step_callback, optimizer_kwargs):

        # @functools.lru_cache()
        def loss_grad_func_wrapper(x):
            # Minuit should work with float64
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')

        minimize_args = [loss_grad_func_wrapper, initial_val]
        minimize_kwargs = {}

        def wrapped_loss_func(x):
            return loss_grad_func_wrapper(x=x)[0]

        def wrapped_loss_grad_func(x):
            return loss_grad_func_wrapper(x=x)[1]


        minimizer = iminuit.Minuit(fcn=wrapped_loss_func, use_array_call=True,
                                   grad=wrapped_loss_grad_func,
                                   forced_parameters="abc",
                                   error_a=0.1,
                                   error_b=0.1,
                                   error_c=0.1,
                                   # limit_a=(-1, 5), a=2.5,
                                   # limit_b=(-1, 8), b=6,

                                   # limit_c=(-3, 12), c=6,
                                   )
        result = minimizer.migrad(ncall=10000, nsplit=8, precision=1e-8)
        params = [p_dict['value'] for p_dict in result[1]]
        return np.array(params)
