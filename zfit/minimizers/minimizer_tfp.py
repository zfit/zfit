import tensorflow as tf
import tensorflow_probability as tfp

from .baseminimizer import BaseMinimizer


class BFGSMinimizer(BaseMinimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def minimize(self, sess=None, params=None):
        # with tf.device("/cpu:0"):
        with tf.name_scope("inside_minimization") as scope:
            # var_a = tf.get_variable
            sess = sess or self.sess
            minimizer_fn = tfp.optimizer.bfgs_minimize

            def to_minimize_func(values):
                # tf.Print(value, [value])
                # print("============value", value)

                # def update_one(param_value):
                #     param, value = param_value
                #     param.update(value=value, session=sess)
                # print("============one param", params[0])
                tf.get_variable_scope().reuse_variables()
                params = [p for p in tf.trainable_variables() if p.floating]
                # params = [g_v[1] for g_v in tfe.implicit_gradients(func)]
                # params = var_list
                tf.get_variable_scope().reuse_variables()
                printed = tf.Print(params, [params], "parameters")

                with tf.control_dependencies([values, printed]):
                    updated_values = []
                    for param, val in zip(params, ztf.unstack_x(values)):
                        updated_values.append(tf.assign(param, value=val))
                    with tf.control_dependencies(updated_values):
                        func_graph = func()
                        # func_graph = tf.identity(func)
                        # tf.get_variable_scope().reuse_variables()
                        with tf.control_dependencies([func_graph]):
                            printed2 = tf.Print(func_graph, [func_graph], "parameters")
                            with tf.control_dependencies([printed2, func_graph]):
                                return func_graph, tf.stack(tf.gradients(func_graph, params))

            params = [p for p in tf.trainable_variables() if p.floating]
            result = minimizer_fn(to_minimize_func,
                                  initial_position=self._extract_start_values(params),
                                  tolerance=self.tolerance, parallel_iterations=1)

        return result
