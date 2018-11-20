# from .minimizers.optimizers_tf import RMSPropMinimizer, GradientDescentMinimizer, AdagradMinimizer, AdadeltaMinimizer,
import zfit.ztf.wrapping_tf
from .minimizers.optimizers_tf import AdamMinimizer
from .minimizers.minimizer_minuit import MinuitMinimizer
from .minimizers.minimizers_scipy import ScipyMinimizer

# from .minimizers.minimizer_tfp import BFGSMinimizer

# WIP below
if __name__ == '__main__':
    from zfit.core.parameter import Parameter
    import time

    from zfit import ztf

    with ztf.Session() as sess:
        with ztf.variable_scope("func1"):
            a = Parameter("variable_a", ztf.constant(1.5),
                          ztf.constant(-1.),
                          ztf.constant(20.),
                          step_size=ztf.constant(0.1))
            b = Parameter("variable_b", 2.)
            c = Parameter("variable_c", 3.1)
        # minimizer_fn = tfp.optimizer.bfgs_minimize

        # sample = ztf.constant(np.random.normal(loc=1., size=100000), dtype=ztf.float64)
        # # sample = np.random.normal(loc=1., size=100000)
        # def func(par_a, par_b, par_c):
        #     high_dim_func = (par_a - sample) ** 2 + \
        #                     (par_b - sample * 4.) ** 2 + \
        #                     (par_c - sample * 8) ** 4
        #     return ztf.reduce_sum(high_dim_func)
        #

        sample = ztf.constant(np.random.normal(loc=1., scale=0.0003, size=10000), dtype=ztf.float64)


        # sample = np.random.normal(loc=1., size=100000)
        def func():
            high_dim_func = (a - sample) ** 2 * abs(ztf.sin(sample * a + b) + 2) + \
                            (b - sample * 4.) ** 2 + \
                            (c - sample * 8) ** 4 + 1.1
            # high_dim_func = 5*high_dim_func*ztf.exp(high_dim_func + 5)
            # high_dim_func = ztf.exp(high_dim_func**3 + 5*high_dim_func)*ztf.sqrt(high_dim_func - 5)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = ztf.sqrt(high_dim_func + 160.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 20.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            # high_dim_func = ztf.sqrt(high_dim_func + 100.4)
            return ztf.reduce_sum(zfit.ztf.wrapping_tf.log(high_dim_func))


        # a = ztf.constant(9.0, dtype=ztf.float64)
        # with ztf.control_dependencies([a]):
        #     def func(a):
        #         return (a - 1.0) ** 2

        n_steps = 0

        #
        # def test_func(val):
        #     global n_steps
        #     print("alive!", n_steps)
        #     global a
        #     print(a)
        #     n_steps += 1
        #     print(val)
        #     # a = val
        #     with ztf.variable_scope("func1", reuse=True):
        #         var1 = ztf.get_variable(name="variable_a", shape=a.shape, dtype=a.dtype)
        #     with ztf.control_dependencies([val, var1]):
        #         f = func(var1)
        #         # a.assign(val, use_locking=True)
        #         with ztf.control_dependencies([var1]):
        #             # grad = ztf.gradients(f, a)[0]
        #             grad = 2. * (var1 - 1.)  # HACK
        #             return f, grad

        # loss_func = func(par_a=a, par_b=b, par_c=c)
        # loss_func = func()
        loss_func = func
        # with ztf.control_dependencies([a]):
        #     min = tfp.optimizer.bfgs_minimize(test_func,
        #                                       initial_position=ztf.constant(10.0,
        # dtype=ztf.float64))
        # minimizer = ztf.train.AdamOptimizer()

        # min = minimizer.minimize(loss=loss_func, var_list=[a, b, c])
        # minimizer = AdamMinimizer(sess=sess, learning_rate=0.3)
        #########################################################################

        # which_minimizer = 'bfgs'
        which_minimizer = 'minuit'
        # which_minimizer = 'tfminuit'
        # which_minimizer = 'scipy'

        print("Running minimizer {}".format((which_minimizer)))

        if which_minimizer == 'minuit':
            minimizer = MinuitMinimizer(sess=sess)

            init = ztf.global_variables_initializer()
            sess.run(init)

            # for _ in range(5):

            n_rep = 1
            start = time.time()
            for _ in range(n_rep):
                value = minimizer.minimize()  # how many times to be serialized
            end = time.time()
            print("value from calculations:", value)
            print("type:", type(value))
            print("time needed", (end - start) / n_rep)
        ##################################################################
        elif which_minimizer == 'tfminuit':
            loss = loss_func()
            minimizer = MinuitTFMinimizer(loss=loss)

            init = ztf.global_variables_initializer()
            sess.run(init)

            # for _ in range(5):

            n_rep = 1
            start = time.time()
            for _ in range(n_rep):
                value = minimizer.minimize(session=sess)
            end = time.time()

            print("value from calculations:", value)
            print("time needed", (end - start) / n_rep)

        #####################################################################
        elif which_minimizer == 'bfgs':
            test1 = BFGSMinimizer(sess=sess, tolerance=1e-6)

            min = test1.minimize()
            last_val = 100000
            cur_val = 9999999
            # HACK
            loss_func = loss_func()
            # HACK END
            # while abs(last_val - cur_val) > 0.00001:
            start = time.time()
            result = sess.run(min)
            end = time.time()
            print("value from calculations:", result)
            print("time needed", (end - start))
            # last_val = cur_val
            # print("running")

            # cur_val = sess.run(loss_func)
            # aval, bval, cval = sess.run([v for v in (a, b, c)])
            # print("a, b, c", aval, bval, cval)
            # minimizer.minimize(loss=loss_func, var_list=[a, b, c])
            cur_val = sess.run(loss_func)
            result = cur_val
            print(sess.run([v for v in (a, b, c)]))
            print(result)
        #####################################################################

        if which_minimizer == 'scipy':
            func = loss_func()
            train_step = ztf.contrib.opt.ScipyOptimizerInterface(
                func,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-8},
                # optimizer_kwargs={'options': {'ftol': 1e-5}},
                tol=1e-10)

            # minimizer = ScipyMinimizer(loss=func,
            #                            method='L-BFGS-B',
            #                            options={'maxiter': 100})

            # with ztf.Session() as sess:
            sess.run(ztf.global_variables_initializer())

            start = time.time()
            for _ in range(1):
                # print(sess.run(func))
                train_step.minimize()
            result = print(sess.run(func))
            # value = minimizer.minimize(loss=loss_func())  # how many times to be serialized
            end = time.time()
            value = result
            print("value from calculations:", value)
            print(sess.run([v for v in (a, b, c)]))

            print("time needed", (end - start))

        print("Result from minimizer {}".format((which_minimizer)))
