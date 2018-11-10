"""
Definition of minimizers, wrappers etc.

"""

import abc
import collections
from collections import OrderedDict
import contextlib

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pep487
from typing import Dict, List, Union, Optional

import zfit.core.math as zmath
from zfit import ztf
from zfit.minimizers.state import MinimizerState
from zfit.util import ztyping


class MinimizerInterface(object):
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self, params=None, sess=None):
        raise NotImplementedError

    def _minimize(self, params):
        raise NotImplementedError

    def _minimize_with_step(self, params):
        raise NotImplementedError

    @abc.abstractmethod
    def edm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fmin(self):
        raise NotImplementedError

    def step(self, params=None, sess=None):
        raise NotImplementedError

    def _step_tf(self, params):
        raise NotImplementedError

    def _step(self, params):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tolerance(self):
        raise NotImplementedError

    def _tolerance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def hesse(self, params=None, sess=None):
        raise NotImplementedError

    def _hesse(self, params):
        raise NotImplementedError

    @abc.abstractmethod
    def error(self, params=None, sigma=1., sess=None):
        raise NotImplementedError

    @abc.abstractmethod
    def set_error_method(self, method):
        raise NotImplementedError

    @abc.abstractmethod
    def set_error_options(self, replace=False, **options):
        raise NotImplementedError


def _raises_error_method(*_, **__):
    raise NotImplementedError("No error method specified or implemented as default")


class BaseMinimizer(MinimizerInterface, pep487.PEP487Object):
    _DEFAULT_name = "BaseMinimizer"

    def __init__(self, loss, params=None, tolerance=None, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if name is None:
            name = self._DEFAULT_name
        self._current_error_method = self._error_methods.get('default', _raises_error_method)
        self._current_error_options = {}
        self._minimizer_state = MinimizerState()
        self.name = name
        self.tolerance = tolerance
        self._sess = None
        self._params = OrderedDict()
        self.loss = loss
        self.set_params(params)

    def __init_subclass__(cls, **kwargs):
        cls._error_methods = {'default': None}  # TODO: set default error method

    @property
    def edm(self):
        """The estimated distance to the minimum.

        Returns:
            numeric
        """
        return self.get_state(copy=False).edm

    @property
    def fmin(self):
        """Function value at the minimum.

        Returns:
            numeric
        """
        return self.get_state(copy=False).fmin

    def hesse(self, params: ztyping.ParamsOrNameType = None, sess: ztyping.SessionType = None) -> Dict:
        """Calculate and set for `params` the symmetric error using the Hessian matrix.

        Args:
            params (list(`zfit.FitParameters` or str)): The parameters or their names to calculate the
                Hessian symmetric error.
            sess (`tf.Session` or None): A TensorFlow session to use for the calculation.

        Returns:
            `dict`: A `dict` containing as keys the parameter names and as values a `dict` which
                contains (next to probably more things) a key 'error', holding the calculated error.
                Example: result['par1']['error'] -> the symmetric error on 'par1'
        """
        params = self._check_input_params(params)

        with self._temp_sess(sess=sess):
            errors = self._hesse(params=params)
            for param in params:
                param.error = errors[param.name]['error']

            return errors

    def _check_input_params(self, params, only_floating=True):
        if isinstance(params, (str, tf.Variable)) or (not hasattr(params, "__len__") and params is not None):
            params = [params, ]
        if params is None or isinstance(params[0], str):
            params = self.get_parameters(names=params, only_floating=only_floating)
        return params

    def error(self, params: ztyping.ParamsOrNameType = None, sess: ztyping.SessionType = None) -> Dict:
        """Calculate and set for `params` the asymmetric error using the set error method.

        Args:
            params (list(`zfit.FitParameters` or str)): The parameters or their names to calculate the
                 errors. If `params` is `None`, use all *floating* parameters.
            sess (`tf.Session` or None): A TensorFlow session to use for the calculation.

        Returns:
            `dict`: A `dict` containing as keys the parameter names and as values a `dict` which
                contains (next to probably more things) two keys 'lower_error' and 'upper_error',
                holding the calculated errors.
                Example: result['par1']['upper_error'] -> the asymmetric upper error of 'par1'
        """
        params = self._check_input_params(params)
        with self._temp_sess(sess=sess):
            error_method = self._current_error_method
            errors = error_method(params=params, **self._current_error_options)
            for param in params:
                param.lower_error = errors[param.name]['lower_error']
                param.upper_error = errors[param.name]['upper_error']
            return errors

    def set_error_method(self, method):
        if isinstance(method, str):
            try:
                method = self._error_methods[method]
            except AttributeError:
                raise AttributeError("The error method '{}' is not registered with the minimizer.".format(method))
        elif callable(method):
            self._current_error_method = method
        else:
            raise ValueError("Method {} neither a valid method name nor a callable function.".format(method))

    def set_error_options(self, replace: bool = False, **options):
        """Specify the options for the `error` calculation.

        Args:
            replace (bool): If True, replace the current options. If False, only update
                (add/overwrite existing).
            **options (keyword arguments): The keyword arguments that will be given to `error`.
        """
        if replace:
            self._current_error_options = {}
        self._current_error_options.update(options)

    def get_state(self, copy: bool = True) -> MinimizerState:
        """Return the current state containing parameters, edm, fmin etc.

        Args:
            copy (bool): If True, return a copy of the (internal) state. Equivalent to a fit result.
                If False, a reference is returned and the object will change if another method of
                the minimizer is invoked (such as `error`, `minimize` etc.)

        Returns:

        """
        import copy as copy_module
        state = self._minimizer_state
        if copy:
            state = copy_module.deepcopy(state)
        return state

    def set_params(self, params: ztyping.ParamsType, update: bool = False):
        """Set the parameters of the minimizer.

        If `None`, the parameters are empty. If a dictionary is given,

        Args:
            params (list, dict, None): The parameters.
            update (bool): If True, add (or overwrite if existing) the `params` to the currently
                stored parameters.
        """
        if params is None:
            if update:
                raise ValueError("Cannot specify `None` as params *and* set `update` to True.")
            self._params = OrderedDict()
            return
        if not hasattr(params, "__len__"):
            params = (params,)

        if not update:  # overwrite: create empty instance
            self._params = OrderedDict()
        if isinstance(params, dict):
            self._params.update(params)
        else:
            for param in params:
                self._params[param.name] = param

    @contextlib.contextmanager
    def _temp_set_parameters(self, params):
        old_params = self._params
        try:
            self.set_params(params)
            yield params
        finally:
            self.set_params(old_params)

    @contextlib.contextmanager
    def _temp_sess(self, sess):
        old_sess = self.sess
        try:
            self.sess = sess
            yield sess
        finally:
            self.sess = old_sess

    def get_parameters(self, names: Optional[Union[List[str], str]] = None,
                       only_floating: bool = True) -> List['FitParameter']:  # TODO: automatically set?
        """Return the parameters. If it is empty, automatically set and return all trainable variables.

        Args:
            names (str, list(str)): The names of the parameters to return.
            only_floating (bool): If True, return only the floating parameters.

        Returns:
            list(`zfit.FitParameters`):
        """
        if not self._params:
            self.set_params(params=tf.trainable_variables())
        if isinstance(names, str):
            names = (names,)
        if names is not None:
            missing_names = set(names).difference(self._params.keys())
            if missing_names:
                raise KeyError("The following names are not valid parameter names")
            params = [self._params[name] for name in names]
        else:
            params = list(self._params.values())

        if only_floating:
            params = self._filter_trainable_params(params=params)
        return params

    @staticmethod
    def _filter_trainable_params(params):
        params = [param for param in params if param.floating]
        return params

    @staticmethod
    def _extract_update_op(params):
        params_update = [param.update_op for param in params]
        return params_update

    @staticmethod
    def _extract_assign_method(params):
        params_assign = [param.assign for param in params]
        return params_assign

    @staticmethod
    def _extract_parameter_names(params):
        names = [param.name for param in params]
        return names

    def _assign_parameters(self, params, values):
        params_assign_op = [param.assign(val) for param, val in zip(params, values)]
        return self.sess.run(params_assign_op)

    def _update_parameters(self, params, values):
        feed_dict = {param.placeholder: val for param, val in zip(params, values)}
        return self.sess.run(self._extract_update_op(params), feed_dict=feed_dict)

    @property
    def sess(self):
        # TODO: return default? or error?
        return self._sess

    @sess.setter
    def sess(self, sess):
        self._sess = sess

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance

    @staticmethod
    def _extract_start_values(params):
        """Extract the current value if defined, otherwise random.

        Arguments:
            params (FitParameter):

        Return:
            list(const): the current values of parameters
        """
        values = [p.read_value() for p in params]
        # TODO: implement if initial val not given
        return values

    def step(self, params: ztyping.ParamsOrNameType = None):
        """Perform a single step in the minimization (if implemented).

        Args:
            params ():

        Returns:

        Raises:
            NotImplementedError: if the `step` method is not implemented in the minimizer.
        """
        params = self._check_input_params(params)
        return self._step(params=params)

    def minimize(self, params: ztyping.ParamsOrNameType = None, sess: ztyping.SessionType = None) -> "TODO":
        """Fully minimize the `loss` with respect to `params` using `sess`.

        Args:
            params (list(str) or list(`zfit.FitParameter`): The parameters with respect to which to
                minimize the `loss`.
            sess (`tf.Session`): The session to use.

        Returns:
            `MinimizerState`: The status of the minimizer, equivalent to a fit result.
        """
        with self._temp_sess(sess=sess):
            params = self._check_input_params(params)
            return self._hook_minimize(params=params)

    def _hook_minimize(self, params):
        return self._call_minimize(params=params)

    def _call_minimize(self, params):
        try:
            return self._minimize(params=params)
        except NotImplementedError as error:
            try:
                return self._minimize_with_step(params=params)
            except NotImplementedError:
                raise error

    def _minimize_with_step(self, params):  # TODO improve
        changes = collections.deque(np.ones(10))
        last_val = -10
        cur_val = 9999999
        try:
            step = self._step_tf(params=params)
        except NotImplementedError:
            step_fn = self.step
        else:
            def step_fn(params):
                return self.sess.run(step)
        while sum(sorted(changes)[-3:]) > self.tolerance:  # TODO: improve condition
            _ = step_fn(params=params)
            changes.popleft()
            changes.append(abs(cur_val - last_val))
            last_val = cur_val
            cur_val = self.sess.run(self.loss.eval())  # TODO: improve...
        fmin = cur_val
        edm = -999  # TODO: get edm
        status = {}  # TODO: create status

        self.get_state(copy=False)._set_new_state(params=params, edm=edm, fmin=fmin, status=status)
        # HACK remove the line, `get_state()` currently cannot return Tensors
        return fmin
        # HACK END
        return self.get_state()


# WIP below
if __name__ == '__main__':
    from zfit.core.parameter import FitParameter
    from zfit.minimizers.minimizer_minuit import MinuitMinimizer, MinuitTFMinimizer
    from zfit.minimizers.minimizer_tfp import BFGSMinimizer

    import time

    with tf.Session() as sess:
        with tf.variable_scope("func1"):
            a = FitParameter("variable_a", ztf.constant(1.5),
                             ztf.constant(-1.),
                             ztf.constant(20.),
                             step_size=ztf.constant(0.1))
            b = FitParameter("variable_b", 2.)
            c = FitParameter("variable_c", 3.1)
        minimizer_fn = tfp.optimizer.bfgs_minimize

        # sample = tf.constant(np.random.normal(loc=1., size=100000), dtype=tf.float64)
        # # sample = np.random.normal(loc=1., size=100000)
        # def func(par_a, par_b, par_c):
        #     high_dim_func = (par_a - sample) ** 2 + \
        #                     (par_b - sample * 4.) ** 2 + \
        #                     (par_c - sample * 8) ** 4
        #     return tf.reduce_sum(high_dim_func)
        #

        sample = tf.constant(np.random.normal(loc=1., scale=0.0003, size=10000), dtype=tf.float64)


        # sample = np.random.normal(loc=1., size=100000)
        def func():
            high_dim_func = (a - sample) ** 2 * abs(tf.sin(sample * a + b) + 2) + \
                            (b - sample * 4.) ** 2 + \
                            (c - sample * 8) ** 4 + 1.1
            # high_dim_func = 5*high_dim_func*tf.exp(high_dim_func + 5)
            # high_dim_func = tf.exp(high_dim_func**3 + 5*high_dim_func)*tf.sqrt(high_dim_func - 5)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4) **3
            # high_dim_func = tf.sqrt(high_dim_func + 160.4)
            # high_dim_func = tf.sqrt(high_dim_func + 20.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            # high_dim_func = tf.sqrt(high_dim_func + 100.4)
            return tf.reduce_sum(tf.log(high_dim_func))


        # a = tf.constant(9.0, dtype=tf.float64)
        # with tf.control_dependencies([a]):
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
        #     with tf.variable_scope("func1", reuse=True):
        #         var1 = tf.get_variable(name="variable_a", shape=a.shape, dtype=a.dtype)
        #     with tf.control_dependencies([val, var1]):
        #         f = func(var1)
        #         # a.assign(val, use_locking=True)
        #         with tf.control_dependencies([var1]):
        #             # grad = tf.gradients(f, a)[0]
        #             grad = 2. * (var1 - 1.)  # HACK
        #             return f, grad

        # loss_func = func(par_a=a, par_b=b, par_c=c)
        # loss_func = func()
        loss_func = func
        # with tf.control_dependencies([a]):
        #     min = tfp.optimizer.bfgs_minimize(test_func,
        #                                       initial_position=tf.constant(10.0,
        # dtype=tf.float64))
        # minimizer = tf.train.AdamOptimizer()

        # min = minimizer.minimize(loss=loss_func, var_list=[a, b, c])
        # minimizer = AdamMinimizer(sess=sess, learning_rate=0.3)
        #########################################################################

        # which_minimizer = 'bfgs'
        which_minimizer = 'minuit'
        # which_minimizer = 'tfminuit'
        # which_minimizer = 'scipy'

        print("Running minimizer {}".format(which_minimizer))

        if which_minimizer == 'minuit':
            minimizer = MinuitMinimizer(sess=sess)

            init = tf.global_variables_initializer()
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

            init = tf.global_variables_initializer()
            sess.run(init)

            # for _ in range(5):

            n_rep = 1
            start = time.time()
            for _ in range(n_rep):
                value = minimizer.minimize()
            end = time.time()

            print("value from calculations:", value)
            print("time needed", (end - start) / n_rep)

        #####################################################################
        elif which_minimizer == 'bfgs':
            test1 = BFGSMinimizer(sess=sess, tolerance=1e-6)

            minimum = test1.minimize(params=[a, b, c])
            last_val = 100000
            cur_val = 9999999
            # HACK
            loss_func = loss_func()
            # HACK END
            # while abs(last_val - cur_val) > 0.00001:
            start = time.time()
            result = sess.run(minimum)
            end = time.time()
            print("value from calculations:", result)
            print("time needed", (end - start))
            # last_val = cur_val
            # print("running")

            # cur_val = sess.run(loss_func)
            # aval, bval, cval = sess.run([v for v in (a, b, c)])
            # aval, bval, cval = sess.run([v.read_value() for v in (a, b, c)])
            # print("a, b, c", aval, bval, cval)
            # minimizer.minimize(loss=loss_func, var_list=[a, b, c])
            cur_val = sess.run(loss_func)
            result = cur_val
            print(sess.run([v.read_value() for v in (a, b, c)]))
            print(result)
        #####################################################################

        if which_minimizer == 'scipy':
            func = loss_func()
            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                func,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-8},
                # optimizer_kwargs={'options': {'ftol': 1e-5}},
                tol=1e-10)

            # minimizer = ScipyMinimizer(loss=func,
            #                            method='L-BFGS-B',
            #                            options={'maxiter': 100})

            # with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            start = time.time()
            for _ in range(1):
                # print(sess.run(func))
                train_step.minimize()
            result = sess.run(func)
            print(result)
            # value = minimizer.minimize(loss=loss_func())  # how many times to be serialized
            end = time.time()
            value = result
            print("value from calculations:", value)
            print(sess.run([v.read_value() for v in (a, b, c)]))

            print("time needed", (end - start))

        print("Result from minimizer {}".format((which_minimizer)))
