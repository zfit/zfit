"""Define FitParameter which holds the values."""

from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

# TF backwards compatibility
try:
    # from tensorflow.python.ops.variables import
    from tensorflow.python.ops.variables import VariableV1
except ImportError:
    from tensorflow import Variable as VariableV1

from zfit.settings import fptype


class FitParameter(VariableV1):
    """
      Class for fit parameters, derived from TF Variable class.
    """

    def __init__(self, name, init_value, lower_limit=0., upper_limit=0., step_size=1e-6):
        """
          Constructor.
            name : name of the parameter (passed on to MINUIT)
            init_value : starting value
            lower_limit : lower limit
            upper_limit : upper limit
            step_size : step size (set to 0 for fixed parameters)
        """
        # TODO: sanitize input
        init_value = tf.cast(init_value, dtype=fptype)
        super(FitParameter, self).__init__(init_value, dtype=fptype,  # PY23: change super
                                           # use_resource=True  # TODO: only 1.11+
                                           )
        self.init_value = init_value
        self.par_name = name
        self.step_size = tf.cast(step_size, dtype=fptype)
        self.lower_limit = tf.cast(lower_limit, dtype=fptype)
        self.upper_limit = tf.cast(upper_limit, dtype=fptype)
        self.placeholder = tf.placeholder(self.dtype, shape=self.get_shape())
        self.update_op = self.assign(self.placeholder)
        self.prev_value = None
        self.error = 0.
        self.positive_error = 0.
        self.negative_error = 0.
        self.fitted_value = 0.

    #    print "new fit parameter %s" % name

    def update(self, session, value):
        """
          Update the value of the parameter. Previous value is remembered in self.prev_value
          and TF update is called only if the value is changed.
            session : TF session
            value   : new value
        """
        if value != self.prev_value:
            if isinstance(value, tf.Tensor):
                # session.run(self.assign(value))
                self.assign(value)
            else:
                session.run(self.update_op, {self.placeholder: value})
                self.prev_value = value

    def floating(self):
        """
          Return True if the parameter is floating (step size>0)
        """
        return self.step_size > 0

    def randomise(self, session, minval, maxval, seed=None):
        """
          Randomise the initial value and update the tf variable value
        """
        if seed:
            np.random.seed(seed)
        val = np.random.uniform(maxval, minval)
        self.init_value = val
        self.update(session, val)
