"""Define FitParameter which holds the values."""

from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from zfit.settings import fptype


class FitParameter(tf.Variable):
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
        tf.Variable.__init__(self, init_value, dtype=fptype)
        self.init_value = init_value
        self.par_name = name
        self.step_size = step_size
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
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
