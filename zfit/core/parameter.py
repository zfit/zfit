"""Define FitParameter which holds the values."""


import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# TF backwards compatibility
from zfit import ztf

try:
    # from tensorflow.python.ops.variables import
    from tensorflow.python.ops.variables import VariableV1
except ImportError:
    from tensorflow import Variable as VariableV1

from zfit.settings import types as ztypes


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
        init_value = tf.cast(init_value, dtype=ztypes.float)
        super().__init__(init_value, dtype=ztypes.float,  # PY23: change super
                                           # use_resource=True  # TODO: only 1.11+
                                           )
        self.init_value = init_value
        self.par_name = name
        self._step_size = None
        self.step_size = tf.cast(step_size, dtype=ztypes.float)
        self.lower_limit = tf.cast(lower_limit, dtype=ztypes.float)
        self.upper_limit = tf.cast(upper_limit, dtype=ztypes.float)
        self.placeholder = tf.placeholder(dtype=self.dtype, shape=self.get_shape())
        self.update_op = self.assign(self.placeholder)  # for performance! Run with sess.run
        self.prev_value = None
        self.error = 0.
        self.positive_error = 0.
        self.negative_error = 0.
        self.fitted_value = 0.

    #    print "new fit parameter %s" % name

    @property
    def step_size(self):
        if self._step_size is None:
            return ztf.constant(0.)
        else:
            return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value is not None:  # use None as zero because cannot test Tensor value
            try:
                if value == 0:
                    value = None
            except TypeError:
                pass
        self._step_size = value



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
                assign_op = self.assign(tf.convert_to_tensor(value))
                # session.run(assign_op)
            else:
                session.run(self.update_op, {self.placeholder: value})
                self.prev_value = value

    @property
    def floating(self):
        """
          Return True if the parameter is floating (step size>0)
        """
        return self._step_size is not None  # None "is" 0 here

    def randomise(self, session, minval, maxval, seed=None):
        """
          Randomise the initial value and update the tf variable value
        """
        if seed:
            np.random.seed(seed)
        val = np.random.uniform(maxval, minval)
        self.init_value = val
        self.update(session, val)
