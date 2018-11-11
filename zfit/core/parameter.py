"""Define FitParameter which holds the values."""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# TF backwards compatibility
from zfit import ztf
from zfit.util import ztyping

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

    def __init__(self, name, init_value, lower_limit=None, upper_limit=None, step_size=None, floating=True):
        """
          Constructor.
            name : name of the parameter,
            init_value : starting value
            lower_limit : lower limit
            upper_limit : upper limit
            step_size : step size (set to 0 for fixed parameters)
        """
        if not isinstance(name, str):
            raise TypeError("Name has to be a string and not {}".format(type(name)))
        # TODO: sanitize input
        init_value = tf.cast(init_value, dtype=ztypes.float)
        super().__init__(init_value, dtype=ztypes.float, name=name,
                         # use_resource=True  # TODO: only 1.11+
                         )
        self.floating = floating
        self.init_value = init_value
        # self.par_name = name
        self._step_size = None
        self.step_size = step_size
        if lower_limit is None:
            lower_limit = -np.infty
        if upper_limit is None:
            upper_limit = np.infty
        self.lower_limit = tf.cast(lower_limit, dtype=ztypes.float)
        self.upper_limit = tf.cast(upper_limit, dtype=ztypes.float)
        self.placeholder = tf.placeholder(dtype=self.dtype, shape=self.get_shape())
        self._update_op = self.assign(self.placeholder)  # for performance! Run with sess.run
        self.prev_value = None
        self.error = 0.
        self.positive_error = 0.
        self.negative_error = 0.
        self.fitted_value = 0.

    @property
    def update_op(self):
        return self._update_op

    @property
    def step_size(self):  # TODO: improve default step_size?
        step_size = self._step_size
        if step_size is None:
            # auto-infer from limits
            step_splits = 1e4
            # step_size = (self.upper_limit - self.lower_limit) / step_splits  # TODO improve? can be tensor?
            step_size = 0.001
            if step_size == np.nan:
                if self.lower_limit == -np.infty or self.upper_limit == np.infty:
                    step_size = 0.0001
                else:
                    raise ValueError("Could not set step size. Is NaN.")
            # TODO: how to deal with infinities?
        step_size = ztf.to_real(step_size)

        return step_size

    @step_size.setter
    def step_size(self, value):
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
        """Return True if the parameter is floating
        """
        return self._floating

    @floating.setter
    def floating(self, floating):
        self._floating = floating

    def randomise(self, session, minval, maxval, seed=None):
        """
          Randomise the initial value and update the tf variable value
        """
        if seed:
            np.random.seed(seed)
        val = np.random.uniform(maxval, minval)
        self.init_value = val
        self.update(session, val)


def convert_to_parameter(value) -> "Parameter":
    """Convert a *numerical* to a fixed parameter or return if already a parameter.

    Args:
        value ():
    """
    if isinstance(value, tf.Variable):
        return value

    # convert to Tensor if not yet
    if not isinstance(value, tf.Tensor):
        if isinstance(value, complex):
            value = ztf.to_complex(value)
            raise ValueError("Complex parameters not yet implemented")  # TODO: complex parameters
        else:
            value = ztf.to_real(value)

    # TODO: check if Tensor is complex
    value = FitParameter("FIXED_autoparam", init_value=value)
    value.floating = False
    return value
