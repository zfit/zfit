"""Define Parameter which holds the values."""
import abc

import numpy as np
import pep487
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# TF backwards compatibility
from tensorflow.python import VariableMetaclass

from zfit import ztf
from zfit.core.baseobject import BaseObject
from zfit.util import ztyping

from tensorflow.python.ops.resource_variable_ops import ResourceVariable as TFBaseVariable

from zfit.settings import types as ztypes


class ZfitParameter(BaseObject):

    @property
    @abc.abstractmethod
    def floating(self):
        raise NotImplementedError

    @floating.setter
    @abc.abstractmethod
    def floating(self):
        raise NotImplementedError

    @abc.abstractmethod
    def value(self):
        raise NotImplementedError



class MetaBaseParameter(type(TFBaseVariable), type(ZfitParameter)):  # resolve metaclasses
    pass


class BaseParameter(TFBaseVariable, ZfitParameter, metaclass=MetaBaseParameter):

    def __init__(self, name, floating=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.floating = floating

    def value(self):
        return self.read_value()

    @property
    def floating(self):
        if self._floating and not self.trainable:
            raise RuntimeError("Floating is set to true but tf Variable is not trainable.")
        return self._floating

    @floating.setter
    def floating(self, value):
        if not isinstance(value, bool):
            raise TypeError("floating has to be a boolean.")
        self._floating = value


class Parameter(BaseParameter):
    """Class for fit parameters, derived from TF Variable class.
    """

    def __init__(self, name, init_value, lower_limit=None, upper_limit=None, step_size=None, floating=True,
                 dtype=ztypes.float):
        """
          Constructor.
            name : name of the parameter,
            init_value : starting value
            lower_limit : lower limit
            upper_limit : upper limit
            step_size : step size (set to 0 for fixed parameters)
        """

        # TODO: sanitize input
        super().__init__(initial_value=init_value, dtype=dtype, name=name)
        init_value = tf.cast(init_value, dtype=ztypes.float)  # TODO: init value mandatory?
        self.floating = floating
        self.init_value = init_value
        self.step_size = step_size
        if lower_limit is None:
            lower_limit = -np.infty
        if upper_limit is None:
            upper_limit = np.infty
        self.lower_limit = tf.cast(lower_limit, dtype=ztypes.float)
        self.upper_limit = tf.cast(upper_limit, dtype=ztypes.float)
        self._placeholder = tf.placeholder(dtype=self.dtype, shape=self.get_shape())
        self._update_op = self.assign(self._placeholder)  # for performance! Run with sess.run

    def _get_dependents(self, only_floating=False):
        return {self}

    # OLD remove? only keep for speed reasons?
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

    def randomise(self, sess, minval=None, maxval=None, seed=None):
        """Update the value with a randomised value between minval and maxval.

        Args:
            sess (`tf.Session`): The TensorFlow session to execute the operation
            minval (Numerical):
            maxval (Numerical):
            seed ():
        """
        if minval is None:
            minval = self.lower_limit
        else:
            minval = tf.cast(minval, dtype=self.dtype)
        if maxval is None:
            maxval = self.upper_limit
        else:
            maxval = tf.cast(maxval, dtype=self.dtype)

        value = ztf.random_uniform(shape=self.shape, minval=minval, maxval=maxval, dtype=self.dtype, seed=seed)
        self.load(value=value, session=sess)
        return value

class ComposedParameter(BaseParameter):

    def __init__(self, params):
        self.parameters = params


    def _get_dependents(self, only_floating):
        dependents = self._extract_dependents(self.parameters.values(), only_floating=only_floating)
        return dependents



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
    value = Parameter("FIXED_autoparam", init_value=value)
    value.floating = False
    return value
