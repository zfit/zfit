"""Define Parameter which holds the values."""

import numpy as np
import tensorflow as tf

# TF backwards compatibility

from zfit import ztf

from tensorflow.python.ops.resource_variable_ops import ResourceVariable as TFBaseVariable

from .baseobject import BaseObject
from .interfaces import ZfitParameter, ZfitObject
from zfit.settings import types as ztypes


class MetaBaseParameter(type(TFBaseVariable), type(ZfitParameter)):  # resolve metaclasses
    pass


class BaseParameter(TFBaseVariable, BaseObject, ZfitParameter, metaclass=MetaBaseParameter):

    def __init__(self, name, initial_value, floating=True, **kwargs):
        super().__init__(initial_value=initial_value, name=name, **kwargs)
        self.floating = floating

    # def value(self):
    #     return self.read_value()

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

    def __add__(self, other):
        if isinstance(other, ZfitObject):
            from . import operations
            return operations.add(self, other, dims=None)
        else:
            return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, ZfitObject):
            from . import operations
            return operations.add(other, self, dims=None)
        else:
            return super().__radd__(other)

    def __mul__(self, other):
        if isinstance(other, ZfitObject):
            from . import operations
            return operations.multiply(self, other, dims=None)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, ZfitObject):
            from . import operations
            return operations.multiply(other, self, dims=None)
        else:
            return super().__rmul__(other)


class Parameter(BaseParameter):
    """Class for fit parameters, derived from TF Variable class.
    """
    _independent = True

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
        if self.independent:
            tf.add_to_collection("zfit_independent", self)
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

    def _get_dependents(self):
        return {self}

    @property
    def independent(self):
        return self._independent

    def __init_subclass__(cls, **kwargs):
        cls._independent = True  # overwritting independent only counnt for subclass/instance

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


class BaseComposedParameter(BaseParameter):

    def __init__(self, params, initial_value, name="BaseComposedParameter", **kwargs):
        super().__init__(initial_value=initial_value, name=name, **kwargs)
        self.parameters = params

    def _get_dependents(self):
        dependents = self._extract_dependents(list(self.parameters.values()))
        return dependents

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, dict):
            raise TypeError("Parameters has to be a dict")
        self._parameters = value


class ComposedParameter(BaseComposedParameter):
    # TODO: raise error if eager is on (because it's very errorprone)
    def __init__(self, name, tensor, **kwargs):
        independend_params = tf.get_collection("zfit_independent")
        grad_indep_params = tf.gradients(tensor, independend_params)
        params = [param for param, grad in zip(independend_params, grad_indep_params) if grad is not None]
        params = {p.name: p for p in params}
        super().__init__(params=params, initial_value=tensor, name=name, **kwargs)

    @property
    def independent(self):
        return False


class ComplexParameter(BaseComposedParameter):
    def __init__(self, name, initial_value, floating=True, dtype=ztypes.complex, **kwargs):
        real_value = tf.real(initial_value)
        real_part = Parameter(name=name + "_real", init_value=real_value, floating=floating, dtype=real_value.dtype)
        imag_value = tf.imag(initial_value)
        imag_part = Parameter(name=name + "_imag", init_value=imag_value, floating=floating, dtype=imag_value.dtype)
        params = {'real': real_part, 'imag': imag_part}
        super().__init__(params=params, initial_value=initial_value, name=name, **kwargs)


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
            value = ComplexParameter("FIXED_autoparam", init_value=value, floating=False)
        else:
            value = ztf.to_real(value)
            value = Parameter("FIXED_autoparam", init_value=value, floating=False)

    # TODO: check if Tensor is complex

    value.floating = False
    return value
