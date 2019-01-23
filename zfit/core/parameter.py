"""Define Parameter which holds the value."""
from contextlib import suppress

import numpy as np
import tensorflow as tf

# TF backwards compatibility
from tensorflow.python import ops, array_ops

import zfit
from zfit import ztf

from tensorflow.python.ops.resource_variable_ops import ResourceVariable as TFBaseVariable
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from .interfaces import ZfitModel, ZfitParameter
from ..util.graph import get_dependents
from ..util.exception import LogicalUndefinedOperationError, NameAlreadyTakenError
from . import baseobject as zbaseobject
from . import interfaces as zinterfaces
from ..settings import ztypes


class MetaBaseParameter(type(TFBaseVariable), type(zinterfaces.ZfitParameter)):  # resolve metaclasses
    pass


# drop-in replacement for ResourceVariable
class ZfitBaseVariable(metaclass=type(TFBaseVariable)):

    def __init__(self, variable: tf.Variable, **kwargs):
        self.variable = variable

    @property
    def name(self):
        return self.variable.name

    @property
    def dtype(self):
        return self.variable.dtype

    def value(self):
        return self.variable.value()

    def assign(self, value, use_locking=False, name=None, read_value=True):
        return self.variable.assign(value=value, use_locking=use_locking,
                                    name=name, read_value=read_value)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype is not None and dtype != self.dtype:
            return NotImplemented
        if as_ref:
            return self.variable.read_value().op.inputs[0]
        else:
            return self.variable.value()

    def _AsTensor(self):
        return self.variable.value()

    @staticmethod
    def _OverloadAllOperators():  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
            ZfitBaseVariable._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        setattr(ZfitBaseVariable, "__getitem__", array_ops._SliceHelperVar)

    @staticmethod
    def _OverloadOperator(operator):  # pylint: disable=invalid-name
        """Defer an operator overload to `ops.Tensor`.
        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
        Args:
          operator: string. The operator name.
        """

        tensor_oper = getattr(ops.Tensor, operator)

        def _run_op(a, *args):
            # pylint: disable=protected-access
            value = a._AsTensor()
            return tensor_oper(value, *args)

        # Propagate __doc__ to wrapper
        try:
            _run_op.__doc__ = tensor_oper.__doc__
        except AttributeError:
            pass

        setattr(ZfitBaseVariable, operator, _run_op)


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(ZfitBaseVariable, _dense_var_to_tensor)
# ops.register_session_run_conversion_functions()

ZfitBaseVariable._OverloadAllOperators()


class ComposedResourceVariable(ResourceVariable):
    def __init__(self, name, initial_value, **kwargs):
        super().__init__(name=name, initial_value=initial_value, **kwargs)
        self._value_tensor = initial_value

    def value(self):
        # with tf.control_dependencies([self._value_tensor]):
        # return 5.
        return self._value_tensor

    def read_value(self):
        # raise RuntimeError()
        return self._value_tensor


class ComposedVariable(tf.Variable, metaclass=type(tf.Variable)):

    def __init__(self, name: str, initial_value: tf.Tensor, **kwargs):
        super().__init__(initial_value=initial_value, **kwargs)
        self._value_tensor = tf.convert_to_tensor(initial_value, preferred_dtype=ztypes.float)
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._value_tensor.dtype

    def value(self):
        return self._value_tensor

    def read_value(self):
        return self.value()

    def assign(self, value, use_locking=False, name=None, read_value=True):
        raise LogicalUndefinedOperationError("Cannot assign to a fixed/composed parameter")

    def load(self, value, session=None):
        raise LogicalUndefinedOperationError("Cannot load to a fixed/composed parameter")

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype is not None and dtype != self.dtype:
            return NotImplemented
        if as_ref:
            # return "NEVER READ THIS"
            raise LogicalUndefinedOperationError("There is no ref for the fixed/composed parameter")
        else:
            return self._value_tensor

    def _AsTensor(self):
        return self._value_tensor

    @staticmethod
    def _OverloadAllOperators():  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
            ComposedVariable._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        setattr(ComposedVariable, "__getitem__", array_ops._SliceHelperVar)

    @staticmethod
    def _OverloadOperator(operator):  # pylint: disable=invalid-name
        """Defer an operator overload to `ops.Tensor`.
        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
        Args:
          operator: string. The operator name.
        """

        tensor_oper = getattr(ops.Tensor, operator)

        def _run_op(a, *args):
            # pylint: disable=protected-access
            value = a._AsTensor()
            return tensor_oper(value, *args)

        # Propagate __doc__ to wrapper
        try:
            _run_op.__doc__ = tensor_oper.__doc__
        except AttributeError:
            pass

        setattr(ComposedVariable, operator, _run_op)


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(ComposedVariable, _dense_var_to_tensor)
fetch_function = lambda variable: ([variable.read_value()],
                                   lambda val: val[0])
feed_function = lambda feed, feed_val: [(feed.read_value(), feed_val)]
feed_function_for_partial_run = lambda feed: [feed.read_value()]

from tensorflow.python.client.session import register_session_run_conversion_functions

# ops.register_dense_tensor_like_type()
register_session_run_conversion_functions(tensor_type=ComposedResourceVariable, fetch_function=fetch_function,
                                          feed_function=feed_function,
                                          feed_function_for_partial_run=feed_function_for_partial_run)

register_session_run_conversion_functions(tensor_type=ComposedVariable, fetch_function=fetch_function,
                                          feed_function=feed_function,
                                          feed_function_for_partial_run=feed_function_for_partial_run)

ComposedVariable._OverloadAllOperators()


class BaseParameter(zbaseobject.BaseNumeric, zinterfaces.ZfitParameter, metaclass=MetaBaseParameter):
    pass


class ZfitParameterMixin:

    def __init__(self, name, initial_value, floating=True, **kwargs):
        super().__init__(initial_value=initial_value, name=name, **kwargs)
        # try:
        #     new_name = self.op.name
        # except AttributeError:  # no `op` attribute -> take normal name
        #     new_name = self.name
        new_name = self.name.rsplit(':', 1)[0]  # get rid of tf node
        new_name = new_name.rsplit('/', 1)[-1]  # get rid of the scope preceding the name
        if not new_name == name:  # name has been mangled because it already exists
            raise NameAlreadyTakenError("Another parameter is already named {}. "
                                        "Use a different, unique one.".format(name))
        self.floating = floating

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
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations
            with suppress(NotImplementedError):
                return operations.add(self, other)
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations
            with suppress(NotImplementedError):
                return operations.add(other, self)
        return super().__radd__(other)

    def __mul__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations
            with suppress(NotImplementedError):
                return operations.multiply(self, other)
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations
            with suppress(NotImplementedError):
                return operations.multiply(other, self)
        return super().__rmul__(other)


class Parameter(ZfitParameterMixin, TFBaseVariable, BaseParameter):
    """Class for fit parameters, derived from TF Variable class.
    """
    _independent = True

    def __init__(self, name, init_value, lower_limit=None, upper_limit=None, step_size=None, floating=True,
                 dtype=ztypes.float, **kwargs):
        """
          Constructor.
            name : name of the parameter,
            init_value : starting value
            lower_limit : lower limit
            upper_limit : upper limit
            step_size : step size (set to 0 for fixed parameters)
        """

        # TODO: sanitize input
        if lower_limit is None:
            lower_limit = -np.infty
        if upper_limit is None:
            upper_limit = np.infty
        # no_limits = -lower_limit == upper_limit == np.infty
        self.lower_limit = tf.cast(lower_limit, dtype=ztypes.float)
        self.upper_limit = tf.cast(upper_limit, dtype=ztypes.float)

        def constraint(x):
            return tf.clip_by_value(x, clip_value_min=self.lower_limit, clip_value_max=self.upper_limit)

        # self.constraint = constraint

        super().__init__(initial_value=init_value, dtype=dtype, name=name, constraint=constraint, **kwargs)
        if self.independent:
            tf.add_to_collection("zfit_independent", self)
        # init_value = tf.cast(init_value, dtype=ztypes.float)  # TODO: init value mandatory?
        # self.init_value = init_value
        self.floating = floating
        self.step_size = step_size
        zfit.run.auto_initialize(self)

        # self._placeholder = tf.placeholder(dtype=self.dtype, shape=self.get_shape())
        # self._update_op = self.assign(self._placeholder)  # for performance! Run with sess.run

    @property
    def lower_limit(self):
        return self._lower_limit

    @lower_limit.setter
    def lower_limit(self, value):
        self._lower_limit = value

    @property
    def upper_limit(self):
        return self._upper_limit

    @upper_limit.setter
    def upper_limit(self, value):
        self._upper_limit = value

    @property
    def has_limits(self):
        no_limits = -self.lower_limit == self.upper_limit == np.infty
        return not no_limits

    def value(self):
        value = super().value()
        if self.has_limits:
            value = self.constraint(value)
        return value

    def read_value(self):
        value = super().read_value()
        if self.has_limits:
            value = self.constraint(value)
        return value

    def _get_dependents(self):
        return {self}

    @property
    def independent(self):
        return self._independent

    def __init_subclass__(cls, **kwargs):
        cls._independent = True  # overwriting independent only for subclass/instance

    # OLD remove? only keep for speed reasons?
    # @property
    # def update_op(self):
    #     return self._update_op

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

    # TODO: make it a random variable? return tensor that evaluates new all the time?
    def randomize(self, sess, minval=None, maxval=None):
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

        # value = ztf.random_uniform(shape=self.shape, minval=minval, maxval=maxval, dtype=self.dtype, seed=seed)
        value = np.random.uniform(size=self.shape, low=minval, high=maxval)
        self.load(value=sess.run(value), session=sess)
        return value


class BaseComposedParameter(ZfitParameterMixin, ComposedVariable, BaseParameter):

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

    @property
    def independent(self):
        return False


class ComposedParameter(BaseComposedParameter):
    # TODO: raise error if eager is on (because it's very errorprone)
    def __init__(self, name, tensor, **kwargs):
        tensor = ztf.convert_to_tensor(tensor)
        independend_params = tf.get_collection("zfit_independent")
        params = get_dependents(tensor=tensor, candidates=independend_params)
        # params_init_op = [param.initializer for param in params]
        params = {p.name: p for p in params}
        # with tf.control_dependencies(params_init_op):
        super().__init__(params=params, initial_value=tensor, name=name, **kwargs)


class ComplexParameter(BaseComposedParameter):
    def __init__(self, name, initial_value, floating=True, dtype=ztypes.complex, **kwargs):
        initial_value = tf.cast(initial_value, dtype=dtype)
        real_value = tf.real(initial_value)
        real_part = Parameter(name=name + "_real", init_value=real_value, floating=floating, dtype=real_value.dtype)
        imag_value = tf.imag(initial_value)
        imag_part = Parameter(name=name + "_imag", init_value=imag_value, floating=floating, dtype=imag_value.dtype)
        params = {'real': real_part, 'imag': imag_part}
        super().__init__(params=params, initial_value=initial_value, name=name, **kwargs)


_auto_number = 0


def get_auto_number():
    global _auto_number
    auto_number = _auto_number
    _auto_number += 1
    return auto_number


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
            value = ComplexParameter("FIXED_autoparam_" + str(get_auto_number()), init_value=value, floating=False)
        else:
            value = ztf.to_real(value)
            value = Parameter("FIXED_autoparam_" + str(get_auto_number()), init_value=value, floating=False)

    # TODO: check if Tensor is complex

    value.floating = False
    return value
