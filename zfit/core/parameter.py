"""Define Parameter which holds the value."""
#  Copyright (c) 2019 zfit

from contextlib import suppress

import numpy as np
import tensorflow as tf

# TF backwards compatibility
from tensorflow.python import ops, array_ops

import zfit
from zfit import ztf

from tensorflow.python.ops.resource_variable_ops import ResourceVariable as TFBaseVariable
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from ..util.temporary import TemporarilySet
from ..core.baseobject import BaseNumeric, BaseObject
from ..util.cache import Cachable, invalidates_cache
from ..util import ztyping
from ..util.execution import SessionHolderMixin
from .interfaces import ZfitModel, ZfitParameter
from ..util.graph import get_dependents_auto
from ..util.exception import LogicalUndefinedOperationError, NameAlreadyTakenError
from . import baseobject as zbaseobject
from . import interfaces as zinterfaces
from ..settings import ztypes, run


class MetaBaseParameter(type(TFBaseVariable), type(zinterfaces.ZfitParameter)):  # resolve metaclasses
    pass


# drop-in replacement for ResourceVariable
# class ZfitBaseVariable(metaclass=type(TFBaseVariable)):
class ZfitBaseVariable(metaclass=MetaBaseParameter):

    def __init__(self, variable: tf.Variable, **kwargs):
        self.variable = variable

    # @property
    # def name(self):
    #     return self.variable.op.name

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


# class ComposedVariable(tf.Variable, metaclass=type(tf.Variable)):
# class ComposedVariable(ResourceVariable, metaclass=type(tf.Variable)):
class ComposedVariable(metaclass=MetaBaseParameter):

    def __init__(self, name: str, initial_value: tf.Tensor, **kwargs):
        # super().__init__(initial_value=initial_value, **kwargs, use_resource=True)
        super().__init__(name=name, **kwargs)
        self._value_tensor = tf.convert_to_tensor(initial_value, preferred_dtype=ztypes.float)
        # self._name = name

    @property
    def name(self):
        return self.name

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


class BaseParameter(ZfitParameter, metaclass=MetaBaseParameter):
    pass


class ZfitParameterMixin(BaseNumeric):
    _existing_names = set()

    def __init__(self, name, initial_value, **kwargs):
        if name in self._existing_names:
            raise NameAlreadyTakenError("Another parameter is already named {}. "
                                        "Use a different, unique one.".format(name))
        self._existing_names.update((name,))
        self._name = name
        super().__init__(initial_value=initial_value, name=name, **kwargs)
        # try:
        #     new_name = self.op.name
        # except AttributeError:  # no `op` attribute -> take normal name
        #     new_name = self.name
        # new_name = self.name.rsplit(':', 1)[0]  # get rid of tf node
        # new_name = self.name  # get rid of tf node
        # new_name = new_name.rsplit('/', 1)[-1]  # get rid of the scope preceding the name
        # if not new_name == name:  # name has been mangled because it already exists
        #     raise NameAlreadyTakenError("Another parameter is already named {}. "
        #                                 "Use a different, unique one.".format(name))

    @property
    def name(self):
        return self._name

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


# solve metaclass confict
class TFBaseVariable(TFBaseVariable, metaclass=MetaBaseParameter):
    pass


class Parameter(SessionHolderMixin, ZfitParameterMixin, TFBaseVariable, BaseParameter):
    """Class for fit parameters, derived from TF Variable class.
    """
    _independent = True

    def __init__(self, name, value, lower_limit=None, upper_limit=None, step_size=None, floating=True,
                 dtype=ztypes.float, **kwargs):
        """
          Constructor.
            name : name of the parameter,
            value : starting value
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
        value = tf.cast(value, dtype=ztypes.float)

        def constraint(x):
            return tf.clip_by_value(x, clip_value_min=self.lower_limit,
                                    clip_value_max=self.upper_limit)

        # self.constraint = constraint

        super().__init__(initial_value=value, dtype=dtype, name=name, constraint=constraint,
                         params={}, **kwargs)
        self.lower_limit = tf.cast(lower_limit, dtype=ztypes.float)
        self.upper_limit = tf.cast(upper_limit, dtype=ztypes.float)
        if self.independent:
            tf.add_to_collection("zfit_independent", self)
        else:
            tf.add_to_collection("zfit_dependent", self)
        # value = tf.cast(value, dtype=ztypes.float)  # TODO: init value mandatory?
        self.floating = floating
        self.step_size = step_size
        zfit.run.auto_initialize(self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._independent = True  # overwriting independent only for subclass/instance

    @property
    def lower_limit(self):
        return self._lower_limit

    @lower_limit.setter
    @invalidates_cache
    def lower_limit(self, value):
        self._lower_limit = value

    @property
    def upper_limit(self):
        return self._upper_limit

    @upper_limit.setter
    @invalidates_cache
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

    @property
    def step_size(self):  # TODO: improve default step_size?
        step_size = self._step_size
        if step_size is None:
            # auto-infer from limits
            # step_splits = 1e4
            # if self.has_limits:
            #     step_size = (self.upper_limit - self.lower_limit) / step_splits  # TODO improve? can be tensor?
            # else:
            step_size = 0.001
            if np.isnan(step_size):
                if self.lower_limit == -np.infty or self.upper_limit == np.infty:
                    step_size = 0.001
                else:
                    raise ValueError("Could not set step size. Is NaN.")
            # TODO: how to deal with infinities?
            step_size = ztf.to_real(step_size)
            self.step_size = step_size

        return step_size

    @step_size.setter
    def step_size(self, value):
        if value is not None:
            value = ztf.convert_to_tensor(value, preferred_dtype=ztypes.float)
            value = tf.cast(value, dtype=ztypes.float)
        self._step_size = value

    def load(self, value: ztyping.NumericalScalarType):
        """:py:class:`~zfit.Parameter` takes on the `value`. Is not part of the graph, does a session run.

        Args:
            value (numerical):
        """
        return super().load(value=value, session=self.sess)

    def set_value(self, value: ztyping.NumericalScalarType):
        """Set the :py:class:`~zfit.Parameter` to `value` (temporarily if used in a context manager).

        Args:
            value (float): The value the parameter will take on.
        """
        super_load = super().load

        def getter():
            return self.sess.run(self)

        def setter(value):
            super_load(value=value, session=self.sess)

        return TemporarilySet(value=value, setter=setter, getter=getter)

    # TODO: make it a random variable? return tensor that evaluates new all the time?
    def randomize(self, minval=None, maxval=None, sampler=np.random.uniform):
        """Update the value with a randomised value between minval and maxval.

        Args:
            minval (Numerical):
            maxval (Numerical):
            sampler ():
        """
        if minval is None:
            minval = self.sess.run(self.lower_limit)
        # else:
        #     minval = tf.cast(minval, dtype=self.dtype)
        if maxval is None:
            maxval = self.sess.run(self.upper_limit)
        # else:
        #     maxval = tf.cast(maxval, dtype=self.dtype)

        # value = ztf.random_uniform(shape=self.shape, minval=minval, maxval=maxval, dtype=self.dtype)
        shape = self.shape.as_list()
        # if shape == []:
        #     size = 1
        # value = self.sess.run(value)
        # eps = 1e-8
        # value = sampler(size=self.shape, low=minval + eps, high=maxval - eps)
        value = sampler(size=self.shape, low=minval, high=maxval)
        # value = np.random.uniform(size=size, low=minval, high=maxval)
        # if shape == []:
        #     value = value[0]
        self.load(value=value)
        return value

    def __repr__(self):
        return f"<zfit.Parameter '{self.name}' floating={self.floating}>"


class BaseComposedParameter(ZfitParameterMixin, ComposedVariable, BaseParameter):

    def __init__(self, params, value, name="BaseComposedParameter", **kwargs):
        super().__init__(initial_value=value, name=name, params=params, **kwargs)
        # self.params = params

    def _get_dependents(self):
        dependents = self._extract_dependents(list(self.params.values()))
        return dependents

    @property
    def floating(self):
        raise LogicalUndefinedOperationError("Cannot be floating or not. Look at the dependents.")

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        if not isinstance(value, dict):
            raise TypeError("Parameters has to be a dict")
        self._params = value

    @property
    def independent(self):
        return False


class ComposedParameter(BaseComposedParameter):
    def __init__(self, name, tensor, dtype=ztypes.float, **kwargs):
        tensor = ztf.convert_to_tensor(tensor, dtype=dtype)
        independent_params = tf.get_collection("zfit_independent")
        params = get_dependents_auto(tensor=tensor, candidates=independent_params)
        # params_init_op = [param.initializer for param in params]
        params = {p.name: p for p in params}
        # with tf.control_dependencies(params_init_op):
        super().__init__(params=params, value=tensor, name=name, dtype=dtype, **kwargs)

    def __repr__(self):
        return f"<zfit.{self.__class__.__name__} '{self.name}' dtype={self.dtype.name}>"


class ComplexParameter(ComposedParameter):
    def __init__(self, name, value, dtype=ztypes.complex, **kwargs):
        self._conj = None
        self._mod = None
        self._arg = None
        self._imag = None
        self._real = None

        super().__init__(name, value, dtype, **kwargs)

    @staticmethod
    def from_cartesian(name, real, imag, dtype=ztypes.complex, floating=True,
                       **kwargs):  # TODO: correct dtype handling, also below
        real = convert_to_parameter(real, name=name + "_real", prefer_floating=floating)
        imag = convert_to_parameter(imag, name=name + "_imag", prefer_floating=floating)
        param = ComplexParameter(name=name, value=tf.cast(tf.complex(real, imag), dtype=dtype),
                                 **kwargs)
        param._real = real
        param._imag = imag
        return param

    @staticmethod
    def from_polar(name, mod, arg, dtype=ztypes.complex, floating=True, **kwargs):
        mod = convert_to_parameter(mod, name=name + "_mod", prefer_floating=floating)
        arg = convert_to_parameter(arg, name=name + "_arg", prefer_floating=floating)
        param = ComplexParameter(name=name, value=tf.cast(tf.complex(mod * tf.math.cos(arg),
                                                                     mod * tf.math.sin(arg)),
                                                          dtype=dtype), **kwargs)
        param._mod = mod
        param._arg = arg
        return param

    @property
    def conj(self):
        if self._conj is None:
            self._conj = ComplexParameter(name='{}_conj'.format(self.name), value=tf.math.conj(self),
                                          dtype=self.dtype)
        return self._conj

    @property
    def real(self):
        real = self._real
        if real is None:
            real = ztf.to_real(self)
        return real

    @property
    def imag(self):
        imag = self._imag
        if imag is None:
            imag = tf.imag(tf.convert_to_tensor(self, preferred_dtype=self.dtype))  # HACK tf bug #30029
        return imag

    @property
    def mod(self):
        mod = self._mod
        if mod is None:
            mod = tf.math.abs(self)
        return mod

    @property
    def arg(self):
        arg = self._arg
        if arg is None:
            arg = tf.math.atan(self.imag / self.real)
        return arg


_auto_number = 0


def get_auto_number():
    global _auto_number
    auto_number = _auto_number
    _auto_number += 1
    return auto_number


def convert_to_parameter(value, name=None, prefer_floating=False) -> "ZfitParameter":
    """Convert a *numerical* to a fixed parameter or return if already a parameter.

    Args:
        value ():
    """
    floating = False
    is_python = False
    if name is not None:
        name = str(name)

    if isinstance(value, ZfitParameter):  # TODO(Mayou36): autoconvert variable. TF 2.0?
        return value
    elif isinstance(value, tf.Variable):
        raise TypeError("Currently, cannot autoconvert tf.Variable to zfit.Parameter.")

    # convert to Tensor if not yet
    if not isinstance(value, tf.Tensor):
        is_python = True
        if isinstance(value, complex):
            value = ztf.to_complex(value)
        else:
            floating = prefer_floating
            value = ztf.to_real(value)

    if not run._enable_parameter_autoconversion:
        return value

    if value.dtype.is_complex:
        if name is None:
            name = "FIXED_complex_autoparam_" + str(get_auto_number())
        value = ComplexParameter(name, value=value, floating=False)

    else:
        # value = Parameter("FIXED_autoparam_" + str(get_auto_number()), value=value, floating=False)
        if is_python:
            params = {}
        else:
            independend_params = tf.get_collection("zfit_independent")
            params = get_dependents_auto(tensor=value, candidates=independend_params)
        if params:
            if name is None:
                name = "composite_autoparam_" + str(get_auto_number())
            value = ComposedParameter(name, tensor=value)
        else:
            if name is None:
                name = "FIXED_autoparam_" + str(get_auto_number())
            value = Parameter(name, value=value, floating=floating)

    # value.floating = False
    return value
