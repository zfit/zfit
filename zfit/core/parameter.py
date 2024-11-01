"""Define Parameter which holds the value."""

#  Copyright (c) 2024 zfit

from __future__ import annotations

import abc
import collections
import copy
import functools
import typing
import weakref
from collections.abc import Callable, Iterable
from contextlib import suppress
from inspect import signature
from typing import Literal, Mapping, Optional, Union
from weakref import WeakSet

import dill
import numpy as np
import pydantic.v1 as pydantic
import tensorflow as tf
import tensorflow_probability as tfp

# TF backwards compatibility
from ordered_set import OrderedSet
from pydantic.v1 import Field, validator
from tensorflow.python.ops import tensor_getitem_override
from tensorflow.python.ops.resource_variable_ops import (
    ResourceVariable as TFVariable,
)
from tensorflow.python.ops.resource_variable_ops import (
    VariableSpec,
)
from tensorflow.python.ops.variables import Variable
from tensorflow.python.types.core import Tensor as TensorType

import zfit.z.numpy as znp

from .. import z
from ..core.baseobject import BaseNumeric, extract_filter_params
from ..minimizers.interface import ZfitResult
from ..serialization.paramrepr import make_param_constructor
from ..serialization.serializer import BaseRepr, Serializer
from ..settings import run, ztypes
from ..util import ztyping
from ..util.checks import NotSpecified
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args
from ..util.exception import (
    BreakingAPIChangeError,
    FunctionNotImplemented,
    IllegalInGraphModeError,
    LogicalUndefinedOperationError,
    ParameterNotIndependentError,
)
from ..util.temporary import TemporarilySet
from . import interfaces as zinterfaces
from .interfaces import ZfitIndependentParameter, ZfitModel, ZfitParameter
from .serialmixin import SerializableMixin

# todo add type hints in this module for api


class MetaBaseParameter(type(tf.Variable), type(zinterfaces.ZfitParameter)):  # resolve metaclasses
    pass


def register_tensor_conversion(
    convertable, name=None, overload_operators=True, priority=10
):  # higher than any tf conversion
    del name

    def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
        return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)

    tf.register_tensor_conversion_function(convertable, _dense_var_to_tensor, priority=priority)

    if overload_operators:
        convertable._OverloadAllOperators()


class OverloadableMixin(ZfitParameter):
    # Conversion to tensor.
    @staticmethod
    def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
        """Utility function for converting a Variable to a Tensor."""
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            msg = (
                f"Incompatible type conversion requested to type '{dtype.name}' for variable "
                f"of type '{v.dtype.name}'"
            )
            raise ValueError(msg)
        if as_ref:
            return v._ref()  # pylint: disable=protected-access
        else:
            return v.value()

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype and not dtype.is_compatible_with(self.dtype):
            msg = (
                f"Incompatible type conversion requested to type '{dtype.name}' for variable "
                f"of type '{self.dtype.name}'"
            )
            raise ValueError(msg)
        if as_ref:
            if hasattr(self, "_ref"):
                return self._ref()
            else:
                msg = "Why is this needed?"
                raise RuntimeError(msg)
        else:
            return self.value()

    def _AsTensor(self):
        return self.value()

    @classmethod
    def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            cls._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        cls.__getitem__ = tensor_getitem_override._slice_helper_var

    @classmethod
    def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
        """Defer an operator overload to ``ops.Tensor``.

        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
        Args:
          operator: string. The operator name.
        """
        # We can't use the overload mechanism on __eq__ & __ne__ since __eq__ is
        # called when adding a variable to sets. As a result we call a.value() which
        # causes infinite recursion when operating within a GradientTape
        # TODO(gjn): Consider removing this
        if operator in ("__eq__", "__ne__"):
            return

        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args, **kwargs):
            # pylint: disable=protected-access
            return tensor_oper(a.value(), *args, **kwargs)

        functools.update_wrapper(_run_op, tensor_oper)
        setattr(cls, operator, _run_op)


register_tensor_conversion(OverloadableMixin, overload_operators=True)


class WrappedVariable(metaclass=MetaBaseParameter):
    def __init__(self, initial_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable = tf.Variable(
            initial_value=initial_value,
            name=self.name,
            dtype=self.dtype,
        )

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    def constraint(self):
        return self.variable.constraint

    @property
    def dtype(self):
        return self.variable.dtype

    def value(self):
        return self.variable.value()

    def read_value(self):  # keep! Needed by TF internally
        return self.variable.read_value()

    @property
    def shape(self):
        return self.variable.shape

    def numpy(self):
        return self.variable.numpy()

    def assign(self, value, use_locking=False, name=None, read_value=True):
        return self.variable.assign(value=value, use_locking=use_locking, name=name, read_value=read_value)

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
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            WrappedVariable._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        WrappedVariable.__getitem__ = tensor_getitem_override._slice_helper_var

    @staticmethod
    def _OverloadOperator(operator):  # pylint: disable=invalid-name
        """Defer an operator overload to ``ops.Tensor``.

        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
        Args:
          operator: string. The operator name.
        """
        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args):
            # pylint: disable=protected-access
            value = a._AsTensor()
            return tensor_oper(value, *args)

        # Propagate __doc__ to wrapper
        with suppress(AttributeError):
            _run_op.__doc__ = tensor_oper.__doc__

        setattr(WrappedVariable, operator, _run_op)


register_tensor_conversion(WrappedVariable, "WrappedVariable", overload_operators=True)


class BaseParameter(Variable, ZfitParameter, TensorType, metaclass=MetaBaseParameter):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except NotImplementedError as error:
            tmp_val = kwargs.pop("name", None)  # remove if name is in there, needs to be passed through
            if args or kwargs:
                kwargs["name"] = tmp_val
                msg = (
                    f"The following arguments reached the top of the inheritance tree, the super "
                    f"init is not implemented (most probably abstract tf.Variable): {args, kwargs}. "
                    f"If you see this error, please post it as an bug at: "
                    f"https://github.com/zfit/zfit/issues/new/choose"
                )
                raise RuntimeError(msg) from error

    def __len__(self):
        return 1


def _finalize_weakref(name):
    if (params := ZfitParameterMixin._existing_params.get(name)) is not None and len(params) == 0:
        # if it's the last one
        ZfitParameterMixin._existing_params.pop(name)


class ZfitParameterMixin(BaseNumeric):
    _existing_params: typing.ClassVar = {}

    def __init__(self, name, label=None, **kwargs):
        if name not in self._existing_params:
            self._existing_params[name] = WeakSet()
            # Is an alternative arg for pop needed in case it fails? Why would it fail?
            weakref.finalize(
                self,
                _finalize_weakref,
                name,
            )
        self._existing_params[name].add(self)
        self._name = name
        self._label = label

        super().__init__(name=name, **kwargs)
        self._assert_params_unique()

    # property needed here to overwrite the name of tf.Variable
    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        if (label := self._label) is None:
            label = self.name
        return label

    def __add__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations

            with suppress(FunctionNotImplemented):
                return operations.add(self, other)
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations

            with suppress(FunctionNotImplemented):
                return operations.add(other, self)
        return super().__radd__(other)

    def __mul__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations

            with suppress(FunctionNotImplemented):
                return operations.multiply(self, other)
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (ZfitModel, ZfitParameter)):
            from . import operations

            with suppress(FunctionNotImplemented):
                return operations.multiply(other, self)
        return super().__rmul__(other)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class TFBaseVariable(TFVariable, metaclass=MetaBaseParameter):
    # class TFBaseVariable(WrappedVariable, metaclass=MetaBaseParameter):

    # Needed, otherwise tf variable complains about the name not having a ':' in there
    @property
    def _shared_name(self):
        return self.name


class Parameter(
    ZfitParameterMixin,
    TFBaseVariable,
    BaseParameter,
    SerializableMixin,
    ZfitIndependentParameter,
):
    """Class for fit parameters that has a default state."""

    _independent = True
    _independent_params = WeakSet()
    DEFAULT_stepsize = 0.01

    def __init__(
        self,
        name: str,
        value: ztyping.NumericalScalarType,
        lower: ztyping.NumericalScalarType | None = None,
        upper: ztyping.NumericalScalarType | None = None,
        stepsize: ztyping.NumericalScalarType | None = None,
        floating: bool = True,
        *,
        label: str | None = None,
        # legacy
        dtype: tf.DType = None,
        step_size: ztyping.NumericalScalarType | None = None,
    ):
        """Fit Parameter that has a default state (value) and limits (lower, upper).

        The name identifies the parameter.
        Multiple parameters with the same name can exist, however,
        they cannot be in the same PDF/func/loss as the value would not be uniquely defined.

        Args:
            name : Name of the parameter. Should be unique within a model/likelihood.
            value : Default value of the parameter. Also used as the starting value in minimization.
            lower : lower limit of the parameter. If the parameter is set to a value below the lower limit, it will raise an error.
            upper : upper limit of the parameter. If the parameter is set to a value above the upper limit, it will raise an error.
            floating : If the parameter is floating (can change value) or fixed (constant) in the minimization.
            label: |@doc:param.init.label||@docend:param.init.label|
            stepsize : Initial step size for minimization. If not set, a default value is used.
        """
        self._independent_params.add(self)

        # legacy start
        if step_size is not None:
            stepsize = step_size
        if dtype is not None:
            msg = "The argument `dtype` has been removed. The dtype is now automatically the default float type."
            raise BreakingAPIChangeError(msg)
        dtype = ztypes.float
        # legacy end

        # TODO: sanitize input for TF2
        self._lower_limit_neg_inf = None
        self._upper_limit_neg_inf = None
        if lower is None:
            self._lower_limit_neg_inf = znp.asarray(-np.inf, dtype)
        if upper is None:
            self._upper_limit_neg_inf = znp.asarray(np.inf, dtype)
        value = znp.asarray(value, dtype=ztypes.float)

        super().__init__(
            initial_value=value,
            dtype=dtype,
            name=name,
            params={},
            label=label,
        )

        self.lower = lower
        self.upper = upper
        self.floating = floating
        self.stepsize = stepsize
        self.set_value(value)  # to check that it is in the limits

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._independent = True  # overwriting independent only for subclass/instance

    @classmethod
    def _from_name(cls, name):
        for param in cls._independent_params:
            if name == param.name:
                return param
        msg = f"Parameter {name} does not exist, please create it first."
        raise ValueError(msg)

    @property
    def lower(self):
        limit = self._lower
        if limit is None:
            limit = self._lower_limit_neg_inf
        return limit

    @lower.setter
    # @invalidate_graph
    def lower(self, value):
        if value is None and self._lower_limit_neg_inf is None:
            self._lower_limit_neg_inf = znp.asarray(-np.inf, dtype=ztypes.float)
        elif value is not None:
            value = znp.asarray(value, dtype=ztypes.float)
        self._lower = value

    @property
    def upper(self):
        limit = self._upper
        if limit is None:
            limit = self._upper_limit_neg_inf
        return limit

    @upper.setter
    # @invalidate_graph
    def upper(self, value):
        if value is None and self._upper_limit_neg_inf is None:
            self._upper_limit_neg_inf = znp.asarray(np.inf, dtype=ztypes.float)
        elif value is not None:
            value = znp.asarray(value, dtype=ztypes.float)
        self._upper = value

    @property
    def has_limits(self) -> bool:
        """If the parameter has limits set or not."""
        no_limits = self._lower is None and self._upper is None
        return not no_limits

    @property
    def at_limit(self) -> tf.Tensor:
        """If the value is at the limit (or over it).

        Returns:
            Boolean ``tf.Tensor`` that tells whether the value is at the limits.
        """
        return self._check_at_limit(self.value())

    def _check_at_limit(self, value, exact=False):
        """The precision is up to 1e-5 relative or 1e-8 absolute if exact is None.

        Args:
            value ():
            exact ():

        Returns:
        """
        if not self.has_limits:
            return tf.constant(False)
        # Adding a slight tolerance to make sure we're not tricked by numerics due to floating point comparison
        diff = znp.abs(self.upper - self.lower)  # catch if it is minus inf
        if not exact:
            reltol = 0.005
            abstol = 1e-5
        else:
            reltol = 1e-5
            abstol = 1e-7
        tol = znp.minimum(diff * reltol, abstol)  # if one limit is inf we would get inf
        if not exact:  # if exact, we wanna allow to set it slightly over the limit.
            tol = -tol  # If not, we wanna make sure it's inside
        at_lower = z.unstable.less_equal(value, self.lower - tol)
        at_upper = z.unstable.greater_equal(value, self.upper + tol)
        return z.unstable.logical_or(at_lower, at_upper)

    def value(self):
        value = super().value()
        if self.has_limits:
            value = tfp.math.clip_by_value_preserve_gradient(
                value, clip_value_min=self.lower, clip_value_max=self.upper
            )
        return value

    @deprecated(None, "Use `value` instead.")
    def read_value(self):
        return self.value()

    @property
    def floating(self):
        if self._floating and (hasattr(self, "trainable") and not self.trainable):
            msg = "Floating is set to true but tf Variable is not trainable."
            raise RuntimeError(msg)
        return self._floating

    @floating.setter
    def floating(self, value):
        if not isinstance(value, bool):
            msg = "floating has to be a boolean."
            raise TypeError(msg)
        self._floating = value

    @property
    def independent(self):
        return self._independent

    @property
    def has_stepsize(self):
        return self._stepsize is not None

    @property
    @deprecated(None, "Use `has_stepsize` instead.")
    def has_step_size(self):
        return self.has_stepsize

    @property
    def stepsize(self) -> tf.Tensor:  # TODO: improve default stepsize?
        """Step size of the parameter, the estimated order of magnitude of the uncertainty.

        This can be crucial to tune for the minimization. A too large ``stepsize`` can produce NaNs, a too small won't
        converge.

        If the step size is not set, the ``DEFAULT_stepsize`` is used.

        Returns:
            The step size
        """
        stepsize = self._stepsize
        if stepsize is None:
            stepsize = self.DEFAULT_stepsize
        return stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value is not None:
            value = float(value)
        self._stepsize = value

    @property
    @deprecated(None, "Use `stepsize` instead.")
    def step_size(self) -> tf.Tensor:
        return self.stepsize

    @step_size.setter
    @deprecated(None, "Use `stepsize` instead.")
    def step_size(self, value):
        self.stepsize = value

    def set_value(self, value: ztyping.NumericalScalarType):
        """Set the :py:class:`~zfit.Parameter` to `value` (temporarily if used in a context manager).

        This operation won't, compared to the assign, return the read value but an object that *can* act as a context
        manager.

        Args:
            value: The value the parameter will take on.
        Raises:
            ValueError: If the value is not inside the limits (in normal Python/eager mode)
            InvalidArgumentError: If the value is not inside the limits (in JIT/traced/graph mode)
        """

        def getter():
            return self.value()

        def setter(value):
            if self.has_limits:
                message = (
                    f"Setting value {value} invalid for parameter {self.name} with limits "
                    f"{self.lower} - {self.upper}. This is changed."
                    f" In order to silence this and clip the value, you can use (with caution,"
                    f" advanced) `Parameter.assign`"
                )
                if run.executing_eagerly():
                    if self._check_at_limit(value, exact=True):
                        raise ValueError(message)
                else:
                    tf.debugging.assert_greater(
                        znp.asarray(value, tf.float64),
                        znp.asarray(self.lower, tf.float64),
                        message=message,
                    )
                    tf.debugging.assert_less(
                        znp.asarray(value, tf.float64),
                        znp.asarray(self.upper, tf.float64),
                        message=message,
                    )
            #     tf.debugging.Assert(self._check_at_limit(value), [value])
            self.assign(value=value, read_value=False)

        return TemporarilySet(value=value, setter=setter, getter=getter)

    def assign(self, value, use_locking: bool | None = None, read_value=False):
        """Set the :py:class:`~zfit.Parameter` to `value` without any checks.

        Compared to ``set_value``, this method cannot be used with a context manager and won't raise an
        error

        Args:
            value: The value the parameter will take on.
        """
        return super().assign(value=value, use_locking=use_locking, read_value=read_value)

    def randomize(
        self,
        minval: ztyping.NumericalScalarType | None = None,
        maxval: ztyping.NumericalScalarType | None = None,
        sampler: Callable = np.random.uniform,
    ) -> tf.Tensor:
        """Update the parameter with a randomised value between minval and maxval and return it.

        Args:
            minval: The lower bound of the sampler. If not given, ``lower_limit`` is used.
            maxval: The upper bound of the sampler. If not given, ``upper_limit`` is used.
            sampler: A sampler with the same interface as ``np.random.uniform``

        Returns:
            The sampled value
        """
        if not tf.executing_eagerly():
            msg = "Randomizing values in a parameter within Graph mode is most probably not" " what is "
            raise IllegalInGraphModeError(msg)
        minval = self.lower if minval is None else znp.asarray(minval, dtype=self.dtype)
        maxval = self.upper if maxval is None else znp.asarray(maxval, dtype=self.dtype)
        if maxval is None or minval is None:
            msg = "Cannot randomize a parameter without limits or limits given."
            raise RuntimeError(msg)
        value = sampler(size=self.shape, low=minval, high=maxval)

        self.set_value(value=value)
        return value

    def get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        *,
        autograd: bool | None = None,
    ) -> OrderedSet[ZfitParameter]:
        del is_yield, extract_independent  # does not make sense for a single parameter
        # if autograd is not None:
        #     raise WorkInProgressError("autograd distinction not available, needed?")
        if autograd is False:
            return OrderedSet()  # we assume that all support autograd
        return extract_filter_params(self, floating=floating, extract_independent=False)

    def __repr__(self):  # many try and except in case it's not fully initialized yet
        if tf.executing_eagerly():  # more explicit: we check for exactly this attribute, nothing inside numpy
            try:
                value = f"{self.numpy():.4g}"
            except Exception as err:
                value = f"errored {err}"
        else:
            value = "graph-node"
        try:
            floating = self.floating
        except Exception as err:
            floating = f"errored {err}"
        try:
            name = self.name
        except Exception as err:
            name = f"errored {err}"
        return f"<zfit.{self.__class__.__name__} '{name}' floating={floating} value={value}>"

    # LEGACY, deprecate?

    @property
    def lower_limit(self):
        msg = "Use `lower` instead of `lower_limit`."
        raise BreakingAPIChangeError(msg)

    @lower_limit.setter
    def lower_limit(self, value):  # noqa: ARG002
        msg = "Use `lower` instead of `lower_limit`."
        raise BreakingAPIChangeError(msg)

    @property
    def upper_limit(self):
        msg = "Use `upper` instead of `upper_limit`."
        raise BreakingAPIChangeError(msg)

    @upper_limit.setter
    def upper_limit(self, value):  # noqa: ARG002
        msg = "Use `upper` instead of `upper_limit`."
        raise BreakingAPIChangeError(msg)

    def __tf_tracing_type__(self, signature_context):
        return ParameterSpec(parameter=self)

    # below needed?
    # CompositeTensor method
    # @property
    # def _type_spec(self):
    #     return ParameterSpec(parameter=self)
    #
    # # CompositeTensor method
    # def _shape_invariant_to_type_spec(self, shape):
    #     return ParameterSpec(shape, self.dtype, self.trainable)

    def __reduce__(self):  # for pickling
        return functools.partial(
            Parameter,
            name=self.name,
            value=self.value(),
            lower=self.lower,
            upper=self.upper,
            floating=self.floating,
            label=self.label,
            stepsize=self.stepsize,
        ), ()


# delattr(Parameter, "__tf_tracing_type__")


class ParameterSpec(VariableSpec):
    value_type = property(lambda _: Parameter)

    def __init__(self, shape=None, dtype=None, trainable=True, alias_id=None, *, parameter=None):
        if parameter is None:
            msg = "Unknown error, please report, parameter should not be none"
            raise RuntimeError(msg)

        shape = parameter.shape
        dtype = parameter.dtype
        trainable = True
        alias_id = hash(parameter)
        self.parameter_value = parameter
        # self._name = parameter.name  # TODO: not needed anymore?
        self.parameter_type = type(parameter)
        if dtype is None:
            dtype = tf.float64
        super().__init__(shape=shape, dtype=dtype, trainable=trainable, alias_id=alias_id)
        self.hash = hash(parameter)

    @classmethod
    def from_value(cls, value):
        return cls(parameter=value)

    def _to_components(self, value):
        return super()._to_components(value)

    def _from_components(self, components):
        return super()._from_components(components)  # checking that there is no error

    def _to_tensors(self, value):
        return [value]

    def is_subtype_of(self, other):
        return (
            type(other) is ParameterSpec
            and self.parameter_type is other.parameter_type
            and self.parameter_value is other.parameter_value
        )

    def most_specific_common_supertype(self, others):
        return self if all(self == other for other in others) else None

    def placeholder_value(self, placeholder_context=None):
        del placeholder_context  # unused, because it's not understood
        return self.parameter_value

    def is_compatible_with(self, spec_or_value):
        return isinstance(spec_or_value, ParameterSpec) and self.parameter_value is spec_or_value.parameter_value

    def __eq__(self, other) -> bool:
        if not isinstance(other, ParameterSpec):
            return False
        return self.parameter_value is other.parameter_value

    def __hash__(self):
        return self.hash


class ParameterRepr(BaseRepr):  # add label?
    _implementation = Parameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(Parameter))
    hs3_type: Literal["Parameter"] = Field("Parameter", alias="type")
    name: str
    value: float
    lower: Optional[float] = Field(None, alias="min")
    upper: typing.Optional[float] = Field(None, alias="max")
    stepsize: Optional[float] = None
    floating: Optional[bool] = None
    label: Optional[str] = None

    @validator("value", pre=True)
    def _validate_value(cls, v):
        if cls.orm_mode(v):
            v = v()
        return v

    def __hash__(self):
        return hash(self.name) + hash(type(self))


class BaseComposedParameter(ZfitParameterMixin, OverloadableMixin, BaseParameter):
    def __init__(self, params, func, dtype=None, name="BaseComposedParameter", **kwargs):
        # 0.4 breaking
        if "value" in kwargs:
            msg = "'value' cannot be provided any longer, `func` is needed."
            raise BreakingAPIChangeError(msg)
        super().__init__(name=name, params=params, **kwargs)
        if not callable(func):
            msg = "`func` is not callable."
            raise TypeError(msg)

        self._func = func
        self._dtype = dtype if dtype is not None else ztypes.float

    @property
    def floating(self):
        msg = "Cannot be floating or not. Look at the dependencies."
        raise LogicalUndefinedOperationError(msg)

    @floating.setter
    def floating(self, value):  # noqa: ARG002
        msg = "Cannot set floating or not. Set in the dependencies (`get_params`)."
        raise LogicalUndefinedOperationError(msg)

    @property
    def params(self):
        return self._params

    def value(self):
        params = self.params
        return znp.asarray(self._func(params), dtype=self.dtype)

        # return tf.convert_to_tensor(value, dtype=self.dtype)

    @deprecated(None, "Use `value` instead.")
    def read_value(self):  # keep! Needed by TF internally
        return self.value()

    @property
    def shape(self):
        return self.value().shape

    def numpy(self):
        return self.value().numpy()

    @property
    def independent(self):
        return False

    def set_value(self, value):  # noqa: ARG002
        """Set the value of the parameter. Cannot be used for composed parameters!

        Args:
            value:
        """
        msg = "Cannot set value of a composed parameter." " Set the value on its components."
        raise LogicalUndefinedOperationError(msg)

    def randomize(self, minval=None, maxval=None, sampler=np.random.uniform):  # noqa: ARG002
        """Randomize the value of the parameter.

        Cannot be used for composed parameters!
        """
        msg = "Cannot randomize a composed parameter."
        raise LogicalUndefinedOperationError(msg)

    def assign(self, value, use_locking=False, name=None, read_value=True):  # noqa: ARG002
        """Assign the value of the parameter.

        Cannot be used for composed parameters!
        """
        msg = "Cannot assign a composed parameter."
        raise LogicalUndefinedOperationError(msg)


class ConstantParameter(OverloadableMixin, ZfitParameterMixin, BaseParameter, SerializableMixin):
    """Constant parameter.

    Value cannot change.
    """

    def __init__(
        self,
        name,
        value,
        *,
        label: str | None = None,
    ):
        """Constant parameter that cannot change its value.

        Args:
            name: Unique identifier of the parameter.
            value: Constant value.
            label: |@doc:param.init.label||@docend:param.init.label|
        """
        super().__init__(name=name, params={}, dtype=ztypes.float, label=label)
        self._value_np = tf.get_static_value(value, partial=True)
        self._value = tf.guarantee_const(znp.array(value, dtype=self.dtype))

    @property
    def shape(self):
        return self.value().shape

    def value(self) -> tf.Tensor:
        return self._value

    @deprecated(None, "Use `value` instead.")
    def read_value(self) -> tf.Tensor:  # keep! Needed by TF internally
        return self.value()

    @property
    def floating(self):
        return False

    @floating.setter
    def floating(self, value):  # noqa: ARG002
        msg = "Cannot set a ConstantParameter to floating. Use a `Parameter` instead."
        raise LogicalUndefinedOperationError(msg)

    @property
    def independent(self) -> bool:
        return False

    @property
    def static_value(self):
        return self._value_np

    def numpy(self):
        return self._value_np

    def __repr__(self):
        value_str = f"{value: .4g}" if (value := self._value_np) is not None else "symbolic"
        return f"<zfit.param.{self.__class__.__name__} '{self.name}' dtype={self.dtype.name} value={value_str}>"


register_tensor_conversion(ConstantParameter, "ConstantParameter", overload_operators=True)
register_tensor_conversion(BaseComposedParameter, "BaseComposedParameter", overload_operators=True)


class ConstantParamRepr(BaseRepr):
    _implementation = ConstantParameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(ConstantParameter))
    hs3_type: Literal["ConstantParameter"] = pydantic.Field("ConstantParameter", alias="type")
    name: str
    value: float
    floating: bool = False
    label: Optional[str] = None

    @validator("value", pre=True)
    def _validate_value(cls, value):
        if cls.orm_mode(value):
            value = value()
        return value

    def _to_orm(self, init):
        init = copy.copy(init)
        init.pop("floating")
        return super()._to_orm(init)


class ComposedParameter(SerializableMixin, BaseComposedParameter):
    @deprecated_args(None, "Use `params` instead.", "dependents")
    @deprecated_args(None, "Use `func` instead.", "value_fn")
    def __init__(
        self,
        name: str,
        func: Optional[Callable] = None,
        *,
        value_fn: Optional[Callable] = None,
        params: (dict[str, ZfitParameter] | Iterable[ZfitParameter] | ZfitParameter) = NotSpecified,
        label: str | None = None,
        unpack_params: bool | None = None,
        dependents: (dict[str, ZfitParameter] | Iterable[ZfitParameter] | ZfitParameter) = NotSpecified,
        dtype: tf.dtypes.DType = ztypes.float,
    ):
        """Arbitrary composition of parameters.

        A `ComposedParameter` allows for arbitrary combinations of parameters and correlations using an arbitrary
        function.

        Examples:
            .. jupyter-execute::

                import zfit

                param1 = zfit.Parameter('param1', 1.0, 0.1, 1.8)
                param2 = zfit.Parameter('param2', 42.0, 0, 100)

                # using a dict for the params
                def mult_dict(params):
                    return params["a"] * params["b"]

                mult_param_dict = zfit.ComposedParameter('mult_dict', mult_dict, params={"a": param1, "b": param2})

                # using a list for the params
                def mult_list(params):
                    return params[0] * params[1]

                mult_param_list = zfit.ComposedParameter('mult_list', mult_list, params=[param1, param2])



        Args:
            name: Unique name of the Parameter.
            func: Function that returns the value of the composed parameter and takes as arguments `params` as
                arguments. The function must be able to be called with the same arguments as `params`.
            params: If it is a `dict`, this will directly be used as the `params` attribute, otherwise the
                parameters will be automatically named with f"param_{i}". The values act as arguments to `func`.
            dtype: Output of `func` dtype
            label: |@doc:param.init.label||@docend:param.init.label|
            unpack_params: If True, the parameters will be unpacked and passed as arguments to `func`. If False, the
                parameters will be passed as a dict/tuple. If None, it will be automatically determined and raise an error
                if it cannot be determined.
            dependents:
                .. deprecated:: unknown
                    use `params` instead.
        """
        # legacy
        if value_fn is not None:
            msg = "Use `func` instead of `value_fn`."
            raise BreakingAPIChangeError(msg)
        # end legacy
        if not isinstance(params, Mapping):
            params = convert_to_container(params)
        original_init = {"name": name, "internal_params": params, "func": func, "unpack_params": unpack_params}

        # legacy
        if dependents is not NotSpecified:
            msg = "Use `params` instead of `dependents`."
            raise BreakingAPIChangeError(msg)
        if params is NotSpecified:
            raise ValueError
        # end legacy

        if params is None:
            msg = "Params needs to be specified."
            raise BreakingAPIChangeError(msg)
        takes_no_params = False
        if unpack_params is None:
            parameters = signature(func).parameters
            if isinstance(params, ZfitParameter):
                params = [params]
                unpack_params = True
            elif len(parameters) == 1 and (len(params) > 1) or "params" in parameters:
                unpack_params = False
            elif len(parameters) - len(params) >= 0:
                unpack_params = True
            elif len(parameters) == 0:
                takes_no_params = True
                unpack_params = False
            else:
                msg = (
                    "Cannot determine if parameter should be unpacked or not. Please specify explicitly `unpack_params`"
                )
                raise ValueError(msg)

        dictlike = isinstance(params, collections.abc.Mapping)

        if dictlike:
            params_dict = params
            if takes_no_params:

                def stratified_fn(params):
                    del params
                    return func()

            elif unpack_params:

                def stratified_fn(params):
                    return func(**params)

            else:
                stratified_fn = func
        else:
            params = convert_to_container(params)

            if params is None or takes_no_params:
                params_dict = {}

                def stratified_fn(params):
                    del params
                    return func()

            else:
                params_dict = {f"param_{i}": p for i, p in enumerate(params)}

                if unpack_params:

                    def stratified_fn(params):
                        return func(*tuple(params.values()))

                else:

                    def stratified_fn(params):
                        return func(tuple(params.values()))

        original_init["params"] = params_dict  # needs to be here, we need the params to be a dict for the serialization

        super().__init__(params=params_dict, func=stratified_fn, name=name, dtype=dtype, label=label)
        self.hs3.original_init.update(original_init)

    def __repr__(self):
        try:
            value = f"{self.numpy():.4g}" if tf.executing_eagerly() else "graph-node"
        except Exception:
            value = "ERROR OCCURRED"
        return f"<zfit.{self.__class__.__name__} '{self.name}' params={[(k, p.name) for k, p in self.params.items()]} value={value}>"


class ComposedParameterRepr(BaseRepr):
    _implementation = ComposedParameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(ComposedParameter))
    hs3_type: Literal["ComposedParameter"] = pydantic.Field("ComposedParameter", alias="type")
    name: str
    func: str
    params: dict[str, Serializer.types.ParamTypeDiscriminated]
    unpack_params: Optional[bool]
    label: Optional[str] = None
    internal_params: Optional[
        Union[
            Serializer.types.ParamTypeDiscriminated,
            list[Serializer.types.ParamTypeDiscriminated],
            dict[str, Serializer.types.ParamTypeDiscriminated],
        ]
    ]

    @validator("func", pre=True)
    def _validate_value_pre(cls, value):
        if cls.orm_mode(value):
            value = dill.dumps(value).hex()
        return value

    @pydantic.root_validator(pre=True)
    def validate_all_functor(cls, values):
        if cls.orm_mode(values):
            values = values["hs3"].original_init
        return values

    def _to_orm(self, init):
        init = copy.copy(init)
        func = init.pop("func")
        params = init.pop("internal_params")

        init["params"] = params
        init["func"] = dill.loads(bytes.fromhex(func))
        return super()._to_orm(init)


class ComplexParameter(ComposedParameter):  # TODO: change to real, imag as input?
    @deprecated_args(None, "Use `func` instead.", "value_fn")
    def __init__(
        self,
        name: str,
        func: Callable | None = None,
        *,
        value_fn: Callable | None = None,
        params,
        dtype=None,
        label: str | None = None,
    ):
        """Create a complex parameter.

        .. note::
            Use the constructor class methods instead of the __init__() constructor:

            - :py:meth:`ComplexParameter.from_cartesian`
            - :py:meth:`ComplexParameter.from_polar`

        Args:
            name: Name of the parameter.
            func: Function that returns the value of the complex parameter and takes as arguments the real and
                imaginary part.
            params: List of the real and imaginary part of the complex parameter.
            label: |@doc:param.init.label||@docend:param.init.label|
        """
        # legacy
        if value_fn is not None:
            msg = "Use `func` instead of `value_fn`."
            raise BreakingAPIChangeError(msg)
        if dtype is not None:
            msg = "The argument `dtype` has been removed. The dtype is now automatically the default complex dtype."
            raise BreakingAPIChangeError(msg)
        dtype = ztypes.complex
        super().__init__(name, func=func, params=params, dtype=dtype, label=label)
        self._conj = None
        self._mod = None
        self._arg = None
        self._imag = None
        self._real = None

    @classmethod
    def from_cartesian(
        cls,
        name: str,
        real: ztyping.NumericalScalarType,
        imag: ztyping.NumericalScalarType,
        floating: bool = True,
        *,
        dtype=ztypes.complex,
        label: str | None = None,
    ) -> ComplexParameter:  # TODO: correct dtype handling, also below
        """Create a complex parameter from cartesian coordinates.

        Args:
            name: Name of the parameter.
            real: Real part of the complex number.
            imag: Imaginary part of the complex number.
            floating: If True, the parameter is floating. If False, the parameter is constant.
            dtype: Data type of the complex parameter.
            label: |@doc:param.init.label||@docend:param.init.label|
        """
        real = convert_to_parameter(real, name=name + "_real", prefer_constant=not floating)
        imag = convert_to_parameter(imag, name=name + "_imag", prefer_constant=not floating)
        param = cls(
            name=name,
            func=lambda _real, _imag: znp.asarray(tf.complex(_real, _imag), dtype=dtype),
            params=[real, imag],
            label=label,
        )
        param._real = real
        param._imag = imag
        return param

    @classmethod
    def from_polar(
        cls,
        name: str,
        mod: ztyping.NumericalScalarType,
        arg: ztyping.NumericalScalarType,
        floating=True,
        *,
        label: str | None = None,
        **__,
    ) -> ComplexParameter:
        """Create a complex parameter from polar coordinates.

        Args:
            name: Name of the parameter.
            mod: Modulus (r) the complex number.
            arg: Argument (phi) of the complex number.
            dtype: Data type of the complex parameter.
            floating: If True, the parameter is floating. If False, the parameter is constant.
            label: |@doc:param.init.label||@docend:param.init.label|
        """
        mod = convert_to_parameter(mod, name=name + "_mod", prefer_constant=not floating)
        arg = convert_to_parameter(arg, name=name + "_arg", prefer_constant=not floating)
        param = cls(
            name=name,
            func=lambda _mod, _arg: znp.asarray(tf.complex(_mod * znp.cos(_arg), _mod * znp.sin(_arg))),
            params=[mod, arg],
            label=label,
        )
        param._mod = mod
        param._arg = arg
        return param

    @property
    def conj(self):
        """Returns a complex conjugated copy of the complex parameter."""

        if self._conj is None:
            name = f"{self.name}_conj"
            if (label := self._label) is not None:
                label = f"Conjugate of {self.label}"
            self._conj = ComplexParameter(
                name=name,
                func=lambda: znp.conj(self),
                params=self.get_params(floating=None),
                label=label,
            )
        return self._conj

    @property
    def real(self) -> tf.Tensor:
        """Real part of the complex parameter."""
        return znp.real(self)

    @property
    def imag(self) -> tf.Tensor:
        """Imaginary part of the complex parameter."""
        return znp.imag(self)

    @property
    def mod(self) -> tf.Tensor:
        """Modulus (r) of the complex parameter."""
        return znp.abs(self)

    @property
    def arg(self) -> tf.Tensor:
        """Argument (phi) of the complex parameter."""
        return znp.angle(self)


# register_tensor_conversion(ConstantParameter, "ConstantParameter", True)
register_tensor_conversion(ComposedParameter, "ComposedParameter", True)

_auto_number = 0


def get_auto_number():
    global _auto_number
    auto_number = _auto_number
    _auto_number += 1
    return auto_number


def _reset_auto_number():
    global _auto_number
    _auto_number = 0


def convert_to_parameters(
    value,
    name: str | list[str] | None = None,
    prefer_constant: bool | None = None,
    lower=None,
    upper=None,
    stepsize=None,
):
    if prefer_constant is None:
        prefer_constant = True
    if isinstance(value, collections.abc.Mapping):
        return convert_to_parameters(**value, prefer_constant=False)
    value = convert_to_container(value)
    is_param_already = [isinstance(val, ZfitParameter) for val in value]
    if all(is_param_already):
        return value
    elif any(is_param_already):
        msg = f"value has to be either ZfitParameters or values, not mixed (currently)." f" Is {value}."
        raise ValueError(msg)
    params_dict = {
        "value": value,
        "name": name,
        "lower": lower,
        "upper": upper,
        "stepsize": stepsize,
    }
    params_dict = {
        key: convert_to_container(val, ignore=np.ndarray) for key, val in params_dict.items() if val is not None
    }
    lengths = {len(v) for v in params_dict.values()}
    if len(lengths) != 1:
        msg = f"Inconsistent length in values when converting the parameters: {params_dict}"
        raise ValueError(msg)

    params = []
    for i in range(len(params_dict["value"])):
        pdict = {k: params_dict[k][i] for k in params_dict}
        params.append(convert_to_parameter(**pdict, prefer_constant=prefer_constant))
    return params


@deprecated_args(None, "Use `params` instead.", "dependents")
def convert_to_parameter(
    value: ztyping.NumericalScalarType | ZfitParameter | Callable,
    name: str | None = None,
    prefer_constant: bool = True,
    params: ZfitParameter | Iterable[ZfitParameter] | None = None,
    lower: ztyping.NumericalScalarType | None = None,
    upper: ztyping.NumericalScalarType | None = None,
    stepsize: ztyping.NumericalScalarType | None = None,
    *,
    label: str | None = None,
    # legacy
    dependents=None,
) -> ZfitParameter:
    """Convert a *numerical* to a constant/floating parameter or return if already a parameter.

    Args:
        value: Value of the parameter. If a `ZfitParameter` is passed, it will be returned as is.
        name: Name of the parameter. If None, a unique name will be created.
        prefer_constant: If True, create a ConstantParameter instead of a Parameter, if possible.
        params: If the value is a callable, the parameters that are passed to the callable.
        lower: Lower limit of the parameter.
        upper: Upper limit of the parameter.
        stepsize: Step size of the parameter.
        label: |@doc:param.init.label||@docend:param.init.label|
    """
    # legacy start
    if dependents is not None:
        params = dependents
    # legacy end
    if name is not None:
        name = str(name)

    if callable(value):
        if params is None:
            msg = "If the value is a callable, the params have to be specified as an empty list/tuple"
            raise ValueError(msg)
        return ComposedParameter(f"Composed_autoparam_{get_auto_number()}", func=value, params=params, label=label)

    if isinstance(value, ZfitParameter):  # TODO(Mayou36): autoconvert variable. TF 2.0?
        return value
    elif isinstance(value, tf.Variable):
        msg = f"Currently, cannot autoconvert tf.Variable ({value}) to zfit.Parameter."
        raise TypeError(msg)

    # convert to Tensor
    if not isinstance(value, tf.Tensor):
        value = z.to_complex(value) if isinstance(value, complex) else z.to_real(value)

    if not run._enable_parameter_autoconversion:
        return value

    if value.dtype.is_complex:
        if name is None:
            name = "FIXED_complex_autoparam_" + str(get_auto_number())
        if prefer_constant:
            complex_params = (
                ConstantParameter(name + "_REALPART", value=znp.real(value)),
                ConstantParameter(name + "_IMAGPART", value=znp.imag(value)),
            )
        else:
            complex_params = (
                Parameter(name + "_REALPART", value=znp.real(value)),
                Parameter(name + "_IMAGPART", value=znp.imag(value)),
            )
        value = ComplexParameter.from_cartesian(name, real=complex_params[0], imag=complex_params[1], label=label)

    elif prefer_constant:
        if name is None:
            name = "FIXED_autoparam_" + str(get_auto_number()) if name is None else name
        value = ConstantParameter(name, value=value, label=label)

    else:
        name = "autoparam_" + str(get_auto_number()) if name is None else name
        value = Parameter(name=name, value=value, lower=lower, upper=upper, stepsize=stepsize, label=label)

    return value


@z.function(wraps="params", keepalive=True)
def assign_values_jit(
    params: Parameter | Iterable[Parameter],
    values: ztyping.NumericalScalarType | Iterable[ztyping.NumericalScalarType],
    use_locking=False,
):
    """Assign values to parameters jitted.

    This method can be significantly faster than `set_values`, however it expects the correct data-type and
    cannot, for example, take a `FitResult` as input or function as a context manager. Only use when
    performance is critical (such as inside a minimizer).

    Args:
        params: The parameters to assign the values to.
        values: Values to assign to the parameters.
        use_locking:
    """
    for i, param in enumerate(params):
        value = values[i]
        if value.dtype != param.dtype:
            value = znp.cast(value, param.dtype)
        param.assign(value, read_value=False, use_locking=use_locking)


def assign_values(
    params: Parameter | Iterable[Parameter],
    values: ztyping.NumericalScalarType | Iterable[ztyping.NumericalScalarType],
    use_locking=False,
    allow_partial: bool | None = None,
):
    """Set the values of multiple parameters in a fast way.

    In general, :meth:`set_values` is to be preferred. `assign_values` will ignore out-of-bounds errors,
     does not offer a context-manager but is in general (an order of magnitude) faster.

    Args:
        params: Parameters to set the values.
        values: List-like object that supports indexing.
        use_locking: if true, lock the parameter to avoid race conditions.
        allow_partial: Allow to set only parts of the parameters in case values is a `ZfitResult`
            and not all are present in the
            *values*. If False, *params* not in *values* will raise an error.
            Note that setting this to true will also go with an empty values container.

    Raises:
        ValueError: If not all *params* are in *values* if *values* is a `FitResult` and `allow_partial` is `False`.
    """
    if allow_partial is None:
        allow_partial = False
    params, values, _ = check_convert_param_values_assign(params, values, allow_partial=allow_partial)
    assign_values_jit(params=params, values=values, use_locking=use_locking)


def set_values(
    params: Parameter | Iterable[Parameter],
    values: (
        ztyping.NumericalScalarType
        | Iterable[ztyping.NumericalScalarType]
        | ZfitResult
        | Mapping[str | ZfitParameter, ztyping.NumericalScalarType]
        | None,
    ) = None,
    allow_partial: bool | None = None,
):
    """Set the values (using a context manager or not) of multiple parameters.

    Args:
        params: Parameters to set the values.
        values: List-like object that supports indexing, a `FitResult` or a `Mapping` of parameters or parameter names to values.
        allow_partial: Allow to set only parts of the parameters in case values is a `ZfitResult`
            and not all are present in the
            *values*. If False, *params* not in *values* will raise an error.
            Note that setting this to true will also go with an empty values container.

    Returns:
        An object for a context manager (but can also be used without), can be ignored.
    Raises:
        ValueError: If the value is not between the limits of the parameter.
        ValueError: If not all *params* are in *values* if *values* is a `FitResult` and `allow_partial` is `False`.
    """
    if allow_partial is None:
        allow_partial = False
    params, values, _ = check_convert_param_values_assign(params, values, allow_partial)

    def setter(values):
        for i, param in enumerate(params):
            param.set_value(values[i])

    def getter():
        return [param.value() for param in params]

    return TemporarilySet(values, setter=setter, getter=getter)


def check_convert_param_values_assign(params, values, allow_partial=False):
    """Check if params are valid and convert them if necessary to be used with assign_values.

    Args:
        params: Parameters to set the values.
        values: List-like object that supports indexing or a `ZfitResult` or a `Mapping`.
        allow_partial: Allow to set only parts of the parameters in case values is a `ZfitResult` or a `Mapping`. Other parameters will not be updated.
            More values than parameters, if they are given in a `Mapping` or `ZfitResult`, are allowed in any case.
    Returns:
        A tuple of (params, values)
    """
    if isinstance(params, ZfitResult) and values is None:
        params, values = None, params
    elif isinstance(params, Mapping):
        if values is not None:
            msg = "Cannot set values to parameters if params are a Mapping."
            raise ValueError(msg)
        params, values = tuple(params.keys()), tuple(params.values())
    elif not isinstance(values, ZfitResult):
        params = convert_to_container(params)
        if params is None:
            msg = "No parameters given to set values to (values={values})."
            raise ValueError(msg)
        noparams = len(params) == 0
        if noparams:
            return params, values, True
    if isinstance(values, ZfitResult):
        result = values
        new_params = []
        values = []
        if params is None:
            params = result.params.keys()
        for param in params:
            if param in result.params:
                values.append(result.params[param]["value"])
                new_params.append(param)
            elif not allow_partial:
                msg = (
                    f"Cannot set {param} with {result} as it is not contained. To partially set"
                    f" the parameters (only those in the result), use allow_partial"
                )
                raise ValueError(msg)
        params = new_params
    elif isinstance(values, Mapping):
        new_params = []
        new_values = []
        for param in params:
            if param in values:
                new_values.append(values[param])
                new_params.append(param)
            elif param.name in values:
                new_values.append(values[param.name])
                new_params.append(param)
            elif not allow_partial:
                msg = (
                    f"Cannot set {param} with {values!r} as it is not contained. To partially set"
                    f" the parameters (only those in the result), use allow_partial"
                )
                raise ValueError(msg)
        values = new_values
        params = new_params

    else:
        if not (tf.is_tensor(values) or isinstance(values, np.ndarray)):
            values = convert_to_container(values)
            lenvalues = len(values)
        else:
            values = znp.asarray(values, dtype=znp.float64)
            values = znp.atleast_1d(values)
            shape = values.shape
            lenvalues = shape[0] if shape is not None else None
        if lenvalues is not None and lenvalues != len(params):
            msg = f"Incompatible length of parameters and values: {params}, {values}"
            raise ValueError(msg)

    not_param = [param for param in params if not isinstance(param, ZfitParameter)]
    if not_param:
        msg = f"The following are not parameters (but should be): {not_param}"
        raise TypeError(msg)
    non_independent_params = [param for param in params if not param.independent]
    if non_independent_params:
        msg = f"trying to set value of parameters that are not independent " f"{non_independent_params}"
        raise ParameterNotIndependentError(msg)
    values = znp.asarray(values, dtype=znp.float64)
    return params, values, False  # if it's empty
