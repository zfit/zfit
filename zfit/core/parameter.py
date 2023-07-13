"""Define Parameter which holds the value."""
#  Copyright (c) 2023 zfit

from __future__ import annotations

import abc
import collections
import copy
import functools
import warnings
from collections.abc import Iterable, Callable
from contextlib import suppress
from inspect import signature
from typing import Optional, Dict
from weakref import WeakValueDictionary

import dill as dill
import numpy as np
import pydantic
import tensorflow as tf
import tensorflow_probability as tfp

# TF backwards compatibility
from ordered_set import OrderedSet
from pydantic import Field, validator
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.resource_variable_ops import (
    ResourceVariable as TFVariable,
    VariableSpec,
)
from tensorflow.python.ops.variables import Variable
from tensorflow.python.types.core import Tensor as TensorType

from typing import Literal

from .serialmixin import SerializableMixin
from .. import z
from ..serialization.paramrepr import make_param_constructor
from ..serialization.serializer import BaseRepr, Serializer

znp = z.numpy
import zfit.z.numpy as znp

from .. import z
from ..core.baseobject import BaseNumeric, extract_filter_params
from ..minimizers.interface import ZfitResult
from ..settings import run, ztypes
from ..util import ztyping
from ..util.cache import invalidate_graph
from ..util.checks import NotSpecified
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args
from ..util.exception import (
    BreakingAPIChangeError,
    FunctionNotImplemented,
    IllegalInGraphModeError,
    LogicalUndefinedOperationError,
    NameAlreadyTakenError,
    ParameterNotIndependentError,
)
from ..util.temporary import TemporarilySet
from . import interfaces as zinterfaces
from .dependents import _extract_dependencies
from .interfaces import ZfitIndependentParameter, ZfitModel, ZfitParameter


# todo add type hints in this module for api


class MetaBaseParameter(
    type(tf.Variable), type(zinterfaces.ZfitParameter)
):  # resolve metaclasses
    pass


def register_tensor_conversion(
    convertable, name=None, overload_operators=True, priority=10
):  # higher than any tf conversion
    def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
        return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)

    tf.register_tensor_conversion_function(
        convertable, _dense_var_to_tensor, priority=priority
    )
    if name:
        pass
        # _pywrap_utils.RegisterType(name, convertable)

    if overload_operators:
        convertable._OverloadAllOperators()


class OverloadableMixin(ZfitParameter):
    # Conversion to tensor.
    @staticmethod
    def _TensorConversionFunction(
        v, dtype=None, name=None, as_ref=False
    ):  # pylint: disable=invalid-name
        """Utility function for converting a Variable to a Tensor."""
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            raise ValueError(
                "Incompatible type conversion requested to type '%s' for variable "
                "of type '%s'" % (dtype.name, v.dtype.name)
            )
        if as_ref:
            return v._ref()  # pylint: disable=protected-access
        else:
            return v.value()

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype and not dtype.is_compatible_with(self.dtype):
            raise ValueError(
                "Incompatible type conversion requested to type '%s' for variable "
                "of type '%s'" % (dtype.name, self.dtype.name)
            )
        if as_ref:
            if hasattr(self, "_ref"):
                return self._ref()
            else:
                raise RuntimeError("Why is this needed?")
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
        setattr(cls, "__getitem__", array_ops._SliceHelperVar)

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
        if operator == "__eq__" or operator == "__ne__":
            return

        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args, **kwargs):
            # pylint: disable=protected-access
            return tensor_oper(a.value(), *args, **kwargs)

        functools.update_wrapper(_run_op, tensor_oper)
        setattr(cls, operator, _run_op)


register_tensor_conversion(OverloadableMixin, overload_operators=True)


class WrappedVariable(metaclass=MetaBaseParameter):
    def __init__(self, initial_value, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable = tf.Variable(
            initial_value=initial_value,
            constraint=constraint,
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

    def read_valu(self):
        return self.variable.read_value()

    @property
    def shape(self):
        return self.variable.shape

    def numpy(self):
        return self.variable.numpy()

    def assign(self, value, use_locking=False, name=None, read_value=True):
        return self.variable.assign(
            value=value, use_locking=use_locking, name=name, read_value=read_value
        )

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
        setattr(WrappedVariable, "__getitem__", array_ops._SliceHelperVar)

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
        try:
            _run_op.__doc__ = tensor_oper.__doc__
        except AttributeError:
            pass

        setattr(WrappedVariable, operator, _run_op)


register_tensor_conversion(WrappedVariable, "WrappedVariable", overload_operators=True)


class BaseParameter(Variable, ZfitParameter, TensorType, metaclass=MetaBaseParameter):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except NotImplementedError:
            tmp_val = kwargs.pop(
                "name", None
            )  # remove if name is in there, needs to be passed through
            if args or kwargs:
                kwargs["name"] = tmp_val
                raise RuntimeError(
                    f"The following arguments reached the top of the inheritance tree, the super "
                    f"init is not implemented (most probably abstract tf.Variable): {args, kwargs}. "
                    f"If you see this error, please post it as an bug at: "
                    f"https://github.com/zfit/zfit/issues/new/choose"
                )

    def __len__(self):
        return 1


class ZfitParameterMixin(BaseNumeric):
    _existing_params = WeakValueDictionary()

    def __init__(self, name, **kwargs):
        if name in self._existing_params:
            raise NameAlreadyTakenError(
                "Another parameter is already named {}. "
                "Use a different, unique one.".format(name)
            )
        self._existing_params.update({name: self})
        self._name = name

        super().__init__(name=name, **kwargs)

    # property needed here to overwrite the name of tf.Variable
    @property
    def name(self) -> str:
        return self._name

    def __del__(self):
        with suppress(
            AttributeError, NotImplementedError
        ):  # if super does not have a __del__
            super().__del__(self)

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
        return id(self) == id(other)

    def __hash__(self):
        if not hasattr(self, "_cached_hash"):
            self._cached_hash = hash(id(self))
        hash_value = self._cached_hash
        return hash_value


class TFBaseVariable(TFVariable, metaclass=MetaBaseParameter):
    # class TFBaseVariable(WrappedVariable, metaclass=MetaBaseParameter):

    # Needed, otherwise tf variable complains about the name not having a ':' in there
    @property
    def _shared_name(self):
        return self.name


from weakref import WeakSet


class Parameter(
    ZfitParameterMixin,
    TFBaseVariable,
    BaseParameter,
    SerializableMixin,
    ZfitIndependentParameter,
):
    """Class for fit parameters, derived from TF Variable class."""

    _independent = True
    _independent_params = WeakSet()
    DEFAULT_STEP_SIZE = 0.001

    @deprecated_args(None, "Use `lower` instead.", "lower_limit")
    @deprecated_args(None, "Use `upper` instead.", "upper_limit")
    def __init__(
        self,
        name: str,
        value: ztyping.NumericalScalarType,
        lower: ztyping.NumericalScalarType | None = None,
        upper: ztyping.NumericalScalarType | None = None,
        step_size: ztyping.NumericalScalarType | None = None,
        floating: bool = True,
        dtype: tf.DType = ztypes.float,
        # legacy
        lower_limit: ztyping.NumericalScalarType | None = None,
        upper_limit: ztyping.NumericalScalarType | None = None,
    ):
        """

        Args:
            name : name of the parameter
            value : starting value
            lower : lower limit
            upper : upper limit
            step_size : step size
        """
        self._independent_params.add(self)

        # legacy start
        if lower_limit is not None:
            lower = lower_limit
        if upper_limit is not None:
            upper = upper_limit
        # legacy end

        # TODO: sanitize input for TF2
        self._lower_limit_neg_inf = None
        self._upper_limit_neg_inf = None
        if lower is None:
            self._lower_limit_neg_inf = tf.cast(-np.infty, dtype)
        if upper is None:
            self._upper_limit_neg_inf = tf.cast(np.infty, dtype)
        value = tf.cast(value, dtype=ztypes.float)

        def constraint(x):
            return tfp.math.clip_by_value_preserve_gradient(
                x, clip_value_min=self.lower, clip_value_max=self.upper
            )

        super().__init__(
            initial_value=value,
            dtype=dtype,
            name=name,
            constraint=constraint,
            params={},
        )

        self.lower = lower
        self.upper = upper
        self.floating = floating
        self.step_size = step_size
        self.set_value(value)  # to check that it is in the limits

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._independent = True  # overwriting independent only for subclass/instance

    @classmethod
    def _from_name(cls, name):
        for param in cls._independent_params:
            if name == param.name:
                return param
        raise ValueError(f"Parameter {name} does not exist, please create it first.")

    @property
    def lower(self):
        limit = self._lower
        if limit is None:
            limit = self._lower_limit_neg_inf
        return limit

    @lower.setter
    @invalidate_graph
    def lower(self, value):
        if value is None and self._lower_limit_neg_inf is None:
            self._lower_limit_neg_inf = tf.cast(-np.infty, dtype=ztypes.float)
        elif value is not None:
            value = tf.cast(value, dtype=ztypes.float)
        self._lower = value

    @property
    def upper(self):
        limit = self._upper
        if limit is None:
            limit = self._upper_limit_neg_inf
        return limit

    @upper.setter
    @invalidate_graph
    def upper(self, value):
        if value is None and self._upper_limit_neg_inf is None:
            self._upper_limit_neg_inf = tf.cast(np.infty, dtype=ztypes.float)
        elif value is not None:
            value = tf.cast(value, dtype=ztypes.float)
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
            value = self.constraint(value)
        return value

    @deprecated(None, "Use `value` instead.")
    def read_value(self):
        value = super().read_value()
        if self.has_limits:
            value = self.constraint(value)
        return value

    @property
    def floating(self):
        if self._floating and (hasattr(self, "trainable") and not self.trainable):
            raise RuntimeError(
                "Floating is set to true but tf Variable is not trainable."
            )
        return self._floating

    @floating.setter
    def floating(self, value):
        if not isinstance(value, bool):
            raise TypeError("floating has to be a boolean.")
        self._floating = value

    def _get_dependencies(self):
        return {self}

    @property
    def independent(self):
        return self._independent

    @property
    def has_step_size(self):
        return self._step_size is not None

    @property
    def step_size(self) -> tf.Tensor:  # TODO: improve default step_size?
        """Step size of the parameter, the estimated order of magnitude of the uncertainty.

        This can be crucial to tune for the minimization. A too large ``step_size`` can produce NaNs, a too small won't
        converge.

        If the step size is not set, the ``DEFAULT_STEP_SIZE`` is used.

        Returns:
            The step size
        """
        step_size = self._step_size
        if step_size is None:
            #     # auto-infer from limits
            #     step_splits = 1e5
            #     if self.has_limits:
            #         step_size = (self.upper_limit - self.lower_limit) / step_splits  # TODO improve? can be tensor?
            #     else:
            #         step_size = self.DEFAULT_STEP_SIZE
            #     if np.isnan(step_size):
            #         if self.lower_limit == -np.infty or self.upper_limit == np.infty:
            #             step_size = self.DEFAULT_STEP_SIZE
            #         else:
            #             raise ValueError("Could not set step size. Is NaN.")
            #     # step_size = z.to_real(step_size)
            #     self.step_size = step_size
            step_size = self.DEFAULT_STEP_SIZE
        # step_size = z.convert_to_tensor(step_size)
        return step_size

    @step_size.setter
    def step_size(self, value):
        if value is not None:
            value = float(value)
            # value = z.convert_to_tensor(value, preferred_dtype=ztypes.float)
            # value = tf.cast(value, dtype=ztypes.float)
        self._step_size = value

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
                        tf.cast(value, tf.float64),
                        tf.cast(self.lower, tf.float64),
                        message=message,
                    )
                    tf.debugging.assert_less(
                        tf.cast(value, tf.float64),
                        tf.cast(self.upper, tf.float64),
                        message=message,
                    )
            #     tf.debugging.Assert(self._check_at_limit(value), [value])
            self.assign(value=value, read_value=False)

        return TemporarilySet(value=value, setter=setter, getter=getter)

    def assign(self, value, use_locking=None, name=None, read_value=False):
        """Set the :py:class:`~zfit.Parameter` to `value` without any checks.

        Compared to ``set_value``, this method cannot be used with a context manager and won't raise an
        error

        Args:
            value: The value the parameter will take on.
        """
        return super().assign(
            value=value, use_locking=use_locking, name=name, read_value=read_value
        )

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
            raise IllegalInGraphModeError(
                "Randomizing values in a parameter within Graph mode is most probably not"
                " what is "
            )
        if minval is None:
            minval = self.lower
        else:
            minval = tf.cast(minval, dtype=self.dtype)
        if maxval is None:
            maxval = self.upper
        else:
            maxval = tf.cast(maxval, dtype=self.dtype)
        if maxval is None or minval is None:
            raise RuntimeError(
                "Cannot randomize a parameter without limits or limits given."
            )
        value = sampler(size=self.shape, low=minval, high=maxval)

        self.set_value(value=value)
        return value

    def get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        only_floating=NotSpecified,
    ) -> set[ZfitParameter]:
        return extract_filter_params(self, floating=floating, extract_independent=False)

    def __repr__(self):
        if (
            tf.executing_eagerly()
        ):  # more explicit: we check for exactly this attribute, nothing inside numpy
            value = f"{self.numpy():.4g}"
        else:
            value = "graph-node"
        return f"<zfit.{self.__class__.__name__} '{self.name}' floating={self.floating} value={value}>"

    # LEGACY, deprecate?
    @property
    def lower_limit(self):
        return self.lower

    @lower_limit.setter
    def lower_limit(self, value):
        self.lower = value

    @property
    def upper_limit(self):
        return self.upper

    @upper_limit.setter
    def upper_limit(self, value):
        self.upper = value

    def __tf_tracing_type__(self, signature_context):
        return ParameterType(parameter=self)


# delattr(Parameter, "__tf_tracing_type__")


class ParameterType(VariableSpec):
    value_type = property(lambda self: Parameter)

    def __init__(
        self, shape=None, dtype=None, trainable=True, alias_id=None, *, parameter=None
    ):
        if parameter is None:
            raise RuntimeError("DEBUGGING HERE")
        if parameter is not None:  # initialize from parameter
            shape = parameter.shape
            dtype = parameter.dtype
            trainable = True
            alias_id = None
        self.parameter_value = parameter
        self._name = parameter.name
        self.parameter_type = type(self)
        if dtype is None:
            dtype = tf.float64
        super().__init__(
            shape=shape, dtype=dtype, trainable=trainable, alias_id=alias_id
        )
        self.hash = hash(self.name)

    @classmethod
    def from_value(cls, value):
        return cls(parameter=value)

    def _to_components(self, value):
        return super()._to_components(value)

    def _from_components(self, components):
        _ = super()._from_components(components)  # checking that there is no error
        return Parameter._from_name(self.name)

    def _to_tensors(self, value):
        return [value]

    def is_subtype_of(self, other):
        return (
            type(other) is ParameterType
            and self.parameter_type is other.parameter_type
            and self.name == other.name
        )

    def most_specific_common_supertype(self, others):
        return self if all(self == other for other in others) else None

    def placeholder_value(self, placeholder_context=None):
        return self.parameter_value

    def __eq__(self, other) -> bool:
        return self.parameter_type == type(other) and self.name == other.name

    def __hash__(self):
        return self.hash


# class ParameterType(VariableSpec):
#     value_type = property(lambda self: Parameter)
#
#     def __init__(self, shape=None, dtype=None, trainable=True, alias_id=None, *, parameter=None):
#         if parameter is not None:  # initialize from parameter
#             shape = parameter.shape
#             dtype = parameter.dtype
#             trainable = True
#             alias_id = None
#         self.parameter_value = parameter
#         if dtype is None:
#             dtype = tf.float64
#         super().__init__(shape=shape, dtype=dtype, trainable=trainable, alias_id=alias_id)
#
#     def is_compatible_with(self, spec_or_value):
#         return super().is_compatible_with(spec_or_value) and self.name == spec_or_value.name
#
#     def is_subtype_of(self, other):
#         return super().is_subtype_of(other) and self.name == other.name
#
#     def most_specific_compatible_type(self, other: "TypeSpec") -> "TypeSpec":
#         if self.is_subtype_of(self, other):
#             return self
#
#     def placeholder_value(self, placeholder_context):
#         return self.parameter_value
#         # return super().placeholder_value(placeholder_context) if self.parameter_value is None else self.parameter_value


# class ParameterType(tf.types.experimental.TraceType):
#     def __init__(self, parameter):
#         self.name = parameter.name
#         self.parameter_value = parameter
#         self.parameter_type = type(parameter)
#
#     def is_subtype_of(self, other) -> bool:
#         """Returns True if `self` is a subtype of `other`.
#
#         Implements the tf.types.experimental.func.TraceType interface.
#
#         If not overridden by a subclass, the default behavior is to assume the
#         TypeSpec is covariant upon attributes that implement TraceType and
#         invariant upon rest of the attributes as well as the structure and type
#         of the TypeSpec.
#
#         Args:
#           other: A TraceType object.
#         """
#
#         if self.parameter_type is not type(other):
#             return False
#
#         return self.name == other.name
#
#     def most_specific_common_supertype(self, others):
#         if self.name == others.name:
#             return self.name
#         else:
#             return None
#
#     def placeholder_value(self, placeholder_context=None):
#         return self.parameter_value
#
#     def __eq__(self, other):
#         return type(other) == self.parameter_type and self.name == other.name


class ParameterRepr(BaseRepr):
    _implementation = Parameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(Parameter))
    hs3_type: Literal["Parameter"] = Field("Parameter", alias="type")
    name: str
    value: float
    lower: Optional[float] = Field(None, alias="min")
    upper: Optional[float] = Field(None, alias="max")
    step_size: Optional[float] = None
    floating: Optional[bool] = None

    @validator("value", pre=True)
    def _validate_value(cls, v):
        if cls.orm_mode(v):
            v = v()
        return v

    def __hash__(self):
        return hash(self.name) + hash(type(self))


class BaseComposedParameter(ZfitParameterMixin, OverloadableMixin, BaseParameter):
    def __init__(
        self, params, value_fn, dtype=None, name="BaseComposedParameter", **kwargs
    ):
        # 0.4 breaking
        if "value" in kwargs:
            raise BreakingAPIChangeError(
                "'value' cannot be provided any longer, `value_fn` is needed."
            )
        super().__init__(name=name, params=params, **kwargs)
        if not hasattr(self, "_composed_param_original_order"):
            self._composed_param_original_order = None
        if not callable(value_fn):
            raise TypeError("`value_fn` is not callable.")
        n_func_params = len(signature(value_fn).parameters)
        # TODO(0.6): change, remove legacy?
        if n_func_params == 0:
            if len(params) == 0:
                warnings.warn(
                    "No `params` specified, the `value_fn` is supposed to return a constant. "
                    "Use preferably `ConstantParameter` instead",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:  # this is the legacy case where the function didn't take arguments
                warnings.warn(
                    "The `value_fn` for composed parameters should take the same number"
                    " of arguments as `params` are given.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                legacy_value_fn = value_fn

                def value_fn(*_):
                    return legacy_value_fn()

            # end legacy

        self._value_fn = value_fn
        self._dtype = dtype if dtype is not None else ztypes.float

    def _get_dependencies(self):
        return _extract_dependencies(list(self.params.values()))

    @property
    def floating(self):
        raise LogicalUndefinedOperationError(
            "Cannot be floating or not. Look at the dependencies."
        )

    @floating.setter
    def floating(self, value):
        raise LogicalUndefinedOperationError(
            "Cannot set floating or not. Set in the dependencies (`get_params`)."
        )

    @property
    def params(self):
        return self._params

    def value(self):
        params = self.params
        parameters = signature(self._value_fn).parameters
        if (
            len(parameters) == 1
            and (len(params) > 1 or "params" in parameters)
            and self._composed_param_original_order is None
        ):
            value = self._value_fn(params)
        elif (
            self._composed_param_original_order is None
        ):  # TODO: this is a temp fix for legacy behavior
            try:
                value = self._value_fn(
                    **params
                )  # since the order is None, it has to be a dict
            except Exception as error:
                raise RuntimeError(
                    "This should not be reached. To fix this, make sure that the params to"
                    " ComposedParameter are a dict and that the function takes one single argument."
                ) from error
        else:
            params = (
                self._composed_param_original_order
            )  # to make sure we have the right order
            value = self._value_fn(*params)
        return tf.convert_to_tensor(value, dtype=self.dtype)

    def read_value(self):
        return tf.identity(self.value())

    @property
    def shape(self):
        return self.value().shape

    def numpy(self):
        return self.value().numpy()

    @property
    def independent(self):
        return False

    def set_value(self, value):
        """Set the value of the parameter. Cannot be used for composed parameters!

        Args:
            value:
        """
        raise LogicalUndefinedOperationError(
            "Cannot set value of a composed parameter."
            " Set the value on its components."
        )

    def randomize(self, minval=None, maxval=None, sampler=np.random.uniform):
        """Randomize the value of the parameter.

        Cannot be used for composed parameters!
        """
        raise LogicalUndefinedOperationError(
            "Cannot randomize a composed parameter." " Randomize the components."
        )

    def assign(self, value, use_locking=False, name=None, read_value=True):
        """Assign the value of the parameter.

        Cannot be used for composed parameters!
        """
        raise LogicalUndefinedOperationError(
            "Cannot assign a composed parameter." " Assign the value on its components."
        )


class ConstantParameter(
    OverloadableMixin, ZfitParameterMixin, BaseParameter, SerializableMixin
):
    """Constant parameter.

    Value cannot change.
    """

    def __init__(self, name, value):
        """Constant parameter that cannot change its value.

        Args:
            name: Unique identifier of the parameter.
            value: Constant value.
        """
        super().__init__(name=name, params={}, dtype=ztypes.float)
        self._value_np = tf.get_static_value(value, partial=True)
        self._value = tf.guarantee_const(tf.convert_to_tensor(value, dtype=self.dtype))

    @property
    def shape(self):
        return self.value().shape

    def value(self) -> tf.Tensor:
        return self._value

    def read_value(self) -> tf.Tensor:
        return self.value()

    @property
    def floating(self):
        return False

    @floating.setter
    def floating(self, value):
        raise LogicalUndefinedOperationError(
            "Cannot set a ConstantParameter to floating. Use a `Parameter` instead."
        )

    @property
    def independent(self) -> bool:
        return False

    def _get_dependencies(self) -> ztyping.DependentsType:
        return OrderedSet()

    @property
    def static_value(self):
        return self._value_np

    def numpy(self):
        return self._value_np

    def __repr__(self):
        value = self._value_np
        if value is not None:
            value_str = f"{value: .4g}"
        else:
            value_str = "symbolic"
        return f"<zfit.param.{self.__class__.__name__} '{self.name}' dtype={self.dtype.name} value={value_str}>"


register_tensor_conversion(
    ConstantParameter, "ConstantParameter", overload_operators=True
)
register_tensor_conversion(
    BaseComposedParameter, "BaseComposedParameter", overload_operators=True
)


class ConstantParamRepr(BaseRepr):
    _implementation = ConstantParameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(ConstantParameter))
    hs3_type: Literal["ConstantParameter"] = pydantic.Field(
        "ConstantParameter", alias="type"
    )
    name: str
    value: float
    floating: bool = False

    # lower: Optional[float] = Field(None, alias="min")
    # upper: Optional[float] = Field(None, alias="max")

    @validator("value", pre=True)
    def _validate_value(cls, value):
        if cls.orm_mode(value):
            value = value()
        return value

    def _to_orm(self, init):
        init = copy.copy(init)
        init.pop("floating")
        out = super()._to_orm(init)
        return out


class ComposedParameter(SerializableMixin, BaseComposedParameter):
    @deprecated_args(None, "Use `params` instead.", "dependents")
    def __init__(
        self,
        name: str,
        value_fn: Callable,
        params: (
            dict[str, ZfitParameter] | Iterable[ZfitParameter] | ZfitParameter
        ) = NotSpecified,
        dtype: tf.dtypes.DType = ztypes.float,
        *,
        dependents: (
            dict[str, ZfitParameter] | Iterable[ZfitParameter] | ZfitParameter
        ) = NotSpecified,
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
            value_fn: Function that returns the value of the composed parameter and takes as arguments `params` as
                arguments. The function must be able to be called with the same arguments as `params`.
            params: If it is a `dict`, this will directly be used as the `params` attribute, otherwise the
                parameters will be automatically named with f"param_{i}". The values act as arguments to `value_fn`.
            dtype: Output of `value_fn` dtype
            dependents:
                .. deprecated:: unknown
                    use `params` instead.
        """
        original_init = {"name": name, "value_fn": value_fn}
        if dependents is not NotSpecified:
            params = dependents
        elif params is NotSpecified:
            raise ValueError
        if isinstance(params, collections.abc.Mapping):
            self._composed_param_original_order = None
        else:
            self._composed_param_original_order = convert_to_container(params)
        if isinstance(params, dict):
            params_dict = params
        else:
            params = convert_to_container(params)
            if params is None:
                params_dict = {}
            else:
                params_dict = {f"param_{i}": p for i, p in enumerate(params)}
        original_init[
            "params"
        ] = params_dict  # needs to be here, we need the params to be a dict for the serialization
        super().__init__(params=params_dict, value_fn=value_fn, name=name, dtype=dtype)
        self.hs3.original_init.update(original_init)

    def __repr__(self):
        if tf.executing_eagerly():
            value = f"{self.numpy():.4g}"
        else:
            value = "graph-node"
        return f"<zfit.{self.__class__.__name__} '{self.name}' params={[(k, p.name) for k, p in self.params.items()]} value={value}>"


class ComposedParameterRepr(BaseRepr):
    _implementation = ComposedParameter
    _constructor = pydantic.PrivateAttr(make_param_constructor(ComposedParameter))
    hs3_type: Literal["ComposedParameter"] = pydantic.Field(
        "ComposedParameter", alias="type"
    )
    name: str
    value_fn: str
    params: Dict[str, Serializer.types.ParamTypeDiscriminated]

    @validator("value_fn", pre=True)
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
        value_fn = init.pop("value_fn")
        init["value_fn"] = dill.loads(bytes.fromhex(value_fn))
        out = super()._to_orm(init)
        return out


class ComplexParameter(ComposedParameter):  # TODO: change to real, imag as input?
    def __init__(self, name, value_fn, params, dtype=ztypes.complex):
        """Create a complex parameter.

        .. note::
            Use the constructor class methods instead of the __init__() constructor:

            - :py:meth:`ComplexParameter.from_cartesian`
            - :py:meth:`ComplexParameter.from_polar`
        """
        super().__init__(name, value_fn=value_fn, params=params, dtype=dtype)
        self._conj = None
        self._mod = None
        self._arg = None
        self._imag = None
        self._real = None

    @classmethod
    def from_cartesian(
        cls, name, real, imag, dtype=ztypes.complex, floating=True
    ) -> ComplexParameter:  # TODO: correct dtype handling, also below
        """Create a complex parameter from cartesian coordinates.

        Args:
            name: Name of the parameter.
            real: Real part of the complex number.
            imag: Imaginary part of the complex number.
        """
        real = convert_to_parameter(
            real, name=name + "_real", prefer_constant=not floating
        )
        imag = convert_to_parameter(
            imag, name=name + "_imag", prefer_constant=not floating
        )
        param = cls(
            name=name,
            value_fn=lambda _real, _imag: tf.cast(
                tf.complex(_real, _imag), dtype=dtype
            ),
            params=[real, imag],
        )
        param._real = real
        param._imag = imag
        return param

    @classmethod
    def from_polar(
        cls, name, mod, arg, dtype=ztypes.complex, floating=True, **kwargs
    ) -> ComplexParameter:
        """Create a complex parameter from polar coordinates.

        Args:
            name: Name of the parameter.
            mod: Modulus (r) the complex number.
            arg: Argument (phi) of the complex number.
        """
        mod = convert_to_parameter(
            mod, name=name + "_mod", prefer_constant=not floating
        )
        arg = convert_to_parameter(
            arg, name=name + "_arg", prefer_constant=not floating
        )
        param = cls(
            name=name,
            value_fn=lambda _mod, _arg: tf.cast(
                tf.complex(_mod * znp.cos(_arg), _mod * znp.sin(_arg)), dtype=dtype
            ),
            params=[mod, arg],
        )
        param._mod = mod
        param._arg = arg
        return param

    @property
    def conj(self):
        """Returns a complex conjugated copy of the complex parameter."""
        if self._conj is None:
            self._conj = ComplexParameter(
                name=f"{self.name}_conj",
                value_fn=lambda: znp.conj(self),
                params=self.get_cache_deps(),
                dtype=self.dtype,
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
    prefer_constant: bool = None,
    lower=None,
    upper=None,
    step_size=None,
):
    if prefer_constant is None:
        prefer_constant = True
    if isinstance(value, collections.abc.Mapping):
        return convert_to_parameters(**value, prefer_constant=False)
    value = convert_to_container(value)
    is_param_already = [isinstance(val, ZfitIndependentParameter) for val in value]
    if all(is_param_already):
        return value
    elif any(is_param_already):
        raise ValueError(
            f"value has to be either ZfitParameters or values, not mixed (currently)."
            f" Is {value}."
        )
    params_dict = {
        "value": value,
        "name": name,
        "lower": lower,
        "upper": upper,
        "step_size": step_size,
    }
    params_dict = {
        key: convert_to_container(val)
        for key, val in params_dict.items()
        if val is not None
    }
    lengths = {len(v) for v in params_dict.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent length in values when converting the parameters: {params_dict}"
        )

    params = []
    for i in range(len(params_dict["value"])):
        pdict = {k: params_dict[k][i] for k in params_dict}
        params.append(convert_to_parameter(**pdict, prefer_constant=prefer_constant))
    return params


@deprecated_args(None, "Use `params` instead.", "dependents")
def convert_to_parameter(
    value,
    name: str | None = None,
    prefer_constant: bool = True,
    params=None,
    lower=None,
    upper=None,
    step_size=None,
    # legacy
    dependents=None,
) -> ZfitParameter:
    """Convert a *numerical* to a constant/floating parameter or return if already a parameter.

    Args:
        value:
        name:
        prefer_constant: If True, create a ConstantParameter instead of a Parameter, if possible.
        params:
        lower:
        upper:
        step_size:
    """
    # legacy start
    if dependents is not None:
        params = dependents
    # legacy end
    if name is not None:
        name = str(name)

    if callable(value):
        if params is None:
            raise ValueError(
                "If the value is a callable, the params have to be specified as an empty list/tuple"
            )
        return ComposedParameter(
            f"Composed_autoparam_{get_auto_number()}", value_fn=value, params=params
        )

    if isinstance(value, ZfitParameter):  # TODO(Mayou36): autoconvert variable. TF 2.0?
        return value
    elif isinstance(value, tf.Variable):
        raise TypeError("Currently, cannot autoconvert tf.Variable to zfit.Parameter.")

    # convert to Tensor
    if not isinstance(value, tf.Tensor):
        if isinstance(value, complex):
            value = z.to_complex(value)
        else:
            value = z.to_real(value)

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
        value = ComplexParameter.from_cartesian(
            name, real=complex_params[0], imag=complex_params[1]
        )

    else:
        if prefer_constant:
            if name is None:
                name = (
                    "FIXED_autoparam_" + str(get_auto_number())
                    if name is None
                    else name
                )
            value = ConstantParameter(name, value=value)

        else:
            name = "autoparam_" + str(get_auto_number()) if name is None else name
            value = Parameter(
                name=name, value=value, lower=lower, upper=upper, step_size=step_size
            )

    return value


@z.function(wraps="params")
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
    params, values = check_convert_param_values_assign(
        params, values, allow_partial=allow_partial
    )
    # params = tuple(params)
    assign_values_jit(params=params, values=values, use_locking=use_locking)


def set_values(
    params: Parameter | Iterable[Parameter],
    values: (
        ztyping.NumericalScalarType | Iterable[ztyping.NumericalScalarType] | ZfitResult
    ),
    allow_partial: bool | None = None,
):
    """Set the values (using a context manager or not) of multiple parameters.

    Args:
        params: Parameters to set the values.
        values: List-like object that supports indexing or a `FitResult`.
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
    params, values = check_convert_param_values_assign(params, values, allow_partial)

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
        values: List-like object that supports indexing or a `ZfitResult`.
        allow_partial: Allow to set only parts of the parameters in case values is a `ZfitResult`

    Returns:
        A tuple of (params, values)
    """
    params = convert_to_container(params)
    if isinstance(values, ZfitResult):
        result = values
        new_params = []
        values = []
        for param in params:
            if param in result.params:
                values.append(result.params[param]["value"])
                new_params.append(param)
            elif not allow_partial:
                raise ValueError(
                    f"Cannot set {param} with {repr(result)} as it is not contained. To partially set"
                    f" the parameters (only those in the result), use allow_partial"
                )
        params = new_params
    elif len(params) > 1:
        if not tf.is_tensor(values) or isinstance(values, np.ndarray):
            values = convert_to_container(values)
            if len(params) != len(values):
                raise ValueError(
                    f"Incompatible length of parameters and values: {params}, {values}"
                )
    not_param = [param for param in params if not isinstance(param, ZfitParameter)]
    if not_param:
        raise TypeError(
            f"The following are not parameters (but should be): {not_param}"
        )
    non_independent_params = [param for param in params if not param.independent]
    if non_independent_params:
        raise ParameterNotIndependentError(
            f"trying to set value of parameters that are not independent "
            f"{non_independent_params}"
        )
    values = znp.asarray(values, dtype=znp.float64)
    return params, values
