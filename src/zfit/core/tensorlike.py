#  Copyright (c) 2025 zfit
from __future__ import annotations

import functools
import typing

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from zfit import _interfaces as zinterfaces

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def register_tensor_conversion(
    convertable, name=None, overload_operators=True, priority=10
):  # higher than any tf conversion
    def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
        return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)

    ops.register_tensor_conversion_function(convertable, _dense_var_to_tensor, priority=priority)
    if name:
        pass
        # _pywrap_utils.RegisterType(name, convertable)

    if overload_operators:
        convertable._OverloadAllOperators(cls=convertable)


class OverloadableMixin:
    # Conversion to tensor.
    @staticmethod
    def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
        """Utility function for converting a Variable to a Tensor."""
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            msg = f"Incompatible type conversion requested to type '{dtype.name}' for variable of type '{v.dtype.name}'"
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

    @staticmethod
    def _OverloadAllOperators(cls):  # noqa: PLW0211
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            cls._OverloadOperator(cls, operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        cls.__getitem__ = array_ops._SliceHelperVar

    @staticmethod
    def _OverloadOperator(cls, operator):  # noqa: PLW0211
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


class OverloadableMixinValues:
    # Conversion to tensor.
    @staticmethod
    def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
        """Utility function for converting a Variable to a Tensor."""
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            msg = f"Incompatible type conversion requested to type '{dtype.name}' for variable of type '{v.dtype.name}'"
            raise ValueError(msg)
        if as_ref:
            return v._ref()  # pylint: disable=protected-access
        else:
            return v.values()

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
            return self.values()

    def _AsTensor(self):
        return self.values()

    @staticmethod
    def _OverloadAllOperators(cls):  # noqa: PLW0211
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            cls._OverloadOperator(cls, operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        cls.__getitem__ = array_ops._SliceHelperVar

    @staticmethod
    def _OverloadOperator(cls, operator):  # noqa: PLW0211
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
            return tensor_oper(a.values(), *args, **kwargs)

        functools.update_wrapper(_run_op, tensor_oper)
        setattr(cls, operator, _run_op)


class MetaBaseParameter(type(tf.Variable), type(zinterfaces.ZfitParameter)):  # resolve metaclasses
    pass
