#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.functions import ProdFunc, SumFunc
    from ..models.functor import ProductPDF, SumPDF

from collections.abc import Callable

import zfit.z.numpy as znp

from ..util import ztyping
from ..util.exception import (
    BreakingAPIChangeError,
    FunctionNotImplemented,
    IntentionAmbiguousError,
    ModelIncompatibleError,
)
from .interfaces import ZfitFunc, ZfitModel, ZfitParameter, ZfitPDF
from .parameter import convert_to_parameter


def multiply(object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType) -> ztyping.BaseObjectType:
    """Multiply two objects and return a new object (may depending on the old).

    Args:
        object1: A ZfitParameter, ZfitFunc or ZfitPDF to multiply with object2
        object2: A ZfitParameter, ZfitFunc or ZfitPDF to multiply with object1
    Raises:
        TypeError: if one of the objects is neither a ZfitFunc, ZfitPDF or convertable to a ZfitParameter
    """
    # converting the objects to known types
    object1, object2 = _convert_to_known(object1, object2)
    new_object = None

    # object 1 is ZfitParameter
    if isinstance(object1, ZfitParameter):
        if isinstance(object2, ZfitParameter):
            new_object = multiply_param_param(param1=object1, param2=object2)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_param_func(param=object1, func=object2)
        elif isinstance(object2, ZfitPDF):
            msg = "Not supported since a while"
            raise BreakingAPIChangeError(msg)
        else:
            msg = "This code should never be reached due to logical reasons. Mistakes happen..."
            raise AssertionError(msg)

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitParameter):
            new_object = multiply_param_func(param=object2, func=object1)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_func_func(func1=object1, func2=object2)
        elif isinstance(object2, ZfitPDF):
            msg = "Cannot multiply a function with a model. Use `func.as_pdf` or `model.as_func`."
            raise ModelIncompatibleError(msg)

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF) and isinstance(object2, ZfitPDF):
        new_object = multiply_pdf_pdf(pdf1=object1, pdf2=object2)

    if new_object is None:
        msg = (
            f"Multiplication for {object1} and {object2} of type {type(object1)} and {type(object2)} is not"
            "properly defined. (may change the order)"
            ""
        )
        raise ModelIncompatibleError(msg)
    return new_object


def multiply_pdf_pdf(pdf1: ZfitPDF, pdf2: ZfitPDF, name: str = "multiply_pdf_pdf") -> ProductPDF:
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        msg = f"`pdf1` and `pdf2` need to be `ZfitPDF` and not {pdf1}, {pdf2}"
        raise TypeError(msg)
    from ..models.functor import ProductPDF

    if not pdf1.is_extended and pdf2.is_extended:
        msg = (
            "Cannot multiply this way a non-extendended PDF with an extended PDF."
            "Only vice-versa is allowed: to multiply an extended PDF with an "
            "non-extended PDF."
        )
        raise IntentionAmbiguousError(msg)

    return ProductPDF(pdfs=[pdf1, pdf2], name=name)


def multiply_func_func(func1: ZfitFunc, func2: ZfitFunc, name: str = "multiply_func_func") -> ProdFunc:
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        msg = f"`func1` and `func2` need to be `ZfitFunc` and not {func1}, {func2}"
        raise TypeError(msg)
    from ..models.functions import ProdFunc

    return ProdFunc(funcs=[func1, func2], name=name)


def multiply_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        msg = "`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not " f"{param}, {func}"
        raise TypeError(msg)
    from ..models.functions import SimpleFuncV1

    def combined_func(x):
        return param * func.func(x=x)

    params = {param.name: param}
    params.update(func.params)
    return SimpleFuncV1(func=combined_func, obs=func.obs, **params)  # TODO: implement with new parameters


def multiply_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        msg = f"`param1` and `param2` need to be `ZfitParameter` and not {param1}, {param2}"
        raise TypeError(msg)
    return znp.multiply(param1, param2)


# Addition logic
def add(object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType) -> ztyping.BaseObjectType:
    """Add two objects and return a new object (may depending on the old).

    Args:
        object1: A ZfitParameter, ZfitFunc or ZfitPDF to add with object2
        object2: A ZfitParameter, ZfitFunc or ZfitPDF to add with object1
    """
    # converting the objects to known types
    object1, object2 = _convert_to_known(object1, object2)
    new_object = None
    # convert anything we can, otherwise raise an FunctionNotImplemented error

    # object 1 is ZfitParameter
    if isinstance(object1, ZfitParameter):
        if isinstance(object2, ZfitFunc):
            new_object = add_param_func(param=object1, func=object2)

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitParameter):
            new_object = add_param_func(param=object2, func=object1)
        elif isinstance(object2, ZfitFunc):
            new_object = add_func_func(func1=object1, func2=object2)
        elif isinstance(object2, ZfitPDF):
            msg = "Cannot add a function with a model. Use `func.as_pdf` or `model.as_func`."
            raise TypeError(msg)

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitFunc):
            msg = "Cannot add a function with a model. Use `func.as_pdf` or `model.as_func`."
            raise TypeError(msg)
        if isinstance(object2, ZfitPDF):
            new_object = add_pdf_pdf(pdf1=object1, pdf2=object2)

    if new_object is None:
        msg = (
            f"Addition for {object1} and {object2} of type {type(object1)} and {type(object2)} is not"
            "properly defined. (may change the order)"
            ""
        )
        raise FunctionNotImplemented(msg)
    return new_object


def _convert_to_known(object1, object2):
    objects = []
    for obj in (object1, object2):
        if not isinstance(obj, (ZfitModel,)):
            try:
                obj = convert_to_parameter(obj)
            except TypeError as error:
                msg = "Object is neither an instance of ZfitModel nor convertible to a Parameter: " f"{obj}"
                raise TypeError(msg) from error
        objects.append(obj)
    object1, object2 = objects
    return object1, object2


def add_pdf_pdf(pdf1: ZfitPDF, pdf2: ZfitPDF, name: str = "add_pdf_pdf") -> SumPDF:
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        msg = f"`pdf1` and `pdf2` need to be `ZfitPDF` and not {pdf1}, {pdf2}"
        raise TypeError(msg)
    if not (pdf1.is_extended and pdf2.is_extended):
        msg = (
            "Adding (non-extended) pdfs is not allowed anymore due to disambiguity."
            "Use the `zfit.pdf.SumPDF([pdf, other_pdf], frac)` syntax instead."
        )
        raise BreakingAPIChangeError(msg)
    from ..models.functor import SumPDF

    return SumPDF(pdfs=[pdf1, pdf2], name=name)


def add_func_func(func1: ZfitFunc, func2: ZfitFunc, name: str = "add_func_func") -> SumFunc:
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        msg = f"`func1` and `func2` need to be `ZfitFunc` and not {func1}, {func2}"
        raise TypeError(msg)
    from ..models.functions import SumFunc

    return SumFunc(funcs=[func1, func2], name=name)


def add_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        msg = "`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not " f"{param}, {func}"
        raise TypeError(msg)
    msg = "This is not supported. Probably in the future."
    raise NotImplementedError(msg)  # TODO: implement with new parameters


def add_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        msg = f"`param1` and `param2` need to be `ZfitParameter` and not {param1}, {param2}"
        raise TypeError(msg)
    # use the default behavior of variables
    return znp.add(param1, param2)


# Conversions


def convert_pdf_to_func(pdf: ZfitPDF, norm: ztyping.LimitsType) -> ZfitFunc:
    def value_func(x):
        return pdf.pdf(x, norm=norm)

    from ..models.functions import SimpleFuncV1

    return SimpleFuncV1(func=value_func, obs=pdf.obs, name=pdf.name + "_as_func", **pdf.params)


def convert_func_to_pdf(func: ZfitFunc | Callable, obs=None, name=None) -> ZfitPDF:
    func_name = "autoconverted_func_to_pdf" if name is None else name
    if not isinstance(func, ZfitFunc) and callable(func):
        if obs is None:
            msg = "If `func` is a function, `obs` has to be specified."
            raise ValueError(msg)
        from ..models.functions import SimpleFuncV1

        func = SimpleFuncV1(func=func, obs=obs, name=func_name)
    from ..models.special import SimplePDF

    name = func.name if name is None else func_name
    return SimplePDF(func=func.func, obs=func.obs, name=name, **func.params)
