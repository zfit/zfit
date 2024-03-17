#  Copyright (c) 2022 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.functor import ProductPDF, SumPDF
    from ..models.functions import ProdFunc, SumFunc

from collections.abc import Callable

import zfit.z.numpy as znp

from ..util import ztyping
from ..util.exception import (
    BreakingAPIChangeError,
    IntentionAmbiguousError,
    ModelIncompatibleError,
)
from .interfaces import ZfitFunc, ZfitModel, ZfitParameter, ZfitPDF
from .parameter import convert_to_parameter


def multiply(
    object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType
) -> ztyping.BaseObjectType:
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
            new_object = multiply_param_pdf(param=object1, pdf=object2)
        else:
            assert (
                False
            ), "This code should never be reached due to logical reasons. Mistakes happen..."

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitParameter):
            new_object = multiply_param_func(param=object2, func=object1)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_func_func(func1=object1, func2=object2)
        elif isinstance(object2, ZfitPDF):
            raise ModelIncompatibleError(
                "Cannot multiply a function with a model. Use `func.as_pdf` or `model.as_func`."
            )

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitPDF):
            new_object = multiply_pdf_pdf(pdf1=object1, pdf2=object2)

    if new_object is None:
        raise ModelIncompatibleError(
            "Multiplication for {} and {} of type {} and {} is not"
            "properly defined. (may change the order)"
            "".format(object1, object2, type(object1), type(object2))
        )
    return new_object


def multiply_pdf_pdf(
    pdf1: ZfitPDF, pdf2: ZfitPDF, name: str = "multiply_pdf_pdf"
) -> ProductPDF:
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError(
            f"`pdf1` and `pdf2` need to be `ZfitPDF` and not {pdf1}, {pdf2}"
        )
    from ..models.functor import ProductPDF

    if not pdf1.is_extended and pdf2.is_extended:
        raise IntentionAmbiguousError(
            "Cannot multiply this way a non-extendended PDF with an extended PDF."
            "Only vice-versa is allowed: to multiply an extended PDF with an "
            "non-extended PDF."
        )

    return ProductPDF(pdfs=[pdf1, pdf2], name=name)


def multiply_func_func(
    func1: ZfitFunc, func2: ZfitFunc, name: str = "multiply_func_func"
) -> ProdFunc:
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError(
            f"`func1` and `func2` need to be `ZfitFunc` and not {func1}, {func2}"
        )
    from ..models.functions import ProdFunc

    return ProdFunc(funcs=[func1, func2], name=name)


def multiply_param_pdf(param: ZfitParameter, pdf: ZfitPDF) -> ZfitPDF:
    # TODO(SUM): remove it completely?
    raise BreakingAPIChangeError(
        "Unfortunately, it is not allowed anymore to multiply pdfs with `frac * pdf + other_pdf` syntax"
        "due to ambiguity. Use the `zfit.pdf.SumPDF([pdf, other_pdf], frac)` syntax instead."
    )

    # if not (isinstance(param, ZfitParameter) and isinstance(pdf, ZfitPDF)):
    #     raise TypeError("`param` and `model` need to be `ZfitParameter` resp. `ZfitPDF` and not "
    #                     "{}, {}".format(param, pdf))
    # if pdf.is_extended:
    #     raise AlreadyExtendedPDFError()
    #
    # new_pdf = pdf.create_extended(param, name_addition="_autoextended")
    # return new_pdf


def multiply_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        raise TypeError(
            "`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not "
            "{}, {}".format(param, func)
        )
    from ..models.functions import SimpleFuncV1

    def combined_func(x):
        return param * func.func(x=x)

    params = {param.name: param}
    params.update(func.params)
    new_func = SimpleFuncV1(
        func=combined_func, obs=func.obs, **params
    )  # TODO: implement with new parameters
    return new_func


def multiply_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        raise TypeError(
            f"`param1` and `param2` need to be `ZfitParameter` and not {param1}, {param2}"
        )
    param = znp.multiply(param1, param2)
    return param


# Addition logic
def add(
    object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType
) -> ztyping.BaseObjectType:
    """Add two objects and return a new object (may depending on the old).

    Args:
        object1: A ZfitParameter, ZfitFunc or ZfitPDF to add with object2
        object2: A ZfitParameter, ZfitFunc or ZfitPDF to add with object1
    """
    # converting the objects to known types
    object1, object2 = _convert_to_known(object1, object2)
    new_object = None

    # object 1 is ZfitParameter
    if isinstance(object1, ZfitParameter):
        if isinstance(object2, ZfitParameter):
            new_object = add_param_param(param1=object1, param2=object2)
        elif isinstance(object2, ZfitFunc):
            new_object = add_param_func(param=object1, func=object2)

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitParameter):
            new_object = add_param_func(param=object2, func=object1)
        elif isinstance(object2, ZfitFunc):
            new_object = add_func_func(func1=object1, func2=object2)
        elif isinstance(object2, ZfitPDF):
            raise TypeError(
                "Cannot add a function with a model. Use `func.as_pdf` or `model.as_func`."
            )

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitFunc):
            raise TypeError(
                "Cannot add a function with a model. Use `func.as_pdf` or `model.as_func`."
            )
        elif isinstance(object2, ZfitPDF):
            new_object = add_pdf_pdf(pdf1=object1, pdf2=object2)

    if new_object is None:
        raise TypeError(
            "Addition for {} and {} of type {} and {} is not"
            "properly defined. (may change the order)"
            "".format(object1, object2, type(object1), type(object2))
        )
    return new_object


def _convert_to_known(object1, object2):
    objects = []
    for obj in (object1, object2):
        if not isinstance(obj, (ZfitModel,)):
            try:
                obj = convert_to_parameter(obj)
            except TypeError as error:
                raise TypeError(
                    "Object is neither an instance of ZfitModel nor convertible to a Parameter: "
                    "{}".format(obj)
                )
        objects.append(obj)
    object1, object2 = objects
    return object1, object2


def add_pdf_pdf(pdf1: ZfitPDF, pdf2: ZfitPDF, name: str = "add_pdf_pdf") -> SumPDF:
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError(
            f"`pdf1` and `pdf2` need to be `ZfitPDF` and not {pdf1}, {pdf2}"
        )
    if not (pdf1.is_extended and pdf2.is_extended):
        raise BreakingAPIChangeError(
            "Adding (non-extended) pdfs is not allowed anymore due to disambiguity."
            "Use the `zfit.pdf.SumPDF([pdf, other_pdf], frac)` syntax instead."
        )
    from ..models.functor import SumPDF

    return SumPDF(pdfs=[pdf1, pdf2], name=name)


def add_func_func(
    func1: ZfitFunc, func2: ZfitFunc, name: str = "add_func_func"
) -> SumFunc:
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError(
            f"`func1` and `func2` need to be `ZfitFunc` and not {func1}, {func2}"
        )
    from ..models.functions import SumFunc

    return SumFunc(funcs=[func1, func2], name=name)


def add_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        raise TypeError(
            "`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not "
            "{}, {}".format(param, func)
        )
    raise NotImplementedError(
        "This is not supported. Probably in the future."
    )  # TODO: implement with new parameters


def add_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        raise TypeError(
            f"`param1` and `param2` need to be `ZfitParameter` and not {param1}, {param2}"
        )
    # use the default behavior of variables
    return znp.add(param1, param2)


# Conversions


def convert_pdf_to_func(pdf: ZfitPDF, norm: ztyping.LimitsType) -> ZfitFunc:
    def value_func(x):
        return pdf.pdf(x, norm=norm)

    from ..models.functions import SimpleFuncV1

    func = SimpleFuncV1(
        func=value_func, obs=pdf.obs, name=pdf.name + "_as_func", **pdf.params
    )
    return func


def convert_func_to_pdf(func: ZfitFunc | Callable, obs=None, name=None) -> ZfitPDF:
    func_name = "autoconverted_func_to_pdf" if name is None else name
    if not isinstance(func, ZfitFunc) and callable(func):
        if obs is None:
            raise ValueError("If `func` is a function, `obs` has to be specified.")
        from ..models.functions import SimpleFuncV1

        func = SimpleFuncV1(func=func, obs=obs, name=func_name)
    from ..models.special import SimplePDF

    name = func.name if name is None else func_name
    pdf = SimplePDF(func=func.func, obs=func.obs, name=name, **func.params)
    return pdf
