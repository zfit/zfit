from typing import Optional, Tuple

from .interfaces import ZfitModel, ZfitFunc, ZfitPDF, ZfitParameter
from .parameter import convert_to_parameter, ComposedParameter
from ..util import ztyping
from ..util.exception import LogicalUndefinedOperationError, AlreadyExtendedPDFError


def multiply(object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType,
             dims: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> ztyping.BaseObjectType:
    """Multiply two objects and return a new object (may depending on the old).

    Args:
        object1 (): A ZfitParameter, ZfitFunc or ZfitPDF to multiply with object2
        object2 ():
        dims (Tuple[Tuple[int, ...], Tuple[int, ...]]): The dimensions to multiply the objects in.
            So for example if object1 and object2 are 1-dimensional objects, then ((0,), (0,)) will
            return a 1-dim object and correspond to the normal multiplication while ((0,), (1,)) will
            create a 2-dim object.
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
            assert False, "This code should never be reached due to logical reasons. Mistakes happen..."

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitParameter):
            new_object = multiply_param_func(param=object2, func=object1)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_func_func(func1=object1, func2=object2, dims=dims)
        elif isinstance(object2, ZfitPDF):
            raise TypeError("Cannot multiply a function with a pdf. Use `func.as_pdf` or `pdf.as_func`.")

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitPDF):
            new_object = multiply_pdf_pdf(pdf1=object1, pdf2=object2, dims=dims)

    if new_object is None:
        raise TypeError("Multiplication for {} and {} of type {} and {} is not"
                        "properly defined. (may change the order)"
                        "".format(object1, object2, type(object1), type(object2)))
    return new_object


def multiply_pdf_pdf(pdf1: ZfitPDF, pdf2: ZfitPDF, dims: ztyping.DimsType = None,
                     name: str = "multiply_pdf_pdf") -> "ProductPDF":
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError("`pdf1` and `pdf2` need to be `ZfitPDF` and not {}, {}".format(pdf1, pdf2))
    from ..models.functor import ProductPDF

    return ProductPDF(pdfs=[pdf1, pdf2], dims=dims, name=name)


def multiply_func_func(func1: ZfitFunc, func2: ZfitFunc, dims: ztyping.DimsType = None,
                       name: str = "multiply_func_func") -> "ProdFunc":
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError("`func1` and `func2` need to be `ZfitFunc` and not {}, {}".format(func1, func2))
    from ..models.functions import ProdFunc

    return ProdFunc(funcs=[func1, func2], dims=dims, name=name)


def multiply_param_pdf(param: ZfitParameter, pdf: ZfitPDF) -> ZfitPDF:
    if not (isinstance(param, ZfitParameter) and isinstance(pdf, ZfitPDF)):
        raise TypeError("`param` and `pdf` need to be `ZfitParameter` resp. `ZfitPDF` and not "
                        "{}, {}".format(param, pdf))
    if pdf.is_extended:
        raise AlreadyExtendedPDFError()
    pdf.set_yield(param)  # TODO: make unmutable with copy?
    return pdf


def multiply_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        raise TypeError("`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not "
                        "{}, {}".format(param, func))
    from ..models.functions import SimpleFunction

    def combined_func(x):
        return param * func.value(x=x)

    params = {param.name: param}
    params.update(func.parameters)
    new_func = SimpleFunction(func=combined_func, **params)  # TODO: implement with new parameters
    return new_func


def multiply_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        raise TypeError("`param1` and `param2` need to be `ZfitParameter` and not {}, {}".format(param1, param2))
    param = param1 * param2
    return ComposedParameter(name=param1.name + "_mult_" + param2.name, tensor=param)


# Addition logic
def add(object1: ztyping.BaseObjectType, object2: ztyping.BaseObjectType,
        dims: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> ztyping.BaseObjectType:
    """Add two objects and return a new object (may depending on the old).

    Args:
        object1 (): A ZfitParameter, ZfitFunc or ZfitPDF to add with object2
        object2 ():
        dims (Tuple[Tuple[int, ...], Tuple[int, ...]]): The dimensions to add the objects in.
            So for example if object1 and object2 are 1-dimensional objects, then ((0,), (0,)) will
            return a 1-dim object and correspond to the normal multiplication while ((0,), (1,)) will
            create a 2-dim object.
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
            new_object = add_func_func(func1=object1, func2=object2, dims=dims)
        elif isinstance(object2, ZfitPDF):
            raise TypeError("Cannot add a function with a pdf. Use `func.as_pdf` or `pdf.as_func`.")

    # object 1 is PDF
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitFunc):
            raise TypeError("Cannot add a function with a pdf. Use `func.as_pdf` or `pdf.as_func`.")
        elif isinstance(object2, ZfitPDF):
            new_object = add_pdf_pdf(pdf1=object1, pdf2=object2, dims=dims)

    if new_object is None:
        raise TypeError("Addition for {} and {} of type {} and {} is not"
                        "properly defined. (may change the order)"
                        "".format(object1, object2, type(object1), type(object2)))
    return new_object


def _convert_to_known(object1, object2):
    objects = []
    for obj in (object1, object2):
        if not isinstance(obj, ZfitModel):
            try:
                obj = convert_to_parameter(obj)
            except TypeError as error:
                raise TypeError("Object is neither an instance of ZfitModel nor convertible to a parameter: "
                                "{}".format(obj))
        objects.append(obj)
    object1, object2 = objects
    return object1, object2


def add_pdf_pdf(pdf1: ZfitPDF, pdf2: ZfitPDF, dims: ztyping.DimsType = None, name: str = "add_pdf_pdf") -> "SumPDF":
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError("`pdf1` and `pdf2` need to be `ZfitPDF` and not {}, {}".format(pdf1, pdf2))
    from ..models.functor import SumPDF

    return SumPDF(pdfs=[pdf1, pdf2], dims=dims, name=name)


def add_func_func(func1: ZfitFunc, func2: ZfitFunc, dims: ztyping.DimsType = None,
                  name: str = "add_func_func") -> "SumFunc":
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError("`func1` and `func2` need to be `ZfitFunc` and not {}, {}".format(func1, func2))
    from ..models.functions import SumFunc

    return SumFunc(funcs=[func1, func2], dims=dims, name=name)


def add_param_func(param: ZfitParameter, func: ZfitFunc) -> ZfitFunc:
    if not (isinstance(param, ZfitParameter) and isinstance(func, ZfitFunc)):
        raise TypeError("`param` and `func` need to be `ZfitParameter` resp. `ZfitFunc` and not "
                        "{}, {}".format(param, func))
    raise NotImplementedError("This is not supported. Probably in the future.")  # TODO: implement with new parameters


def add_param_param(param1: ZfitParameter, param2: ZfitParameter) -> ZfitParameter:
    if not (isinstance(param1, ZfitParameter) and isinstance(param2, ZfitParameter)):
        raise TypeError("`param1` and `param2` need to be `ZfitParameter` and not {}, {}".format(param1, param2))

    param = param1 + param2
    return ComposedParameter(name=param1.name + "_add_" + param2.name, tensor=param)


# Conversions

def convert_pdf_to_func(pdf: ZfitPDF, norm_range: ztyping.LimitsType) -> ZfitFunc:
    def value_func(x):
        return pdf.pdf(x, norm_range=norm_range)

    from ..models.functions import SimpleFunction

    func = SimpleFunction(func=value_func, name=pdf.name + "_as_func", **pdf.parameters)
    return func


def convert_func_to_pdf(func: ZfitFunc) -> ZfitPDF:
    if not isinstance(func, ZfitFunc) and callable(func):
        from ..models.functions import SimpleFunction
        func = SimpleFunction(func=func)
    from ..models.special import SimplePDF
    pdf = SimplePDF(func=func.value, name=func.name, **func.parameters)
    return pdf
