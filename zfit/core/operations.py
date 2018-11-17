from typing import Optional, Tuple

import zfit
from zfit.core.basefunc import ZfitFunc
from zfit.core.basemodel import ZfitModel
from zfit.core.basepdf import ZfitPDF
from zfit.core.parameter import convert_to_parameter
from zfit.pdfs.functions import ProdFunc, SimpleFunction, SumFunc
from zfit.pdfs.functor import ProductPDF, SumPDF
from zfit.pdfs.special import SimplePDF
from zfit.util import ztyping
from zfit.util.exception import LogicalUndefinedOperationError


def multiply(object1: ztyping.ZfitObjectType, object2: ztyping.ZfitObjectType,
             dims: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> ztyping.ZfitObjectType:
    """Multiply two objects and return a new object (may depending on the old).

    Args:
        object1 (): A Parameter, ZfitFunc or ZfitPDF to multiply with object2
        object2 ():
        dims (Tuple[Tuple[int, ...], Tuple[int, ...]]): The dimensions to multiply the objects in.
            So for example if object1 and object2 are 1-dimensional objects, then ((0,), (0,)) will
            return a 1-dim object and correspond to the normal multiplication while ((0,), (1,)) will
            create a 2-dim object.
    Raises:
        TypeError: if one of the objects is neither a ZfitFunc, ZfitPDF or convertable to a Parameter
    """
    # converting the objects to known types
    object1, object2 = _convert_to_known(object1, object2)
    new_object = None

    # object 1 is Parameter
    if isinstance(object1, zfit.Parameter):
        if isinstance(object2, zfit.Parameter):
            new_object = multiply_param_param(param1=object1, param2=object2)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_param_func(param=object1, func=object2)
        elif isinstance(object2, ZfitPDF):
            new_object = multiply_param_pdf(param=object1, pdf=object2)
        else:
            assert False, "This code should never be reached due to logical reasons. Mistakes happen..."

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, zfit.Parameter):
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
        raise LogicalUndefinedOperationError("Multiplication for {} and {} of type {} and {} is not"
                                             "properly defined. (may change the order)"
                                             "".format(object1, object2, type(object1), type(object2)))


def multiply_pdf_pdf(pdf1, pdf2, dims=None, name="multiply_pdf_pdf"):
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError("`pdf1` and `pdf2` need to be `ZfitPDF` and not {}, {}".format(pdf1, pdf2))

    return ProductPDF(pdfs=[pdf1, pdf2], dims=dims, name=name)


def multiply_func_func(func1, func2, dims=None, name="multiply_func_func"):
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError("`func1` and `func2` need to be `ZfitFunc` and not {}, {}".format(func1, func2))

    return ProdFunc(funcs=[func1, func2], dims=dims, name=name)


def multiply_param_pdf(param, pdf):
    if not (isinstance(param, zfit.Parameter) and isinstance(pdf, ZfitPDF)):
        raise TypeError("`param` and `pdf` need to be `zfit.Parameter` resp. `ZfitPDF` and not "
                        "{}, {}".format(param, pdf))
    raise NotImplementedError("TODO")  # TODO: implement


def multiply_param_func(param, func):
    if not (isinstance(param, zfit.Parameter) and isinstance(func, ZfitFunc)):
        raise TypeError("`param` and `func` need to be `zfit.Parameter` resp. `ZfitFunc` and not "
                        "{}, {}".format(param, func))
    raise NotImplementedError("TODO")  # TODO: implement with new parameters


def multiply_param_param(param1, param2):
    if not (isinstance(param1, zfit.Parameter) and isinstance(param2, zfit.Parameter)):
        raise TypeError("`param1` and `param2` need to be `zfit.Parameter` and not {}, {}".format(param1, param2))

    raise NotImplementedError("TODO")  # TODO: implement with new parameters


# Addition logic
def add(object1: ztyping.ZfitObjectType, object2: ztyping.ZfitObjectType,
        dims: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> ztyping.ZfitObjectType:
    """Add two objects and return a new object (may depending on the old).

    Args:
        object1 (): A Parameter, ZfitFunc or ZfitPDF to add with object2
        object2 ():
        dims (Tuple[Tuple[int, ...], Tuple[int, ...]]): The dimensions to add the objects in.
            So for example if object1 and object2 are 1-dimensional objects, then ((0,), (0,)) will
            return a 1-dim object and correspond to the normal multiplication while ((0,), (1,)) will
            create a 2-dim object.
    """
    # converting the objects to known types
    object1, object2 = _convert_to_known(object1, object2)
    new_object = None

    # object 1 is Parameter
    if isinstance(object1, zfit.Parameter):
        if isinstance(object2, zfit.Parameter):
            new_object = add_param_param(param1=object1, param2=object2)
        elif isinstance(object2, ZfitFunc):
            new_object = add_param_func(param=object1, func=object2)
        else:
            assert False, "This code should never be reached due to logical reasons. Mistakes happen..."

    # object 1 is Function
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitFunc):
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
        raise LogicalUndefinedOperationError("Multiplication for {} and {} of type {} and {} is not"
                                             "properly defined. (may change the order)"
                                             "".format(object1, object2, type(object1), type(object2)))


def _convert_to_known(object1, object2):
    objects = []
    for obj in (object1, object2):
        if not isinstance(object1, ZfitModel):
            try:
                obj = convert_to_parameter(obj)
            except TypeError as error:
                raise TypeError("Object is neither an instance of ZfitModel nor convertible to parameter: "
                                "{}".format(obj))
            else:
                objects.append(obj)
    object1, object2 = objects
    return object1, object2


def add_pdf_pdf(pdf1, pdf2, dims=None, name="add_pdf_pdf"):
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError("`pdf1` and `pdf2` need to be `ZfitPDF` and not {}, {}".format(pdf1, pdf2))

    return SumPDF(pdfs=[pdf1, pdf2], dims=dims, name=name)


def add_func_func(func1, func2, dims=None, name="add_func_func"):
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError("`func1` and `func2` need to be `ZfitFunc` and not {}, {}".format(func1, func2))

    return SumFunc(funcs=[func1, func2], dims=dims, name=name)


def add_param_func(param, func):
    if not (isinstance(param, zfit.Parameter) and isinstance(func, ZfitFunc)):
        raise TypeError("`param` and `func` need to be `zfit.Parameter` resp. `ZfitFunc` and not "
                        "{}, {}".format(param, func))
    raise NotImplementedError("This is not supported. Probably in the future.")  # TODO: implement with new parameters


def add_param_param(param1, param2):
    if not (isinstance(param1, zfit.Parameter) and isinstance(param2, zfit.Parameter)):
        raise TypeError("`param1` and `param2` need to be `zfit.Parameter` and not {}, {}".format(param1, param2))

    raise NotImplementedError("TODO")  # TODO: implement with new parameters

# Conversions

def convert_pdf_to_func(pdf, norm_range):
    def value_func(x):
        return pdf.prob(x, norm_range=norm_range)

    func = SimpleFunction(func=value_func, name=pdf.name + "_as_func", **pdf.get_parameters(only_floating=False))
    return func

def convert_func_to_pdf(func):
    if not isinstance(func, ZfitFunc) and callable(func):
        func = SimpleFunction(func=func)
    pdf = SimplePDF(func=func.value, name=func.name, **func.get_parameters(only_floating=False))
    return pdf
