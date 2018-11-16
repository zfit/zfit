from typing import Optional, Tuple

import zfit
from zfit.core.basefunc import ZfitFunc
from zfit.core.basemodel import ZfitModel
from zfit.core.basepdf import ZfitPDF
from zfit.core.parameter import convert_to_parameter
from zfit.pdfs.functions import ProdFunc
from zfit.pdfs.functor import ProductPDF
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
    """
    # converting the objects to known types
    objects = []
    for obj in (object1, object2):
        if not isinstance(object1, ZfitModel):
            try:
                obj = convert_to_parameter(obj)
            except TypeError as error:
                raise TypeError("Object is neither an instance of ZfitModel nor convertable to parameter: "
                                "{}".format(obj))
            else:
                objects.append(obj)
    object1, object2 = objects
    new_object = None

    if isinstance(object1, zfit.FitParameter):
        if isinstance(object2, zfit.FitParameter):
            new_object = multiply_param_param(param1=object1, param2=object2)
        elif isinstance(object2, ZfitFunc):
            new_object = multiply_param_func(param=object1, func=object2, dims=dims)
        elif isinstance(object2, ZfitPDF):
            new_object = multiply_param_pdf(param=object1, pdf=object2, dims=dims)
        else:
            assert False, "This code should never be reached due to logical reasons. Mistakes happen..."
    elif isinstance(object1, ZfitFunc):
        if isinstance(object2, ZfitFunc):
            new_object = multiply_func_func(func1=object1, func2=object2, dims=dims)
        elif isinstance(object2, ZfitPDF):
            raise TypeError("Cannot multiply a function with a pdf. Use `func.as_pdf` or `pdf.as_func`.")
    elif isinstance(object1, ZfitPDF):
        if isinstance(object2, ZfitPDF):
            new_object = mutliply_pdf_pdf(pdf1=object1, pdf2=object2, dims=dims)

    if new_object is None:
        raise LogicalUndefinedOperationError("Multiplication for {} and {} of type {} and {} is not"
                                             "properly defined. (may change the order)"
                                             "".format(object1, object2, type(object1), type(object2)))


def mutliply_pdf_pdf(pdf1, pdf2, dims=None):
    if not (isinstance(pdf1, ZfitPDF) and isinstance(pdf2, ZfitPDF)):
        raise TypeError("`pdf1` and `pdf2` need to be `ZfitPDF` and not {}, {}".format(pdf1, pdf2))

    return ProductPDF(pdfs=[pdf1, pdf2])


def multiply_func_func(func1, func2, dims=None):
    if not (isinstance(func1, ZfitFunc) and isinstance(func2, ZfitFunc)):
        raise TypeError("`param1` and `param2` need to be `ZfitFunc` and not {}, {}".format(func1, func2))

    return ProdFunc(funcs=[func1, func2])


def multiply_param_pdf(param, pdf, dims=None):
    if not (isinstance(param, zfit.FitParameter) and isinstance(pdf, ZfitPDF)):
        raise TypeError("`param` and `param` need to be `zfit.Parameter` resp. `ZfitPDF` and not "
                        "{}, {}".format(param, pdf))
    raise NotImplementedError("TODO")  # TODO: implement


def multiply_param_func(param, func, dims=None):
    if not (isinstance(param, zfit.FitParameter) and isinstance(func, ZfitFunc)):
        raise TypeError("`param` and `param` need to be `zfit.Parameter` resp. `ZfitFunc` and not "
                        "{}, {}".format(param, func))
    raise NotImplementedError("TODO")  # TODO: implement with new parameters


def multiply_param_param(param1, param2):
    if not (isinstance(param1, zfit.FitParameter) and isinstance(param2, zfit.FitParameter)):
        raise TypeError("`param1` and `param2` need to be `zfit.Parameter` and not {}, {}".format(param1, param2))

    raise NotImplementedError("TODO")  # TODO: implement with new parameters


