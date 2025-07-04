#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable, Mapping

# ruff: noqa: F821
from typing import (
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import tensorflow as tf

# space
from ordered_set import OrderedSet
from tensorflow.python.types.core import TensorLike

# The #: symbols at the end of every type alias are for marking the module level variables
# as documented, such that sphinx will document them.
from uhi.typing.plottable import PlottableHistogram

if typing.TYPE_CHECKING:
    import zfit
LowerTypeInput = Union[tf.Tensor, np.ndarray, tuple[float], list[float], float]  #:
LowerTypeReturn = Union[np.ndarray, tf.Tensor, None, bool]  #:

UpperTypeInput = LowerTypeInput  #:

UpperTypeReturn = LowerTypeReturn  #:

LowerRectTypeInput = Union[tf.Tensor, np.ndarray, Iterable[float], float]  #:
LowerRectTypeReturn = Union[np.ndarray, tf.Tensor, None, bool]  #:

UpperRectTypeInput = LowerTypeInput  #:
UpperRectTypeReturn = LowerTypeReturn  #:

RectLowerReturnType = Union[np.ndarray, tf.Tensor, float]  #:
RectUpperReturnType = RectLowerReturnType  #:
RectLimitsReturnType = tuple[RectLowerReturnType, RectUpperReturnType]  #:
RectLimitsTFReturnType = tuple[tf.Tensor, tf.Tensor]  #:
RectLimitsNPReturnType = tuple[np.ndarray, np.ndarray]  #:

RectLimitsInputType = Union[LowerRectTypeInput, UpperRectTypeInput]  #:

LimitsType = Union[tuple[tuple[float, ...]], tuple[float, ...], bool, "zfit.Space"]  #:
LimitsTypeSimpleInput = Union[tuple[float, float], bool]  #:
LimitsTypeInput = Union[tuple[tuple[tuple[float, ...]]], tuple[float, float], bool]  #:
LimitsTypeReturn = Union[tuple[tuple[tuple[float, ...]], tuple[tuple[float, ...]]], None, bool]  #:

NumericalType = Union[int, float, np.ndarray, TensorLike]  #:
LimitsTypeInput = Union["zfit.core.interfaces.ZfitLimit", RectLimitsInputType, bool, None]  #:
LimitsTypeInputV1 = Union[Iterable[NumericalType], NumericalType, bool, None]  #:
LimitsFuncTypeInput = Union[LimitsTypeInput, Callable]  #:
LimitsTypeReturn = Union[tuple[np.ndarray, np.ndarray], None, bool]  #:

_IterLimitsTypeReturn = Union[tuple["zfit.Space"], tuple[tuple[tuple[float]]], tuple[tuple[float]]]  #:

AxesTypeInput = Union[int, Iterable[int]]  #:
AxesTypeReturn = Union[tuple[int], None]  #:

ObsTypeInput = Union[str, Iterable[str], "zfit.Space"]  #:
ObsTypeReturn = Union[tuple[str, ...], None]  #:
ObsType = tuple[str]  #:

# Space
SpaceOrSpacesTypeInput = Union["zfit.Space", Iterable["zfit.Space"]]  #:
SpaceType = "zfit.Space"  #:
NormInputType = Optional["zfit.Space"]  #:

# Data
XType = Union[float, tf.Tensor]  #:
XTypeInput = Union[np.ndarray, tf.Tensor, "zfit.Data"]  #:
XTypeReturnNoData = Union[np.ndarray, tf.Tensor]  #:
XTypeReturn = Union[XTypeReturnNoData, "zfit.Data"]  #:
NumericalTypeReturn = Union[tf.Tensor, np.array]  #:

DataInputType = Union["zfit.Data", Iterable["zfit.Data"]]  #:
BinnedDataInputType = Union[PlottableHistogram, Iterable[PlottableHistogram]]  #:
ZfitBinnedDataInputType = Union["zfit.data.BinnedData", Iterable["zfit.data.BinnedData"]]  #:
AnyDataInputType = Union[DataInputType, BinnedDataInputType]  #:

WeightsStrInputType = Union[tf.Tensor, None, np.ndarray, str]  #:
WeightsInputType = Union[tf.Tensor, None, np.ndarray]  #:

# Models
ModelsInputType = Union["zfit.core.interfaces.ZfitModel", Iterable["zfit.core.interfaces.ZfitModel"]]  #:

PDFInputType = Union["zfit.core.interfaces.ZfitPDF", Iterable["zfit.core.interfaces.ZfitPDF"]]  #:
BinnedPDFInputType = Union["zfit.core.interfaces.ZfitBinnedPDF", Iterable["zfit.core.interfaces.ZfitBinnedPDF"]]  #:
BinnedHistPDFInputType = Union[BinnedPDFInputType, PlottableHistogram, Iterable[PlottableHistogram]]  #:

FuncInputType = Union["zfit.core.interfaces.ZfitFunc", Iterable["zfit.core.interfaces.ZfitFunc"]]  #:

NumericalScalarType = Union[int, float, complex, tf.Tensor, "zfit.core.interfaces.ZfitParameter"]  #:
NumericalType = Union[int, float, np.ndarray, tf.Tensor, "zfit.core.interfaces.ZfitParameter"]  #:

# Integer sampling
nSamplingTypeIn = Union[int, tf.Tensor, str]  #:

ConstraintsTypeInput = Optional[
    Union[
        Iterable[Union["zfit.core.interfaces.ZfitConstraint", Callable]],
        "zfit.core.interfaces.ZfitConstraint",
        Callable,
    ]
]  #:

# Parameter
ParamsTypeOpt = Optional[Iterable["zfit.core.interfaces.ZfitParameter"]]  #:
ParamsNameOpt = Optional[str | list[str]]  #:
ParamsOrNameType = Optional[ParamsTypeOpt | Iterable[str]]  #:
ParameterType = TypeVar("ParameterType", bound=dict[str, "zfit.core.interfaces.ZfitParameter"])  #:
ParametersType = Iterable[ParameterType]
ParamTypeInput = TypeVar("ParamTypeInput", "zfit.core.interfaces.ZfitParameter", NumericalScalarType)  #:
ParamsTypeInput = Mapping[Union[str, "zfit.core.interfaces.ZfitParameter"], ParamTypeInput]  #:

ExtendedInputType = Optional[bool | ParamTypeInput]  #:

# Zfit Structure
BaseObjectType = Union[
    "zfit.core.interfaces.ZfitParameter",
    "zfit.core.interfaces.ZfitFunction",
    "zfit.core.interfaces.ZfitPDF",
]  #:
DependentsType = OrderedSet  #:

# Caching
CacherOrCachersType = Union[
    "zfit.core.interfaces.ZfitGraphCachable",
    Iterable["zfit.core.interfaces.ZfitGraphCachable"],
]  #:

LimitsDictAxes = dict[tuple[int], "zfit.core.interfaces.ZfitLimit"]  #:
LimitsDictObs = dict[tuple[str], "zfit.core.interfaces.ZfitLimit"]  #:
LimitsDictNoCoords = Union[LimitsDictAxes, LimitsDictObs]  #:
LimitsDictWithCoords = dict[str, LimitsDictNoCoords]  #:

BinningTypeInput = Union[Iterable["ZfitBinning"], "ZfitBinning", int]  #:
OptionsInputType = Optional[Mapping[str, object]]  #:
ConstraintsInputType = Optional[
    Union[
        "zfit.core.interfaces.ZfitConstraint",
        Iterable["zfit.core.interfaces.ZfitConstraint"],
    ]
]  #:
ArrayLike = tf.types.experimental.TensorLike  #:

ParamValuesMap = Optional[Mapping[Union[str, "zfit.core.interfaces.ZfitParameter"], NumericalScalarType]]
