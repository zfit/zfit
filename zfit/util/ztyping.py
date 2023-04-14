#  Copyright (c) 2023 zfit
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Mapping,
)

import numpy as np
import tensorflow as tf

# space
from ordered_set import OrderedSet

# The #: symbols at the end of every type alias are for marking the module level variables
# as documented, such that sphinx will document them.
from uhi.typing.plottable import PlottableHistogram

LowerTypeInput = Union[tf.Tensor, np.ndarray, Tuple[float], List[float], float]  #:
LowerTypeReturn = Union[np.ndarray, tf.Tensor, None, bool]  #:

UpperTypeInput = LowerTypeInput  #:

UpperTypeReturn = LowerTypeReturn  #:

LowerRectTypeInput = Union[tf.Tensor, np.ndarray, Iterable[float], float]  #:
LowerRectTypeReturn = Union[np.ndarray, tf.Tensor, None, bool]  #:

UpperRectTypeInput = LowerTypeInput  #:
UpperRectTypeReturn = LowerTypeReturn  #:

RectLowerReturnType = Union[np.ndarray, tf.Tensor, float]  #:
RectUpperReturnType = RectLowerReturnType  #:
RectLimitsReturnType = Tuple[RectLowerReturnType, RectUpperReturnType]  #:
RectLimitsTFReturnType = Tuple[tf.Tensor, tf.Tensor]  #:
RectLimitsNPReturnType = Tuple[np.ndarray, np.ndarray]  #:

RectLimitsInputType = Union[LowerRectTypeInput, UpperRectTypeInput]  #:

LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], bool, "zfit.Space"]  #:
LimitsTypeSimpleInput = Union[Tuple[float, float], bool]  #:
LimitsTypeInput = Union[Tuple[Tuple[Tuple[float, ...]]], Tuple[float, float], bool]  #:
LimitsTypeReturn = Union[
    Tuple[Tuple[Tuple[float, ...]], Tuple[Tuple[float, ...]]], None, bool
]  #:

LimitsTypeInput = Union[
    "zfit.core.interfaces.ZfitLimit", RectLimitsInputType, bool, None
]  #:
LimitsFuncTypeInput = Union[LimitsTypeInput, Callable]  #:
LimitsTypeReturn = Union[Tuple[np.ndarray, np.ndarray], None, bool]  #:

_IterLimitsTypeReturn = Union[
    Tuple["zfit.Space"], Tuple[Tuple[Tuple[float]]], Tuple[Tuple[float]]
]  #:

AxesTypeInput = Union[int, Iterable[int]]  #:
AxesTypeReturn = Union[Tuple[int], None]  #:

ObsTypeInput = Union[str, Iterable[str], "zfit.Space"]  #:
ObsTypeReturn = Union[Tuple[str, ...], None]  #:
ObsType = Tuple[str]  #:

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
ZfitBinnedDataInputType = Union[
    "zfit.data.BinnedData", Iterable["zfit.data.BinnedData"]
]  #:
AnyDataInputType = Union[DataInputType, BinnedDataInputType]  #:

WeightsStrInputType = Union[tf.Tensor, None, np.ndarray, str]  #:
WeightsInputType = Union[tf.Tensor, None, np.ndarray]  #:

# Models
ModelsInputType = Union[
    "zfit.core.interfaces.ZfitModel", Iterable["zfit.core.interfaces.ZfitModel"]
]  #:

PDFInputType = Union[
    "zfit.core.interfaces.ZfitPDF", Iterable["zfit.core.interfaces.ZfitPDF"]
]  #:
BinnedPDFInputType = Union[
    "zfit.core.interfaces.ZfitBinnedPDF", Iterable["zfit.core.interfaces.ZfitBinnedPDF"]
]  #:

FuncInputType = Union[
    "zfit.core.interfaces.ZfitFunc", Iterable["zfit.core.interfaces.ZfitFunc"]
]  #:

NumericalScalarType = Union[
    int, float, complex, tf.Tensor, "zfit.core.interfaces.ZfitParameter"
]  #:
NumericalType = Union[
    int, float, np.ndarray, tf.Tensor, "zfit.core.interfaces.ZfitParameter"
]  #:

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
ParamsNameOpt = Optional[Union[str, List[str]]]  #:
ParamsOrNameType = Optional[Union[ParamsTypeOpt, Iterable[str]]]  #:
ParameterType = TypeVar(
    "ParameterType", bound=Dict[str, "zfit.core.interfaces.ZfitParameter"]
)  #:
ParametersType = Iterable[ParameterType]
ParamTypeInput = TypeVar(
    "ParamTypeInput", "zfit.core.interfaces.ZfitParameter", NumericalScalarType
)  #:

ExtendedInputType = Optional[Union[bool, ParamTypeInput]]  #:

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

OrderedDict = Dict

LimitsDictAxes = Dict[Tuple[int], "zfit.core.interfaces.ZfitLimit"]  #:
LimitsDictObs = Dict[Tuple[str], "zfit.core.interfaces.ZfitLimit"]  #:
LimitsDictNoCoords = Union[LimitsDictAxes, LimitsDictObs]  #:
LimitsDictWithCoords = Dict[str, LimitsDictNoCoords]  #:

BinningInput = Union[Iterable["ZfitBinning"], "ZfitBinning", int]  #:
OptionsInputType = Optional[Mapping[str, object]]  #:
ConstraintsInputType = Optional[
    Union[
        "zfit.core.interfaces.ZfitConstraint",
        Iterable["zfit.core.interfaces.ZfitConstraint"],
    ]
]  #:
ArrayLike = tf.types.experimental.TensorLike  #:
