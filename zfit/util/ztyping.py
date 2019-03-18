from typing import Union, Tuple, Iterable, Optional, List, Dict, Set

import numpy as np
import tensorflow as tf

# space

LowerTypeInput = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
LowerTypeReturn = Union[Tuple[Tuple[float, ...]], None, bool]

UpperTypeInput = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
UpperTypeReturn = Union[Tuple[Tuple[float, ...]], None, bool]

LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], bool]
LimitsTypeSimpleInput = Union[Tuple[float, float], bool]
LimitsTypeInput = Union[Tuple[Tuple[Tuple[float, ...]]], Tuple[float, float], bool]
LimitsTypeReturn = Union[Tuple[Tuple[Tuple[float, ...]], Tuple[Tuple[float, ...]]], None, bool]

_IterLimitsTypeReturn = Union[Tuple['Space'], Tuple[Tuple[Tuple[float]]], Tuple[Tuple[float]]]

AxesTypeInput = Union[int, Iterable[int]]
AxesTypeReturn = Union[Tuple[int], None]

ObsTypeInput = Union[str, Iterable[str], "Space"]
ObsTypeReturn = Union[Tuple[str, ...], None]

# Space
SpaceOrSpacesTypeInput = Union["zfit.Space", Iterable["zfit.Space"]]
SpaceTypeReturn = "zfit.Space"

# Data
XType = Union[float, tf.Tensor]
XTypeInput = Union[np.ndarray, tf.Tensor, "zfit.Data"]
XTypeReturn = Union[tf.Tensor, "zfit.Data"]
NumericalTypeReturn = Union[tf.Tensor, np.array]

DataInputType = Union["zfit.Data", Iterable["zfit.Data"]]

WeightsStrInputType = Union[tf.Tensor, None, np.ndarray, str]
WeightsInputType = Union[tf.Tensor, None, np.ndarray]

# Models
ModelsInputType = Union['ZfitModel', Iterable['ZfitModel']]

PDFInputType = Union['ZfitPDF', Iterable['ZfitPDF']]

FuncInputType = Union['ZfitFunc', Iterable['ZfitFunc']]

NumericalScalarType = Union[int, float, complex, tf.Tensor]

# Integer sampling
nSamplingTypeIn = Union[int, tf.Tensor, str]

# Parameter
ParamsTypeOpt = Optional[Iterable['ZfitParameter']]
ParamsNameOpt = Optional[Union[str, List[str]]]
ParamsOrNameType = Optional[Union[ParamsTypeOpt, Iterable[str]]]
ParametersType = Dict[str, "ZfitParameter"]
ParamTypeInput = Union['ZfitParameter', NumericalScalarType]

# TensorFlow specific
SessionType = Optional[tf.Session]

# Zfit Structure
BaseObjectType = Union['ZfitParameter', 'ZfitFunction', 'ZfitPDF']
DependentsType = Set['zfit.Parameter']

# Caching
CacherOrCachersType = Union['ZfitCachable', Iterable['ZfitCachable']]

try:
    from typing import OrderedDict
except ImportError:  # < python 3.7
    OrderedDict = Dict

