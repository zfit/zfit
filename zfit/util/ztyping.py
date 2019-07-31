#  Copyright (c) 2019 zfit

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

_IterLimitsTypeReturn = Union[Tuple['zfit.Space'], Tuple[Tuple[Tuple[float]]], Tuple[Tuple[float]]]

AxesTypeInput = Union[int, Iterable[int]]
AxesTypeReturn = Union[Tuple[int], None]

ObsTypeInput = Union[str, Iterable[str], "zfit.Space"]
ObsTypeReturn = Union[Tuple[str, ...], None]

# Space
SpaceOrSpacesTypeInput = Union["zfit.Space", Iterable["zfit.Space"]]
SpaceType = "zfit.Space"

# Data
XType = Union[float, tf.Tensor]
XTypeInput = Union[np.ndarray, tf.Tensor, "zfit.Data"]
XTypeReturn = Union[tf.Tensor, "zfit.Data"]
NumericalTypeReturn = Union[tf.Tensor, np.array]

DataInputType = Union["zfit.Data", Iterable["zfit.Data"]]

WeightsStrInputType = Union[tf.Tensor, None, np.ndarray, str]
WeightsInputType = Union[tf.Tensor, None, np.ndarray]

# Models
ModelsInputType = Union['zfit.core.interfaces.ZfitModel',
                        Iterable['zfit.core.interfaces.ZfitModel']]

PDFInputType = Union['zfit.core.interfaces.ZfitPDF',
                     Iterable['zfit.core.interfaces.ZfitPDF']]

FuncInputType = Union['zfit.core.interfaces.ZfitFunc',
                      Iterable['zfit.core.interfaces.ZfitFunc']]

NumericalScalarType = Union[int, float, complex, tf.Tensor]

# Integer sampling
nSamplingTypeIn = Union[int, tf.Tensor, str]

# Parameter
ParamsTypeOpt = Optional[Iterable['zfit.core.interfaces.ZfitParameter']]
ParamsNameOpt = Optional[Union[str, List[str]]]
ParamsOrNameType = Optional[Union[ParamsTypeOpt, Iterable[str]]]
ParametersType = Dict[str, "zfit.core.interfaces.ZfitParameter"]
ParamTypeInput = Union['zfit.core.interfaces.ZfitParameter', NumericalScalarType]

# TensorFlow specific
SessionType = Optional[tf.Session]

# Zfit Structure
BaseObjectType = Union['zfit.core.interfaces.ZfitParameter',
                       'zfit.core.interfaces.ZfitFunction',
                       'zfit.core.interfaces.ZfitPDF']
DependentsType = Set['zfit.Parameter']

# Caching
CacherOrCachersType = Union['zfit.core.interfaces.ZfitCachable',
                            Iterable['zfit.core.interfaces.ZfitCachable']]

try:
    from typing import OrderedDict
except ImportError:  # < python 3.7
    OrderedDict = Dict

