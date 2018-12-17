import typing
from typing import Union, Tuple, Iterable, Optional, List, Dict, Set

import numpy as np
import tensorflow as tf

# space
LowerTypeInput = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
LowerTypeReturn = Union[Tuple[Tuple[float, ...]], None, "False"]

UpperTypeInput = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
UpperTypeReturn = Union[Tuple[Tuple[float, ...]], None, "False"]

LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], "False"]
LimitsTypeSimpleInput = Union[Tuple[float, float], "False"]
LimitsTypeInput = Union[Tuple[Tuple[Tuple[float, ...]], Tuple[Tuple[float, ...]]], "False"]
LimitsTypeReturn = Union[Tuple[Tuple[Tuple[float, ...]], Tuple[Tuple[float, ...]]], None, "False"]

_IterLimitsTypeReturn = Union[List['NamedSpace'], List[Tuple[Tuple[float]]], Tuple[Tuple[float]]]

AxesTypeInput = Union[int, Tuple[int, ...]]
AxesTypeReturn = Union[List[int], None]

ObsTypeInput = Union[str, List[str], "NamedSpace"]
ObsTypeReturn = Union[Tuple[str, ...], None]
SpaceTypeReturn = "NamedSpace"

# Data
XType = Union[float, tf.Tensor]
XTypeInput = Union[np.ndarray, tf.Tensor, "Data"]
XTypeReturn = Union[tf.Tensor, "Data"]
NumericalTypeReturn = Union[tf.Tensor, np.array]

# Parameter
ParamsType = Optional[Iterable['ZfitParameter']]
ParamsNameOpt = Optional[Union[str, List[str]]]
ParamsOrNameType = Optional[Union[ParamsType, Iterable[str]]]
ParametersType = Dict[str, "ZfitParameter"]

# TensorFlow specific
SessionType = Optional[tf.Session]

# Zfit Structure
BaseObjectType = Union['ZfitParameter', 'ZfitFunction', 'ZfitPDF']
DependentsType = Set['Parameter']

try:
    from typing import OrderedDict
except ImportError:  # < python 3.7
    OrderedDict = Dict
