import typing
from typing import Union, Tuple, Iterable, Optional, List, Dict, Set

import numpy as np
import tensorflow as tf

InputLowerType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
ReturnLowerType = Tuple[Tuple[float, ...]]
InputUpperType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], float]
ReturnUpperType = Tuple[Tuple[float, ...]]
LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...], "False"]
DimsType = Tuple[str, ...]
XType = Union[float, tf.Tensor]
ParamsType = Optional[Iterable['ZfitParameter']]
ParamsNameOpt = Optional[Union[str, List[str]]]
ParamsOrNameType = Optional[Union[ParamsType, Iterable[str]]]
ParametersType = Dict[str, "ZfitParameter"]
SessionType = Optional[tf.Session]
BaseObjectType = Union['ZfitParameter', 'ZfitFunction', 'ZfitPDF']
DependentsType = Set['Parameter']
ReturnNumericalType = Union[tf.Tensor, np.array]
InputObservableType = Union[str, List[str]]

try:
    from typing import OrderedDict
except ImportError:  # < python 3.7
    OrderedDict = Dict
