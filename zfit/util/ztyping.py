from collections import OrderedDict
from typing import Union, Tuple, Iterable, Optional, List, Dict

import tensorflow as tf

LowerType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
UpperType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
DimsType = Union[Tuple[int, ...], Tuple[int, ...]]
XType = Union[float, tf.Tensor]
ParamsType = Optional[Iterable['ZfitParameter']]
ParamsNameOpt = Optional[Union[str, List[str]]]
ParamsOrNameType = Optional[Union[ParamsType, Iterable[str]]]
ParametersType = Dict[str, "ZfitParameter"]
SessionType = Optional[tf.Session]
BaseObjectType = Union['ZfitParameter', 'ZfitFunction', 'ZfitPDF']
DependentsType = List['Parameter']
