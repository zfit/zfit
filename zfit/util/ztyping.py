from typing import Union, Tuple

import tensorflow as tf

LowerType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
UpperType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
LimitsType = Union[Tuple[Tuple[float, ...]], Tuple[float, ...]]
DimsType = Union[Tuple[int, ...], Tuple[int, ...]]
XType = Union[float, tf.Tensor]
