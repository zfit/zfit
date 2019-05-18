#  Copyright (c) 2019 zfit

from typing import Callable, Any, Iterable, Union

import tensorflow as tf


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_to_container(value: Any, container: Callable = list, non_containers=None,
                         convert_none=False) -> Union[None, Iterable]:
    """Convert `value` into a `container` storing `value` if `value` is not yet a python container.

    Args:
        value (object):
        container (callable): Converts a tuple to a container.
        non_containers (Optional[List[Container]]): Types that do not count as a container. Has to
            be a list of types. As an example, if `non_containers` is [list, tuple] and the value
            is [5, 3] (-> a list with two entries),this won't be converted to the `container` but end
            up as (if the container is e.g. a tuple): ([5, 3],) (a tuple with one entry).

    Returns:

    """
    from ..core.interfaces import ZfitData, ZfitLoss, ZfitModel  # here due to dependency
    if non_containers is None:
        non_containers = []
    if not isinstance(non_containers, list):
        raise TypeError("`non_containers` have to be a list or a tuple")
    if value is None and not convert_none:
        return value
    if not isinstance(value, container):
        try:
            non_containers.extend([str, tf.Tensor, ZfitData, ZfitLoss, ZfitModel])
            if isinstance(value, tuple(non_containers)):
                raise TypeError
            value = container(value)
        except TypeError:
            value = container((value,))
    return value


def is_container(obj):
    """Check if `object` is a list or a tuple.

    Args:
        obj ():

    Returns:
        bool: True if it is a *container*, otherwise False
    """
    return isinstance(obj, (list, tuple))
