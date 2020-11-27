#  Copyright (c) 2021 zfit

from typing import Any, Callable, Iterable, Union

import tensorflow as tf


def convert_to_container(value: Any, container: Callable = list, non_containers=None,
                         convert_none=False) -> Union[None, Iterable]:
    """Convert `value` into a `container` storing `value` if `value` is not yet a python container.

    Args:
        value:
        container: Converts a tuple to a container.
        non_containers: Types that do not count as a container. Has to
            be a list of types. As an example, if `non_containers` is [list, tuple] and the value
            is [5, 3] (-> a list with two entries),this won't be converted to the `container` but end
            up as (if the container is e.g. a tuple): ([5, 3],) (a tuple with one entry).

    Returns:
    """
    from ..core.interfaces import ZfitData  # here due to dependency
    from ..core.interfaces import ZfitLoss, ZfitModel, ZfitParameter, ZfitSpace
    if non_containers is None:
        non_containers = []
    if not isinstance(non_containers, list):
        raise TypeError("`non_containers` have to be a list or a tuple")
    if value is None and not convert_none:
        return value
    if not isinstance(value, container):
        try:
            non_containers.extend([str, tf.Tensor, ZfitData, ZfitLoss, ZfitModel, ZfitSpace, ZfitParameter])
            if isinstance(value, tuple(non_containers)):
                raise TypeError
            value = container(value)
        except TypeError:
            value = container((value,))
    return value


def is_container(obj):
    """Check if `object` is a list or a tuple.

    Args:
        obj:

    Returns:
        True if it is a *container*, otherwise False
    """
    return isinstance(obj, (list, tuple))
