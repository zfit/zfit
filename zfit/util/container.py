#  Copyright (c) 2021 zfit

from typing import Any, Callable, Iterable, Union

import tensorflow as tf
from uhi.typing.plottable import PlottableHistogram

import zfit.binned


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
    from ..core.interfaces import ZfitData, ZfitBinnedData  # here due to dependency
    from ..core.interfaces import ZfitLoss, ZfitModel, ZfitParameter, ZfitSpace
    import zfit
    if non_containers is None:
        non_containers = []
    if not isinstance(non_containers, list):
        raise TypeError("`non_containers` have to be a list or a tuple")
    if value is None and not convert_none:
        return value
    if not type(value) == container:
        non_containers.extend([str, tf.Tensor, ZfitData, ZfitLoss, ZfitModel, ZfitSpace, ZfitParameter,
                               ZfitBinnedData, zfit.binned.RegularBinning, zfit.binned.VariableBinning,
                               PlottableHistogram])
        try:
            if isinstance(value, tuple(non_containers)):
                raise TypeError  # we can't convert, it's a non-container
            value = container(value)
        except (TypeError, AttributeError):  # by tf, it can't convert
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
