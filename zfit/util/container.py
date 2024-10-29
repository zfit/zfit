#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from collections.abc import Callable, Iterable
from typing import Any

import tensorflow as tf
from uhi.typing.plottable import PlottableHistogram


def convert_to_container(
    value: Any, container: Callable = list, non_containers=None, ignore=None, convert_none=False
) -> None | Iterable:
    """Convert `value` into a `container` storing `value` if `value` is not yet a python container.

    Args:
        value:
        container: Converts a tuple to a container.
        non_containers: Types that do not count as a container. Has to
            be a list of types. As an example, if `non_containers` is [list, tuple] and the value
            is [5, 3] (-> a list with two entries),this won't be converted to the `container` but end
            up as (if the container is e.g. a tuple): ([5, 3],) (a tuple with one entry).

            By default, the following types are added to `non_containers`:
            [str, tf.Tensor, ZfitData, ZfitLoss, ZfitModel, ZfitSpace, ZfitParameter, ZfitBinnedData,
            ZfitBinning, PlottableHistogram, pd.DataFrame, pd.Series, np.ndarray]

    Returns:
    """
    from ..core.interfaces import (  # here due to dependency
        ZfitBinnedData,
        ZfitBinning,
        ZfitData,
        ZfitLoss,
        ZfitModel,
        ZfitParameter,
        ZfitSpace,
    )

    if non_containers is None:
        non_containers = []
    if not isinstance(non_containers, list):
        msg = "`non_containers` have to be a list or a tuple"
        raise TypeError(msg)
    if value is None and not convert_none:
        return value
    if type(value) is not container and non_containers is not False:
        import hist

        non_containers.extend(
            [
                str,
                tf.Tensor,
                np.ndarray,
                ZfitData,
                ZfitLoss,
                ZfitModel,
                ZfitSpace,
                ZfitParameter,
                ZfitBinnedData,
                ZfitBinning,
                PlottableHistogram,
                pd.DataFrame,
                pd.Series,
                hist.axis.Regular,
                hist.axis.Variable,
            ]
        )
        non_containers = tuple(non_containers)
        if ignore is not None and isinstance(value, ignore):
            return value
        try:
            if isinstance(value, non_containers):
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
