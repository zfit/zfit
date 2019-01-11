from typing import Callable, Any

import tensorflow as tf


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_to_container(value: Any, container: Callable = list, non_containers=None, convert_none=False) -> "container":
    """Convert `value` into a `container` storing `value` if `value` is not yet a python container.

    Args:
        value (object):
        container (callable): Converts a tuple to a container.

    Returns:

    """
    if non_containers is None:
        non_containers = ()
    if not isinstance(non_containers, (list, tuple)):
        raise TypeError("`non_containers` have to be a list or a tuple")
    if value is None and not convert_none:
        return value
    if not isinstance(value, container):
        try:
            if isinstance(value, [str, tf.Tensor].extend(non_containers)):
                raise TypeError
            value = container(value)
        except TypeError:
            value = container((value,))
    return value


def is_container(object):
    """Check if `object` is a list or a tuple.

    Args:
        object ():

    Returns:

    """
    if isinstance(object, (list, tuple)):
        return True
    else:
        return False
