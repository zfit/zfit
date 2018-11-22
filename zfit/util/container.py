class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_to_container(value, container=list):
    """Convert `value` into a `container` storing `value` if `value` is not yet a python container.

    Args:
        value (object):
        container (callable): Converts an iterable to a container.

    Returns:

    """
    if not is_container(value):
        try:
            if not isinstance(value, str):
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
