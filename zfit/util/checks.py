def isiterable(object):
    """Check if `object` is a list or a tuple.

    Args:
        object ():

    Returns:

    """
    if isinstance(object, (list, tuple)):
        return True
    else:
        return False
