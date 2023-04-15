#  Copyright (c) 2023 zfit

from __future__ import annotations


def make_param_constructor(constructor):
    """Create a constructor for a parameter class avoiding the `NameAlreadyTakenError`.

    If the parameter already exists, it is returned instead of creating a new one.

    Args:
        constructor: Callable that creates the parameter.

    Returns:
        Callable that creates the parameter.
    """

    def param_constructor(name, **kwargs):
        from ..core.parameter import Parameter

        if name in Parameter._existing_params:
            return Parameter._existing_params[name]
        else:
            return constructor(name=name, **kwargs)

    return param_constructor
