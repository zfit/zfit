#  Copyright (c) 2024 zfit

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
        from zfit.serialization import Serializer

        previous_existing = Serializer._existing_params

        if (param := previous_existing.get(name)) is None:
            Serializer._existing_params[name] = (param := constructor(name=name, **kwargs))
        return param

    return param_constructor
