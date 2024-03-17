#  Copyright (c) 2023 zfit
"""Zfit serialization module.

This module contains the serialization functionality of zfit. It is used to serialize and deserialize
zfit objects to a string representation.
The output format adheres to the HS3 standard (NOT YET, but will be in the future).

The serialization is done via the :py:class:`~zfit.serializer.Serializer` class.
"""

from __future__ import annotations

from .core.serialmixin import ZfitSerializable
from .serialization import Serializer
from .util.exception import WorkInProgressError
from .util.warnings import warn_experimental_feature

__all__ = ["dumps", "loads"]


@warn_experimental_feature
def dumps(obj: ZfitSerializable):
    """Serialize a PDF or a list of PDFs to a JSON string according to the HS3 standard.

    .. warning::
        This is an experimental feature and the API might change in the future. DO NOT RELY ON THE OUTPUT FOR
        ANYTHING ELSE THAN TESTING.

    THIS FUNCTION DOESN'T YET ADHERE TO HS3 (but just as a proxy).

    |@doc:hs3.explain| HS3 is the `HEP Statistics Serialization Standard <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_.
                   It is a JSON/YAML-based serialization that is a
                   coordinated effort of the HEP community to standardize the serialization of statistical models. The standard
                   is still in development and is not yet finalized. This function is experimental and may change in the future. |@docend:hs3.explain|



    Args:
    obj (ZfitSerializable): The object to dump.

    Returns:
    str: The string representation of the object.
    """
    return Serializer.to_hs3(obj)


@warn_experimental_feature
def loads(string: str):
    """Load a zfit object from a string representation in the HS3 format.

    .. warning::
        This is an experimental feature and the API might change in the future. DO NOT RELY ON THE OUTPUT FOR
        ANYTHING ELSE THAN TESTING.

    THIS FUNCTION DOESN'T YET ADHERE TO HS3 (but just as a proxy).

    |@doc:hs3.explain| HS3 is the `HEP Statistics Serialization Standard <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_.
                   It is a JSON/YAML-based serialization that is a
                   coordinated effort of the HEP community to standardize the serialization of statistical models. The standard
                   is still in development and is not yet finalized. This function is experimental and may change in the future. |@docend:hs3.explain|


    Args:
        string (str): The string representation of the object.

    Returns:
        ZfitSerializable: The object.
    """
    return Serializer.from_hs3(string)


def dump(*_, **__):
    raise WorkInProgressError(
        "Not yet implemented, use `dumps` and manually dump the string."
    )


def load(*_, **__):
    raise WorkInProgressError(
        "Not yet implemented, use `loads` and manually load the string."
    )
