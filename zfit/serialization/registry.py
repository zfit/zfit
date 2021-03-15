#  Copyright (c) 2020 zfit
from typing import Mapping, Set

import bidict

from zfit.serialization.interfaces import ZfitSerializable, ZfitArranger, ZfitRepr

SUPPORTED_VERSIONS: Set[str] = {'0.1.0'}


class SerializationRegistry:
    """Registry to keep a map of the default serialization path for each :py:class:`Zfit.Serializable`."""

    def __init__(self,
                 version: str,
                 serializable_by_uid: Mapping[str, ZfitSerializable],
                 arranger_by_uid: Mapping[str, ZfitArranger],
                 repr_by_uid: Mapping[str, ZfitRepr],
                 supported_versions: Set[str] = SUPPORTED_VERSIONS, ):
        """

        Args:
            version:
            serializable_by_uid:
            arranger_by_uid:
            repr_by_uid:
            supported_versions: Supported version strings.
        """
        if version not in supported_versions:
            raise KeyError(f"{version} is not supported.")
        self.versions[version] = self
        self.serializable_by_uid = bidict.bidict(serializable_by_uid)
        self.arranger_by_uid = bidict.bidict(arranger_by_uid)
        self.repr_by_uid = repr_by_uid

    def register(self, serializable: ZfitSerializable, uid: str):
        self.serializable_by_uid[uid] = serializable
        self.repr_by_uid[uid] = serializable.Repr


default_serialization_registry = SerializationRegistry(version='0.1.0')


def register(serializable: ZfitSerializable, uid: str):
    """"""
    default_serialization_registry
    return serializable
