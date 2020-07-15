#  Copyright (c) 2020 zfit
from typing import Optional, Mapping
import bidict
from zfit.serialization.interfaces import ZfitSerializable, ZfitArranger


class SerializationRegistry:
    """Registry to keep a map of the default serialization path for each :py:class:`Zfit.Serializable`."""
    versions = {}

    def __init__(self,
                 version: str,
                 uid_impl_map: Mapping[str, ZfitSerializable],
                 uid_arr_map: Mapping[str, ZfitArranger]):
        super().__init__()
        if version in self.versions:
            raise KeyError(f"{version} already exists. Cannot create version.")
        self.versions[version] = self
        self.serializable_by_uid = bidict.bidict(uid_impl_map)
        self.arranger_by_uid = bidict.bidict(uid_arr_map)
        self.repr_by_uid = {}

    def register(self, serializable: ZfitSerializable, uid: str):
        self.serializable_by_uid[uid] = serializable
        self.repr_by_uid[uid] = serializable.Repr
