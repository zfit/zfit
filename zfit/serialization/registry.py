#  Copyright (c) 2020 zfit
from typing import Optional, Mapping
import bidict
from zfit.serialization.interfaces import ZfitSerializable, ZfitArranger


class SerializationRegistry:
    versions = {}

    def __init__(self,
                 version: str,
                 uid_impl_map: Mapping[str, ZfitSerializable],
                 uid_arr_map: Mapping[str, ZfitArranger]):
        super().__init__()
        if version in self.versions:
            raise KeyError(f"{version} already a defined version.")
        self.versions[version] = self
        self.uid_impl_map = bidict.bidict(uid_impl_map)
        self.uid_arr_map = bidict.bidict(uid_arr_map)
        self.uid_repr_map = {}

    def register_implementation(self, impl: ZfitSerializable, uid: str):
        self.uid_impl_map[uid] = impl
        self.uid_repr_map[uid] = impl.Repr
