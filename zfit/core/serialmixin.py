#  Copyright (c) 2022 zfit
from __future__ import annotations

import pydantic

from zfit.util.warnings import warn_experimental_feature


class ZfitSerializable:
    hs3_type: str = None

    @classmethod
    def get_repr(cls) -> pydantic.BaseModel:
        from zfit.serialization import Serializer

        return Serializer.type_repr[cls.hs3_type]


class SerializableMixin(ZfitSerializable):
    hs3 = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hs3 = type(self).hs3(self)

    def __init_subclass__(cls, **kwargs):
        cls.hs3 = create_HS3(cls)

    @warn_experimental_feature
    def to_json(self):

        from zfit.serialization import Serializer

        Serializer.initialize()
        orm = self.get_repr().from_orm(self)
        return orm.json(exclude_none=True, by_alias=True)

    @warn_experimental_feature
    @classmethod
    def from_json(cls, json):
        from zfit.serialization import Serializer

        Serializer.initialize()
        orm = cls.get_repr().parse_raw(json)
        return orm.to_orm()

    @warn_experimental_feature
    def to_dict(self):
        from zfit.serialization import Serializer

        Serializer.initialize()
        orm = self.get_repr().from_orm(self)
        return orm.dict(exclude_none=True, by_alias=True)

    @warn_experimental_feature
    def from_dict(self, dict_):
        from zfit.serialization import Serializer

        Serializer.initialize()
        orm = self.get_repr().parse_obj(dict_)
        return orm.to_orm()

    @classmethod
    def get_repr(cls):
        try:
            from ..serialization import Serializer
        except ImportError:
            return None
        return Serializer.constructor_repr.get(cls)


class HS3:
    implementation = None

    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.original_init = {}

    def to_json(self):
        orm = self.repr.from_orm(self)
        return orm.json(exclude_none=True, by_alias=True)

    def to_dict(self):
        orm = self.repr.from_orm(self)
        return orm.dict(exclude_none=True, by_alias=True)

    @classmethod
    def from_json(cls, json):
        orm = cls.implementation.get_repr().parse_raw(json)
        return orm.to_orm()

    @property
    def repr(self):
        return self.obj.get_repr()


def create_HS3(cls):
    class HS3Specialized(HS3):
        implementation = cls

    return HS3Specialized
