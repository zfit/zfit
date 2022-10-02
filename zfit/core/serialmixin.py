#  Copyright (c) 2022 zfit
import pydantic


class ZfitSerializable:
    hs3_type = None

    @classmethod
    @property
    def repr(cls) -> pydantic.BaseModel:
        from zfit.serialization import Serializer

        return Serializer.type_repr[cls.hs3_type]


class SerializableMixin(ZfitSerializable):
    def to_json(self):
        orm = self.repr.from_orm(self)
        return orm.json(exclude_none=True, by_alias=True)

    @classmethod
    @property
    def repr(cls):
        try:
            from ..serialization import Serializer
        except ImportError:
            return None
        return Serializer.constructor_repr.get(cls)
