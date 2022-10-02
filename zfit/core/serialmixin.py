#  Copyright (c) 2022 zfit
import pydantic


class ZfitSerializable:
    hs3_type: str = None

    @classmethod
    def get_repr(cls) -> pydantic.BaseModel:
        from zfit.serialization import Serializer

        return Serializer.type_repr[cls.hs3_type]


class SerializableMixin(ZfitSerializable):
    def to_json(self):
        orm = self.get_repr().from_orm(self)
        return orm.json(exclude_none=True, by_alias=True)

    @classmethod
    def get_repr(cls):
        try:
            from ..serialization import Serializer
        except ImportError:
            return None
        return Serializer.constructor_repr.get(cls)
