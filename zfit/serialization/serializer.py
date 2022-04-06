#  Copyright (c) 2022 zfit


def convert_to_repr(repr):
    cleaned = {}
    for key, val in repr.items():
        if isinstance(val, Serializable):
            val = val.obj_to_repr()
        cleaned[key] = val
    return cleaned


class Serializable:
    def obj_to_repr(self):
        repr_uncleaned = self._obj_to_repr()
        return convert_to_repr(repr_uncleaned)

    def _obj_to_repr(self):
        raise NotImplementedError


class Repr:
    registered_classes = {}

    @classmethod
    def register_serializable(cls, uid: str, constructor):
        if uid in cls.registered_classes:
            raise ValueError(f"uid {uid} already exists.")
        cls.registered_classes[uid] = constructor


def register(uid):
    def func(cls):
        if not issubclass(cls, Serializable):
            raise TypeError(f"Class {cls} has to be subclass of Serializable.")
        Repr.register_serializable(uid=uid, constructor=cls)
        return cls

    return func


# class AtomicSerializable(Serializable):
#
#     def to_repr(self):
