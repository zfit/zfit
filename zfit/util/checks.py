#  Copyright (c) 2024 zfit
from __future__ import annotations


class Singleton:
    __instance = None

    def __new__(cls, *_, **__):
        instance = cls.__instance
        if instance is None:
            instance = super().__new__(cls)
            cls.__instance = instance

        return instance


class NotSpecified(Singleton):
    def __bool__(self):
        return False

    def __repr__(self):
        return "None"


NONE = NotSpecified()


class ZfitNotImplemented:
    def __new__(cls, *_, **__):
        msg = "Cannot create an instance of it, meant to be used as a single object reference."
        raise RuntimeError(msg)


class RuntimeDependency:
    def __init__(self, name, how=None, error_msg=None):
        if how is None:
            how = ""
        self.__name = name
        self.__how = how
        self.__error_msg = error_msg

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        raise ImportError(
            (f"Original import error {self.__error_msg}.\n" if self.__error_msg is not None else "")
            + f"This requires {self.__name}"
            " to be installed. You can usually install it with"
            f"`pip install zfit[{self.__name}]` or"
            f"`pip install zfit[all]`."
            f"{self.__how}"
        )
