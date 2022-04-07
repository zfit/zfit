#  Copyright (c) 2022 zfit


class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
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
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "Cannot create an instance of it, meant to be used as a single object reference."
        )


class RuntimeDependency:
    def __init__(self, name, how=None):
        if how is None:
            how = ""
        self.__name = name
        self.__how = how

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        raise ImportError(
            f"This requires {self.__name}"
            " to be installed. You can usually install it with"
            f"`pip install zfit[{self.__name}]` or"
            f"`pip install zfit[all]`."
            f"{self.__how}"
        )
