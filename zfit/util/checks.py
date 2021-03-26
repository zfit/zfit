#  Copyright (c) 2021 zfit

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

    def __str__(self):
        return "A not specified state"


NONE = NotSpecified()


class ZfitNotImplemented:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Cannot create an instance of it, meant to be used as a single object reference.")
